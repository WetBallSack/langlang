"""brooks.py — Al Brooks price action, mechanised and backtested (specification BPA-1.0, frozen 2026-10-09).

    python brooks.py download --market spy|crypto     # SPY via Alpaca (ALPACA_API_KEY / ALPACA_SECRET_KEY), perps via Binance
    python brooks.py claims   --market spy|crypto     # D1-D8: Brooks' probability claims vs two random nulls (dev data)
    python brooks.py firerate --market spy|crypto     # trades per symbol-year per variant (counts only, before returns)
    python brooks.py dev      --market spy|crypto     # V1-V8 + B0/B1 benchmarks + gates 2-7 (+ SPY validation)
    python brooks.py holdout  --market spy|crypto     # ONE pass of the selected variant on the untouched holdout
    python brooks.py backtest --market crypto --sym LTCUSDT --start 2023-01-01 --end 2023-07-01 --variant V1_full

Holdouts: SPY 2023-01..2026-09 (QQQ/IWM replication); crypto 2024-07..2026-09 on the dev basket + FIL/NEAR/UNI/XLM/AAVE.
Blocked (Vegas holdout): ETHUSDT XRPUSDT BNBUSDT DOGEUSDT ADAUSDT. BTCUSDT/SOLUSDT: debugging only.
Fees = 0 (assumed). Slippage per side on stop/market fills: crypto 1 bp (stress 2.5 bp), SPY $0.005 (stress $0.01).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numba import njit

import brooks_core as BC
import brooks_sim as BS
import langlang as LL
import floop as F
import qstats as Q

BLOCKED = {"ETHUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT"}
DEBUG_ONLY = {"BTCUSDT", "SOLUSDT"}
LEDGER = "brooks_ledger.jsonl"
DEFAULT_TICKS = {"LTCUSDT": 0.01, "LINKUSDT": 0.001, "AVAXUSDT": 0.001, "BCHUSDT": 0.01, "TRXUSDT": 0.00001,
                 "DOTUSDT": 0.001, "ATOMUSDT": 0.001, "ETCUSDT": 0.001, "FILUSDT": 0.001, "NEARUSDT": 0.001,
                 "UNIUSDT": 0.001, "XLMUSDT": 0.00001, "AAVEUSDT": 0.01, "BTCUSDT": 0.1, "SOLUSDT": 0.01}


@dataclass
class BConfig:
    cache_dir: str = "./cache"
    vision_workers: int = 8
    interval: str = "5m"
    crypto_dev: tuple = ("LTCUSDT", "LINKUSDT", "AVAXUSDT", "BCHUSDT", "TRXUSDT", "DOTUSDT", "ATOMUSDT", "ETCUSDT")
    crypto_dev_start: str = "2021-01-01"
    crypto_dev_end: str = "2024-06-30"
    crypto_hold_start: str = "2024-07-01"
    crypto_hold_end: str = "2026-09-30"
    crypto_hold_syms: tuple = ("FILUSDT", "NEARUSDT", "UNIUSDT", "XLMUSDT", "AAVEUSDT")
    spy_sym: str = "SPY"
    spy_repl: tuple = ("QQQ", "IWM")
    spy_dev_start: str = "2016-01-01"
    spy_dev_end: str = "2020-12-31"
    spy_val_start: str = "2021-01-01"
    spy_val_end: str = "2022-12-31"
    spy_hold_start: str = "2023-01-01"
    spy_hold_end: str = "2026-09-30"
    slip_crypto: float = 1e-4
    slip_crypto_stress: float = 2.5e-4
    slip_spy: float = 0.005            # dollars per share per side
    slip_spy_stress: float = 0.01
    fee_side: float = 0.0
    n_bench: int = 1000
    seed: int = 7
    min_dev_trades: int = 300
    params: str = ""                   # "key=value,key=value" overrides of brooks_core.DEFAULTS


VARIANTS = OrderedDict([
    ("V1_full",       dict(fam=(1, 1, 1, 1, 1, 1), te=1, scalp=0)),
    ("V2_F1_H2L2",    dict(fam=(1, 0, 0, 0, 0, 0), te=1, scalp=0)),
    ("V3_F2F3_range", dict(fam=(0, 1, 1, 0, 0, 0), te=1, scalp=0)),
    ("V4_F4_reversal", dict(fam=(0, 0, 0, 1, 0, 0), te=1, scalp=0)),
    ("V5_F5_breakout", dict(fam=(0, 0, 0, 0, 1, 0), te=1, scalp=0)),
    ("V6_F6_opening", dict(fam=(0, 0, 0, 0, 0, 1), te=1, scalp=0)),
    ("V7_full_scalp", dict(fam=(1, 1, 1, 1, 1, 1), te=1, scalp=1)),
    ("V8_full_noTE",  dict(fam=(1, 1, 1, 1, 1, 1), te=0, scalp=0)),
])
NEIGHBOURS = {"doji": (0.15, 0.35), "sig_min": (3, 5), "zz_atr": (0.75, 1.5), "dt_tol": (0.1, 0.5), "tr_min": (10, 30),
              "bo_fail_k": (1, 5), "micro_ch": (4, 7), "spike_move": (2.0, 3.0), "tl_break": (0.1, 0.5),
              "strong_br": (0.5, 0.7), "strong_r": (0.8, 1.2), "h_tol": (0.0, 0.25)}


def params_of(cfg):
    over = {}
    for kv in filter(None, cfg.params.split(",")):
        k, v = kv.split("=")
        over[k] = float(v)
    if cfg.interval == "15m":
        over.setdefault("sess_or", 6)
        over.setdefault("tstop_swing", 16)
        over.setdefault("tstop_scalp", 4)
    return over


def freeze_hash(cfg):
    d = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in ("cache_dir", "vision_workers", "n_bench")}
    blob = json.dumps({"cfg": d, "variants": VARIANTS, "defaults": BC.DEFAULTS, "neighbours": NEIGHBOURS,
                       "design": "BPA-1.0"}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def ledger_append(kind, **kw):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(time=pd.Timestamp.utcnow().isoformat(), kind=kind, **kw), default=str) + "\n")


def ledger_find(kind, fh, market):
    if not os.path.exists(LEDGER):
        return None
    hit = None
    for line in open(LEDGER):
        r = json.loads(line)
        if r.get("kind") == kind and r.get("freeze") == fh and r.get("market") == market:
            hit = r
    return hit


def ledger_trials():
    if not os.path.exists(LEDGER):
        return 0
    return sum(int(json.loads(l).get("trials", 0)) for l in open(LEDGER))


# ====================================================================== DATA
def check_universe(syms):
    bad = sorted(set(syms) & BLOCKED)
    if bad:
        raise SystemExit(f"blocked symbols (Vegas holdout): {bad}")


def _vis(cfg):
    return F.FVision(replace(LL.Config(), cache_dir=cfg.cache_dir, vision_workers=cfg.vision_workers))


def binance_bars(cfg, sym, start, end, tf):
    check_universe([sym])
    months = [LL.mkey(m) for m in LL.month_starts(LL._utc(start) - pd.Timedelta(days=40), LL._utc(end))]
    df = _vis(cfg).series(sym, "klines", tf, months)
    if df is None or df.empty:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.loc[df.index < LL._utc(end) + pd.Timedelta(days=1), ["open", "high", "low", "close", "volume"]]


ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"


def _alpaca_get(params):
    import requests
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("set ALPACA_API_KEY and ALPACA_SECRET_KEY (free account at alpaca.markets) to download SPY")
    for k in range(5):
        r = requests.get(ALPACA_URL, params=params, timeout=30,
                         headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(15)
            continue
        if r.status_code in (402, 403) and params.get("feed") == "sip":
            params = dict(params, feed="iex")
            continue
        time.sleep(2 ** k)
    raise IOError(f"Alpaca {r.status_code}: {r.text[:200]}")


ALPACA_GET = _alpaca_get


def alpaca_year(cfg, sym, tf, year):
    path = os.path.join(cfg.cache_dir, "alpaca", f"{sym}_{tf}_{year}.pkl")
    now = LL._now()
    if os.path.exists(path) and (year < now.year or time.time() - os.path.getmtime(path) < 6 * 3600):
        return pd.read_pickle(path)
    rows, token = [], None
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), now - pd.Timedelta(minutes=20))
    while True:
        p = dict(symbols=sym, timeframe=tf, start=f"{year}-01-01T00:00:00Z", end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 limit=10000, adjustment="raw", feed="sip")
        if token:
            p["page_token"] = token
        js = ALPACA_GET(p)
        rows += js.get("bars", {}).get(sym, []) or []
        token = js.get("next_page_token")
        if not token:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], utc=True).dt.as_unit("ns")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
        ["open", "high", "low", "close", "volume"]].astype(float)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)
    return df


def rth(df):
    loc = df.index.tz_convert("America/New_York")
    mins = loc.hour * 60 + loc.minute
    return df[(mins >= 570) & (mins < 960) & (loc.weekday < 5)]


def spy_bars(cfg, sym, start, end, tf):
    y0, y1 = (LL._utc(start) - pd.Timedelta(days=40)).year, LL._utc(end).year
    parts = [d for y in range(y0, y1 + 1) if (d := alpaca_year(cfg, sym, tf, y)) is not None]
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = rth(df)
    return df.loc[(df.index >= LL._utc(start) - pd.Timedelta(days=40)) & (df.index < LL._utc(end) + pd.Timedelta(days=1))]


def resample(df, rule):
    g = df.groupby(df.index.floor(rule))
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                         "volume": g.volume.sum()})


def load_ticks(cfg):
    path = os.path.join(cfg.cache_dir, "brooks_ticks.json")
    t = dict(DEFAULT_TICKS)
    if os.path.exists(path):
        t.update(json.load(open(path)))
    return t


# ====================================================================== PREPARATION
@dataclass
class Prep:
    sym: str
    rth: bool
    t5: np.ndarray
    FE: np.ndarray
    OUT: np.ndarray
    IT: np.ndarray
    GATE: np.ndarray
    EV: dict
    plo: tuple
    o1: np.ndarray
    h1: np.ndarray
    l1: np.ndarray
    c1: np.ndarray
    mlo: np.ndarray
    mhi: np.ndarray
    tick: float
    lo: int
    hi: int
    month: np.ndarray
    bucket: np.ndarray


def bars_for(cfg, market, sym, start, end):
    if market == "spy":
        b1 = spy_bars(cfg, sym, start, end, "1Min")
        b5 = spy_bars(cfg, sym, start, end, "5Min")
        if b5 is None and b1 is not None:
            b5 = resample(b1, "5min")
    else:
        b5 = binance_bars(cfg, sym, start, end, "5m")
        b1 = binance_bars(cfg, sym, start, end, "1m")
    if b5 is None or b5.empty:
        return None, None
    if cfg.interval == "15m":
        b5 = resample(b5, "15min")
    return b5, b1


FEAT_KEYS = ("doji", "strong_br", "strong_cpos", "strong_r", "surprise_r", "sig_overlap")
_FE_CACHE = {}


def prepare_from(cfg, market, sym, b5, b1, start, end, P, tick, cache_key=None):
    fk = None
    if cache_key is not None:
        fk = cache_key + tuple(float(P[BC.pidx(k)]) for k in FEAT_KEYS)
    if fk is not None and fk in _FE_CACHE:
        FE = _FE_CACHE[fk]
    else:
        FE = BC.features(b5, P, "rth" if market == "spy" else "crypto")
        if fk is not None:
            if len(_FE_CACHE) > 24:
                _FE_CACHE.clear()
            _FE_CACHE[fk] = FE
    OUT, IT, GATE, EV = BC.run_engine(FE, P, tick, market == "spy")
    t5 = b5.index.as_unit("ns").asi8
    lo = int(np.searchsorted(t5, LL._utc(start).value))
    hi = int(np.searchsorted(t5, (LL._utc(end) + pd.Timedelta(days=1)).value))
    IT[:lo] = 0.0
    IT[hi:] = 0.0
    if b1 is not None and len(b1):
        t1 = b1.index.as_unit("ns").asi8
        o1, h1, l1, c1 = (b1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    else:
        t1 = np.zeros(0, np.int64)
        o1 = h1 = l1 = c1 = np.zeros(0)
    step = (15 if cfg.interval == "15m" else 5) * 60 * 10**9
    mlo, mhi = BS.minute_map(t5, t1, step)
    idx = pd.DatetimeIndex(b5.index)
    month = (idx.year * 12 + idx.month).to_numpy(np.int64)
    if market == "spy":
        bucket = (FE[:, BC.C_SBAR] // 6).astype(np.int64)
    else:
        bucket = idx.hour.to_numpy(np.int64)
    return Prep(sym, market == "spy", t5, FE, OUT, IT, GATE, EV, BS.pivot_tracks(FE), o1, h1, l1, c1, mlo, mhi,
                float(tick), lo, hi, month, bucket)


_RAW = {}


def prepare(cfg, market, sym, start, end, P):
    key = (market, sym, start, end, cfg.interval)
    if key not in _RAW:
        if len(_RAW) > 16:
            _RAW.clear()
        _RAW[key] = bars_for(cfg, market, sym, start, end)
    b5, b1 = _RAW[key]
    if b5 is None:
        return None
    tick = 0.01 if market == "spy" else load_ticks(cfg).get(sym, 0.0)
    return prepare_from(cfg, market, sym, b5, b1, start, end, P, tick, cache_key=key)


# ====================================================================== SIMULATION + BENCHMARKS
def slip_frac(cfg, pr: Prep, stress=False):
    if pr.rth:
        px = np.nanmedian(pr.FE[pr.lo:pr.hi, BC.C_C]) if pr.hi > pr.lo else 100.0
        return (cfg.slip_spy_stress if stress else cfg.slip_spy) / px
    return cfg.slip_crypto_stress if stress else cfg.slip_crypto


def simulate(cfg, pr: Prep, var: dict, stress=False):
    allow = np.array([bool(x) for x in var["fam"]], np.bool_)
    if not pr.rth:
        allow[5] = False
    out = np.zeros((max(1000, (pr.hi - pr.lo) // 3), BS.NTC))
    FE = pr.FE
    plo, ploi, phi, phii = pr.plo
    nt = BS.simulate(pr.IT, allow, int(var["te"]), bool(var["scalp"]), FE[:, BC.C_O], FE[:, BC.C_H], FE[:, BC.C_L],
                     FE[:, BC.C_C], FE[:, BC.C_ATR], FE[:, BC.C_SB], FE[:, BC.C_SS], FE[:, BC.C_INS], FE[:, BC.C_SLAST],
                     plo, ploi, phi, phii, pr.o1, pr.h1, pr.l1, pr.c1, pr.mlo, pr.mhi, pr.tick,
                     slip_frac(cfg, pr, stress), cfg.fee_side, True, out)
    tr = pd.DataFrame(out[:nt], columns=BS.TRADE_COLS)
    for c in ("fam", "side", "sig_bar", "entry_bar", "exit_bar", "reason"):
        tr[c] = tr[c].astype(np.int64)
    tr["sym"] = pr.sym
    tr["entry_time"] = pd.to_datetime(pr.t5[tr.entry_bar.to_numpy()], utc=True)
    tr["exit_time"] = pd.to_datetime(pr.t5[tr.exit_bar.to_numpy()], utc=True)
    tr["month"] = pr.month[tr.sig_bar.to_numpy()]
    tr["bucket"] = pr.bucket[tr.sig_bar.to_numpy()]
    return tr


def benchmark(cfg, pr: Prep, tr: pd.DataFrame, kind="B0", reps=None):
    """per-replication mean R of random entries with identical exits. B0: same month and time-of-day bucket.
    B1: same month, bars where the family's context gate points the same way (context without the setup)."""
    reps = reps or cfg.n_bench
    if len(tr) == 0:
        return np.zeros(0)
    cand = np.arange(max(pr.lo, 1), min(pr.hi, len(pr.t5) - 2))
    cand = cand[np.isfinite(pr.FE[cand, BC.C_ATR]) & (pr.FE[cand, BC.C_SLAST] == 0)]
    if kind == "B0":
        ckey = pr.month[cand] * 100 + pr.bucket[cand]
        items = cand
        tkey = tr.month.to_numpy() * 100 + tr.bucket.to_numpy()
    else:                                    # a bar can belong to several families' gates
        ks, its = [], []
        for f in range(6):
            g = pr.GATE[cand, f]
            m = g != 0
            ks.append((pr.month[cand[m]] * 10 + f) * 3 + (g[m].astype(np.int64) + 1))
            its.append(cand[m])
        ckey = np.concatenate(ks)
        items = np.concatenate(its)
        tkey = (tr.month.to_numpy() * 10 + tr.fam.to_numpy()) * 3 + (tr.side.to_numpy() + 1)
    order = np.argsort(ckey, kind="stable")
    ckey, items = ckey[order], items[order].astype(np.int64)
    uk, gid = np.unique(tkey, return_inverse=True)
    plo = np.searchsorted(ckey, uk, side="left").astype(np.int64)
    phi = np.searchsorted(ckey, uk, side="right").astype(np.int64)
    FE = pr.FE
    po, poi, ph, phii = pr.plo
    return BS.bench(reps, cfg.seed, tr[BS.TRADE_COLS].to_numpy(float), plo, phi, items, gid.astype(np.int64),
                    FE[:, BC.C_O], FE[:, BC.C_H], FE[:, BC.C_L], FE[:, BC.C_C], FE[:, BC.C_ATR], FE[:, BC.C_SB],
                    FE[:, BC.C_SS], FE[:, BC.C_SLAST], po, poi, ph, phii, pr.o1, pr.h1, pr.l1, pr.c1, pr.mlo,
                    pr.mhi, pr.tick, slip_frac(cfg, pr), cfg.fee_side)


def summarize(tr):
    if tr is None or len(tr) == 0:
        return dict(n=0, mean_R=np.nan, gross_R=np.nan, PF=np.nan, win=np.nan)
    R = tr.R.to_numpy()
    pos, neg = R[R > 0].sum(), -R[R < 0].sum()
    return dict(n=len(R), mean_R=float(R.mean()), gross_R=float(tr.R_gross.mean()),
                PF=float(pos / neg) if neg > 0 else np.inf, win=float((R > 0).mean()))


def day_stats(tr):
    if len(tr) == 0:
        return np.zeros(0), np.zeros(0)
    d = tr.exit_time.dt.floor("D")
    g = tr.groupby(d).R
    full = pd.date_range(d.min(), d.max(), freq="D")
    return g.sum().reindex(full, fill_value=0.0).to_numpy(), g.count().reindex(full, fill_value=0).to_numpy()


# ====================================================================== PERIODS
def periods(cfg, market, which):
    if market == "spy":
        return {"dev": ([cfg.spy_sym], cfg.spy_dev_start, cfg.spy_dev_end),
                "val": ([cfg.spy_sym], cfg.spy_val_start, cfg.spy_val_end),
                "hold": ([cfg.spy_sym], cfg.spy_hold_start, cfg.spy_hold_end),
                "repl": (list(cfg.spy_repl), cfg.spy_hold_start, cfg.spy_hold_end)}[which]
    return {"dev": (list(cfg.crypto_dev), cfg.crypto_dev_start, cfg.crypto_dev_end),
            "hold": (list(cfg.crypto_dev), cfg.crypto_hold_start, cfg.crypto_hold_end),
            "repl": (list(cfg.crypto_hold_syms), cfg.crypto_dev_start, cfg.crypto_hold_end)}[which]


def preps(cfg, market, which, P):
    syms, a, z = periods(cfg, market, which)
    check_universe(syms)
    out = []
    for s in syms:
        t0 = time.time()
        pr = prepare(cfg, market, s, a, z, P)
        if pr is None or pr.hi - pr.lo < 2000:
            print(f"   {s}: no/insufficient data for {a}..{z}")
            continue
        print(f"   {s} {a}..{z}: {pr.hi - pr.lo:,} bars, 1m coverage {np.mean(pr.mhi[pr.lo:pr.hi] > pr.mlo[pr.lo:pr.hi]):.0%}"
              f" ({time.time() - t0:.0f}s)")
        out.append(pr)
    return out


def run_variant(cfg, prs, var, stress=False):
    trs = [simulate(cfg, pr, var, stress) for pr in prs]
    return pd.concat(trs, ignore_index=True) if trs else pd.DataFrame(columns=BS.TRADE_COLS)


def pooled_bench(cfg, prs, tr, kind):
    parts, w = [], []
    for pr in prs:
        sub = tr[tr.sym == pr.sym]
        if len(sub):
            parts.append(benchmark(cfg, pr, sub, kind))
            w.append(len(sub))
    if not parts:
        return np.zeros(1)
    w = np.array(w, float)
    return sum(p * wi for p, wi in zip(parts, w)) / w.sum()


# ====================================================================== CLAIMS (D1-D8) — strategy-independent
@njit(cache=True)
def race(H, L, start, side, entry, stop, k, maxb):
    """+k·R before -1R from bar `start` (stop first on ties). 1 = target, 0 = stop, -1 = neither within maxb"""
    R = abs(entry - stop)
    tgt = entry + side * k * R
    for b in range(start, min(start + maxb, len(H))):
        if (side > 0 and L[b] <= stop) or (side < 0 and H[b] >= stop):
            return 0
        if (side > 0 and H[b] >= tgt) or (side < 0 and L[b] <= tgt):
            return 1
    return -1


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def claims_from(pr: Prep):
    """measure every claim on one prepared series; returns {claim: (successes, trials)}"""
    FE, OUT, EV = pr.FE, pr.OUT, pr.EV
    H, L, C, A = FE[:, BC.C_H], FE[:, BC.C_L], FE[:, BC.C_C], FE[:, BC.C_ATR]
    n = len(C)
    lo, hi = pr.lo, pr.hi
    inwin = lambda t: lo <= t < hi - 30
    res = {}

    def add(key, ok):
        a, b = res.get(key, (0, 0))
        res[key] = (a + int(ok), b + 1)
    # D1 trading-range breakout failure (80% broad, ~50% tight/breakout mode)
    for row in EV["BO"]:
        t, side, bh, bl, broad, tight = int(row[0]), int(row[1]), row[2], row[3], row[4], row[5]
        if not inwin(t):
            continue
        hgt = bh - bl
        lvl = bh if side > 0 else bl
        for K in (5, 10, 20):
            fail = False
            for b in range(t + 1, min(t + 1 + K, n)):
                if (side > 0 and H[b] >= lvl + hgt) or (side < 0 and L[b] <= lvl - hgt):
                    break
                if (side > 0 and C[b] < bh) or (side < 0 and C[b] > bl):
                    fail = True
                    break
            kind = "broad" if broad > 0 else ("tight" if tight > 0 else "mid")
            add(f"D1 TR breakout fails within {K} bars ({kind} range)", fail)
    # D2 reversal attempts in trends fail (80%); minor reversal becomes an opposite trend (~20%)
    for row in EV["REV"]:
        t, side, entry, ext, a = int(row[0]), int(row[1]), row[2], row[3], row[4]
        if not inwin(t) or t + 1 >= n:
            continue
        if not ((side < 0 and L[t + 1] <= entry) or (side > 0 and H[t + 1] >= entry)):
            continue
        fail, done = False, False
        for b in range(t + 1, min(t + 100, n)):
            if (side < 0 and H[b] > ext) or (side > 0 and L[b] < ext):
                fail, done = True, True
                break
            if (side < 0 and L[b] <= entry - a) or (side > 0 and H[b] >= entry + a):
                done = True
                break
        if done:
            add("D2 trend reversal attempt fails (new extreme before 1 ATR)", fail)
        flip = np.any(OUT[t + 1:min(t + 21, n), 1] == side)
        far = (np.min(L[t + 1:min(t + 51, n)]) <= entry - 2 * a) if side < 0 else (np.max(H[t + 1:min(t + 51, n)]) >= entry + 2 * a)
        add("D2b minor reversal grows into an opposite trend (AI flips, 2 ATR)", bool(flip and far))
    # D3 H/L counts: +1R and +2R before -1R
    for row in EV["H"]:
        sig, side, nl, nb, entry, stop, wai, st, emat, code = row
        sig = int(sig)
        if not inwin(sig):
            continue
        pb = int(code // 10)
        for k, lab in ((1, "1R"), (2, "2R")):
            r = race(H, L, sig + 1, int(side), entry, stop, k, 288)
            if r < 0:
                continue
            ctx = "with-trend" if wai > 0 else "counter/none"
            if 1 <= nl <= 4:
                add(f"D3 {'H' if side > 0 else 'L'}{int(nl)} (leg count) {ctx} +{lab} first", r == 1)
            if nb == 2:
                add(f"D3 bar-count 2 {ctx} +{lab} first", r == 1)
            if nl == 2 and wai > 0:
                add(f"D3 H2/L2 with-trend, pullback {'<20' if pb < 20 else '>=20'} bars +{lab} first", r == 1)
                add(f"D3 H2/L2 with-trend, EMA test={int(emat)} +{lab} first", r == 1)
    # D4 measured moves: leg1 = leg2 from zig-zag swings
    sw = EV["SW"]
    for i in range(2, len(sw)):
        a_, b_, c_ = sw[i - 2], sw[i - 1], sw[i]
        if not (a_[1] == -1 and b_[1] == 1 and c_[1] == -1):
            continue
        tconf = int(c_[3])
        if not inwin(tconf):
            continue
        tgt = c_[2] + (b_[2] - a_[2])
        hit = None
        for b in range(tconf + 1, min(tconf + 288, n)):
            if L[b] < c_[2]:
                hit = False
                break
            if H[b] >= tgt:
                hit = True
                break
        if hit is not None:
            add("D4 leg1=leg2 measured move hit before leg-2 start", hit)
    for row in EV["BO"]:
        t, side, bh, bl = int(row[0]), int(row[1]), row[2], row[3]
        if not inwin(t) or t + 1 >= n:
            continue
        hgt = bh - bl
        lvl = bh if side > 0 else bl
        ft = (C[t + 1] > FE[t + 1, BC.C_O]) if side > 0 else (C[t + 1] < FE[t + 1, BC.C_O])
        hit = None
        for b in range(t + 1, min(t + 288, n)):
            if (side > 0 and L[b] < lvl) or (side < 0 and H[b] > lvl):
                hit = False
                break
            if (side > 0 and H[b] >= lvl + hgt) or (side < 0 and L[b] <= lvl - hgt):
                hit = True
                break
        if hit is not None:
            add(f"D4 TR-height measured move hit ({'with' if ft else 'no'} follow-through)", hit)
    # D5 channels: break of the channel bottom before a 1-height extension above (proxy for the 75%/25% claim)
    for row in EV["CH"]:
        t, side, top, bot = int(row[0]), int(row[1]), row[2], row[3]
        if not inwin(t) or top <= bot:
            continue
        hgt = top - bot
        out_ = None
        for b in range(t + 1, min(t + 288, n)):
            if (side > 0 and C[b] < bot) or (side < 0 and C[b] > top):
                out_ = True
                break
            if (side > 0 and H[b] >= top + hgt) or (side < 0 and L[b] <= bot - hgt):
                out_ = False
                break
        if out_ is not None:
            add("D5 channel breaks against the trend before a 1-height extension", out_)
    # D6 micro channels: ~10 bars is a climax -> TR or opposite 1 ATR within 20 bars
    for row in EV["MC"]:
        t, side, ln, x, a, flag = int(row[0]), int(row[1]), int(row[2]), row[3], row[4], row[5]
        if not inwin(t):
            continue
        seg = slice(t + 1, min(t + 21, n))
        opp = (np.min(L[seg]) <= x - a) if side > 0 else (np.max(H[seg]) >= x + a)
        tr_ = np.any(OUT[seg, 0] == 0)
        add(f"D6 micro channel of {ln} bars → TR or 1-ATR reversal within 20", bool(opp or tr_))
    # D8 MTR and final flag: +2R before -1R
    for key, rows in (("MTR", EV["MTR"]), ("final flag", EV["FF"])):
        for row in rows:
            t, side, entry, stop = int(row[0]), int(row[1]), row[2], row[3]
            if not inwin(t) or t + 1 >= n:
                continue
            if not ((side > 0 and H[t + 1] >= entry) or (side < 0 and L[t + 1] <= entry)):
                continue
            r = race(H, L, t + 1, side, entry, stop, 2, 288)
            if r >= 0:
                add(f"D8 {key} +2R before -1R", r == 1)
    # D7 session statistics (US sessions)
    if pr.rth:
        day = FE[:, BC.C_DAY].astype(np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
        ends = np.r_[starts[1:], len(day)]
        for d0, d1 in zip(starts, ends):
            if d0 < lo or d1 > hi or d1 - d0 < 70:
                continue
            idx = np.arange(d0, d1)
            hh, ll = H[idx], L[idx]
            ih, il = int(np.argmax(hh)), int(np.argmin(ll))
            first = min(ih, il)
            for kk in (1, 7, 12, 18):
                add(f"D7 day's high or low set by bar {kk}", first <= kk - 1)
            b1h, b1l = hh[0], ll[0]
            up = np.flatnonzero(hh[1:] > b1h)
            dn = np.flatnonzero(ll[1:] < b1l)
            fu = up[0] + 1 if len(up) else 10**9
            fd = dn[0] + 1 if len(dn) else 10**9
            if min(fu, fd) < 10**9:
                if fu < fd:
                    add("D7 first breakout of bar 1 reverses (opposite side broken later)", fd < 10**9)
                else:
                    add("D7 first breakout of bar 1 reverses (opposite side broken later)", fu < 10**9)
            if len(idx) > 20:
                h18, l18 = hh[:18].max(), ll[:18].min()
                rest_h, rest_l = hh[18:], ll[18:]
                bu = np.flatnonzero(rest_h > h18)
                bd = np.flatnonzero(rest_l < l18)
                if len(bu) or len(bd):
                    if len(bu) and (not len(bd) or bu[0] < bd[0]):
                        add("D7 after the 18-bar range breaks, the opposite extreme holds", rest_l[bu[0]:].min() >= l18)
                    else:
                        add("D7 after the 18-bar range breaks, the opposite extreme holds", rest_h[bd[0]:].max() <= h18)
            a0 = np.nanmean(A[idx])
            tfo_up = il <= 2 and C[idx[-1]] > FE[idx[0], BC.C_O] + 2 * a0
            tfo_dn = ih <= 2 and C[idx[-1]] < FE[idx[0], BC.C_O] - 2 * a0
            add("D7 trend-from-the-open day", tfo_up or tfo_dn)
    return res


BROOKS_BANDS = [("D1 TR breakout fails within 10 bars (broad range)", 0.70, 0.90, "80% of TR breakouts fail"),
                ("D1 TR breakout fails within 10 bars (tight range)", 0.40, 0.60, "~50% in breakout mode"),
                ("D2 trend reversal attempt fails (new extreme before 1 ATR)", 0.70, 0.90, "80% of reversals fail"),
                ("D2b minor reversal grows into an opposite trend (AI flips, 2 ATR)", 0.10, 0.30, "~20% become trends"),
                ("D3 H2 (leg count) with-trend +1R first", 0.55, 0.70, "with-trend ~60% for 1R"),
                ("D3 L2 (leg count) with-trend +1R first", 0.55, 0.70, "with-trend ~60% for 1R"),
                ("D3 H2 (leg count) with-trend +2R first", 0.35, 0.50, "~40% for 2R"),
                ("D4 TR-height measured move hit (with follow-through)", 0.60, 0.80, "~70% after strong breakout"),
                ("D5 channel breaks against the trend before a 1-height extension", 0.65, 0.85, "75% break the trend line"),
                ("D6 micro channel of 10 bars → TR or 1-ATR reversal within 20", 0.60, 1.00, "10-bar micro channel = climax"),
                ("D8 MTR +2R before -1R", 0.35, 0.50, "MTR ~40%"),
                ("D8 final flag +2R before -1R", 0.35, 0.50, "final flag ~40%"),
                ("D7 day's high or low set by bar 1", 0.15, 0.30, "bar 1 ~20%"),
                ("D7 day's high or low set by bar 7", 0.40, 0.60, "50% by bar 7"),
                ("D7 day's high or low set by bar 12", 0.60, 0.80, "70% by bar 12"),
                ("D7 day's high or low set by bar 18", 0.85, 0.95, "90% by bar 18"),
                ("D7 first breakout of bar 1 reverses (opposite side broken later)", 0.40, 0.60, "50% of opening moves reverse"),
                ("D7 after the 18-bar range breaks, the opposite extreme holds", 0.70, 0.80, "70-80% hold"),
                ("D7 trend-from-the-open day", 0.15, 0.25, "~20% of days")]


def null_paths(b1: pd.DataFrame, kind, seed, rth=False, block=None):
    """1m null paths with the original timestamps: 'boot' = stationary blocks of whole 1m bars (relative OHLC),
    'rw' = Gaussian random walk with the same 1m volatility and symmetric wicks"""
    rng = np.random.default_rng(seed)
    o, h, l, c = (b1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    prev = np.r_[o[0], c[:-1]]
    g = np.log(o / prev); ho = np.log(h / o); lo_ = np.log(l / o); co = np.log(c / o)
    n = len(o)
    if kind == "boot":
        blk = block or (30 if rth else 360)
        idx = np.empty(n, np.int64)
        k = 0
        while k < n:
            s = rng.integers(0, n - blk)
            m = min(blk, n - k)
            idx[k:k + m] = np.arange(s, s + m)
            k += m
        G, HO, LO, CO = g[idx], ho[idx], lo_[idx], co[idx]
        if rth:                                                   # no overnight gap pasted inside the session
            G = np.where(np.abs(G) > 5 * np.std(g), 0.0, G)
    else:
        sd = np.std(co)                                           # intrabar volatility (no overnight gaps)
        CO = rng.normal(0, sd, n)
        G = np.zeros(n)
        HO = np.maximum(CO, 0) + np.abs(rng.normal(0, sd / 2, n))
        LO = np.minimum(CO, 0) - np.abs(rng.normal(0, sd / 2, n))
    oo = np.empty(n); cc = np.empty(n)
    px = o[0]
    for i in range(n):
        oo[i] = px * math.exp(G[i])
        cc[i] = oo[i] * math.exp(CO[i])
        px = cc[i]
    return pd.DataFrame({"open": oo, "high": oo * np.exp(np.maximum(HO, np.maximum(CO, 0))),
                         "low": oo * np.exp(np.minimum(LO, np.minimum(CO, 0))), "close": cc,
                         "volume": b1["volume"].to_numpy(float)}, index=b1.index)


def cmd_claims(cfg, market):
    P = BC.pvec(**params_of(cfg))
    syms, a, z = periods(cfg, market, "dev")
    check_universe(syms)
    obs, nb, nr = {}, {}, {}
    for s in syms:
        b5, b1 = bars_for(cfg, market, s, a, z)
        if b5 is None:
            continue
        tick = 0.01 if market == "spy" else load_ticks(cfg).get(s, 0.0)
        pr = prepare_from(cfg, market, s, b5, b1, a, z, P, tick)
        for k, v in claims_from(pr).items():
            obs[k] = tuple(np.add(obs.get(k, (0, 0)), v))
        if b1 is not None and len(b1):
            rule = "15min" if cfg.interval == "15m" else "5min"
            for kind, acc in (("boot", nb), ("rw", nr)):
                p1 = null_paths(b1, kind, cfg.seed, rth=(market == "spy"))
                p5 = resample(p1, rule)
                if market == "spy":
                    p5 = p5.loc[p5.index.isin(b5.index)]
                prn = prepare_from(cfg, market, s, p5, None, a, z, P, tick)
                for k, v in claims_from(prn).items():
                    acc[k] = tuple(np.add(acc.get(k, (0, 0)), v))
        print(f"   {s}: measured")
    rows = []
    for key in sorted(obs):
        k, n_ = obs[key]
        p, lo, hi = wilson(k, n_)
        pb = wilson(*nb.get(key, (0, 0)))[0]
        pr_ = wilson(*nr.get(key, (0, 0)))[0]
        band = next(((b0, b1_, why) for kk, b0, b1_, why in BROOKS_BANDS if kk == key), None)

        def differs(other):
            k2, n2 = other.get(key, (0, 0))
            if n2 == 0 or n_ == 0:
                return False
            p2 = k2 / n2
            pp = (k + k2) / (n_ + n2)
            se = math.sqrt(max(pp * (1 - pp) * (1 / n_ + 1 / n2), 1e-12))
            return abs(p - p2) / se > 2.58
        verdict = ""
        if band:
            inside = band[0] <= p <= band[1]
            verdict = ("CONFIRMED" if inside and differs(nb) and differs(nr) else
                       "in band but = random" if inside else "OUTSIDE band")
        rows.append(dict(claim=key, n=n_, observed=p, ci_lo=lo, ci_hi=hi, null_boot=pb, null_rw=pr_,
                         brooks=f"{band[2]} [{band[0]:.0%}-{band[1]:.0%}]" if band else "", verdict=verdict))
    tab = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    tab.to_csv(f"reports/brooks_claims_{market}.csv", index=False)
    with pd.option_context("display.max_rows", 200, "display.width", 250, "display.max_colwidth", 70,
                           "display.float_format", "{:.3f}".format):
        print(tab.to_string(index=False))
    ledger_append("claims", market=market, freeze=freeze_hash(cfg), trials=0)
    return tab


# ====================================================================== EXPERIMENT
def _vars_for(market):
    return OrderedDict((k, v) for k, v in VARIANTS.items() if market == "spy" or k != "V6_F6_opening")


def cmd_firerate(cfg, market):
    fh = freeze_hash(cfg)
    P = BC.pvec(**params_of(cfg))
    prs = preps(cfg, market, "dev", P)
    yrs = sum((pr.hi - pr.lo) for pr in prs) / ((252 * 78) if market == "spy" else 105_120) * (3 if cfg.interval == "15m" else 1)
    rows = []
    for name, var in _vars_for(market).items():
        tr = run_variant(cfg, prs, var)
        by = tr.groupby("fam").size().to_dict() if len(tr) else {}
        rows.append(dict(variant=name, trades=len(tr), per_symbol_year=round(len(tr) / max(yrs, 1e-9), 1),
                         testable=len(tr) >= cfg.min_dev_trades,
                         **{BC.FAM[f]: int(by.get(f, 0)) for f in range(6)}))
    rep = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    rep.to_json(f"reports/brooks_firerate_{market}.json", orient="records", indent=1)
    print(rep.to_string(index=False))
    rh = hashlib.sha256(rep.to_json().encode()).hexdigest()[:16]
    ledger_append("firerate", market=market, freeze=fh, report=rh, trials=0)
    print(f"\nlogged fire-rate report {rh} (freeze {fh}). Next: python brooks.py dev --market {market}")


def neighbour_runs(cfg, market, prs_dev, var):
    out = {}
    base = params_of(cfg)
    for k, vals in NEIGHBOURS.items():
        for v in vals:
            over = dict(base, **{k: v})
            P = BC.pvec(**over)
            prs = [prepare(cfg, market, pr.sym, *periods(cfg, market, "dev")[1:], P) for pr in prs_dev]
            tr = run_variant(cfg, [p for p in prs if p is not None], var)
            out[f"{k}={v}"] = summarize(tr)["mean_R"]
    return out


def cmd_dev(cfg, market):
    fh = freeze_hash(cfg)
    if ledger_find("firerate", fh, market) is None:
        raise SystemExit(f"run `python brooks.py firerate --market {market}` first (counts before returns)")
    fr = {r["variant"]: r for r in json.load(open(f"reports/brooks_firerate_{market}.json"))}
    P = BC.pvec(**params_of(cfg))
    print(f"BPA-1.0 freeze {fh} — DEV ({market})")
    prs = preps(cfg, market, "dev", P)
    res = OrderedDict()
    for name, var in _vars_for(market).items():
        tr = run_variant(cfg, prs, var)
        st = summarize(tr)
        t_s = run_variant(cfg, prs, var, stress=True)
        b0 = pooled_bench(cfg, prs, tr, "B0")
        b1 = pooled_bench(cfg, prs, tr, "B1")
        ds, dc = day_stats(tr)
        m, lo, hi = Q.per_trade_mean_ci(ds, dc, B=2000, mean_block=1, seed=cfg.seed, q=(2.5, 97.5))
        res[name] = dict(var=var, tr=tr, s=st, stress=summarize(t_s)["mean_R"], b0=b0, b1=b1, ci=(lo, hi))
    days = pd.date_range(*[LL._utc(x) for x in periods(cfg, market, "dev")[1:]], freq="D")
    names = [k for k in res if res[k]["s"]["n"] > 0]
    M = np.column_stack([res[k]["tr"].groupby(res[k]["tr"].exit_time.dt.floor("D")).R.sum()
                         .reindex(days, fill_value=0.0).to_numpy() for k in names]) if names else np.zeros((0, 0))
    pbo = Q.pbo_cscv(M) if len(names) >= 2 else np.nan
    srs = [res[k]["s"]["mean_R"] / res[k]["tr"].R.std(ddof=1) for k in names if res[k]["s"]["n"] > 30]
    n_trials = max(30, ledger_trials() + len(VARIANTS) + 2 * len(NEIGHBOURS))
    rows = []
    for name, r in res.items():
        s = r["s"]
        rows.append(dict(variant=name, n=s["n"], mean_R=s["mean_R"], gross_R=s["gross_R"], PF=s["PF"], win=s["win"],
                         B0_95=float(np.percentile(r["b0"], 95)) if len(r["b0"]) else np.nan, B1=float(np.mean(r["b1"])),
                         ci_lo=r["ci"][0], stress_R=r["stress"],
                         g1=fr.get(name, {}).get("testable", False),
                         g2=s["mean_R"] > np.percentile(r["b0"], 95) and s["mean_R"] > np.mean(r["b1"]) if s["n"] else False,
                         g5=r["ci"][0] > 0, g6=r["stress"] > 0))
    tab = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.float_format", "{:+.3f}".format):
        print(tab.to_string(index=False))
    cand = tab[tab.g1 & tab.g2 & tab.g5 & tab.g6]
    chosen, gates = None, OrderedDict()
    if len(cand):
        best = cand.sort_values("mean_R", ascending=False).iloc[0]
        chosen = best.variant
        dsr = Q.deflated_sharpe(res[chosen]["tr"].R.to_numpy(), srs, n_trials)
        nb = neighbour_runs(cfg, market, prs, res[chosen]["var"])
        frac_pos = float(np.mean(np.array(list(nb.values())) > 0)) if nb else 0.0
        gates["3 DSR > 0.95"] = (np.isfinite(dsr) and dsr > 0.95, f"{dsr:.2f} (N={n_trials})")
        gates["4 PBO < 0.20"] = (np.isfinite(pbo) and pbo < 0.2, f"{pbo:.2f}")
        gates["7 ≥70% of neighbours keep a positive expectancy"] = (frac_pos >= 0.7, f"{frac_pos:.0%} of {len(nb)}")
        if market == "spy":
            prv = preps(cfg, market, "val", P)
            tv = run_variant(cfg, prv, res[chosen]["var"])
            sv = summarize(tv)
            gates["7b validation 2021-22: same sign, ≥50% of dev effect"] = (
                sv["n"] > 0 and sv["mean_R"] >= 0.5 * best.mean_R, f"{sv['mean_R']:+.3f}R (n={sv['n']}) vs dev {best.mean_R:+.3f}R")
        passed = all(v[0] for v in gates.values())
        print(f"\ncandidate (passed gates 1, 2, 5, 6): {chosen}")
        for k, (ok, d) in gates.items():
            print(f"   [{'PASS' if ok else 'FAIL'}] {k}: {d}")
        if not passed:
            print("Verdict: the selected variant fails a dev gate → stop here (no holdout run).")
            chosen = None
    else:
        print(f"\nno variant passes gates 1 (≥{cfg.min_dev_trades} trades), 2 (> B0 95th pct and > B1), "
              f"5 (day-block CI > 0) and 6 (cost stress). PBO={pbo:.2f}")
        print("Verdict: the mechanised Brooks rules show no edge on dev data → stop (no holdout run).")
    os.makedirs("reports", exist_ok=True)
    tab.to_csv(f"reports/brooks_dev_{market}.csv", index=False)
    json.dump(dict(freeze=fh, chosen=chosen, pbo=pbo, dev_mean=float(tab.set_index("variant").mean_R.get(chosen, np.nan))
                   if chosen else None, gates={k: [bool(v[0]), v[1]] for k, v in gates.items()}),
              open(f"reports/brooks_dev_{market}.json", "w"), indent=1, default=float)
    ledger_append("dev", market=market, freeze=fh, chosen=chosen, trials=len(VARIANTS) + (2 * len(NEIGHBOURS) if chosen else 0))
    if chosen:
        print(f"Next (ONE run only): python brooks.py holdout --market {market}")


def cmd_holdout(cfg, market, force=False):
    fh = freeze_hash(cfg)
    lock = f"brooks_holdout_{market}.txt"
    if ledger_find("dev", fh, market) is None:
        raise SystemExit("run dev first")
    dev = json.load(open(f"reports/brooks_dev_{market}.json"))
    if not dev.get("chosen"):
        raise SystemExit("dev selected nothing — the Brooks family is closed for this market")
    if os.path.exists(lock) and not force:
        print(open(lock).read())
        raise SystemExit("holdout already run once; --force-new-trial logs a re-run as a new trial")
    P = BC.pvec(**params_of(cfg))
    var = VARIANTS[dev["chosen"]]
    lines = [f"HOLDOUT {market} — {dev['chosen']} (freeze {fh})"]
    ok_all = True
    for leg in ("hold", "repl"):
        prs = preps(cfg, market, leg, P)
        tr = run_variant(cfg, prs, var)
        s = summarize(tr)
        b0 = pooled_bench(cfg, prs, tr, "B0")
        good = s["n"] > 0 and s["mean_R"] >= 0.5 * dev["dev_mean"] and s["mean_R"] > np.percentile(b0, 95)
        ok_all &= good
        lines.append(f"[{'PASS' if good else 'FAIL'}] {leg}: n={s['n']} mean {s['mean_R']:+.3f}R (gross {s['gross_R']:+.3f}) "
                     f"PF {s['PF']:.2f}; B0 95th {np.percentile(b0, 95):+.3f}; needs ≥ {0.5 * dev['dev_mean']:+.3f}")
    lines.append("Verdict: " + ("holdout passes → 3-month paper trading before any money" if ok_all else
                               "holdout fails → the mechanised Brooks rules are closed for this market"))
    txt = "\n".join(lines)
    print(txt)
    open(lock, "w").write(txt + "\n")
    ledger_append("holdout", market=market, freeze=fh, passed=bool(ok_all), trials=1)


def cmd_download(cfg, market):
    for which in (("dev", "val", "hold", "repl") if market == "spy" else ("dev", "hold", "repl")):
        syms, a, z = periods(cfg, market, which)
        check_universe(syms)
        for s in syms:
            t0 = time.time()
            b5, b1 = bars_for(cfg, market, s, a, z)
            print(f"   {s} {a}..{z}: {0 if b5 is None else len(b5):,} {cfg.interval} bars, "
                  f"{0 if b1 is None else len(b1):,} 1m bars ({time.time() - t0:.0f}s)")


def cmd_ticks(cfg):
    info = F.PUBLIC_GET("/fapi/v1/exchangeInfo", {})
    want = set(cfg.crypto_dev) | set(cfg.crypto_hold_syms)
    ticks = {sy["symbol"]: float(f["tickSize"]) for sy in info.get("symbols", []) if sy["symbol"] in want
             for f in sy["filters"] if f["filterType"] == "PRICE_FILTER"}
    os.makedirs(cfg.cache_dir, exist_ok=True)
    json.dump(ticks, open(os.path.join(cfg.cache_dir, "brooks_ticks.json"), "w"), indent=1)
    print(ticks)


def cmd_backtest(cfg, market, sym, start, end, variant):
    P = BC.pvec(**params_of(cfg))
    pr = prepare(cfg, market, sym, start, end, P)
    tr = simulate(cfg, pr, VARIANTS[variant])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summarize(tr).items()}))
    if len(tr):
        print(tr.groupby(tr.fam.map(lambda f: BC.FAM[f])).R.agg(["count", "mean"]).to_string())
    ledger_append("adhoc", market=market, freeze=freeze_hash(cfg), trials=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "claims", "firerate", "dev", "holdout", "backtest", "ticks"])
    ap.add_argument("--market", choices=["spy", "crypto"], default="crypto")
    ap.add_argument("--sym", default="LTCUSDT")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-07-01")
    ap.add_argument("--variant", default="V1_full")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--force-new-trial", action="store_true")
    a = ap.parse_args(argv)
    cfg = BConfig()
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        cfg = replace(cfg, **{k: tuple(v.split(",")) if isinstance(cur, tuple) else type(cur)(v)})
    {"download": lambda: cmd_download(cfg, a.market), "claims": lambda: cmd_claims(cfg, a.market),
     "firerate": lambda: cmd_firerate(cfg, a.market), "dev": lambda: cmd_dev(cfg, a.market),
     "holdout": lambda: cmd_holdout(cfg, a.market, a.force_new_trial), "ticks": lambda: cmd_ticks(cfg),
     "backtest": lambda: cmd_backtest(cfg, a.market, a.sym, a.start, a.end, a.variant)}[a.cmd]()


if __name__ == "__main__":
    main()
