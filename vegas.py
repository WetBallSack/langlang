"""vegas.py — 1-minute Vegas tunnel trend-following backtest (pre-registered design A-1.0, frozen 2026-10-09).

Workflow (each step refuses to run until the previous one is logged in vegas_ledger.jsonl):
    python vegas.py download           # 1m contract + mark klines + funding (dev and holdout; holdout is not read)
    python vegas.py firerate           # counts only (no returns): regime occupancy, touches, triggers, trades
    python vegas.py dev                # returns on 2021-01..2026-10 BTC/SOL: variants, B0, PBO, DSR, selection
    python vegas.py holdout            # ONE run of the selected variant on H1 (BTC 2019-20) + H2 (ETH/XRP/BNB/DOGE/ADA)
    python vegas.py backtest --sym BTCUSDT --start 2024-01-01 --end 2024-07-01 [--set exit=XC]   # ad-hoc, logged

Rules (Document A): signal at 1m bar i close, market entry at bar i+1 open; direction from the 1h EMA144/169 regime
(R1) or 1m EMA576/676 (R2); E-P pullback into the 1m EMA144/169 tunnel or E-B breakout from it; stop
d = max(structural, d_min); exits X-V (Fibonacci thirds), X-C (chandelier) or X-F (2R bracket); 24h time stop;
mark-price stop triggers, stop-first ties, gap fills, funding at archive timestamps. Fees = 0 (assumed).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numba import njit, prange

import langlang as LL
import floop as F

NS_MIN = 60_000_000_000
NS_H = 60 * NS_MIN
LEDGER = "vegas_ledger.jsonl"
HOLDOUT_LOCK = "vegas_holdout_result.json"


# ====================================================================== 0. STATISTICS (self-contained)
EULER = 0.5772156649015329


@njit(cache=True)
def _boot_paths(day_sum, day_cnt, B, mean_block, seed, want_dd):
    """stationary (Politis-Romano) bootstrap over days. want_dd=0: per-trade mean (sum R / sum trades) per path;
    want_dd=1: maximum drawdown in R of the resampled day path"""
    np.random.seed(seed)
    n = len(day_sum)
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty(B)
    for b in range(B):
        i = np.random.randint(0, n)
        s = 0.0
        c = 0.0
        peak = 0.0
        dd = 0.0
        for k in range(n):
            if k > 0 and np.random.random() < p:
                i = np.random.randint(0, n)
            s += day_sum[i]
            c += day_cnt[i]
            if s > peak:
                peak = s
            if peak - s > dd:
                dd = peak - s
            i += 1
            if i >= n:
                i = 0
        out[b] = dd if want_dd else (s / c if c > 0 else 0.0)
    return out


def per_trade_mean_ci(day_sum, day_cnt, B=10_000, mean_block=5, seed=7, q=(5, 95)):
    """mean R per trade with a day-block (stationary) bootstrap: resample days, ratio of sums"""
    day_sum, day_cnt = np.asarray(day_sum, float), np.asarray(day_cnt, float)
    if len(day_sum) == 0 or day_cnt.sum() == 0:
        return np.nan, np.nan, np.nan
    m = _boot_paths(day_sum, day_cnt, int(B), float(mean_block), int(seed), 0)
    return float(day_sum.sum() / day_cnt.sum()), float(np.percentile(m, q[0])), float(np.percentile(m, q[1]))


def boot_max_dd(day_sum, B=10_000, mean_block=5, seed=7, pct=95):
    """95th percentile of the maximum drawdown (in R) over resampled day paths of the same length"""
    day_sum = np.asarray(day_sum, float)
    if len(day_sum) == 0:
        return np.nan
    dds = _boot_paths(day_sum, np.ones(len(day_sum)), int(B), float(mean_block), int(seed), 1)
    return float(np.percentile(dds, pct))


def deflated_sharpe(R, trial_srs, n_trials):
    """Bailey & López de Prado (2014): P(true SR > expected max SR of n_trials zero-skill trials)"""
    from scipy.stats import kurtosis, norm, skew
    R = np.asarray(R, float)
    if len(R) < 30 or R.std(ddof=1) == 0 or len(trial_srs) < 2:
        return np.nan
    sr = R.mean() / R.std(ddof=1)
    N = max(int(n_trials), len(trial_srs), 2)
    var = float(np.var(trial_srs, ddof=1))
    sr0 = math.sqrt(max(var, 1e-12)) * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * math.e)))
    g3, g4 = skew(R), kurtosis(R, fisher=False)
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if den <= 0:
        return np.nan
    return float(norm.cdf((sr - sr0) * math.sqrt(len(R) - 1) / math.sqrt(den)))


def pbo_cscv(M, S=16):
    """probability of backtest overfitting (CSCV) on a T x N matrix of per-period returns"""
    import itertools
    M = np.asarray(M, float)
    if M.ndim != 2:
        return np.nan
    T, N = M.shape
    if N < 2 or T < S * 2:
        return np.nan
    edges = np.linspace(0, T, S + 1).astype(int)
    bsum = np.array([M[edges[i]:edges[i + 1]].sum(0) for i in range(S)])
    bsq = np.array([(M[edges[i]:edges[i + 1]] ** 2).sum(0) for i in range(S)])
    bn = np.diff(edges).astype(float)
    lam = []
    for combo in itertools.combinations(range(S), S // 2):
        m = np.zeros(S, bool)
        m[list(combo)] = True

        def sharpe(mask):
            k = bn[mask].sum()
            mu = bsum[mask].sum(0) / k
            var = bsq[mask].sum(0) / k - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-18))
        s_is, s_oos = sharpe(m), sharpe(~m)
        best = int(np.argmax(s_is))
        rank = (s_oos < s_oos[best]).sum() + 0.5 * ((s_oos == s_oos[best]).sum() - 1) + 1
        w = rank / (N + 1)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.asarray(lam) <= 0))


# ====================================================================== 1. CONFIG
@dataclass
class VConfig:
    cache_dir: str = "./cache"
    vision_workers: int = 8
    n_bench: int = 1000
    # data (Document A §A.2)
    dev_syms: tuple = ("BTCUSDT", "SOLUSDT")
    dev_start: str = "2021-01-01"
    dev_end: str = "2026-10-08"
    h1_syms: tuple = ("BTCUSDT",)
    h1_start: str = "2019-09-01"
    h1_end: str = "2020-12-31"
    h2_syms: tuple = ("ETHUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT")
    h2_start: str = "2019-09-01"
    h2_end: str = "2026-10-08"
    # rules (§A.3)
    entry: str = "EP"            # EP pullback | EB breakout
    regime: str = "R1"           # R1 1h EMA144/169 + 24h slope | R2 1m EMA576/676 + 1440-bar slope
    exit: str = "XV"             # XV Fibonacci thirds | XC chandelier | XF 1R/2R bracket
    session: str = "vegas"       # vegas: no entries when the signal bar closes 17:00-24:00 New York | none
    d_min: float = 0.008
    ema_fast: int = 12
    tun_a: int = 144
    tun_b: int = 169
    slow_a: int = 576
    slow_b: int = 676
    h1_a: int = 144
    h1_b: int = 169
    slope_h: int = 24
    atr_n: int = 60
    atr15_n: int = 14
    prox: float = 0.25
    brk: float = 0.10
    eq_look: int = 60
    touch_look: int = 30
    inval: float = 0.5
    stop_buf: float = 0.10
    xv_t1: float = 1.0
    xv_t2: float = 1.618
    xv_t3: float = 4.236
    xv_ratchet: float = 2.618
    xc_mult: float = 3.0
    xf_target: float = 2.0
    time_stop: int = 1440
    entry_slip: float = 1e-4
    tp_slip: float = 1e-4
    stop_slip: float = 3e-4
    cost_mult: float = 1.0
    fee_side: float = 0.0        # assumed zero (KOL); stress row adds 5 bp/side
    burn_mult: int = 5
    seed: int = 7
    min_dev_trades: int = 5000          # selection / testability floors (Document A §A.7-A.8)
    min_proj_holdout: int = 2000
    min_holdout_trades: int = 2000


VARIANTS = OrderedDict([
    ("V1_breakout",   dict(entry="EB", regime="R1", exit="XV", session="vegas", d_min=0.008)),
    ("V2_pullback",   dict(entry="EP", regime="R1", exit="XV", session="vegas", d_min=0.008)),
    ("V3_chandelier", dict(entry="EP", regime="R1", exit="XC", session="vegas", d_min=0.008)),
    ("V4_fixed_2R",   dict(entry="EP", regime="R1", exit="XF", session="vegas", d_min=0.008)),
    ("V5_no_session", dict(entry="EP", regime="R1", exit="XV", session="none",  d_min=0.008)),
    ("V6_R2_576_676", dict(entry="EP", regime="R2", exit="XV", session="vegas", d_min=0.008)),
    ("V7_dmin_0.40",  dict(entry="EP", regime="R1", exit="XV", session="vegas", d_min=0.004)),
])
V8_NOTE = ("V8_live_vegas_sar: NOT RUNNABLE until the vegas-sar-scalper source is pasted and ported "
           "(Document A §A.4); it still counts as a trial")
EXIT_PREF = {"XF": 0, "XC": 1, "XV": 2}
NEIGHBOUR_PARAMS = ("prox", "brk", "eq_look", "touch_look", "inval", "stop_buf", "d_min", "atr_n", "time_stop",
                    "xc_mult", "xf_target")


def freeze_hash(cfg: VConfig):
    d = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in ("cache_dir", "vision_workers", "n_bench")}
    blob = json.dumps({"cfg": d, "variants": VARIANTS, "design": "A-1.1"}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def ledger_append(kind, **kw):
    rec = dict(time=pd.Timestamp.utcnow().isoformat(), kind=kind, **kw)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def ledger_find(kind, fh):
    if not os.path.exists(LEDGER):
        return None
    hit = None
    for line in open(LEDGER):
        r = json.loads(line)
        if r.get("kind") == kind and r.get("freeze") == fh:
            hit = r
    return hit


def ledger_trials():
    """trials logged so far for this family (every variant run counts); the decision floor is 30"""
    if not os.path.exists(LEDGER):
        return 0
    return sum(int(json.loads(l).get("trials", 0)) for l in open(LEDGER))


def apply_sets(cfg, sets):
    for s in sets or []:
        k, v = s.split("=", 1)
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown setting {k}")
        cur = getattr(cfg, k)
        cfg = replace(cfg, **{k: type(cur)(v) if not isinstance(cur, tuple) else tuple(v.split(","))})
    return cfg


# ====================================================================== 2. DATA
def _rest_month(sym, kind, month, cache_dir):
    """REST fallback for months the archive lacks (e.g. BTCUSDT before 2020)"""
    path = os.path.join(cache_dir, "vegas_rest", f"{sym}_{kind}_{month}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    lo = pd.Timestamp(month + "-01", tz="UTC")
    hi = min(lo + pd.offsets.MonthBegin(1), LL._now().floor("D"))
    ep = "/fapi/v1/klines" if kind == "klines" else "/fapi/v1/markPriceKlines"
    rows, cur = [], int(lo.value // 1_000_000)
    end = int(hi.value // 1_000_000)
    while cur < end:
        raw = F.PUBLIC_GET(ep, dict(symbol=sym, interval="1m", startTime=cur, endTime=end - 1, limit=1500))
        if not raw:
            break
        rows += raw
        cur = int(raw[-1][0]) + 60_000
        if len(raw) < 1500:
            break
    if not rows:
        return None
    a = np.array([[float(x) for x in r[:6]] + [float(r[9]) if len(r) > 9 else 0.0] for r in rows])
    idx = pd.to_datetime(a[:, 0].astype("int64"), unit="ms", utc=True).as_unit("ns")
    df = pd.DataFrame({"open": a[:, 1], "high": a[:, 2], "low": a[:, 3], "close": a[:, 4]}, index=idx)
    if kind == "klines":
        df["volume"], df["tbv"], df["count"] = a[:, 5], a[:, 6], 1.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)
    return df


def load_raw(cfg: VConfig, sym, start, end, rest_fallback=True):
    """1m contract klines, 1m mark klines, funding — from 45 days before start (burn-in) to end"""
    vis = F.FVision(replace(LL.Config(), cache_dir=cfg.cache_dir, vision_workers=cfg.vision_workers))
    months = [LL.mkey(m) for m in LL.month_starts(LL._utc(start) - pd.Timedelta(days=45), LL._utc(end))]
    k = vis.series(sym, "klines", "1m", months)
    mk = vis.series(sym, "mark", "1m", months)
    fu = vis.funding(sym, months)
    if rest_fallback:
        have = set(k.index.strftime("%Y-%m")) if k is not None and len(k) else set()
        miss = [m for m in months if m not in have and m < LL.mkey(LL._now())]
        if miss:
            try:
                extra = [d for m in miss if (d := _rest_month(sym, "klines", m, cfg.cache_dir)) is not None]
                if extra:
                    k = pd.concat(([k] if k is not None else []) + extra).sort_index()
                    k = k[~k.index.duplicated(keep="last")]
                    mx = [d for m in miss if (d := _rest_month(sym, "mark", m, cfg.cache_dir)) is not None]
                    if mx:
                        mk = pd.concat(([mk] if mk is not None else []) + mx).sort_index()
                        mk = mk[~mk.index.duplicated(keep="last")]
            except Exception as e:                                   # noqa: BLE001
                print(f"   {sym}: REST fallback failed ({type(e).__name__}); using archive months only")
    if k is None or k.empty:
        return None
    k = k.loc[(k.index < LL._utc(end) + pd.Timedelta(days=1))]
    return k, mk, fu


@dataclass
class Bars:
    sym: str
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    mo: np.ndarray
    mh: np.ndarray
    ml: np.ndarray
    gap: np.ndarray        # 0 contiguous, 1 small gap (1-4 missing bars), 2 maintenance gap (5 min-24 h), 3 reseed
    fund: np.ndarray       # funding rate applying within the bar (0 if none)
    fprice: np.ndarray     # mark (or contract) price at the funding minute
    month: np.ndarray      # year*12+month of the bar
    eval_from: int         # first bar inside the evaluation window
    eval_to: int           # one past the last bar inside the window
    mark_share: float


def make_bars(sym, raw, start, end):
    k, mk, fu = raw
    k = k[~k.index.duplicated(keep="last")].sort_index()
    t = k.index.as_unit("ns").asi8
    o, h, l, c = (k[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    if mk is not None and len(mk):
        m = mk.reindex(k.index)
        mo, mh, ml = (m[x].to_numpy(float) for x in ("open", "high", "low"))
    else:
        mo = mh = ml = np.full(len(t), np.nan)
    dt = np.diff(t, prepend=t[0] - NS_MIN)
    missing = dt // NS_MIN - 1
    gap = np.zeros(len(t), np.int8)
    gap[(missing >= 1) & (missing < 5)] = 1
    gap[(missing >= 5) & (dt <= 24 * NS_H)] = 2
    gap[dt > 24 * NS_H] = 3
    gap[0] = 3
    fund = np.zeros(len(t))
    fprice = np.zeros(len(t))
    if fu is not None and len(fu):
        ft = fu.index.as_unit("ns").asi8
        j = np.searchsorted(t, ft, side="right") - 1
        ok = (j >= 0) & (ft - t[np.clip(j, 0, None)] < NS_MIN)
        for jj, r in zip(j[ok], fu["rate"].to_numpy(float)[ok]):
            fund[jj] += r
            fprice[jj] = mo[jj] if mo[jj] == mo[jj] else o[jj]
    ts = pd.DatetimeIndex(k.index)
    month = (ts.year * 12 + ts.month).to_numpy(np.int64)
    lo, hi = np.searchsorted(t, LL._utc(start).value), np.searchsorted(t, (LL._utc(end) + pd.Timedelta(days=1)).value)
    return Bars(sym, t, o, h, l, c, mo, mh, ml, gap, fund, fprice, month, int(lo), int(hi),
                float(np.isfinite(mo[lo:hi]).mean()) if hi > lo else 0.0)


# ====================================================================== 3. INDICATORS (TradingView semantics)
@njit(cache=True)
def ema_tv(x, n, seg):
    """Pine ta.ema: ema_0 = x_0, then alpha*x + (1-alpha)*ema; reseeded where seg is set"""
    a = 2.0 / (n + 1.0)
    out = np.empty(len(x))
    e = x[0]
    for i in range(len(x)):
        if seg[i]:
            e = x[i]
        else:
            e = a * x[i] + (1.0 - a) * e
        out[i] = e
    return out


@njit(cache=True)
def rma_tv(x, n, seg):
    """Pine ta.rma: NaN until n values, SMA seed, then (prev*(n-1)+x)/n; reseeded where seg is set"""
    out = np.full(len(x), np.nan)
    s = 0.0
    cnt = 0
    prev = np.nan
    for i in range(len(x)):
        if seg[i]:
            s, cnt, prev = 0.0, 0, np.nan
        if prev != prev:
            s += x[i]
            cnt += 1
            if cnt == n:
                prev = s / n
                out[i] = prev
        else:
            prev = (prev * (n - 1) + x[i]) / n
            out[i] = prev
    return out


@njit(cache=True)
def true_range(h, l, c, seg):
    tr = np.empty(len(h))
    for i in range(len(h)):
        if seg[i]:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return tr


@njit(cache=True)
def seg_age(seg):
    a = np.empty(len(seg), np.int64)
    k = 0
    for i in range(len(seg)):
        k = 0 if seg[i] else k + 1
        a[i] = k
    return a


def _agg(t, o, h, l, c, period_ns):
    key = t // period_ns
    br = np.flatnonzero(np.diff(key)) + 1
    st = np.r_[0, br]
    en = np.r_[br, len(t)]
    return (key[st] * period_ns, o[st], np.maximum.reduceat(h, st), np.minimum.reduceat(l, st), c[en - 1])


def _map_completed(t1m, htf_open, period_ns):
    """index of the last HTF bar completed at the close of each 1m bar"""
    return np.searchsorted(htf_open, t1m + NS_MIN - period_ns, side="right") - 1


@dataclass
class Ind:
    U: np.ndarray
    L: np.ndarray
    M: np.ndarray
    E12: np.ndarray
    atr: np.ndarray
    atr15: np.ndarray
    reg: np.ndarray        # +1 / -1 / 0 (int8)
    sess_ok: np.ndarray    # entries allowed (bool) under the Vegas session rule
    valid: np.ndarray      # all inputs past burn-in


_IND_CACHE = {}


def indicators(b: Bars, cfg: VConfig, regime=None) -> Ind:
    regime = regime or cfg.regime
    key = (b.sym, int(b.t[0]), len(b.t), float(b.c[0]), float(b.c[-1]), float(b.c[len(b.c) // 2]), regime, cfg.atr_n, cfg.ema_fast, cfg.tun_a, cfg.tun_b, cfg.slow_a,
           cfg.slow_b, cfg.h1_a, cfg.h1_b, cfg.slope_h, cfg.atr15_n, cfg.burn_mult)
    if key in _IND_CACHE:
        return _IND_CACHE[key]
    seg = b.gap == 3
    age = seg_age(seg)
    bm = cfg.burn_mult
    ea, eb = ema_tv(b.c, cfg.tun_a, seg), ema_tv(b.c, cfg.tun_b, seg)
    U, L = np.maximum(ea, eb), np.minimum(ea, eb)
    E12 = ema_tv(b.c, cfg.ema_fast, seg)
    atr = rma_tv(true_range(b.h, b.l, b.c, seg), cfg.atr_n, seg)
    valid = (age >= bm * max(cfg.tun_b, cfg.ema_fast)) & np.isfinite(atr)
    # 15m ATR for the chandelier
    t15, o15, h15, l15, c15 = _agg(b.t, b.o, b.h, b.l, b.c, 15 * NS_MIN)
    seg15 = np.r_[True, np.diff(t15) > 24 * NS_H]
    atr15h = rma_tv(true_range(h15, l15, c15, seg15), cfg.atr15_n, seg15)
    j15 = _map_completed(b.t, t15, 15 * NS_MIN)
    atr15 = np.where(j15 >= 0, atr15h[np.clip(j15, 0, None)], np.nan)
    if regime == "R1":
        th, oh, hh, lh, ch = _agg(b.t, b.o, b.h, b.l, b.c, NS_H)
        segh = np.r_[True, np.diff(th) > 24 * NS_H]
        ageh = seg_age(segh)
        e1, e2 = ema_tv(ch, cfg.h1_a, segh), ema_tv(ch, cfg.h1_b, segh)
        lag = cfg.slope_h
        prev = np.r_[np.full(lag, np.nan), e2[:-lag]]
        okh = (ageh >= bm * cfg.h1_b) & (ageh >= lag)
        up = okh & (ch > np.maximum(e1, e2)) & (e2 > prev)
        dn = okh & (ch < np.minimum(e1, e2)) & (e2 < prev)
        rh = np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)
        jh = _map_completed(b.t, th, NS_H)
        reg = np.where(jh >= 0, rh[np.clip(jh, 0, None)], 0).astype(np.int8)
    else:
        e1, e2 = ema_tv(b.c, cfg.slow_a, seg), ema_tv(b.c, cfg.slow_b, seg)
        prev = np.r_[np.full(1440, np.nan), e2[:-1440]]
        ok = (age >= bm * cfg.slow_b) & (age >= 1440)
        up = ok & (b.c > np.maximum(e1, e2)) & (e2 > prev)
        dn = ok & (b.c < np.minimum(e1, e2)) & (e2 < prev)
        reg = np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)
    ny = pd.DatetimeIndex(b.t + NS_MIN, tz="UTC").tz_convert("America/New_York")
    sess_ok = ~np.asarray(ny.hour >= 17)
    out = Ind(U, L, (U + L) / 2, E12, atr, atr15, reg, sess_ok, valid)
    if len(_IND_CACHE) > 24:
        _IND_CACHE.clear()
    _IND_CACHE[key] = out
    return out


# ====================================================================== 4. EXITS (shared by strategy and benchmark)
# float params P: 0 d_min 1 prox 2 brk 3 inval 4 stop_buf 5 t1 6 t2 7 t3 8 ratchet 9 xc_mult 10 xf_target
#                 11 entry_slip 12 tp_slip 13 stop_slip 14 fee_side
# int params IP:  0 entry(0 EB,1 EP) 1 exit(0 XV,1 XC,2 XF) 2 use_session 3 eq_look 4 touch_look 5 time_stop
# exit reasons: 1 stop, 2 stop gapped through at the open, 3 target(s), 4 tunnel violation, 5 time stop,
#               6 maintenance gap, 7 end of data
@njit(cache=True)
def scan_exit(k, s, Pf, P0, d, stop0, o, h, l, c, mo, mh, ml, U, L, atr15, gap, fund, fprice, P, IP):
    n = len(o)
    xmode = IP[1]
    es, tps, ss, fee = P[11], P[12], P[13], P[14]
    stop = stop0
    rem = 1.0
    net = 0.0          # sum frac * s * (fill - Pf)
    gross = 0.0        # sum frac * s * (raw - P0)
    fills_px = 0.0     # sum frac * fill (for fees)
    fcost = 0.0
    tr_open = np.ones(3, np.bool_)
    T = np.empty(3)
    T[0], T[1], T[2] = Pf + s * P[5] * d, Pf + s * P[6] * d, Pf + s * P[7] * d
    TF = Pf + s * P[10] * d
    t1_done = False
    ratchet = False
    pend_viol = False
    pend_time = False
    ext = Pf
    mae, mfe = 0.0, 0.0
    reason = 7
    j = k
    for j in range(k, n):
        if j > k:
            if fund[j] != 0.0:                       # held through this bar's funding timestamp (its start)
                fcost += s * fund[j] * fprice[j] * rem
            if gap[j] >= 2:
                px = o[j] * (1 - s * ss * 3)
                net += rem * s * (px - Pf); gross += rem * s * (o[j] - P0); fills_px += rem * px
                rem = 0.0; reason = 6
                break
            if pend_time:
                px = o[j] * (1 - s * es)
                net += rem * s * (px - Pf); gross += rem * s * (o[j] - P0); fills_px += rem * px
                rem = 0.0; reason = 5
                break
            if pend_viol:
                px = o[j] * (1 - s * es)
                f = 1.0 / 3.0
                net += f * s * (px - Pf); gross += f * s * (o[j] - P0); fills_px += f * px
                rem -= f; tr_open[2] = False; pend_viol = False
                if rem <= 1e-9:
                    reason = 4
                    break
            mop = mo[j] if mo[j] == mo[j] else o[j]
            if s * (mop - stop) <= 0:
                px = o[j] * (1 - s * ss)
                net += rem * s * (px - Pf); gross += rem * s * (o[j] - P0); fills_px += rem * px
                rem = 0.0; reason = 2
                break
        mlo = ml[j] if ml[j] == ml[j] else l[j]
        mhi = mh[j] if mh[j] == mh[j] else h[j]
        if (s > 0 and mlo <= stop) or (s < 0 and mhi >= stop):          # stop first; targets void
            px = stop * (1 - s * ss)
            net += rem * s * (px - Pf); gross += rem * s * (stop - P0); fills_px += rem * px
            rem = 0.0; reason = 1
            break
        best = h[j] if s > 0 else l[j]
        worst = l[j] if s > 0 else h[j]
        if xmode == 2:
            if s * (best - TF) > 0:
                px = TF * (1 - s * tps)
                net += rem * s * (px - Pf); gross += rem * s * (TF - P0); fills_px += rem * px
                rem = 0.0; reason = 3
                break
        elif xmode == 0:
            for q in range(3):
                if tr_open[q] and s * (best - T[q]) > 0:
                    f = 1.0 / 3.0
                    px = T[q] * (1 - s * tps)
                    net += f * s * (px - Pf); gross += f * s * (T[q] - P0); fills_px += f * px
                    rem -= f; tr_open[q] = False
                    if q == 0:
                        t1_done = True
            if rem <= 1e-9:
                reason = 3
                break
        fav = s * (best - Pf) / d
        adv = s * (worst - Pf) / d
        if fav > mfe:
            mfe = fav
        if adv < mae:
            mae = adv
        if s * (best - ext) > 0:
            ext = best
        # close-of-bar updates, effective from the next bar
        if xmode == 0:
            if t1_done and s * (Pf - stop) > 0:
                stop = Pf
            if (not ratchet) and s * (ext - (Pf + s * P[8] * d)) >= 0:
                ns = Pf + s * 1.0 * d
                if s * (ns - stop) > 0:
                    stop = ns
                ratchet = True
            if tr_open[2] and ((s > 0 and c[j] < L[j]) or (s < 0 and c[j] > U[j])):
                pend_viol = True
        elif xmode == 1:
            a15 = atr15[j]
            dist = P[0] * Pf
            if a15 == a15 and P[9] * a15 > dist:
                dist = P[9] * a15
            tr = ext - s * dist
            if s * (tr - stop) > 0:
                stop = tr
        if j - k + 1 >= IP[5]:
            pend_time = True
    if rem > 1e-9:                                                       # end of data
        px = c[n - 1]
        net += rem * s * (px - Pf); gross += rem * s * (px - P0); fills_px += rem * px
        reason = 7
        j = n - 1
    fees = fee * (Pf + fills_px)
    Rnet = (net - fees - fcost) / d
    Rgross = gross / d
    return j, Rnet, Rgross, reason, mae, mfe


# ====================================================================== 5. STRATEGY SIMULATION (one pass)
# fire-rate counters per month: 0 regime long bars 1 regime short bars 2 equilibrium bars 3 touches
# 4 new episodes 5 same-bar touch triggers 6 triggers 7 blocked: in position 8 blocked: session
# 9 blocked: gap/maintenance 10 blocked: next bar missing 11 trades 12 bars evaluated
NCNT = 13


@njit(cache=True)
def simulate(lo, hi, t, o, h, l, c, mo, mh, ml, gap, fund, fprice, month, m0,
             U, L, M, E12, atr, atr15, reg, sess_ok, valid, P, IP, cnt, out):
    n = len(o)
    emode, use_sess, eq_look, tlook = IP[0], IP[2], IP[3], IP[4]
    prox, brk, inval, sbuf, dmin, es = P[1], P[2], P[3], P[4], P[0], P[11]
    armL = False; armS = False
    lastL = -10**9; lastS = -10**9
    extL = 0.0; extS = 0.0
    last_eq = -10**9
    busy = -1
    block = -1
    nt = 0
    for i in range(1, n - 1):
        g = gap[i]
        if g >= 2:
            armL = False; armS = False
            block = i + 240
        elif g == 1:
            if i + 60 > block:
                block = i + 60
        if not valid[i]:
            armL = False; armS = False
            continue
        inwin = i >= lo and i < hi
        mi = month[i] - m0
        r = reg[i]
        if inwin:
            cnt[mi, 12] += 1
            if r == 1:
                cnt[mi, 0] += 1
            elif r == -1:
                cnt[mi, 1] += 1
        a = atr[i]
        sig = 0
        S = 0.0
        if emode == 0:                                                   # E-B breakout
            if abs(E12[i] - M[i]) <= prox * a and abs(c[i] - M[i]) <= prox * a:
                if inwin:
                    cnt[mi, 2] += 1
                eq_prev = last_eq
                last_eq = i
            else:
                eq_prev = last_eq
            if eq_prev >= i - eq_look and eq_prev < i:
                if r == 1 and c[i - 1] <= U[i - 1] and c[i] > U[i] + brk * a:
                    sig = 1; S = L[i] - sbuf * a
                elif r == -1 and c[i - 1] >= L[i - 1] and c[i] < L[i] - brk * a:
                    sig = -1; S = U[i] + sbuf * a
        else:                                                            # E-P pullback state machines
            # long side
            trigL = False
            if r != 1:
                armL = False
            elif c[i] < L[i] - inval * a:
                armL = False
            else:
                if l[i] <= U[i] + prox * a:
                    if inwin:
                        cnt[mi, 3] += 1
                        if not armL:
                            cnt[mi, 4] += 1
                    if not armL or l[i] < extL:
                        extL = l[i]
                    armL = True; lastL = i
                elif armL and l[i] < extL:
                    extL = l[i]
                if armL:
                    if i - lastL <= tlook:
                        if c[i] > max(U[i], E12[i]):
                            trigL = True
                    else:
                        armL = False
            trigS = False
            if r != -1:
                armS = False
            elif c[i] > U[i] + inval * a:
                armS = False
            else:
                if h[i] >= L[i] - prox * a:
                    if inwin:
                        cnt[mi, 3] += 1
                        if not armS:
                            cnt[mi, 4] += 1
                    if not armS or h[i] > extS:
                        extS = h[i]
                    armS = True; lastS = i
                elif armS and h[i] > extS:
                    extS = h[i]
                if armS:
                    if i - lastS <= tlook:
                        if c[i] < min(L[i], E12[i]):
                            trigS = True
                    else:
                        armS = False
            if trigL:
                sig = 1; S = min(L[i], extL) - sbuf * a
                if inwin and lastL == i:
                    cnt[mi, 5] += 1
            elif trigS:
                sig = -1; S = max(U[i], extS) + sbuf * a
                if inwin and lastS == i:
                    cnt[mi, 5] += 1
        if sig == 0 or not inwin:
            continue
        cnt[mi, 6] += 1
        if i < busy:
            cnt[mi, 7] += 1
            continue
        if use_sess == 1 and not sess_ok[i]:
            cnt[mi, 8] += 1
            continue
        if i < block:
            cnt[mi, 9] += 1
            continue
        if gap[i + 1] != 0:
            cnt[mi, 10] += 1
            continue
        k = i + 1
        P0 = o[k]
        Pf = P0 * (1 + sig * es)
        dd = sig * (Pf - S)
        dm = dmin * Pf
        bind = 0
        if dd < dm:
            dd = dm; bind = 1
        stop0 = Pf - sig * dd
        j, Rn, Rg, why, mae, mfe = scan_exit(k, sig, Pf, P0, dd, stop0, o, h, l, c, mo, mh, ml, U, L, atr15,
                                             gap, fund, fprice, P, IP)
        if nt < out.shape[0]:
            out[nt, 0] = i; out[nt, 1] = k; out[nt, 2] = j; out[nt, 3] = sig; out[nt, 4] = Rn; out[nt, 5] = Rg
            out[nt, 6] = why; out[nt, 7] = mae; out[nt, 8] = mfe; out[nt, 9] = dd / Pf; out[nt, 10] = bind
            out[nt, 11] = 1.0 if sess_ok[i] else 0.0
            nt += 1
        cnt[mi, 11] += 1
        busy = j
        if emode == 1:
            if sig == 1:
                armL = False
            else:
                armS = False
    return nt


def params(cfg: VConfig):
    P = np.array([cfg.d_min, cfg.prox, cfg.brk, cfg.inval, cfg.stop_buf, cfg.xv_t1, cfg.xv_t2, cfg.xv_t3,
                  cfg.xv_ratchet, cfg.xc_mult, cfg.xf_target, cfg.entry_slip * cfg.cost_mult,
                  cfg.tp_slip * cfg.cost_mult, cfg.stop_slip * cfg.cost_mult, cfg.fee_side], float)
    IP = np.array([1 if cfg.entry == "EP" else 0, {"XV": 0, "XC": 1, "XF": 2}[cfg.exit],
                   1 if cfg.session == "vegas" else 0, cfg.eq_look, cfg.touch_look, cfg.time_stop], np.int64)
    return P, IP


TRADE_COLS = ["sig_bar", "entry_bar", "exit_bar", "side", "R", "R_gross", "reason", "mae", "mfe", "d_rel",
              "dmin_bind", "sess"]
REASONS = {1: "stop", 2: "stop_gap", 3: "target", 4: "tunnel_viol", 5: "time", 6: "maint_gap", 7: "end"}


def run_symbol(b: Bars, cfg: VConfig):
    ind = indicators(b, cfg)
    P, IP = params(cfg)
    m0 = int(b.month.min())
    cnt = np.zeros((int(b.month.max()) - m0 + 1, NCNT), np.int64)
    out = np.zeros((max(1000, (b.eval_to - b.eval_from) // 20), len(TRADE_COLS)))
    nt = simulate(b.eval_from, b.eval_to, b.t, b.o, b.h, b.l, b.c, b.mo, b.mh, b.ml, b.gap, b.fund, b.fprice,
                  b.month, m0, ind.U, ind.L, ind.M, ind.E12, ind.atr, ind.atr15, ind.reg, ind.sess_ok, ind.valid,
                  P, IP, cnt, out)
    tr = pd.DataFrame(out[:nt], columns=TRADE_COLS)
    for col in ("sig_bar", "entry_bar", "exit_bar", "side", "reason", "dmin_bind"):
        tr[col] = tr[col].astype(np.int64)
    tr["sym"] = b.sym
    tr["entry_time"] = pd.to_datetime(b.t[tr.entry_bar.to_numpy()], utc=True)
    tr["exit_time"] = pd.to_datetime(b.t[tr.exit_bar.to_numpy()], utc=True)
    tr["hold"] = tr.exit_bar - tr.entry_bar + 1
    tr["month"] = b.month[tr.sig_bar.to_numpy()]
    return tr, cnt, m0


# ====================================================================== 6. RANDOM-ENTRY BENCHMARK (B0, B0b)
@njit(cache=True)
def _rng(state):
    state ^= (state << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state ^= state >> np.uint64(7)
    state ^= (state << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state


@njit(cache=True, parallel=True)
def bench(reps, seed, tr_side, tr_drel, tr_hold, tr_group, pool_ptr, pool_items,
          o, h, l, c, mo, mh, ml, U, L, atr15, gap, fund, fprice, P, IP):
    nrep = reps
    meanA = np.zeros(nrep)
    meanB = np.zeros(nrep)
    ntr = len(tr_side)
    es, fee = P[11], P[14]
    n = len(o)
    for r in prange(nrep):
        st = np.uint64(seed) * np.uint64(1000003) + np.uint64(r) * np.uint64(2654435761) + np.uint64(1)
        sa = 0.0
        sb = 0.0
        na = 0
        for q in range(ntr):
            g = tr_group[q]
            a0, a1 = pool_ptr[g], pool_ptr[g + 1]
            if a1 <= a0:
                continue
            st = _rng(st)
            pick = pool_items[a0 + int(st % np.uint64(a1 - a0))]
            if pick + 1 >= n:
                continue
            k = pick + 1
            s = tr_side[q]
            P0 = o[k]
            Pf = P0 * (1 + s * es)
            d = tr_drel[q] * Pf
            j, Rn, Rg, why, mae, mfe = scan_exit(k, s, Pf, P0, d, Pf - s * d, o, h, l, c, mo, mh, ml, U, L,
                                                 atr15, gap, fund, fprice, P, IP)
            sa += Rn
            jx = k + tr_hold[q]
            if jx >= n:
                jx = n - 1
            px = o[jx] * (1 - s * es)
            sb += (s * (px - Pf) - fee * (Pf + px)) / d
            na += 1
        meanA[r] = sa / na if na else 0.0
        meanB[r] = sb / na if na else 0.0
    return meanA, meanB


def bench_pools(b: Bars, ind, cfg: VConfig, tr: pd.DataFrame, match_regime=True):
    """candidate entry bars for each trade's random replacement: same month, same session rule, and (B0) the
    regime pointing the trade's way. Returns CSR pointers, pool items and each trade's pool id."""
    lo, hi = b.eval_from, b.eval_to
    cand = np.arange(lo, hi - 1)
    ok = ind.valid[cand] & (b.gap[cand + 1] == 0)
    if cfg.session == "vegas":
        ok &= ind.sess_ok[cand]
    cand = cand[ok]
    ckey = b.month[cand] * 3 + ((ind.reg[cand].astype(np.int64) + 1) if match_regime else 0)
    srt = np.argsort(ckey, kind="stable")
    ckey, cs = ckey[srt], cand[srt]
    tkey = tr.month.to_numpy().astype(np.int64) * 3 + ((tr.side.to_numpy().astype(np.int64) + 1) if match_regime else 0)
    uk, gid = np.unique(tkey, return_inverse=True)
    lo_ = np.searchsorted(ckey, uk, side="left")
    hi_ = np.searchsorted(ckey, uk, side="right")
    ptr = np.zeros(len(uk) + 1, np.int64)
    ptr[1:] = np.cumsum(hi_ - lo_)
    items = np.concatenate([cs[x:y] for x, y in zip(lo_, hi_)]).astype(np.int64) if len(uk) else np.zeros(0, np.int64)
    return ptr, items, gid.astype(np.int64)


def run_bench(b: Bars, cfg: VConfig, tr: pd.DataFrame, match_regime=True, reps=None):
    """B0: for every real trade, one independent random entry bar matched on month, regime side, session flag and
    direction, with the same relative stop and the same exit module; B0b: same draws, fixed-time exit after the real
    holding time. Returns per-replication mean R. (Change vs design A-1.0, flagged: no sequential overlap rejection —
    with month-wide pools it silently dropped most draws and biased the rest toward month-end; the per-trade null
    does not need it.)"""
    reps = reps or cfg.n_bench
    if tr.empty:
        return np.zeros(0), np.zeros(0)
    ind = indicators(b, cfg)
    P, IP = params(cfg)
    ptr, items, gid = bench_pools(b, ind, cfg, tr, match_regime)
    order = np.argsort(tr.sig_bar.to_numpy(), kind="stable")
    return bench(reps, cfg.seed, tr.side.to_numpy()[order].astype(np.int64), tr.d_rel.to_numpy()[order],
                 tr.hold.to_numpy()[order].astype(np.int64), np.array(gid, np.int64)[order], ptr, items,
                 b.o, b.h, b.l, b.c, b.mo, b.mh, b.ml, ind.U, ind.L, ind.atr15, b.gap, b.fund, b.fprice, P, IP)


# ====================================================================== 7. RUNNING VARIANTS AND STATS
def load_set(cfg: VConfig, syms, start, end, rest_fallback=False):
    out = {}
    for s in syms:
        raw = load_raw(cfg, s, start, end, rest_fallback=rest_fallback)
        if raw is None:
            print(f"   {s}: no data in {start}..{end}")
            continue
        b = make_bars(s, raw, start, end)
        if b.eval_to - b.eval_from < 10_000:
            print(f"   {s}: only {b.eval_to - b.eval_from} bars in window — skipped")
            continue
        out[s] = b
        print(f"   {s}: {b.eval_to - b.eval_from:,} 1m bars in window, mark coverage {b.mark_share:.1%}")
    return out


def run_cfg(bars: dict, cfg: VConfig):
    trs, cnts = [], {}
    for s, b in bars.items():
        tr, cnt, m0 = run_symbol(b, cfg)
        trs.append(tr)
        cnts[s] = (cnt, m0)
    tr = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame(columns=TRADE_COLS + ["sym"])
    return tr, cnts


def summarize(tr: pd.DataFrame):
    if tr is None or len(tr) == 0:
        return dict(n=0, mean_R=np.nan, gross_R=np.nan, PF=np.nan, win=np.nan, median_hold=np.nan, dmin_share=np.nan,
                    top1_share=np.nan, mean_ex_top1=np.nan)
    R = tr.R.to_numpy()
    pos, neg = R[R > 0].sum(), -R[R < 0].sum()
    srt = np.sort(R)[::-1]
    k = max(1, int(len(R) * 0.01))
    return dict(n=len(R), mean_R=float(R.mean()), gross_R=float(tr.R_gross.mean()), PF=float(pos / neg) if neg else np.inf,
                win=float((R > 0).mean()), median_hold=float(tr.hold.median()), dmin_share=float(tr.dmin_bind.mean()),
                top1_share=float(srt[:k].sum() / R.sum()) if R.sum() != 0 else np.nan,
                mean_ex_top1=float(srt[k:].mean()) if len(R) > k else np.nan)


def daily(tr: pd.DataFrame):
    if len(tr) == 0:
        return np.zeros(0), np.zeros(0), pd.DatetimeIndex([])
    d = tr.exit_time.dt.floor("D")
    g = tr.groupby(d).R
    full = pd.date_range(d.min(), d.max(), freq="D")
    s = g.sum().reindex(full, fill_value=0.0)
    c = g.count().reindex(full, fill_value=0)
    return s.to_numpy(), c.to_numpy(), full


def firerate_table(cnts, cfg_name):
    rows = []
    names = ["reg_long", "reg_short", "eq_bars", "touches", "episodes", "same_bar_trig", "triggers",
             "blk_in_position", "blk_session", "blk_gap", "blk_next_bar", "trades", "bars"]
    for s, (cnt, m0) in cnts.items():
        for mi in range(cnt.shape[0]):
            if cnt[mi, 12] == 0:
                continue
            yy, mm = divmod(m0 + mi - 1, 12)
            rows.append(dict(variant=cfg_name, sym=s, month=f"{yy}-{mm + 1:02d}", **dict(zip(names, cnt[mi].tolist()))))
    return pd.DataFrame(rows)


def _cfgs(base: VConfig):
    return OrderedDict((k, replace(base, **v)) for k, v in VARIANTS.items())


def holdout_symbol_years(cfg: VConfig):
    vis = F.FVision(replace(LL.Config(), cache_dir=cfg.cache_dir, vision_workers=cfg.vision_workers))
    yrs = 0.0
    for syms, a, z in ((cfg.h1_syms, cfg.h1_start, cfg.h1_end), (cfg.h2_syms, cfg.h2_start, cfg.h2_end)):
        want = {LL.mkey(m) for m in LL.month_starts(LL._utc(a), LL._utc(z))}
        for s in syms:
            yrs += len(want & vis.avail(s, "klines", "1m")) / 12
    return yrs


# ====================================================================== 8. COMMANDS
def cmd_download(cfg):
    for syms, a, z, rest in ((cfg.dev_syms, cfg.dev_start, cfg.dev_end, False), (cfg.h1_syms, cfg.h1_start, cfg.h1_end, True),
                             (cfg.h2_syms, cfg.h2_start, cfg.h2_end, False)):
        for s in syms:
            t0 = time.time()
            raw = load_raw(cfg, s, a, z, rest_fallback=rest)
            n = 0 if raw is None else len(raw[0])
            print(f"   {s} {a}..{z}: {n:,} 1m bars cached ({time.time() - t0:.0f}s)")   # counts only, no prices shown


def cmd_firerate(cfg):
    fh = freeze_hash(cfg)
    print(f"design A-1.1 freeze {fh} — fire-rate report (counts only; no returns are computed or shown)")
    bars = load_set(cfg, cfg.dev_syms, cfg.dev_start, cfg.dev_end)
    tabs, rows = [], []
    dev_years = sum((b.eval_to - b.eval_from) / 525_600 for b in bars.values())
    hold_years = holdout_symbol_years(cfg)
    for name, c in _cfgs(cfg).items():
        tr, cnts = run_cfg(bars, c)
        tabs.append(firerate_table(cnts, name))
        n = len(tr)
        proj = n / dev_years * hold_years if dev_years else 0
        testable = n >= cfg.min_dev_trades and proj >= cfg.min_proj_holdout
        rows.append(dict(variant=name, trades=n, per_symbol_year=round(n / dev_years, 1) if dev_years else 0,
                         projected_holdout=int(proj), median_hold_min=float(tr.hold.median()) if n else 0,
                         dmin_share=float(tr.dmin_bind.mean()) if n else 0, testable=testable))
    rep = pd.DataFrame(rows)
    full = pd.concat(tabs, ignore_index=True)
    os.makedirs("reports", exist_ok=True)
    full.to_csv("reports/vegas_firerate_monthly.csv", index=False)
    blob = rep.to_json(orient="records") + full.to_json(orient="records")
    rh = hashlib.sha256(blob.encode()).hexdigest()[:16]
    rep.to_json("reports/vegas_firerate.json", orient="records", indent=1)
    print(rep.to_string(index=False))
    tot = full.groupby("variant")[["triggers", "blk_in_position", "blk_session", "blk_gap", "blk_next_bar", "trades"]].sum()
    print("\nwhere triggers went:\n" + tot.to_string())
    print(f"\n{V8_NOTE}")
    print(f"holdout symbol-years available: {hold_years:.1f} (dev {dev_years:.1f})")
    ledger_append("firerate", freeze=fh, report=rh, trials=0)
    print(f"\nlogged fire-rate report {rh} for freeze {fh}. Next: python vegas.py dev")


def dev_eval(cfg, bars, verbose=True):
    cfgs = _cfgs(cfg)
    res = OrderedDict()
    for name, c in cfgs.items():
        tr, _ = run_cfg(bars, c)
        res[name] = dict(cfg=c, trades=tr, s=summarize(tr))
    return res


def cmd_dev(cfg):
    fh = freeze_hash(cfg)
    fr = ledger_find("firerate", fh)
    if fr is None:
        raise SystemExit("run `python vegas.py firerate` first (the fire-rate report must be logged before returns)")
    testable = {r["variant"]: r["testable"] for r in json.load(open("reports/vegas_firerate.json"))}
    print(f"design A-1.1 freeze {fh} — DEV returns on {', '.join(cfg.dev_syms)} {cfg.dev_start}..{cfg.dev_end}")
    bars = load_set(cfg, cfg.dev_syms, cfg.dev_start, cfg.dev_end)
    res = dev_eval(cfg, bars)
    rows = []
    for name, r in res.items():
        c, tr = r["cfg"], r["trades"]
        s = r["s"]
        t2, _ = run_cfg(bars, replace(c, cost_mult=2.0))
        t3, _ = run_cfg(bars, replace(c, cost_mult=3.0))
        tf, _ = run_cfg(bars, replace(c, fee_side=5e-4))
        # pooled B0 mean per replication = trade-weighted mean across symbols
        if len(tr):
            parts = [run_bench(b, c, tr[tr.sym == sy]) for sy, b in bars.items() if (tr.sym == sy).any()]
            w = np.array([(tr.sym == sy).sum() for sy in bars if (tr.sym == sy).any()], float)
            b0 = sum(p[0] * wi for p, wi in zip(parts, w)) / w.sum()
            b0b = sum(p[1] * wi for p, wi in zip(parts, w)) / w.sum()
        else:
            b0 = b0b = np.zeros(1)
        r["b0"] = b0
        rows.append(dict(variant=name, n=s["n"], mean_R=s["mean_R"], gross_R=s["gross_R"], PF=s["PF"], win=s["win"],
                         R_2x=summarize(t2)["mean_R"], R_3x=summarize(t3)["mean_R"], R_fees5bp=summarize(tf)["mean_R"],
                         B0=float(b0.mean()), vs_B0=s["mean_R"] - float(b0.mean()),
                         p_B0=float((b0 >= s["mean_R"]).mean()), B0b=float(b0b.mean()),
                         testable=testable.get(name, False), PF_2x=summarize(t2)["PF"]))
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 200):
        print(tab.to_string(index=False))
    # selection rule (frozen): testable, >= 5,000 trades, PF >= 1.0 at 2x cost; highest mean net R; ties by exit pref
    ok = tab[(tab.testable) & (tab.n >= cfg.min_dev_trades) & (tab.PF_2x >= 1.0)]
    chosen = None
    if len(ok):
        ok = ok.assign(pref=[EXIT_PREF[res[v]["cfg"].exit] for v in ok.variant])
        ok = ok.sort_values(["mean_R", "pref"], ascending=[False, True])
        chosen = ok.variant.iloc[0]
    # PBO on daily net R of testable variants, DSR with the trial floor
    names = [v for v in res if testable.get(v, False) and res[v]["s"]["n"] > 0] or [v for v in res if res[v]["s"]["n"] > 0]
    days = pd.date_range(pd.Timestamp(cfg.dev_start, tz="UTC"), pd.Timestamp(cfg.dev_end, tz="UTC"), freq="D")
    Mx = np.column_stack([res[v]["trades"].groupby(res[v]["trades"].exit_time.dt.floor("D")).R.sum()
                          .reindex(days, fill_value=0.0).to_numpy() for v in names]) if names else np.zeros((0, 0))
    pbo = pbo_cscv(Mx) if len(names) >= 2 else np.nan
    srs = [res[v]["s"]["mean_R"] / res[v]["trades"].R.std(ddof=1) for v in names if res[v]["s"]["n"] > 30]
    n_trials = max(30, ledger_trials() + len(VARIANTS) + 1)          # +1: V8 counts as a trial
    out = dict(freeze=fh, table=tab.to_dict(orient="records"), chosen=chosen, pbo=pbo, n_trials=n_trials)
    if chosen:
        tr = res[chosen]["trades"]
        dsr = deflated_sharpe(tr.R.to_numpy(), srs, n_trials)
        out["dsr"] = dsr
        out["neighbours"] = neighbours(bars, res[chosen]["cfg"])
        out["walk_forward"] = walk_forward(res, cfg, testable)
        print(f"\nselected (frozen rule): {chosen}   PBO={pbo:.2f}   DSR={dsr:.2f} (N={n_trials})")
        nb = out["neighbours"]
        print(f"neighbours ±20%: {sum(v > 0 for v in nb.values())}/{len(nb)} positive, worst {min(nb.values()):+.3f}R")
        print("walk-forward (variant chosen on data before each 6-month fold → its fold mean R): "
              + ", ".join(f"{k}:{v['variant'][:2]} {v['mean_R']:+.3f}" for k, v in out["walk_forward"].items()))
    else:
        print(f"\nno variant meets the selection rule (testable, ≥5,000 trades, PF ≥ 1.0 at 2× cost). PBO={pbo:.2f}")
        print("verdict: the family stops here — there is nothing to send to the holdout.")
    print(V8_NOTE)
    json.dump(out, open("reports/vegas_dev.json", "w"), indent=1, default=float)
    ledger_append("dev", freeze=fh, chosen=chosen, trials=len(VARIANTS) + len(out.get("neighbours", {})))
    if chosen:
        print("Next (ONE run only): python vegas.py holdout")


def neighbours(bars, c: VConfig):
    out = {}
    for p in NEIGHBOUR_PARAMS:
        if p == "xc_mult" and c.exit != "XC" or p == "xf_target" and c.exit != "XF" or p == "eq_look" and c.entry != "EB" \
                or p == "brk" and c.entry != "EB" or p in ("touch_look", "inval") and c.entry != "EP":
            continue
        v = getattr(c, p)
        for f in (0.8, 1.2):
            nv = type(v)(round(v * f)) if isinstance(v, int) else v * f
            tr, _ = run_cfg(bars, replace(c, **{p: nv}))
            out[f"{p}={nv}"] = summarize(tr)["mean_R"]
    return out


def walk_forward(res, cfg, testable):
    folds = OrderedDict()
    a = pd.Timestamp("2021-07-01", tz="UTC")
    end = pd.Timestamp(cfg.dev_end, tz="UTC")
    while a < end:
        z = min(a + pd.DateOffset(months=6), end)
        best, bv = None, -np.inf
        for v, r in res.items():
            tr = r["trades"]
            past = tr[tr.entry_time < a]
            if testable.get(v, False) and len(past) >= min(300, cfg.min_dev_trades) and past.R.mean() > bv:
                best, bv = v, past.R.mean()
        if best:
            tr = res[best]["trades"]
            cur = tr[(tr.entry_time >= a) & (tr.entry_time < z)]
            folds[f"{a:%Y-%m}"] = dict(variant=best, mean_R=float(cur.R.mean()) if len(cur) else float("nan"), n=len(cur))
        a = z
    return folds


def cmd_holdout(cfg, force=False):
    fh = freeze_hash(cfg)
    if ledger_find("dev", fh) is None or not os.path.exists("reports/vegas_dev.json"):
        raise SystemExit("run `python vegas.py dev` first")
    dev = json.load(open("reports/vegas_dev.json"))
    chosen = dev.get("chosen")
    if not chosen:
        raise SystemExit("dev selected no variant — the family is closed; there is no holdout run to do")
    if os.path.exists(HOLDOUT_LOCK) and not force:
        print(open(HOLDOUT_LOCK).read())
        raise SystemExit("the holdout has already been run once (result above). Re-running is a new trial: "
                         "--force-new-trial logs it as such.")
    c = replace(cfg, **VARIANTS[chosen])
    print(f"HOLDOUT — {chosen} (freeze {fh}); this is the single pre-registered run")
    legs = {}
    for leg, syms, a, z, rest in (("H1", c.h1_syms, c.h1_start, c.h1_end, True), ("H2", c.h2_syms, c.h2_start, c.h2_end, False)):
        bars = load_set(c, syms, a, z, rest_fallback=rest)
        tr, _ = run_cfg(bars, c)
        t2, _ = run_cfg(bars, replace(c, cost_mult=2.0))
        parts = [(run_bench(b, c, tr[tr.sym == s])[0], (tr.sym == s).sum()) for s, b in bars.items() if (tr.sym == s).any()]
        b0 = sum(p * w for p, w in parts) / sum(w for _, w in parts) if parts else np.zeros(1)
        legs[leg] = dict(tr=tr, t2=t2, b0=b0, n=len(tr))
    tr = pd.concat([legs[k]["tr"] for k in legs], ignore_index=True)
    t2 = pd.concat([legs[k]["t2"] for k in legs], ignore_index=True)
    s = summarize(tr)
    ds, dc, _ = daily(tr)
    mean, lo5, _ = per_trade_mean_ci(ds, dc, seed=cfg.seed)
    nb = sum(legs[k]["n"] for k in legs)
    b0 = sum(legs[k]["b0"] * legs[k]["n"] for k in legs if legs[k]["n"]) / max(nb, 1)
    yrs = tr.groupby(tr.entry_time.dt.year).R.agg(["mean", "count"])
    yrs = yrs[yrs["count"] >= 100]
    g = OrderedDict()
    leg_ok = all(legs[k]["tr"].R.mean() > 0 for k in legs if legs[k]["n"] > 0)
    g[f"1 ≥{cfg.min_holdout_trades:,} trades; H1 and H2 means > 0"] = (s["n"] >= cfg.min_holdout_trades and leg_ok,
                                                f"{s['n']} trades; " + ", ".join(f"{k} {legs[k]['tr'].R.mean():+.3f} (n={legs[k]['n']})"
                                                                                for k in legs if legs[k]["n"]))
    g["2 mean ≥ +0.03R, day-block lower bound > 0"] = (s["mean_R"] >= 0.03 and lo5 > 0, f"{s['mean_R']:+.3f}R, 5% bound {lo5:+.3f}")
    g["3 PF ≥ 1.08"] = (s["PF"] >= 1.08, f"{s['PF']:.2f}")
    g["4 beats B0 by ≥ 0.03R, p < 0.05"] = (s["mean_R"] - b0.mean() >= 0.03 and (b0 >= s["mean_R"]).mean() < 0.05,
                                           f"Δ {s['mean_R'] - b0.mean():+.3f}R, p {(b0 >= s['mean_R']).mean():.3f}")
    g["5 DSR ≥ 0.95 at N_eff ≥ 30 (dev)"] = (np.isfinite(dev.get("dsr", np.nan)) and dev.get("dsr", 0) >= 0.95,
                                            f"{dev.get('dsr', float('nan')):.2f} (N={dev['n_trials']})")
    g["6 PBO ≤ 0.20 (dev)"] = (np.isfinite(dev["pbo"]) and dev["pbo"] <= 0.20, f"{dev['pbo']:.2f}")
    g["7 positive at 2× cost"] = (summarize(t2)["mean_R"] > 0, f"{summarize(t2)['mean_R']:+.3f}R")
    nbv = list(dev.get("neighbours", {}).values())
    g["8 neighbours ≥75% positive, none < −0.03R (dev)"] = (bool(nbv) and np.mean(np.array(nbv) > 0) >= 0.75 and min(nbv) >= -0.03,
                                                         f"{np.mean(np.array(nbv) > 0):.0%} positive, worst {min(nbv) if nbv else float('nan'):+.3f}")
    g["9 drop top 1% still > 0; ≥4/6 years positive"] = (s["mean_ex_top1"] > 0 and (yrs["mean"] > 0).sum() >= min(4, len(yrs)),
                                                      f"ex-top1% {s['mean_ex_top1']:+.3f}R; {(yrs['mean'] > 0).sum()}/{len(yrs)} years")
    dd95 = boot_max_dd(ds, seed=cfg.seed)
    g["10 bootstrap 95th-pct DD ≤ 25R"] = (dd95 <= 25, f"{dd95:.1f}R")
    passed = all(v[0] for v in g.values())
    lines = [f"{'[PASS]' if v[0] else '[FAIL]'} {k}: {v[1]}" for k, v in g.items()]
    verdict = ("ALL GATES 1-10 PASS → gate 11: 3-month paper trading before any money" if passed else
               "AT LEAST ONE GATE FAILED → the Vegas family is closed (no re-tuning, no new filters)")
    txt = "\n".join([f"holdout {chosen}: n={s['n']} mean {s['mean_R']:+.3f}R (gross {s['gross_R']:+.3f}) PF {s['PF']:.2f} "
                     f"win {s['win']:.1%}; B0 {b0.mean():+.3f}R"] + lines + [f"Verdict: {verdict}"])
    print(txt)
    with open(HOLDOUT_LOCK, "w") as f:
        f.write(txt + "\n")
    ledger_append("holdout", freeze=fh, chosen=chosen, passed=passed, trials=1)


def cmd_backtest(cfg, sym, start, end):
    bars = load_set(cfg, [sym], start, end, rest_fallback=True)
    tr, cnts = run_cfg(bars, cfg)
    s = summarize(tr)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, indent=1))
    if len(tr):
        print(tr.groupby("reason").R.agg(["count", "mean"]).rename(index=REASONS).to_string())
    ledger_append("adhoc", freeze=freeze_hash(cfg), trials=1, sym=sym, start=start, end=end)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "firerate", "dev", "holdout", "backtest"])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--sym", default="BTCUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-07-01")
    ap.add_argument("--force-new-trial", action="store_true")
    a = ap.parse_args(argv)
    cfg = apply_sets(VConfig(), a.set)
    {"download": lambda: cmd_download(cfg), "firerate": lambda: cmd_firerate(cfg), "dev": lambda: cmd_dev(cfg),
     "holdout": lambda: cmd_holdout(cfg, a.force_new_trial),
     "backtest": lambda: cmd_backtest(cfg, a.sym, a.start, a.end)}[a.cmd]()


if __name__ == "__main__":
    main()
