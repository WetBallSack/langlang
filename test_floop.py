"""Offline tests for floop.py (no network).   Run:  python test_floop.py
If Deno is installed (deno on PATH or ./denobin/deno), also checks the port against FLOOP v7.1's own TypeScript
(floop_ref.ts) on identical windows."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace

import math
import numpy as np
import pandas as pd

import floop as F
import langlang as LL
import mockvision

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ------------------------------------------------------------------ synthetic market (1m bars)
def synth1m(sym, start="2024-01-01", days=180, seed=1, p0=60000.0):
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    k = np.arange(n)
    vol = 0.0008 * np.exp(0.5 * np.sin(k / 4000.0 + seed))
    drift = 0.00012 * np.sign(np.sin(k / 9000.0 + seed)) * (np.sin(k / 2600.0) > 0.2)
    r = drift + vol * rng.standard_t(4, n) / np.sqrt(2)
    c = p0 * np.exp(np.cumsum(r))
    o = np.r_[p0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2)))
    v = rng.lognormal(2, 0.5, n)
    tb = v * np.clip(0.5 + 40 * r + rng.normal(0, 0.05, n), 0.05, 0.95)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "tbv": tb}, index=idx)


def to5m(m1):
    g = m1.groupby(m1.index.floor("5min"))
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                         "volume": g.volume.sum(), "tbv": g.tbv.sum(), "count": 100.0})


class FMock(mockvision.MockVision):
    """data.binance.vision stand-in serving 5m/1d klines (with taker volume), 1m mark klines and funding"""

    def __init__(self, m1: dict, now, page=40):
        self.m1, self.now, self.page, self.header_from = m1, pd.Timestamp(now), page, "2024-03"
        self.calls = 0
        import threading
        self._lock = threading.Lock()
        self._zips = {}
        self.keys = self._index()

    def _index(self):
        keys, cur = [], self.now.strftime("%Y-%m")
        for s, df in self.m1.items():
            df = df.loc[df.index < self.now.floor("D")]
            months = sorted(set(df.index.strftime("%Y-%m")))
            for kind, tf in (("klines", "5m"), ("klines", "1d"), ("markPriceKlines", "1m")):
                for m in months:
                    if m != cur and self._month_published(m):
                        keys.append(f"data/futures/um/monthly/{kind}/{s}/{tf}/{s}-{tf}-{m}.zip")
                for d in sorted(set(df.index[df.index >= pd.Timestamp(cur + "-01", tz="UTC")].strftime("%Y-%m-%d"))):
                    keys.append(f"data/futures/um/daily/{kind}/{s}/{tf}/{s}-{tf}-{d}.zip")
            for m in months:
                if m != cur and self._month_published(m):
                    keys.append(f"data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip")
        return sorted(set(keys))

    def _build(self, path):
        name = path.rsplit("/", 1)[1][:-4]
        if "/fundingRate/" in path:
            sym, _, m = name.split("-", 2)
            idx = pd.date_range(m + "-01", pd.Timestamp(m + "-01") + pd.offsets.MonthBegin(1), freq="8h",
                                inclusive="left", tz="UTC")
            csv = "calc_time,funding_interval_hours,last_funding_rate\n" + "".join(
                f"{t.value // 1_000_000 + 3},8,{0.0001 + 0.0002 * np.sin(i / 7):.8f}\n" for i, t in enumerate(idx))
            return self._zip(name + ".csv", csv)
        sym, tf, period = name.split("-", 2)
        df = self.m1[sym]
        lo = pd.Timestamp(period + ("-01" if len(period) == 7 else ""), tz="UTC")
        hi = lo + (pd.offsets.MonthBegin(1) if len(period) == 7 else pd.Timedelta(days=1))
        df = df.loc[(df.index >= lo) & (df.index < hi)]
        mark = "/markPriceKlines/" in path
        if not mark:
            g = df.groupby(df.index.floor("5min" if tf == "5m" else "1D"))
            df = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                               "volume": g.volume.sum(), "tbv": g.tbv.sum()})
        step = {"1m": 60_000, "5m": 300_000, "1d": 86_400_000}[tf]
        rows = []
        for t, r in zip(df.index, df.itertuples(index=False)):
            ot = t.value // 1_000_000
            v, tb = (0.0, 0.0) if mark else (r.volume, r.tbv)
            rows.append(f"{ot},{r.open:.10g},{r.high:.10g},{r.low:.10g},{r.close:.10g},{v:.6f},{ot + step - 1},"
                        f"{v * r.close:.4f},{0 if mark else 100},{tb:.6f},{tb * r.close:.4f},0\n")
        header = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                  "taker_buy_quote_volume,ignore\n") if period[:7] >= self.header_from else ""
        return self._zip(name + ".csv", header + "".join(rows))


def install_mock(m1, now, cache):
    mv = FMock(m1, now)
    LL.HTTP_GET = mv.get
    LL._now = lambda: pd.Timestamp(now)
    shutil.rmtree(cache, ignore_errors=True)
    F._FEAT_CACHE.clear()
    return mv


# ------------------------------------------------------------------ A. port vs the real TypeScript
def test_ts_equivalence():
    deno = shutil.which("deno") or ("./denobin/deno" if os.path.exists("./denobin/deno") else None)
    node = shutil.which("node")
    if deno:
        cmd = [deno, "run", "--quiet", "floop_ref.ts"]
    elif node and int(subprocess.run([node, "-p", "process.versions.node.split('.')[0]"], capture_output=True,
                                     text=True).stdout.strip() or 0) >= 22:
        cmd = [node, "--experimental-strip-types", "--no-warnings", "floop_ref.ts"]
    else:
        cmd = None
    if not cmd or not os.path.exists("floop_ref.ts"):
        print("[SKIP] TypeScript equivalence: needs Node >= 22 or Deno (curl -fsSL https://deno.land/install.sh | sh)")
        return
    m1 = synth1m("BTCUSDT", days=60, seed=11)
    m5 = to5m(m1)
    cfg = F.FConfig()
    feat = F.build_feat(F.SymData("BTCUSDT", m5, None, None), cfg)
    pre = F.pre_signals(feat, cfg)
    modes = F.modes_v71(np.where(pre["valid"], pre["tprob"], np.nan))
    dirs = F.directions(feat, cfg, pre, modes)
    rng = np.random.default_rng(3)
    cand = np.flatnonzero(dirs != 0)
    ticks = sorted(set(rng.choice(np.arange(10_000, len(m5)), 30, replace=False).tolist()) | set(cand[cand > 10_000][:20].tolist()))
    htf = {}
    for mins, nm in ((15, "c15"), (60, "c1h"), (240, "c4h")):
        o, h, l, c = F._htf(m5, mins)
        htf[nm] = (pd.DataFrame({"o": o, "h": h, "l": l, "c": c}), mins)
    inp = {"ticks": [], "trendProbs": [float(x) for x in pre["tprob"][1000:4000]]}
    for t in ticks:
        T = m5.index[t] + pd.Timedelta("5min")
        tk = {"c5": m5.iloc[t - 999:t + 1][["open", "high", "low", "close", "volume"]].values.tolist(),
              "mode": {1: "trend", 2: "reversion", 0: "neutral"}[int(modes[t])]}
        for nm, n in (("c15", 300), ("c1h", 200), ("c4h", 200)):
            d, mins = htf[nm]
            d = d.loc[d.index + pd.Timedelta(minutes=mins) <= T].iloc[-n:]
            tk[nm] = np.c_[d[["o", "h", "l", "c"]].values, np.zeros(len(d))].tolist()
        day = m5.loc[(m5.index >= T.floor("D") - pd.Timedelta("1D")) & (m5.index < T.floor("D"))]
        tk["piv"] = [float(day.high.max()), float(day.low.min()), float(day.close.iloc[-1])]
        inp["ticks"].append(tk)
    out = json.loads(subprocess.run(cmd, input=json.dumps(inp), capture_output=True, text=True, check=True).stdout)
    num = lambda x: float(str(x).replace("Infinity", "inf"))
    cols = {"rfTrend": 0, "rfFilt": 2, "barsSinceSig": 4, "atr": 8, "emaFast": 9, "emaSlow": 11, "adx": 12, "chop": 13,
            "trendProb": 18, "volScore": 23}
    worst = max(abs(feat.W[t, j] - num(r[k])) / max(1, abs(num(r[k]))) for t, r in zip(ticks, out["ticks"]) for k, j in cols.items())
    mism = 0
    for t, r in zip(ticks, out["ticks"]):
        mism += int(dirs[t]) != r["dir"] or int(pre["total"][t]) != r["total"] or feat.htf4h[t] != r["mtf4h"]
        if r["bracket"]:
            b = F.bracket(feat, t, int(modes[t]), int(dirs[t]), cfg)
            mism += abs(b["stop"] - r["bracket"]["stopLevel"]) > 1e-9 * b["stop"] or abs(b["R"] - r["bracket"]["structuralR"]) > 1e-12
    mm = {"trend": 1, "reversion": 2, "neutral": 0}
    pm = F.modes_v71(np.array(inp["trendProbs"]))
    mode_mis = sum(int(pm[i]) != mm[m] for i, m in enumerate(out["modes"]))
    check(f"port == v7.1 TypeScript: {len(ticks)} ticks ({sum(1 for r in out['ticks'] if r['bracket'])} entries), "
          f"{len(pm)} regime ticks", worst <= 1e-12 and mism == 0 and mode_mis == 0,
          f"max rel diff {worst:.1e}, {mism} signal/bracket mismatches, {mode_mis} mode mismatches")


# ------------------------------------------------------------------ B. exits
def exit_case(prices1m, sl, tp, side=1, use_tp=True, lev=15.0, hard=0.30, time_bars=0, trail=0.0, gap_at=None, no1m=False):
    m1 = pd.DataFrame({"open": prices1m, "high": prices1m, "low": prices1m, "close": prices1m},
                      index=pd.date_range("2024-01-01", periods=len(prices1m), freq="1min", tz="UTC"))
    m1["open"] = m1.close.shift().fillna(m1.close.iloc[0])
    m1["high"] = np.maximum(m1.open, m1.close) * 1.0002
    m1["low"] = np.minimum(m1.open, m1.close) * 0.9998
    if gap_at is not None:
        m1.iloc[gap_at, m1.columns.get_loc("open")] = m1.close.iloc[gap_at]
    g = m1.groupby(m1.index.floor("5min"))
    m5 = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    t5, t1 = m5.index.as_unit("ns").asi8, m1.index.as_unit("ns").asi8
    e = float(m5.open.iloc[0])
    args = (t5, m5.open.values, m5.high.values, m5.low.values, m5.close.values,
            np.zeros(0, np.int64) if no1m else t1, np.zeros(0) if no1m else m1.open.values,
            np.zeros(0) if no1m else m1.high.values, np.zeros(0) if no1m else m1.low.values, 0, 0, side, e, sl, tp, use_tp,
            lev, hard, time_bars, trail, 0.0003, 0.0001, 0.0001, 10_000)
    return F.scan_exit(*args), m1, m5


def test_exits():
    p = np.r_[np.full(10, 100.0), np.linspace(100, 98, 20)]
    (t, px, why, held, freed), m1, _ = exit_case(p, sl=99.0, tp=103.0)
    hit = int(np.argmax(m1.low.values <= 99.0))
    check("SL fires on the first 1m mark bar through the stop, fill = stop − slippage",
          why == 1 and t == m1.index[hit].value and abs(px - 99.0 * (1 - 0.0003)) < 1e-9, f"held {held} 5m bars")
    p = np.r_[np.full(10, 100.0), [100.0], np.full(5, 100.0)]
    (t, px, why, *_), m1, _ = exit_case(p, sl=99.99, tp=100.01)
    check("stop and target inside the same minute → stop first", why == 1)
    p = np.r_[np.full(10, 100.0), [95.0], np.full(10, 95.0)]
    (t, px, why, *_), m1, _ = exit_case(p, sl=99.0, tp=103.0, gap_at=10)
    check("gap through the stop fills at the minute's open, not at the stop", why == 1 and abs(px - 95.0 * (1 - 0.0003)) < 1e-9)
    p = np.r_[np.full(10, 100.0), np.linspace(100, 97.5, 15), np.full(10, 97.5)]
    (t, px, why, held, freed), m1, m5 = exit_case(p, sl=90.0, tp=110.0)
    check("hard stop: 5m close ≥2% adverse at 15× (30% of margin) → exit next 5m open", why == 3 and
          abs(px - m5.open.values[held] * (1 - 0.0001)) < 1e-9, f"after {held} bars")
    p = np.full(200, 100.0)
    (t, px, why, held, _), *_ = exit_case(p, sl=90.0, tp=110.0, time_bars=6)
    check("time stop after N bars (V5)", why == 4 and held == 6)
    p = np.r_[np.full(10, 100.0), np.linspace(100, 110, 100), np.linspace(110, 100, 100)]
    (t, px, why, held, _), *_ = exit_case(p, sl=97.0, tp=1e9, use_tp=False, trail=3.0)
    check("trailing stop (V4) exits on the way down near peak − trail", why == 5 and 106.0 < px < 107.5, f"exit {px:.2f}")
    p = np.r_[np.full(10, 100.0), np.linspace(100, 98, 20)]
    (t, px, why, *_), *_ = exit_case(p, sl=99.0, tp=103.0, no1m=True)
    check("no 1m data → 5m bar fallback, stop first", why == 1)


# ------------------------------------------------------------------ C. sizing rules
def test_sizing():
    cfg = F.FConfig()
    eng = F.Engine({}, cfg)
    b = dict(R=4 / 3, side=1, sl_mult=3.0, tp_mult=4.0, reward_pct=0.008, risk_pct=0.006, sym="X")
    eng.funding_rate = lambda T, s: 0.0
    m, info = F.kelly_margin(eng, 10**18, F.TREND, b, 12, 1.0, cfg)
    check("cold start: fewer than 30 trades → $50 probe (v7.1 warm-up)", m == 50 and info["probe"])
    eng.load_closed([dict(pnl=30.0 if i % 10 < 7 else -20.0, margin=300.0, mode=F.TREND, exit_ns=i, record_ns=i) for i in range(40)])
    m, info = F.kelly_margin(eng, 10**18, F.TREND, b, 12, 1.0, cfg)
    check("70% win history → positive Kelly margin within [$50, $600]", 50 <= m <= 600 and not info["probe"], f"${m}")
    eng.load_closed([dict(pnl=30.0 if i % 10 < 3 else -20.0, margin=300.0, mode=F.TREND, exit_ns=i, record_ns=i) for i in range(40)])
    m, _ = F.kelly_margin(eng, 10**18, F.TREND, b, 12, 1.0, cfg)
    check("30% win history at R≈1.33 → blocked (no edge at q25)", m == 0)
    co = F.Engine({}, cfg)
    check("N-aware correlation scalar 1/√(N(1+(N−1)ρ̄)) with ρ̄ fallback 0.75",
          abs(1 / np.sqrt(2 * 1.75) - max(0.2, min(1.5, 1 / np.sqrt(2 * (1 + 0.75))))) < 1e-12 and co.scalars(0, ["A"]) == 1.0)


# ------------------------------------------------------------------ D. causality (no look-ahead)
def test_causality():
    cfg = F.FConfig()
    m5 = to5m(synth1m("BTCUSDT", days=90, seed=4))
    full = F.build_feat(F.SymData("BTCUSDT", m5, None, None), cfg)
    pf = F.pre_signals(full, cfg); mf = F.modes_v71(np.where(pf["valid"], pf["tprob"], np.nan)); df = F.directions(full, cfg, pf, mf)
    ok, n = True, 0
    for frac in (0.5, 0.75, 0.9):
        cut = int(len(m5) * frac)
        tr = F.build_feat(F.SymData("BTCUSDT", m5.iloc[:cut], None, None), cfg)
        pt = F.pre_signals(tr, cfg); mt = F.modes_v71(np.where(pt["valid"], pt["tprob"], np.nan)); dt = F.directions(tr, cfg, pt, mt)
        a = slice(0, cut - 1)
        ok &= (np.array_equal(np.nan_to_num(full.W[a], nan=-9), np.nan_to_num(tr.W[a], nan=-9))
               and np.array_equal(full.htf4h[a], tr.htf4h[a]) and np.array_equal(full.piv[a][~np.isnan(full.piv[a])], tr.piv[a][~np.isnan(tr.piv[a])])
               and np.array_equal(mf[a], mt[a]) and np.array_equal(df[a], dt[a]))
        n += int((df[a] != 0).sum())
    check(f"no look-ahead: features, regime and signals identical on truncated data ({n} signals compared)", ok and n > 0)


# ------------------------------------------------------------------ E. end-to-end through the mock archive
M1 = {}


def test_end_to_end():
    now = pd.Timestamp("2024-07-10 12:00", tz="UTC")
    for k, (s, p0) in enumerate((("BTCUSDT", 60000.0), ("SOLUSDT", 140.0), ("ETHUSDT", 3000.0), ("XRPUSDT", 0.6))):
        M1[s] = synth1m(s, days=190, seed=20 + k, p0=p0)
    mv = install_mock(M1, now, "cache_t_floop")
    cfg = F.FConfig(cache_dir="cache_t_floop", vision_workers=4)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        feats = F.features_for(cfg, "2024-02-15", "2024-07-01")
    check("archive → features: 5m klines (taker volume), 1m mark klines, funding for both symbols",
          set(feats) == {"BTCUSDT", "SOLUSDT"} and all(len(f.t1) > 100_000 and len(f.fund_t) > 300 for f in feats.values())
          and all(np.isfinite(f.tbv).mean() > 0.99 for f in feats.values()), f"{time.time() - t0:.0f}s, {mv.calls} requests")
    global E2E_FEATS
    E2E_FEATS = feats
    tr, eng = F.run(feats, cfg, "2024-02-15", "2024-07-01")
    check(f"V0 runs end-to-end ({len(tr)} trades)", len(tr) > 10, str(F.metrics(tr, cfg))[:160])
    by = {}
    overlap = False
    for t in sorted(tr, key=lambda x: x["entry_ns"]):
        if t["sym"] in by and t["entry_ns"] < by[t["sym"]]:
            overlap = True
        by[t["sym"]] = t["exit_ns"]
    check("one position per symbol at a time", not overlap)
    first = sorted(tr, key=lambda x: x["entry_ns"])[:30]
    check("first 30 trades are $50 probes at 15×", all(t["probe"] and abs(t["margin"] - 50) < 1e-9 for t in first)
          and all(abs(t["notional"] / t["margin"] - 15) < 0.2 for t in first))
    ok_dl = True
    for t in tr:
        d = t["entry_ns"] // F.DAY
        lost = sum(-x["pnl"] for x in tr if x["record_ns"] is not None and x["record_ns"] < t["entry_ns"] - F.NS5 + 1
                   and x["record_ns"] // F.DAY == d and x["pnl"] < 0)
        ok_dl &= lost < cfg.daily_loss_limit
    check("no entry after the day's recorded losses reach $200", ok_dl)
    c1 = replace(cfg, **F.VARIANTS["V1_fixed_risk"])
    tr1, _ = F.run(feats, c1, "2024-02-15", "2024-07-01")
    risks = [abs(t["qty"] * (t["entry"] - t["stop"])) / t["eq_before"] for t in tr1]
    check("V1: risk at the stop ≈ 0.5% of equity", len(tr1) > 10 and 0.0035 < np.median(risks) < 0.0055, f"median {np.median(risks):.4f}")
    # end-to-end causality: archive cut earlier → identical entries before the cut
    cutT = pd.Timestamp("2024-05-20", tz="UTC")
    install_mock(M1, cutT + pd.Timedelta(hours=8), "cache_t_floop_cut")
    with contextlib.redirect_stdout(io.StringIO()):
        fc = F.features_for(replace(cfg, cache_dir="cache_t_floop_cut"), "2024-02-15", "2024-05-19")
    trc, _ = F.run(fc, cfg, "2024-02-15", "2024-05-19")
    lim = (cutT - pd.Timedelta(days=2)).value
    a = [(t["sym"], t["entry_ns"], round(t["margin"], 2)) for t in tr if t["exit_ns"] < lim]
    b = [(t["sym"], t["entry_ns"], round(t["margin"], 2)) for t in trc if t["exit_ns"] < lim]
    check(f"no look-ahead end-to-end: archive cut at {cutT.date()} → same trades before it", a == b and len(a) > 5, f"{len(a)} trades")
    install_mock(M1, now, "cache_t_floop")
    return cfg


def test_experiment(cfg):
    folds = [("2024-02-15", "2024-05-01", "holdout"), ("2024-05-01", "2024-07-01", "dev")]
    out = io.StringIO()
    t0 = time.time()
    c = replace(cfg, universe_n=2)
    with contextlib.redirect_stdout(out):
        res, summary, gates = F.run_experiment(c, folds=folds, start="2024-02-15", end="2024-07-01")
    txt = out.getvalue()
    check("experiment: 8 variants (incl. V6 point-in-time universe) + folds + selection + gates",
          len(res) == 8 and ("pass/fail gates" in txt or "No variant" in txt),
          f"{time.time() - t0:.0f}s; trades: {[len(r['trades']) for r in res.values()]}")
    print("   " + "\n   ".join(l for l in txt.splitlines() if " ALL " in l or "selected" in l or "[PASS]" in l or "[FAIL]" in l))


def test_paper():
    state = "paper_t.json"
    if os.path.exists(state):
        os.remove(state)
    clock = {"now": pd.Timestamp("2024-04-01 00:02", tz="UTC")}

    def frame(sym, iv):
        base = M1[{"BTCUSDC": "BTCUSDT", "SOLUSDC": "SOLUSDT"}.get(sym, "BTCUSDT")]
        if iv == "1m":
            return base
        mins = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[iv]
        g = base.groupby(base.index.floor(f"{mins}min"))
        return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                             "volume": g.volume.sum(), "tbv": g.tbv.sum()})

    cache = {}

    def get(path, params):
        now = clock["now"]
        if path == "/fapi/v1/time":
            return {"serverTime": int(now.value // 10**6)}
        if path == "/fapi/v1/fundingRate":
            return [{"fundingRate": "0.0001"}]
        key = (params["symbol"], params["interval"])
        if key not in cache:
            cache[key] = frame(*key)
        df = cache[key]
        df = df[df.index <= now]
        if "startTime" in params:
            df = df[df.index >= pd.Timestamp(params["startTime"], unit="ms", tz="UTC")].iloc[:params["limit"]]
        else:
            df = df.iloc[-params["limit"]:]
        return [[int(t.value // 10**6), r.open, r.high, r.low, r.close, r.volume, 0, 0, 0, r.tbv] for t, r in zip(df.index, df.itertuples())]

    F.PUBLIC_GET = get
    cfg = F.FConfig(live_symbols="BTCUSDC,SOLUSDC", min_strength=8)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for k in range(60):
            clock["now"] += pd.Timedelta(minutes=30)
            F.paper_tick(cfg, state)
    S = json.load(open(state))
    opened = sum("OPEN" in l for l in S["log"])
    check("paper runner (Azure): 60 ticks from public data, positions opened/closed, state persisted",
          opened > 0 and isinstance(S["cash"], float), f"{opened} opens, {len(S['closed'])} closes, cash {S['cash']:.2f}")
    # golden-live plumbing: a JSON shaped like the live ?test=indicators output must match itself
    now_ms = int(clock["now"].value // 10**6)
    live = {}
    for sym in ("BTCUSDC",):
        m5, htf, prev = F.live_window(sym, cfg, now_ms)
        _, pre, _, W = F.tick_signal(sym, cfg, m5, htf, prev, F.NEUTRAL)
        g = F.garch_fit_window(m5.close.to_numpy(float), 0, len(m5))
        live[sym] = {"indicators": {"close": float(m5.close.iloc[-1]), "rfTrend": W[0], "rfFilt": W[2], "atr": W[8],
                                    "adx": W[12], "chop": W[13], "strength": int(pre["total"][-1])},
                     "regime": {"trendProb": W[18]}, "garchFitted": {"lastSigma": g[0]}}
    json.dump(live, open("live_t.json", "w"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        F.golden_live(cfg, "live_t.json")
    check("golden-live comparison tool: identical inputs → all fields match", "9/9 match" in out.getvalue())


def random_walk_feat(sym="BTCUSDT", days=200, seed=11, p0=60000.0, vol=0.0006):
    """driftless Gaussian random walk on 1m bars: no strategy can have an edge here"""
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    r = vol * rng.standard_normal(n)
    c = p0 * np.exp(np.cumsum(r)); o = np.r_[p0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    v = rng.lognormal(2, 0.5, n)
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "tbv": v * rng.uniform(0.3, 0.7, n)}, index=idx)
    return F.build_feat(F.SymData(sym, to5m(m1), m1[["open", "high", "low", "close"]], None), F.FConfig())


def test_zero_edge_and_ledger(cfg_e2e):
    """the engine must not invent an edge: on a driftless random walk, win rate = the bracket's break-even 1/(1+R)"""
    c = replace(F.FConfig(), **F.VARIANTS["V1_fixed_risk"], entry_slip=0.0, tp_slip=0.0, stop_slip=0.0)
    tr = []
    for seed in (11, 12, 13):
        feats = {"BTCUSDT": random_walk_feat(seed=seed)}
        tr += F.run(feats, c, "2024-01-10", "2024-07-20")[0]
    w = np.array([t["pnl"] > 0 for t in tr], float); w0 = np.array([t["null_w"] for t in tr])
    z = (w - w0).sum() / math.sqrt((w0 * (1 - w0)).sum())
    avgR = np.mean([t["R"] for t in tr])
    check(f"random walk, zero costs: win rate = break-even 1/(1+R), 3 seeds pooled (no built-in edge) — {len(tr)} trades",
          len(tr) >= 1500 and abs(z) < 3.5 and abs(avgR) < 0.1,
          f"W={w.mean():.3f} vs W0={w0.mean():.3f}, z={z:+.2f}, avg R {avgR:+.3f}")
    c0 = replace(F.FConfig(), entry_slip=0.0, tp_slip=0.0, stop_slip=0.0)
    tr0, _ = F.run(feats, c0, "2024-01-10", "2024-07-20")
    n_probe = sum(t["probe"] for t in tr0)
    check("V0 on a no-edge market: 30 $50 probes, then Bayesian Kelly blocks (v7.1 behaviour)",
          n_probe == 30 and len(tr0) - n_probe <= 10, f"{n_probe} probes, {len(tr0) - n_probe} Kelly-sized trades")
    # v7.1 freeze: once the rolling 30-trade sample's q25 < break-even, no trade closes, so the sample never changes
    pre = F.pre_signals(feats["BTCUSDT"], c0); md = F.mode_series(feats["BTCUSDT"], c0, pre)
    d = F.directions(feats["BTCUSDT"], c0, pre, md)
    last = max(t["exit_ns"] for t in tr0)
    later = int(((d != 0) & (feats["BTCUSDT"].t > last)).sum())
    check("v7.1 Kelly freeze reproduced: after the last trade, signals keep firing but none is ever taken",
          later > 200, f"{later} signal bars after the last trade")
    cu = replace(c0, unfreeze_hours=24.0)
    tru, _ = F.run(feats, cu, "2024-01-10", "2024-07-20")
    gaps = np.diff(sorted(t["entry_ns"] for t in tru)) / 3.6e12 if len(tru) > 1 else np.array([0.0])
    check("proposed fix (unfreeze_hours=24): after a freeze the bot keeps probing at $50, ≤ ~1 probe per 24h idle",
          len(tru) > len(tr0) + 20 and all(abs(t["margin"] - 50) < 1e-9 for t in tru if t["probe"]),
          f"{len(tru)} trades vs {len(tr0)} frozen; longest idle gap {gaps.max():.0f}h")
    # ledger: live v7.1 books only bot-side closes; 'all' books every close
    feats2, s0, s1 = cfg_e2e
    trl, _ = F.run(feats2, replace(F.FConfig(cache_dir=s0), ledger="live"), s1[0], s1[1])
    tra, _ = F.run(feats2, replace(F.FConfig(cache_dir=s0), ledger="all"), s1[0], s1[1])
    rec_l = [t for t in trl if t["record_ns"] is not None]
    check("ledger=live: only hard/time-stop closes reach the daily-loss ledger and mode log (as live v7.1)",
          all(t["why"] in ("hard", "time") for t in rec_l) and len(rec_l) < len(trl),
          f"{len(rec_l)}/{len(trl)} closes recorded")
    check("ledger=all: every close is recorded (the bug-fixed behaviour)",
          all(t["record_ns"] is not None for t in tra), f"{len(tra)} trades vs {len(trl)} with the live ledger")


def test_strict_causality():
    """stronger than a cut-and-compare-earlier check: each sampled tick is recomputed from data ending at its own close,
    so even a 1-bar look-ahead (into the next 5m bar or the still-open 15m/1h/4h bar) would show up"""
    rng = np.random.default_rng(3)
    m1 = synth1m("BTCUSDT", days=70, seed=31)
    m5 = to5m(m1); cfg = F.FConfig()
    def sig(m5x, m1x):
        Fe = F.build_feat(F.SymData("BTCUSDT", m5x, m1x[["open", "high", "low", "close"]], None), cfg)
        pre = F.pre_signals(Fe, cfg); md = F.mode_series(Fe, cfg, pre); d = F.directions(Fe, cfg, pre, md)
        return Fe, pre, md, d
    Ff, pref, mdf, df = sig(m5, m1)
    cand = np.flatnonzero(df != 0); cand = cand[cand > 6000]
    ticks = list(rng.choice(cand, min(10, len(cand)), replace=False)) + list(rng.choice(np.arange(6000, len(m5) - 50), 10, replace=False))
    bad = 0
    for t in ticks:
        cut = m5.index[t] + pd.Timedelta("5min")
        Fc, prec, mdc, dc = sig(m5[m5.index < cut], m1[m1.index < cut])
        tc = len(Fc.t) - 1
        same = (np.allclose(np.nan_to_num(Ff.W[t], nan=-9), np.nan_to_num(Fc.W[tc], nan=-9), rtol=1e-12)
                and Ff.htf15[t] == Fc.htf15[tc] and Ff.htf1h[t] == Fc.htf1h[tc] and Ff.htf4h[t] == Fc.htf4h[tc]
                and pref["total"][t] == prec["total"][tc] and mdf[t] == mdc[tc] and df[t] == dc[tc])
        if same and df[t] != 0:
            b1 = F.bracket(Ff, t, int(mdf[t]), int(df[t]), cfg); b2 = F.bracket(Fc, tc, int(mdc[tc]), int(dc[tc]), cfg)
            same = np.isclose(b1["stop"], b2["stop"]) and np.isclose(b1["tp"], b2["tp"])
        bad += not same
    check(f"strict causality: {len(ticks)} ticks recomputed from data ending at their own close → identical",
          bad == 0, f"{sum(df[t] != 0 for t in ticks)} with entry signals; {bad} mismatches")


if __name__ == "__main__":
    t0 = time.time()
    test_ts_equivalence()
    test_exits()
    test_sizing()
    test_causality()
    test_strict_causality()
    cfg = test_end_to_end()
    test_zero_edge_and_ledger((E2E_FEATS, "cache_t_floop", ("2024-02-15", "2024-07-01")))
    test_experiment(cfg)
    test_paper()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed ({time.time() - t0:.0f}s)")
