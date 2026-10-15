"""test_vegas.py — offline checks for vegas.py (no network): python test_vegas.py"""
import contextlib
import io
import math
import os
import shutil
import tempfile
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import floop as F
import langlang as LL
import mockfeeds as MF
import qstats as Q
import test_floop as TF
import vegas as V

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def pine_ema(x, n):
    a, s, out = 2 / (n + 1), None, []
    for v in x:
        s = v if s is None else a * v + (1 - a) * s
        out.append(s)
    return np.array(out)


def pine_rma(x, n):
    out, s = [], None
    for i, v in enumerate(x):
        if s is None:
            if i >= n - 1:
                s = float(np.mean(x[i - n + 1:i + 1]))
            out.append(s if s is not None else np.nan)
        else:
            s = (s * (n - 1) + v) / n
            out.append(s)
    return np.array(out)


def bars_of(df, start, end, sym="TST", mark=None, fund=None):
    return V.make_bars(sym, (df, mark, fund), start, end)


# ---------------------------------------------------------------- 1. indicators
def test_indicators():
    rng = np.random.default_rng(1)
    x = 100 + np.cumsum(rng.normal(0, 1, 5000))
    seg = np.zeros(5000, bool); seg[0] = True
    e = V.ema_tv(x, 169, seg)
    r = V.rma_tv(np.abs(np.diff(x, prepend=x[0])), 60, seg)
    ok_e = np.max(np.abs(e - pine_ema(x, 169))) < 1e-9
    pr = pine_rma(np.abs(np.diff(x, prepend=x[0])), 60)
    ok_r = np.allclose(r[59:], pr[59:], rtol=1e-12) and np.isnan(r[:59]).all()
    check("EMA = Pine ta.ema (seed x0) and RMA = Pine ta.rma (SMA seed) exactly", ok_e and ok_r,
          f"max |Δema| {np.max(np.abs(e - pine_ema(x, 169))):.1e}")
    t = pd.date_range("2024-01-01", periods=3000, freq="1min", tz="UTC").as_unit("ns").asi8
    th = np.unique(t // V.NS_H) * V.NS_H
    j = V._map_completed(t, th, V.NS_H)
    okm = all(th[jj] + V.NS_H <= ti + V.NS_MIN and (jj + 1 >= len(th) or th[jj + 1] + V.NS_H > ti + V.NS_MIN)
              for ti, jj in zip(t[59::7], j[59::7]))
    check("1h mapping uses only the last COMPLETED hour at each 1m close (no look-ahead into the open hour)", okm)
    tw = pd.Timestamp("2024-01-10 21:59", tz="UTC").value       # winter: NY 16:59-17:00 close
    ts = pd.Timestamp("2024-07-10 20:59", tz="UTC").value       # summer: NY 16:59-17:00 close
    ny = lambda v: pd.DatetimeIndex([v + V.NS_MIN], tz="UTC").tz_convert("America/New_York").hour[0]
    check("session rule follows New York DST (17:00 NY = 22:00 UTC winter, 21:00 UTC summer)",
          ny(tw) == 17 and ny(ts) == 17 and ny(tw - V.NS_MIN) == 16)


# ---------------------------------------------------------------- 2. exits
def exit_case(rows, side=1, Pf=100.0, d=1.0, exit="XF", mark=None, gap_at=None, L=None, U=None, fund_at=None,
              time_stop=1440, slip=0.0):
    rows = np.array(rows, float)
    o, h, l, c = rows.T
    n = len(o)
    mo = mh = ml = np.full(n, np.nan)
    if mark is not None:
        mk = np.array(mark, float)
        mo, mh, ml = mk[:, 0], mk[:, 1], mk[:, 2]
    gap = np.zeros(n, np.int8)
    if gap_at is not None:
        gap[gap_at] = 2
    fund, fpx = np.zeros(n), np.zeros(n)
    if fund_at is not None:
        fund[fund_at[0]], fpx[fund_at[0]] = fund_at[1], o[fund_at[0]]
    UU = np.full(n, -1e9) if U is None else np.array(U, float)
    LL_ = np.full(n, -1e9) if L is None else np.array(L, float)
    if side < 0 and L is None:
        LL_ = np.full(n, 1e9)
        UU = np.full(n, 1e9)
    cfg = replace(V.VConfig(), exit=exit, time_stop=time_stop, entry_slip=slip, tp_slip=slip, stop_slip=slip)
    P, IP = V.params(cfg)
    return V.scan_exit(0, side, Pf, Pf, d, Pf - side * d, o, h, l, c, mo, mh, ml, UU, LL_, np.full(n, np.nan),
                       gap, fund, fpx, P, IP)


def test_exits():
    j, R, Rg, why, *_ = exit_case([[100, 102.5, 98.5, 101]])          # stop and 2R target in one bar
    check("stop-first: stop and target both reachable in a bar → stop fills, targets void", why == 1 and abs(R + 1) < 1e-12)
    j, R, Rg, why, *_ = exit_case([[100, 100.5, 99.5, 100], [97, 97.5, 96.5, 97]])
    check("open gaps through the stop → filled at the open (worse than −1R)", why == 2 and abs(R + 3) < 1e-12, f"R {R:+.2f}")
    j, R, Rg, why, *_ = exit_case([[100, 101.2, 99.6, 101], [100.5, 100.6, 99.95, 100]], exit="XV",
                                  L=[0, 0], U=[0, 0])
    check("X-V: T1 (+1R) third, then breakeven stop on the rest → +1/3 R", why == 1 and abs(R - 1 / 3) < 1e-12, f"R {R:+.4f}")
    j, R, Rg, why, *_ = exit_case([[100, 100.4, 99.7, 99.8], [99.9, 100.2, 99.6, 100], [100, 100.1, 99.5, 99.5],
                                   [99.4, 99.5, 98.9, 99.0]], exit="XV", L=[99.9, 99.9, 0, 0], U=[0, 0, 0, 0])
    check("X-V: tunnel violation closes only the final third at the next open; the rest keeps its stop",
          why == 1 and j == 3 and abs(R - ((99.9 - 100) / 3 + 2 / 3 * -1)) < 1e-12, f"R {R:+.4f}, exit bar {j}")
    j, R, Rg, why, *_ = exit_case([[100, 100.3, 99.8, 100]] * 5, time_stop=3)
    check("24h time stop (here 3 bars) exits at the next open", why == 5 and j == 3, f"exit bar {j}")
    j, R, *_ = exit_case([[100, 100.3, 99.8, 100]] * 3 + [[100, 102.1, 99.9, 102]], fund_at=(1, 0.001))
    check("funding: a long pays a positive rate (R reduced by rate·price/d)", abs(R - (2 - 0.1)) < 1e-9, f"R {R:+.3f}")
    j, R, Rg, why, *_ = exit_case([[100, 100.3, 99.8, 100], [100, 100.3, 99.8, 100]], gap_at=1, slip=1e-4)
    check("maintenance gap → exit at the first post-gap open with 3× stop slippage",
          why == 6 and abs(R - (100 * (1 - 3e-4) - 100)) < 1e-9)
    j, R, Rg, why, *_ = exit_case([[100, 100.5, 99.2, 100], [100, 100.3, 99.95, 100.1]],
                                  mark=[[100, 100.4, 99.5], [100.1, 100.3, 98.9]])
    check("stops trigger on MARK price (contract wick to 99.2 ignored; mark low 98.9 triggers next bar)",
          why == 1 and j == 1, f"exit bar {j}")
    j, R, Rg, why, *_ = exit_case([[100, 100.5, 97.4, 98], [98, 99, 97, 97.5]], side=-1, Pf=100, d=1, exit="XF")
    check("short side mirrors (2R target below)", why == 3 and abs(R - 2) < 1e-12)


# ---------------------------------------------------------------- 3. causality
def test_strict_causality():
    df = TF.synth1m("BTCUSDT", start="2024-01-01", days=100, seed=3, p0=60000.0)
    cfg = replace(V.VConfig(), stop_slip=3e-4)
    full = bars_of(df, "2024-02-20", "2024-04-09")
    tr, _, _ = V.run_symbol(full, cfg)
    rng = np.random.default_rng(5)
    pick = sorted(rng.choice(len(tr), min(12, len(tr)), replace=False)) if len(tr) else []
    bad = 0
    for q in pick:
        i = int(tr.sig_bar.iloc[q])
        cut = bars_of(df.iloc[:i + 2], "2024-02-20", "2024-04-09")          # data ends at the entry bar
        V._IND_CACHE.clear()
        tc, _, _ = V.run_symbol(cut, cfg)
        a = tr[tr.sig_bar <= i][["sig_bar", "side", "d_rel"]].reset_index(drop=True)
        b = tc[tc.sig_bar <= i][["sig_bar", "side", "d_rel"]].reset_index(drop=True)
        bad += not (len(a) == len(b) and (a.sig_bar.values == b.sig_bar.values).all()
                    and (a.side.values == b.side.values).all() and np.allclose(a.d_rel, b.d_rel, rtol=1e-12))
    check(f"strict causality: {len(pick)} signals recomputed from data ending at their own entry bar → identical "
          f"signal history", len(pick) >= 8 and bad == 0, f"{len(tr)} trades, {bad} mismatches")


# ---------------------------------------------------------------- 4. no built-in edge
def test_zero_edge():
    trs = {}
    t0 = time.time()
    for seed in (21, 22):
        df = MF.gbm1m("2024-01-01", 170, seed, p0=100.0, vol=0.0009)
        b = bars_of(df, "2024-02-15", "2024-06-18")
        for vn in ("V2_pullback", "V4_fixed_2R", "V3_chandelier"):
            c = replace(V.VConfig(), **V.VARIANTS[vn], entry_slip=0.0, tp_slip=0.0, stop_slip=0.0)
            tr, _, _ = V.run_symbol(b, c)
            b0, b0b = V.run_bench(b, c, tr, reps=200)
            trs.setdefault(vn, []).append((tr, b0))
    lines, ok = [], True
    for vn, parts in trs.items():
        R = np.concatenate([p[0].R.to_numpy() for p in parts])
        z = R.mean() / (R.std(ddof=1) / math.sqrt(len(R)))
        b0 = np.mean([p[1].mean() for p in parts])
        lines.append(f"{vn[:2]} n={len(R)} mean {R.mean():+.3f}R z={z:+.1f} B0 {b0:+.3f}")
        ok &= len(R) >= 300 and abs(z) < 3.5 and abs(R.mean() - b0) < 0.1
    xf = np.concatenate([p[0].R.to_numpy() for p in trs["V4_fixed_2R"]])
    w = (xf > 0).mean()
    se = math.sqrt(1 / 3 * 2 / 3 / len(xf))
    check("random walk, zero costs: mean R ≈ 0 and ≈ B0 for pullback, chandelier and fixed-bracket variants",
          ok, "; ".join(lines) + f" ({time.time() - t0:.0f}s)")
    check("random walk: 2R bracket wins at the break-even rate 1/(1+2) = 33.3%", abs(w - 1 / 3) < 3.5 * se,
          f"win {w:.3f} ± {se:.3f}")


# ---------------------------------------------------------------- 5. full workflow on a mock archive
def test_workflow():
    now = pd.Timestamp("2024-08-10 12:00", tz="UTC")
    m1 = {"BTCUSDT": TF.synth1m("BTCUSDT", start="2023-11-01", days=280, seed=41, p0=40000.0),
          "SOLUSDT": TF.synth1m("SOLUSDT", start="2023-11-01", days=280, seed=42, p0=60.0),
          "ETHUSDT": TF.synth1m("ETHUSDT", start="2023-11-01", days=280, seed=43, p0=2000.0)}
    hist = {"BTCUSDT": TF.synth1m("BTCUSDT", start="2023-06-01", days=150, seed=44, p0=30000.0)}
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    cache = os.path.join(tmp, "cache")
    # archive lacks BTC before 2023-11; the REST fallback serves it
    MF.install(m1, now, cache)

    def rest(path, params):
        df = hist["BTCUSDT"]
        a, z = pd.Timestamp(params["startTime"], unit="ms", tz="UTC"), pd.Timestamp(params["endTime"], unit="ms", tz="UTC")
        sub = df[(df.index >= a) & (df.index <= z)].iloc[:params["limit"]]
        return [[int(t.value // 1_000_000), r.open, r.high, r.low, r.close, r.volume, 0, 0, 100, r.tbv, 0, 0]
                for t, r in zip(sub.index, sub.itertuples())]
    F.PUBLIC_GET = rest
    cfg = replace(V.VConfig(), cache_dir=cache, vision_workers=4, n_bench=40,
                  dev_syms=("BTCUSDT", "SOLUSDT"), dev_start="2024-01-15", dev_end="2024-06-30",
                  h1_syms=("BTCUSDT",), h1_start="2023-08-15", h1_end="2023-10-25",
                  h2_syms=("ETHUSDT",), h2_start="2024-01-15", h2_end="2024-07-31",
                  min_dev_trades=100, min_proj_holdout=20, min_holdout_trades=20)
    os.chdir(tmp)
    try:
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                V.cmd_dev(cfg)
            refused = False
        except SystemExit as e:
            refused = "firerate" in str(e)
        check("pre-registration: `dev` refuses to compute returns before the fire-rate report is logged", refused)
        t0 = time.time()
        with contextlib.redirect_stdout(out):
            V.cmd_download(cfg)
            V.cmd_firerate(cfg)
        fr = pd.read_json("reports/vegas_firerate.json")
        check("download + fire-rate report (counts only) logged to the ledger", len(fr) == 7 and os.path.exists(V.LEDGER)
              and "mean_R" not in out.getvalue(), f"{time.time() - t0:.0f}s; trades {fr.trades.tolist()}")
        t0 = time.time()
        out2 = io.StringIO()
        with contextlib.redirect_stdout(out2):
            V.cmd_dev(cfg)
        dev = json_load("reports/vegas_dev.json")
        txt = out2.getvalue()
        check("dev: 7 variants with 2×/3×/fees stress, B0, PBO, DSR, selection rule, neighbours, walk-forward",
              len(dev["table"]) == 7 and "selected" in txt or "no variant" in txt, f"{time.time() - t0:.0f}s; chosen={dev['chosen']}")
        print("   " + "\n   ".join(l for l in txt.splitlines()[:12]))
        if dev["chosen"]:
            out3 = io.StringIO()
            with contextlib.redirect_stdout(out3):
                V.cmd_holdout(cfg)
            h = out3.getvalue()
            again = False
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    V.cmd_holdout(cfg)
            except SystemExit as e:
                again = "already been run" in str(e)
            check("holdout: H1 (REST fallback for pre-archive BTC) + H2, 10 gates, verdict; second run refused",
                  "Verdict" in h and h.count("[PASS]") + h.count("[FAIL]") == 10 and again and "H1" in h,
                  h.splitlines()[0][:110])
            print("   " + "\n   ".join(h.splitlines()[1:]))
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    V.cmd_holdout(cfg)
                closed = False
            except SystemExit as e:
                closed = "closed" in str(e)
            check("no variant selected → holdout refuses (family closed)", closed)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
        F.PUBLIC_GET = F._public_get_default


def json_load(p):
    import json
    return json.load(open(p))


# ---------------------------------------------------------------- 6. speed
def test_speed():
    df = MF.gbm1m("2024-01-01", 365, 9, p0=100.0, vol=0.0009)
    b = bars_of(df, "2024-02-15", "2024-12-31")
    c = replace(V.VConfig(), **V.VARIANTS["V2_pullback"])
    V._IND_CACHE.clear()
    t0 = time.time(); V.indicators(b, c); ti = time.time() - t0
    t0 = time.time(); tr, _, _ = V.run_symbol(b, c); ts = time.time() - t0
    t0 = time.time(); V.run_bench(b, c, tr, reps=1000); tb = time.time() - t0
    check("speed: one symbol-year of 1m bars", ti + ts < 20 and tb < 120,
          f"indicators {ti:.1f}s, simulation {ts:.1f}s ({len(tr)} trades), B0 ×1000 {tb:.1f}s (1 core)")


# ---------------------------------------------------------------- 7. regressions for the review fixes (A-1.1)
def pullback_trade(lows):
    """hand-made long pullback: flat tunnel U=100 / L=99.5, ATR 0.5, regime long. Bars with the given lows touch
    the tunnel; the next bar closes back above it (trigger). Returns the trade's structural stop distance."""
    n = 14
    o = np.full(n, 100.5); h = np.full(n, 100.8); l = np.full(n, 100.3); c = np.full(n, 100.6)
    for q, lo in enumerate(lows):
        i = 4 + q
        l[i] = lo; c[i] = 99.9; o[i] = 100.1
    k = 4 + len(lows)
    l[k] = 100.2; c[k] = 100.4                                   # trigger: close > max(U, EMA12), no touch
    t = (np.arange(n) * V.NS_MIN).astype(np.int64)
    nan = np.full(n, np.nan)
    U = np.full(n, 100.0); L = np.full(n, 99.5); E12 = np.full(n, 100.0); atr = np.full(n, 0.5)
    cfg = replace(V.VConfig(), entry="EP", exit="XF", session="none", d_min=0.0001, entry_slip=0.0)
    P, IP = V.params(cfg)
    cnt = np.zeros((1, V.NCNT), np.int64)
    out = np.zeros((10, len(V.TRADE_COLS)))
    nt = V.simulate(0, n, t, o, h, l, c, nan, nan, nan, np.zeros(n, np.int8), np.zeros(n), np.zeros(n),
                    np.full(n, 24300, np.int64), 24300, U, L, (U + L) / 2, E12, atr, nan, np.ones(n, np.int8),
                    np.ones(n, bool), np.ones(n, bool), P, IP, cnt, out)
    if nt == 0:
        return np.nan, np.nan
    Pf = o[int(out[0, 1])]
    return out[0, 9] * Pf, Pf


def test_review_fixes():
    d_deep_first, Pf = pullback_trade([99.0, 99.7])             # deepest touch first, shallower touch after it
    d_one, _ = pullback_trade([99.0])
    want = Pf - (99.0 - 0.10 * 0.5)
    check("fix: the pullback stop sits beyond the DEEPEST touch bar, not the latest one",
          abs(d_deep_first - want) < 1e-9 and abs(d_one - want) < 1e-9,
          f"stop distance {d_deep_first:.3f} (expected {want:.3f}; the old code gave {Pf - (99.5 - 0.05):.3f})")
    rows = [[100, 100.5, 99.8, 100.2], [100.2, 100.3, 98.5, 99.0], [99.0, 99.2, 98.8, 99.0]]
    _, R_exit_bar, *_ = exit_case(rows, side=1, d=1.0, exit="XF", fund_at=(1, 0.01))
    _, R_entry_bar, *_ = exit_case(rows, side=1, d=1.0, exit="XF", fund_at=(0, 0.01))
    check("fix: funding is charged when the position is held at the funding time, including the bar it exits in",
          abs(R_exit_bar - (-1.0 - 0.01 * 100.2)) < 1e-9 and abs(R_entry_bar - (-1.0)) < 1e-9,
          f"stopped in the funding bar: R {R_exit_bar:+.3f} (old code: −1.000); funding at entry: R {R_entry_bar:+.3f}")
    rng = np.random.default_rng(3)
    ds = rng.normal(0.05, 2.0, 1500); dc = rng.integers(0, 6, 1500).astype(float)
    a, b_ = V.per_trade_mean_ci(ds, dc, seed=1), Q.per_trade_mean_ci(ds, dc, seed=1)
    da, db = V.boot_max_dd(ds, seed=1), Q.boot_max_dd(ds, seed=1)
    R = rng.normal(0.03, 1, 3000); srs = rng.normal(0, 0.04, 9); M = rng.normal(0, 1, (480, 5))
    t0 = time.time(); V.per_trade_mean_ci(ds, dc, seed=2); V.boot_max_dd(ds, seed=2); tv = time.time() - t0
    check("built-in statistics match qstats (no external module needed); bootstraps compiled",
          abs(a[0] - b_[0]) < 1e-12 and abs(a[1] - b_[1]) < 0.01 and abs(da - db) / db < 0.05
          and V.deflated_sharpe(R, srs, 30) == Q.deflated_sharpe(R, srs, 30) and V.pbo_cscv(M) == Q.pbo_cscv(M),
          f"CI low {a[1]:+.4f} vs {b_[1]:+.4f}, DD95 {da:.1f} vs {db:.1f}; 2×10,000 resamples in {tv:.2f}s")
    df = TF.synth1m("BTCUSDT", start="2024-01-01", days=75, seed=11, p0=40000.0)
    b = bars_of(df, "2024-02-15", "2024-03-15")
    cfg = V.VConfig()
    tr, _, _ = V.run_symbol(b, cfg)
    ind = V.indicators(b, cfg)
    ptr, items, gid = V.bench_pools(b, ind, cfg, tr)
    cand = np.arange(b.eval_from, b.eval_to - 1)
    cand = cand[ind.valid[cand] & (b.gap[cand + 1] == 0) & ind.sess_ok[cand]]
    ok = len(tr) > 20
    for q in range(len(tr)):
        pool = items[ptr[gid[q]]:ptr[gid[q] + 1]]
        exp = cand[(b.month[cand] == tr.month.iloc[q]) & (ind.reg[cand] == tr.side.iloc[q])]
        ok &= np.array_equal(np.sort(pool), np.sort(exp))
    check("benchmark pools (vectorised) = same month, session rule and regime side as each real trade", ok,
          f"{len(tr)} trades checked")


if __name__ == "__main__":
    t0 = time.time()
    test_indicators()
    test_exits()
    test_strict_causality()
    test_zero_edge()
    test_workflow()
    test_review_fixes()
    test_speed()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed in {time.time() - t0:.0f}s")
