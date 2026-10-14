"""test_brooks.py — offline checks for the mechanised Brooks backtest: python test_brooks.py"""
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

import brooks as BR
import brooks_core as BC
import brooks_sim as BS
import floop as F
import langlang as LL
import mockfeeds as MF
import test_floop as TF

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def bars(rows, start="2024-01-01"):
    a = np.array(rows, float)
    idx = pd.date_range(start, periods=len(a), freq="5min", tz="UTC")
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "volume": 1.0}, index=idx)


def warm(n=40, p=100.0):
    return [[p, p + 0.5, p - 0.5, p + (0.05 if i % 2 else -0.05)] for i in range(n)]


def to5(df1):
    return BR.resample(df1, "5min")


# ---------------------------------------------------------------- 1. definitions
def test_definitions():
    rows = warm() + [[99.9, 100.5, 99.5, 100.0], [100.4, 101.0, 99.4, 100.95], [100.5, 100.6, 99.6, 99.7]]
    P = BC.pvec()
    FE = BC.features(bars(rows), P)
    t = len(rows) - 2
    check("signal bar: ideal bull reversal bar (body, close near high, 1/3+ lower tail, small upper tail) scores 6/6",
          FE[t, BC.C_SIGB] == 6, f"score {FE[t, BC.C_SIGB]:.0f}")
    check("signal bar: a bull bar closing below its midpoint scores 0 (close position is required)",
          FE[t + 1, BC.C_SIGB] == 0)
    rows = warm() + [[100, 101, 99, 100.5], [100.4, 100.8, 99.4, 100.6], [100.5, 101.2, 98.8, 100.0], [100, 100, 100, 100]]
    FE = BC.features(bars(rows), P)
    n = len(rows)
    r1, r2, r3, r4 = n - 4, n - 3, n - 2, n - 1
    outside = FE[r3, BC.C_H] > FE[r2, BC.C_H] and FE[r3, BC.C_L] < FE[r2, BC.C_L]
    check("inside / outside / doji bars per glossary (body/range 0.25 is a trend bar, a zero-range bar is a doji)",
          FE[r2, BC.C_INS] == 1 and FE[r3, BC.C_INS] == 0 and outside and FE[r4, BC.C_DOJI] == 1
          and FE[r1, BC.C_DOJI] == 0 and FE[r1, BC.C_BTB] == 1)


def h_fixture(lower_low=True):
    rows = warm(40)
    p = 100.0
    for i in range(3):                                   # swing low
        p -= 0.6
        rows.append([p + 0.5, p + 0.6, p - 0.4, p])
    for i in range(6):                                   # up leg
        rows.append([p, p + 0.9, p - 0.1, p + 0.8])
        p += 0.8
    peak = p + 0.4
    rows.append([p, peak, p - 0.3, p + 0.2])             # leg high
    A = [p + 0.1, peak - 0.4, p - 0.6, p - 0.4]          # pullback bar (lower high) -> H1 signal
    rows.append(A)
    rows.append([A[3], peak - 0.1, A[3] - 0.1, peak - 0.2])          # H1 trigger (no new high)
    lo = A[2] - 0.6 if lower_low else A[2] + 0.2
    rows.append([peak - 0.2, peak - 0.9, lo, lo + 0.3])              # second push down
    rows.append([lo + 0.3, peak - 0.3, lo + 0.2, peak - 0.35])       # trigger again
    return rows


def test_counting():
    P = BC.pvec()
    out = {}
    for ll in (True, False):
        rows = h_fixture(ll)
        FE = BC.features(bars(rows), P)
        _, _, _, EV = BC.run_engine(FE, P, 0.01, False)
        h = EV["H"]
        h = h[(h[:, 1] == 1) & (h[:, 0] >= len(rows) - 6)]
        out[ll] = [(int(r[2]), int(r[3])) for r in h]
    check("H-count: H1 trigger, one more leg down (lower low), trigger again → H2 (leg count 2, bar count 2)",
          out[True] == [(1, 1), (2, 2)], f"(leg, bar) = {out[True]}")
    check("H-count: second trigger without a lower low stays H1 in leg mode (bar mode counts 2)",
          out[False] == [(1, 1), (1, 2)], f"(leg, bar) = {out[False]}")


# ---------------------------------------------------------------- 2. causality
def test_causality():
    df1 = TF.synth1m("LTCUSDT", start="2024-01-01", days=40, seed=7, p0=80.0)
    d5 = to5(df1)
    P = BC.pvec()
    FE = BC.features(d5, P)
    OUT, IT, GATE, EV = BC.run_engine(FE, P, 0.01, False)
    sig = np.argwhere(IT[:, :, 0] != 0)
    rng = np.random.default_rng(1)
    pick = sig[rng.choice(len(sig), min(25, len(sig)), replace=False)]
    bad = 0
    for t, f in pick:
        FEc = BC.features(d5.iloc[:t + 1], P)
        _, ITc, Gc, _ = BC.run_engine(FEc, P, 0.01, False)
        bad += not (np.allclose(ITc[t, f], IT[t, f], equal_nan=True) and np.allclose(Gc[t], GATE[t]))
    check(f"strict causality: {len(pick)} intents recomputed from data ending at their own signal bar → identical",
          len(pick) >= 15 and bad == 0, f"{len(sig)} intents in 40 days, families {sorted(set(pick[:, 1].tolist()))}, {bad} mismatches")


# ---------------------------------------------------------------- 3. fills and exits
def test_fills_exits():
    O = np.array([100.0, 100.0, 100.0]); H = np.array([100.5, 101.3, 101]); L = np.array([99.5, 99.8, 99])
    INS = np.zeros(3); SL = np.zeros(3)
    o1 = np.array([100.0, 100.4, 100.9]); h1 = np.array([100.5, 100.95, 101.3]); l1 = np.array([99.8, 100.3, 100.8])
    mlo = np.array([0, 0, 3]); mhi = np.array([0, 3, 3])
    ok, fb, fj, fill, raw = BS.try_fill(0, BC.T_STOP, 101.0, 1, O, H, L, INS, SL, o1, h1, l1, mlo, mhi, 0.01, 0.0)
    check("buy stop fills on the first 1m trade-through at the stop (never better)", ok and fj == 2 and raw == 101.0)
    o1g = np.array([101.4, 101.5, 101.6])
    ok, fb, fj, fill, raw = BS.try_fill(0, BC.T_STOP, 101.0, 1, O, H, L, INS, SL, o1g, h1 + 1, l1 + 1, mlo, mhi, 0.01, 0.0)
    check("gap through a buy stop fills at the open", ok and raw == 101.4)
    ok, *_ = BS.try_fill(0, BC.T_LIMIT, 99.8, 1, O, H, L, INS, SL, o1, h1, l1, mlo, mhi, 0.01, 0.0)
    check("buy limit needs a trade-through by a tick (touch is not a fill)", not ok)
    # exits on a hand-made path: fill 100, stop 99, target 102
    n = 6
    O5 = np.full(n, 100.0); H5 = np.array([100.4, 101.2, 100.6, 100.5, 100.5, 100.5]); L5 = np.array([99.9, 100.2, 99.95, 99.9, 99.9, 99.9])
    C5 = np.full(n, 100.2); z = np.zeros(n); pl = np.full(n, np.nan); pli = np.full(n, -1, np.int64)
    e = np.zeros(0)
    m0 = np.zeros(n, np.int64)
    args = (O5, H5, L5, C5, z, z, z, pl, pli, pl, pli, e, e, e, e, m0, m0, 0.01, 0.0, 0.0, False,
            np.zeros((1, 6, BC.NI)), np.zeros(6, np.bool_), 0)
    eb, pnl, gross, wr, why, *_ = BS.run_pos(0, -1, 1, 100.0, 100.0, 1.0, 99.0, 102.0, 0.0, 99.0, 102.0, 0.0, 1, 99, *args)
    check("breakeven: after +1R at a 5m close the stop moves to entry (exit at 100 on the dip)", why == BS.R_STOP and eb == 2
          and abs(pnl) < 1e-12, f"exit bar {eb}, pnl {pnl:+.3f}")
    eb, pnl, gross, wr, why, *_ = BS.run_pos(0, -1, 1, 100.0, 100.0, 1.0, 99.0, 102.0, 0.0, 99.0, 102.0, 0.0, 0, 3, *args)
    check("time stop exits at the open of the bar after the limit", why == BS.R_TIME and eb == 3)
    H6 = H5.copy(); H6[1] = 102.5; L6 = L5.copy(); L6[1] = 98.5
    args2 = (O5, H6, L6) + args[3:]
    eb, pnl, *_ , = BS.run_pos(0, -1, 1, 100.0, 100.0, 1.0, 99.0, 102.0, 0.0, 99.0, 102.0, 0.0, 0, 99, *args2)
    check("stop-first when stop and target are both inside one bar", abs(pnl + 1.0) < 1e-12)
    SL1 = z.copy(); SL1[2] = 1
    args3 = args[:6] + (SL1,) + args[7:]
    eb, pnl, gross, wr, why, *_ = BS.run_pos(0, -1, 1, 100.0, 100.0, 1.0, 98.0, 103.0, 0.0, 98.0, 103.0, 0.0, 0, 99, *args3)
    check("flat at the US close (last bar of the session)", why == BS.R_SESSION and eb == 2)


# ---------------------------------------------------------------- 4. no built-in edge
def rw_prep(seed, days=150, session="crypto"):
    df1 = MF.gbm1m("2024-01-01", days, seed, p0=100.0, vol=0.0009)
    d5 = to5(df1)
    cfg = BR.BConfig()
    P = BC.pvec()
    return BR.prepare_from(cfg, "crypto", "LTCUSDT", d5, df1, "2024-02-01", "2024-05-25", P, 0.01)


def test_zero_edge():
    cfg = replace(BR.BConfig(), slip_crypto=0.0, n_bench=200)
    trs, b0s, b1s = [], [], []
    t0 = time.time()
    for seed in (31, 32, 33):
        pr = rw_prep(seed)
        tr = BR.simulate(cfg, pr, BR.VARIANTS["V1_full"])
        trs.append(tr)
        b0s.append(BR.benchmark(cfg, pr, tr, "B0").mean())
        b1s.append(BR.benchmark(cfg, pr, tr, "B1").mean())
    tr = pd.concat(trs)
    R = tr.R.to_numpy()
    z = R.mean() / (R.std(ddof=1) / math.sqrt(len(R)))
    fam = tr.groupby("fam").R.agg(["count", "mean"])
    check("random walk, zero costs: BPA-Full mean R ≈ 0 and B0 ≈ B1 ≈ 0 (no built-in edge)",
          len(R) > 400 and abs(z) < 3.5 and abs(np.mean(b0s)) < 0.1 and abs(np.mean(b1s)) < 0.1,
          f"n={len(R)} mean {R.mean():+.3f}R (z={z:+.1f}); B0 {np.mean(b0s):+.3f} B1 {np.mean(b1s):+.3f}; "
          + ", ".join(f"{BC.FAM[int(f)][:2]} {r['count']:.0f}@{r['mean']:+.2f}" for f, r in fam.iterrows())
          + f" ({time.time() - t0:.0f}s)")


def test_claims_null():
    """the bar-18 claim against pure noise: arcsine law says ~55-65% even with no structure"""
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2016-01-04", periods=150)
    rows, idx = [], []
    px = 200.0
    for d in days:
        t0 = pd.Timestamp(d.strftime("%Y-%m-%d") + " 09:30", tz="America/New_York").tz_convert("UTC")
        for k in range(390):
            r = rng.normal(0, 0.0007)
            o = px; c = px * math.exp(r)
            rows.append([o, max(o, c) * (1 + abs(rng.normal(0, 0.0002))), min(o, c) * (1 - abs(rng.normal(0, 0.0002))), c, 100])
            idx.append(t0 + pd.Timedelta(minutes=k))
            px = c
    df1 = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(idx))
    d5 = BR.resample(df1, "5min")
    cfg = BR.BConfig()
    pr = BR.prepare_from(cfg, "spy", "SPY", d5, df1, "2016-02-15", "2016-08-01", BC.pvec(), 0.01)
    res = BR.claims_from(pr)
    k, n = res["D7 day's high or low set by bar 18"]
    p = k / n
    check("claims on a random walk: 'high or low of day by bar 18' ≈ 55-70% from noise alone (Brooks says 90%)",
          0.48 <= p <= 0.70 and n > 80, f"{p:.0%} of {n} random sessions")
    return res


# ---------------------------------------------------------------- 5. workflows on mocked data
def test_crypto_workflow():
    now = pd.Timestamp("2024-11-20 12:00", tz="UTC")
    m1 = {s: TF.synth1m(s, start="2024-01-01", days=320, seed=50 + k, p0=p) for k, (s, p) in
          enumerate((("LTCUSDT", 80.0), ("LINKUSDT", 15.0), ("FILUSDT", 5.0)))}
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    MF.install(m1, now, os.path.join(tmp, "cache"))
    cfg = replace(BR.BConfig(), cache_dir=os.path.join(tmp, "cache"), vision_workers=4, n_bench=50,
                  crypto_dev=("LTCUSDT", "LINKUSDT"), crypto_dev_start="2024-02-15", crypto_dev_end="2024-07-31",
                  crypto_hold_start="2024-08-01", crypto_hold_end="2024-11-10", crypto_hold_syms=("FILUSDT",),
                  min_dev_trades=50)
    os.chdir(tmp)
    try:
        blocked = False
        try:
            BR.check_universe(["LTCUSDT", "ETHUSDT"])
        except SystemExit:
            blocked = True
        check("universe guard: Vegas-holdout symbols (ETH/XRP/BNB/DOGE/ADA) are refused in code", blocked)
        refused = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                BR.cmd_dev(cfg, "crypto")
        except SystemExit as e:
            refused = "firerate" in str(e)
        check("pre-registration: dev refuses to run before the fire-rate report", refused)
        t0 = time.time()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            BR.cmd_download(cfg, "crypto")
            BR.cmd_firerate(cfg, "crypto")
            BR.cmd_claims(cfg, "crypto")
            BR.cmd_dev(cfg, "crypto")
        txt = out.getvalue()
        fr = pd.read_json("reports/brooks_firerate_crypto.json")
        check("crypto: download → fire-rate → claims (vs 2 nulls) → dev (V1-V8 minus F6, B0/B1, gates)",
              len(fr) == 7 and "D3" in txt and ("Verdict" in txt or "candidate" in txt), f"{time.time() - t0:.0f}s")
        print("   " + "\n   ".join(l for l in txt.splitlines() if l.strip().startswith(("V", "candidate", "[", "Verdict", "no variant"))))
        dev = BR.json.load(open("reports/brooks_dev_crypto.json"))
        if dev["chosen"]:
            with contextlib.redirect_stdout(io.StringIO()) as h:
                BR.cmd_holdout(cfg, "crypto")
            again = False
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    BR.cmd_holdout(cfg, "crypto")
            except SystemExit as e:
                again = "already" in str(e)
            check("holdout: time + symbol legs, verdict; second run refused", again, h.getvalue().splitlines()[-1])
        else:
            closed = False
            try:
                BR.cmd_holdout(cfg, "crypto")
            except SystemExit as e:
                closed = "closed" in str(e)
            check("no variant passed dev → holdout refuses (family closed)", closed)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_spy_workflow():
    rng = np.random.default_rng(9)
    days = pd.bdate_range("2015-11-02", "2016-09-30")
    store = {}
    px = 200.0
    for sym in ("SPY", "QQQ", "IWM"):
        rows = []
        for d in days:
            t0 = pd.Timestamp(d.strftime("%Y-%m-%d") + " 09:30", tz="America/New_York").tz_convert("UTC")
            drift = rng.normal(0, 0.0003)
            for k in range(390):
                r = drift + rng.normal(0, 0.0006)
                o = px; c = px * math.exp(r); px = c
                rows.append(dict(t=(t0 + pd.Timedelta(minutes=k)).strftime("%Y-%m-%dT%H:%M:%SZ"), o=o,
                                 h=max(o, c) * (1 + abs(rng.normal(0, 0.0002))), l=min(o, c) * (1 - abs(rng.normal(0, 0.0002))),
                                 c=c, v=1000))
        store[sym] = pd.DataFrame(rows)

    def fake(params):
        s, tf = params["symbols"], params["timeframe"]
        df = store[s]
        ts = pd.to_datetime(df.t, utc=True)
        m = (ts >= pd.Timestamp(params["start"])) & (ts < pd.Timestamp(params["end"]))
        sub = df[m]
        if tf == "5Min":
            g = sub.assign(k=ts[m].dt.floor("5min")).groupby("k")
            sub = pd.DataFrame({"t": g.t.first(), "o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "v": g.v.sum()})
        off = int(params.get("page_token") or 0)
        page = sub.iloc[off:off + params["limit"]]
        nxt = str(off + params["limit"]) if off + params["limit"] < len(sub) else None
        return {"bars": {s: page.to_dict("records")}, "next_page_token": nxt}
    BR.ALPACA_GET = fake
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    LL._now = lambda: pd.Timestamp("2016-10-10", tz="UTC")
    cfg = replace(BR.BConfig(), cache_dir=os.path.join(tmp, "cache"), n_bench=50, min_dev_trades=30,
                  spy_dev_start="2016-01-04", spy_dev_end="2016-05-31", spy_val_start="2016-06-01", spy_val_end="2016-07-31",
                  spy_hold_start="2016-08-01", spy_hold_end="2016-09-30")
    os.chdir(tmp)
    try:
        t0 = time.time()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            BR.cmd_download(cfg, "spy")
            BR.cmd_claims(cfg, "spy")
            BR.cmd_firerate(cfg, "spy")
            BR.cmd_dev(cfg, "spy")
        txt = out.getvalue()
        fr = pd.read_json("reports/brooks_firerate_spy.json")
        check("SPY (Alpaca, RTH): download → claims incl. session stats D7 → fire-rate (8 variants incl. F6) → dev",
              len(fr) == 8 and "D7" in txt and fr.F6_opening.sum() > 0, f"{time.time() - t0:.0f}s; F6 trades {fr.F6_opening.iloc[0]}")
        print("   " + "\n   ".join(l for l in txt.splitlines() if "D7" in l or "bar 18" in l)[:900])
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_speed():
    df1 = MF.gbm1m("2024-01-01", 365, 3, p0=100.0, vol=0.0009)
    d5 = to5(df1)
    cfg = BR.BConfig()
    t0 = time.time()
    pr = BR.prepare_from(cfg, "crypto", "LTCUSDT", d5, df1, "2024-01-15", "2024-12-31", BC.pvec(), 0.01)
    tp = time.time() - t0
    t0 = time.time()
    tr = BR.simulate(cfg, pr, BR.VARIANTS["V1_full"])
    ts = time.time() - t0
    t0 = time.time()
    BR.benchmark(cfg, pr, tr, "B0", reps=1000)
    tb = time.time() - t0
    check("speed: one symbol-year of 5m bars with 1m fills", tp + ts < 30 and tb < 120,
          f"features+engine {tp:.1f}s, simulation {ts:.1f}s ({len(tr)} trades), B0×1000 {tb:.1f}s (1 core)")


# ---------------------------------------------------------------- 6. regressions for bugs fixed in review
def test_regressions():
    P = BC.pvec()
    rows = warm(60)
    p = 100.0
    rows.append([p - 0.2, p + 3.5, p - 0.7, p + 3.4])            # surprise outside bar: low below the prior low
    FE = BC.features(bars(rows), P)
    OUT, *_ = BC.run_engine(FE, P, 0.01, False)
    check("fix 1: a spike bar is not its own pullback (stays BULL_SPIKE on its bar)", OUT[len(rows) - 1, 0] == 1)
    df1 = MF.gbm1m("2024-01-01", 120, 5, p0=100.0, vol=0.0012, drift=0.00004)
    d5 = to5(df1)
    FE = BC.features(d5, P)
    OUT, IT, GATE, EV = BC.run_engine(FE, P, 0.01, False)
    L = FE[:, BC.C_L]
    f1 = np.flatnonzero(IT[:, 0, 0] == 1)
    n_h1 = len([i for i in f1 if OUT[i, 6] == 0 and OUT[i, 0] == 2 and L[i] < L[i - 1]])
    check("fix 2: H1 after a micro channel fires on the pullback bar (channel length read from the prior bar)", n_h1 > 0,
          f"{n_h1} such intents")
    f3 = np.flatnonzero(IT[:, 2, 0] != 0)
    rr = np.abs(IT[f3, 2, 5] - IT[f3, 2, 2]) / np.abs(IT[f3, 2, 2] - IT[f3, 2, 3])
    check("fix 3: F3 obeys the trader's equation (no tradable F3 with reward/risk below 0.4/0.6)",
          len(f3) > 50 and int(((IT[f3, 2, 11] > 0) & (rr < 2 / 3)).sum()) == 0, f"{len(f3)} F3 intents")
    # fix 4: reversal vs protective stop in the same minute follows price order
    n = 4
    O5 = np.full(n, 100.0); H5 = np.array([100.3, 100.3, 100.2, 100.2]); L5 = np.array([99.9, 99.9, 98.5, 98.5])
    C5 = np.full(n, 100.0); z = np.zeros(n); pl = np.full(n, np.nan); pli = np.full(n, -1, np.int64); e = np.zeros(0)
    m0 = np.zeros(n, np.int64)
    allow = np.ones(6, np.bool_)
    for rev_px, want in ((99.5, BS.R_REVERSE), (98.8, BS.R_STOP)):
        IT_ = np.zeros((n, 6, BC.NI))
        IT_[1, 3, 0] = -1; IT_[1, 3, 1] = BC.T_STOP; IT_[1, 3, 2] = rev_px; IT_[1, 3, 11] = 1
        r = BS.run_pos(0, -1, 1, 100.0, 100.0, 1.0, 99.0, 103.0, 0.0, 99.0, 103.0, 0.0, 0, 99, O5, H5, L5, C5, z, z, z,
                       pl, pli, pl, pli, e, e, e, e, m0, m0, 0.01, 0.0, 0.0, True, IT_, allow, 1)
        check(f"fix 4: long stop 99 vs opposite reversal stop {rev_px} in one bar → "
              f"{'reversal first (higher level hit first)' if want == BS.R_REVERSE else 'protective stop first'}",
              r[4] == want and (r[6] == 3 if want == BS.R_REVERSE else r[6] == -1), f"reason {r[4]}, family {r[6]}")
    sw = {}
    for zz in (0.75, 1.5):
        Pz = BC.pvec(zz_atr=zz)
        _, _, _, EVz = BC.run_engine(BC.features(d5, Pz), Pz, 0.01, False)
        sw[zz] = len(EVz["SW"])
    check("parameters are wired by name (a larger zig-zag threshold gives fewer swings)", sw[0.75] > sw[1.5] > 0, str(sw))


# ---------------------------------------------------------------- 7. BPA-1.1 fidelity fixes (from the chart audit)
def rth_sessions(n_days=60, seed=8, drift_sd=0.0004, start="2016-03-01", vol_pm=1.0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    rows, idx = [], []
    px = 200.0
    for d in days:
        t0 = pd.Timestamp(d.strftime("%Y-%m-%d") + " 09:30", tz="America/New_York").tz_convert("UTC")
        px *= math.exp(rng.normal(0, 0.004))                      # overnight gap
        drift = rng.normal(0, drift_sd)
        for k in range(390):
            r = drift + rng.normal(0, 0.0007)
            o = px; c = px * math.exp(r); px = c
            v = 1000.0 if k < 210 else 1000.0 * vol_pm
            rows.append([o, max(o, c) * (1 + abs(rng.normal(0, 0.0002))), min(o, c) * (1 - abs(rng.normal(0, 0.0002))), c, v])
            idx.append(t0 + pd.Timedelta(minutes=k))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(idx))


def test_v11():
    df1 = rth_sessions()
    d5 = BR.resample(df1, "5min")
    P = BC.pvec()
    FE = BC.features(d5, P, "rth")
    OUT, IT, GATE, EV = BC.run_engine(FE, P, 0.01, True)
    sbar, left = FE[:, BC.C_SBAR], FE[:, BC.C_LEFT]
    has = (IT[:, :, 0] != 0).any(axis=1)
    first = sbar == 0
    check("v1.1: no entries on the session's first bar or in its last 20 minutes",
          has.sum() > 50 and not has[first].any() and not has[left < 4].any(), f"{int(has.sum())} signal bars")
    fo = first & ~np.isnan(OUT[:, 0])                              # skip the indicator warm-up bars
    check("v1.1: counts, trend state and always-in reset at the US open",
          fo.sum() > 40 and np.all(OUT[fo, 6] == 0) and np.all(OUT[fo, 7] == 0) and np.all(OUT[fo, 1] == 0)
          and np.all(OUT[fo, 0] == 0), f"{int(fo.sum())} session opens")
    # gap up, then a strong sell-off that never makes a new 20-bar (prior-day) low → always-in flips short
    rows = []
    p = 100.0
    for i in range(80):                                            # day 1: grind up to 104
        rows.append([p, p + 0.3, p - 0.2, p + 0.05]); p += 0.05
    t1 = pd.date_range("2016-03-01 14:30", periods=78, freq="5min", tz="UTC")
    t2 = pd.date_range("2016-03-02 14:30", periods=78, freq="5min", tz="UTC")
    gap = p + 3.0
    day2 = [[gap, gap + 0.1, gap - 0.05, gap]]
    q = gap
    for i in range(4):                                             # four big bear bars, still far above day-1 lows
        day2.append([q, q + 0.02, q - 0.62, q - 0.6]); q -= 0.6
    for i in range(73):
        day2.append([q, q + 0.2, q - 0.2, q])
    idx = t1.append(t2)
    rows = rows[:78] + day2
    d = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)
    FE2 = BC.features(d, P, "rth")
    OUT2, *_ = BC.run_engine(FE2, P, 0.01, True)
    k = 78 + 4
    check("v1.1: a strong sell-off from a gap-up flips always-in short (spikes measured within the session)",
          OUT2[k, 1] == -1 and FE2[k, BC.C_L] > FE2[:78, BC.C_L].min(), f"always-in {OUT2[k, 1]:+.0f}, state {OUT2[k, 0]:+.0f}")
    # F4: every trade is a real second entry
    df1c = MF.gbm1m("2024-01-01", 120, 5, p0=100.0, vol=0.0012, drift=0.00004)
    d5c = to5(df1c)
    FEc = BC.features(d5c, P)
    OUTc, ITc, _, _ = BC.run_engine(FEc, P, 0.01, False)
    H, L, A = FEc[:, BC.C_H], FEc[:, BC.C_L], FEc[:, BC.C_ATR]
    bad = 0
    f4 = np.flatnonzero(ITc[:, 3, 0] != 0)
    for t_ in f4:
        side = ITc[t_, 3, 0]
        ok = False
        for t1_ in range(max(2, t_ - 15), t_ - 1):
            if side > 0:
                if FEc[t1_, BC.C_SIGB] >= 4 and H[t1_ + 1:t_].max(initial=-1e18) >= H[t1_] + 0.01 and \
                        L[t1_ + 1:t_ + 1].min() <= min(L[t1_], L[t1_ - 1]) + 0.25 * A[t1_:t_ + 1].max() + 1e-9:
                    ok = True
                    break
            else:
                if FEc[t1_, BC.C_SIGS] >= 4 and L[t1_ + 1:t_].min(initial=1e18) <= L[t1_] - 0.01 and \
                        H[t1_ + 1:t_ + 1].max() >= max(H[t1_], H[t1_ - 1]) - 0.25 * A[t1_:t_ + 1].max() - 1e-9:
                    ok = True
                    break
        bad += not ok
    check("v1.1: every F4 reversal is a real second entry (first signal triggered, failed, re-tested, new signal)",
          len(f4) > 10 and bad == 0, f"{len(f4)} F4 intents, {bad} without the structure")
    f2 = np.argwhere(ITc[:, 1, 0] != 0).ravel()
    viol = 0
    for t_ in f2:
        side = ITc[t_, 1, 0]
        w = slice(t_ - 4, t_ + 1)
        if side > 0 and (FEc[w, BC.C_SS].sum() >= 2 or FEc[w, BC.C_STB].sum() >= 4):
            viol += 1
        if side < 0 and (FEc[w, BC.C_SB].sum() >= 2 or FEc[w, BC.C_BTB].sum() >= 4):
            viol += 1
    check("v1.1: F2 never fades a strong spike into the range edge", len(f2) > 10 and viol == 0, f"{len(f2)} F2 intents")
    # F6 opening reversals follow a move from the open into the magnet
    f6 = [t_ for t_ in np.flatnonzero(IT[:, 5, 0] != 0) if IT[t_, 5, 1] == BC.T_STOP]
    bad6 = 0
    for t_ in f6:
        d0 = t_ - int(sbar[t_])
        side = IT[t_, 5, 0]
        lo, hi = FE[d0:t_ + 1, BC.C_L].min(), FE[d0:t_ + 1, BC.C_H].max()
        op = FE[t_, BC.C_DOP]
        if side > 0 and not (op - lo >= 0.5 * FE[t_, BC.C_ATR] - 1e-9):
            bad6 += 1
        if side < 0 and not (hi - op >= 0.5 * FE[t_, BC.C_ATR] - 1e-9):
            bad6 += 1
    check("v1.1: F6 opening reversals follow a ≥0.5 ATR move from the open (no more EMA-pullback 'reversals')",
          len(f6) > 5 and bad6 == 0, f"{len(f6)} opening reversals")
    # data cleaning: a bad print and an early-close afternoon
    raw = rth_sessions(n_days=3, seed=3, vol_pm=1.0)
    raw.iloc[100, raw.columns.get_loc("high")] = raw.iloc[100]["close"] * 1.03        # isolated spike, ~20x the 1m range
    hd = rth_sessions(n_days=1, seed=4, start="2018-12-24", vol_pm=0.001)              # after 13:00: almost no volume
    allb = pd.concat([raw, hd])
    with contextlib.redirect_stdout(io.StringIO()):
        cl = BR.clean_rth_1m(allb)
    spike_fixed = cl.loc[raw.index[100], "high"] < raw.iloc[100]["close"] * 1.01        # was +3%
    hd_loc = cl.index.tz_convert("America/New_York")
    late = ((hd_loc.normalize() == pd.Timestamp("2018-12-24", tz="America/New_York")) & (hd_loc.hour >= 13)).sum()
    normal_pm = ((hd_loc.normalize() != pd.Timestamp("2018-12-24", tz="America/New_York")) & (hd_loc.hour >= 13)).sum()
    check("v1.1 data: bad-print wick clipped; half-day after-hours bars dropped; normal afternoons kept",
          spike_fixed and late == 0 and normal_pm == 3 * 180, f"half-day late bars {late}, normal afternoon bars {normal_pm}")


if __name__ == "__main__":
    t0 = time.time()
    test_definitions()
    test_counting()
    test_causality()
    test_fills_exits()
    test_zero_edge()
    test_claims_null()
    test_regressions()
    test_v11()
    test_crypto_workflow()
    test_spy_workflow()
    test_speed()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed in {time.time() - t0:.0f}s")
