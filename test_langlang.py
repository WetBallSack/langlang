"""Offline tests for langlang.py v2.1 (no network).  Run:  python test_langlang.py
Uses synthetic markets: mockex.py (ccxt stand-in) and mockvision.py (data.binance.vision stand-in).
If langlang_v2.py is present, also checks that variant V0 reproduces v2 exactly."""
import contextlib
import importlib.util
import io
import shutil
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import langlang as L
import mockex
import mockvision

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------- helpers
def path_df(prices, start="2024-01-01", wick=0.001):
    idx = pd.date_range(start, periods=len(prices), freq="15min", tz="UTC")
    c = np.asarray(prices, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + wick), "low": np.minimum(o, c) * (1 - wick),
                         "close": c, "volume": 1.0}, index=idx)


def plan_for(m, i_sig, stop, tp1=1e9, tp2=1e9, tp_real=None, sym="TESTUSDT"):
    return dict(sym=sym, ts=int(m.index[i_sig].value), stop=stop, tp1=tp1, tp2=tp2, tp_real=tp_real, stage=1,
                ep=1, kind="收敛三角/楔形", risk_mult=1.0)


def zero_cost(**kw):
    return replace(L.Config(), taker_fee=0.0, slippage=0.0, funding_rate=0.0, funding_source="flat", **kw)


# ---------------------------------------------------------------- A. basics
def test_basics():
    c = L.apply_overrides(L.Config(), ["exit_mode=donchian", "risk_pct=0.005", "universe_n=40", "be_after_tp1=false"])
    check("--set overrides (str/float/int/bool)", (c.exit_mode, c.risk_pct, c.universe_n, c.be_after_tp1)
          == ("donchian", 0.005, 40, False))
    check("next_round: 262→500, 0.42→0.5, 0.99→1, 0.0629→0.1",
          [L.next_round(x) for x in (262, 0.42, 0.99, 0.0629)] == [500, 0.5, 1, 0.1])
    check("plain/to_ccxt", L.plain("sol/usdt:usdt") == "SOLUSDT" and L.to_ccxt("SOLUSDT") == "SOL/USDT:USDT")


# ---------------------------------------------------------------- B/C/D. sizing, account, cooldown
def test_sizing_account_cooldown():
    cfg = L.Config()
    sz = L.size_risk("SOLUSDT", 100, 98, 90, 3000, cfg)                 # 1% of 9k, 2% stop
    check("risk sizing: $90 at a 2% stop → 2x, notional ≈ $4,091, loss at stop incl. costs = $90",
          sz["lev"] == 2 and abs(sz["notional"] - 90 / 0.022) < 1e-6
          and abs(sz["notional"] * (0.02 + 0.002) - 90) < 1e-9, f"{sz['lev']}x, {sz['notional']:.0f}")
    sz = L.size_risk("SOLUSDT", 100, 98.5, 135, 3000, cfg)              # 1.5% risk at a 1.5% stop
    check("leverage rises only when margin would exceed one tranche (→3x)", sz["lev"] == 3 and sz["margin"] <= 3000,
          f"{sz['lev']}x margin {sz['margin']:.0f}")
    sz = L.size_risk("SOLUSDT", 100, 92, 90, 3000, cfg)
    check("stop stays well before liquidation", (100 - 92) <= 0.5 * (100 - sz["liq"]) or sz["lev"] == 1)
    acct = L.Account(cfg)
    acct.settle(-90); acct.settle(+500)
    check("account: loss refilled from reserve, profit swept to vault",
          acct.active == 3000 and abs(acct.reserve - 5910) < 1e-9 and abs(acct.vault - 500) < 1e-9)
    acct = L.Account(cfg)
    for _ in range(3):
        acct.settle(-80)
    check("after 3 straight losses risk is halved", abs(acct.risk_usd() - 0.5 * 0.01 * acct.risk_base()) < 1e-9)
    acct.settle(+10)
    check("…and restored after a win", abs(acct.risk_usd() - 0.01 * acct.risk_base()) < 1e-9)
    acct = L.Account(cfg)
    while not acct.halted:
        acct.settle(-100)
    check("halts once trading capital < 75% of start", acct.risk_base() < 6750 and acct.risk_base() > 6600)
    cd = L.Cooldown(cfg)
    H = 3600 * 10**9
    cd.record(dict(sym="X", ep=1, pnl=-1, exit_ts=100 * H))
    check("cooldown: blocked 24h after a loss", not cd.allowed("X", 1, 124 * H, 110 * H))
    check("cooldown: allowed after 48h with a NEW structure", cd.allowed("X", 1, 149 * H, 120 * H))
    check("cooldown: old structure (started before the loss) is refused", not cd.allowed("X", 1, 149 * H, 90 * H))
    cd.record(dict(sym="X", ep=1, pnl=-1, exit_ts=150 * H))
    check("max 2 losses per (coin, episode)", not cd.allowed("X", 1, 400 * H, 300 * H) and cd.allowed("X", 2, 400 * H, 300 * H))


# ---------------------------------------------------------------- E. exits on hand-built paths
def test_exits():
    rng = np.random.default_rng(1)
    flat = 100 + rng.normal(0, 0.15, 192)
    up = np.linspace(100, 130, 288)
    down = np.linspace(130, 95, 192)
    m = path_df(np.r_[flat, up, down])
    cfg = zero_cost()
    X = L.ExecData(m, cfg)
    acct = L.Account(cfg)
    tr = L.simulate(X, plan_for(m, 191, stop=97.0), acct, cfg)
    # independent check of the 4h chandelier with pandas resampling
    b4 = m.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    pc = b4.close.shift().fillna(b4.close)
    tr4 = np.maximum(b4.high - b4.low, np.maximum((b4.high - pc).abs(), (b4.low - pc).abs()))
    atr4 = tr4.ewm(alpha=1 / 14, adjust=False).mean()
    entry = m.open.iloc[192]
    R = entry - 97.0
    stop, hh, exp_exit = 97.0, entry, None
    for i in range(192, len(m)):
        if m.low.iloc[i] <= stop:
            exp_exit = min(m.open.iloc[i], stop)
            break
        hh = max(hh, m.high.iloc[i])
        t_close = m.index[i] + pd.Timedelta(minutes=15)
        if t_close.hour % 4 == 0 and t_close.minute == 0 and hh >= entry + R:
            stop = max(stop, hh - 3 * atr4.loc[m.index[i].floor("4h")])
    check("chandelier: exits on the trail after the peak, price matches an independent pandas calc",
          tr["why"] == "trail" and exp_exit is not None and abs(tr["exit"] - exp_exit) < 1e-9 and tr["R"] > 3,
          f"exit {tr['exit']:.3f} vs {exp_exit:.3f}, {tr['R']:+.2f}R, peak 130")
    # never armed: +0.8R then down to the stop
    p = np.r_[flat, np.linspace(100, 102.4, 96), np.linspace(102.4, 95, 96)]
    m2 = path_df(p)
    tr = L.simulate(L.ExecData(m2, cfg), plan_for(m2, 191, stop=97.0), L.Account(cfg), cfg)
    check("not armed below +1R: full −1R at the initial stop (no breakeven move)",
          tr["why"] == "stop" and abs(tr["R"] + 1) < 0.02 and tr["mfe_R"] < 1, f"{tr['R']:+.3f}R, MFE {tr['mfe_R']:.2f}R")
    # stale: nothing happens for days
    m3 = path_df(np.r_[flat, 100 + rng.normal(0, 0.1, 700)])
    tr = L.simulate(L.ExecData(m3, cfg), plan_for(m3, 191, stop=97.0), L.Account(cfg), cfg)
    check("stale exit after 5 days without reaching +1R", tr["why"] == "stale" and abs(tr["hold_d"] - 479 / 96) < 1e-9,
          f"{tr['hold_d']:.2f} days")
    # donchian: exits at the 10-day low of completed daily bars
    p = np.r_[flat, np.linspace(100, 160, 96 * 15), np.linspace(160, 100, 96 * 3)]
    m4 = path_df(p)
    cd_ = replace(cfg, exit_mode="donchian")
    tr = L.simulate(L.ExecData(m4, cd_), plan_for(m4, 191, stop=97.0), L.Account(cd_), cd_)
    dlow = m4.low.resample("1D").min()
    exp_stop = dlow.loc[: tr["exit_time"].floor("D") - pd.Timedelta(days=1)].iloc[-10:].min()
    check("donchian: exit at the prior 10-day low", tr["why"] == "trail" and abs(tr["exit"] - exp_stop) < 1e-9,
          f"exit {tr['exit']:.3f} vs {exp_stop:.3f}")
    # partial: 1/3 at +2R (no real resistance) then trail
    cp = replace(cfg, exit_mode="partial")
    trp = L.simulate(L.ExecData(m, cp), plan_for(m, 191, stop=97.0), L.Account(cp), cp)
    trc = L.simulate(L.ExecData(m, cfg), plan_for(m, 191, stop=97.0), L.Account(cfg), cfg)
    expR = cp.partial_frac * 2 + (1 - cp.partial_frac) * trc["R"]
    check("partial: 1/3 banked at +2R, rest trails like the chandelier", abs(trp["R"] - expR) < 1e-6,
          f"{trp['R']:+.3f}R vs expected {expR:+.3f}R")
    # gap in the data: exit at the last bar before it
    mg = pd.concat([m.iloc[:300], m.iloc[300:].set_axis(m.index[300:] + pd.Timedelta(days=3))])
    tr = L.simulate(L.ExecData(mg, cfg), plan_for(mg, 191, stop=97.0), L.Account(cfg), cfg)
    check("data gap → exit at the last bar before it", tr["why"] == "gap" and tr["exit_time"] == mg.index[299])
    # real funding: charged at each settlement held, at that bar's price
    mf = path_df(np.r_[flat, np.full(400, 100.0)], wick=0)
    fund = pd.DataFrame({"rate": 0.001, "interval_h": 8},
                        index=pd.date_range("2024-01-01", periods=20, freq="8h", tz="UTC"))
    cr = replace(cfg, funding_source="real", stale_bars=10**6, trail_activate_R=99)
    X = L.ExecData(mf, cr, fund)
    tr = L.simulate(X, plan_for(mf, 191, stop=97.0), L.Account(cr), cr)
    held = fund.index[(fund.index > mf.index[192]) & (fund.index <= tr["exit_time"])]
    qty = tr["notional"] / tr["entry"]
    exp_pnl = qty * (tr["exit"] - tr["entry"]) - sum(0.001 * qty * mf.open.loc[t] for t in held)
    check("real funding: paid at every settlement while held", abs(tr["pnl"] - exp_pnl) < 1e-6,
          f"{len(held)} settlements, pnl {tr['pnl']:.2f} vs {exp_pnl:.2f}")
    # gap through the stop: isolated loss capped at the margin
    ml = path_df(np.r_[flat, 100, 100, 100, 40, 40])
    ml.iloc[195, ml.columns.get_loc("open")] = 40.0                     # opens 60% below: gaps through the stop
    ml.iloc[195, ml.columns.get_loc("high")] = 40.1
    cl = L.Config()
    tr = L.simulate(L.ExecData(ml, cl), plan_for(ml, 191, stop=97.0), L.Account(cl), cl)
    check("gap through stop: loss capped at isolated margin (+fee)",
          tr["why"] == "liq" and abs(tr["pnl"] + tr["margin"] + tr["entry"] * tr["notional"] / tr["entry"] * cl.taker_fee) < 1e-6,
          f"{tr['why']}, pnl {tr['pnl']:.0f}, margin {tr['margin']:.0f}")
    # fixed mode in v2.1 = v2 behaviour (TP1 half, breakeven, TP2)
    cfx = replace(cfg, exit_mode="fixed")
    tr = L.simulate(L.ExecData(m, cfx), plan_for(m, 191, stop=97.0, tp1=110, tp2=125), L.Account(cfx), cfx)
    check("fixed mode: 50% at TP1, rest at TP2", tr["why"] == "tp2" and abs(tr["R"] - (0.5 * (110 - entry) + 0.5 * (125 - entry)) / R) < 1e-6)


# ---------------------------------------------------------------- F. no look-ahead (static data, all event features)
def test_causality_static():
    data = mockex.synth()
    syms = [s for s, tf in data if tf == "1d"]
    start = data[(syms[0], "15m")].index[0] + pd.Timedelta(days=420)
    end = max(v.index[-1] for v in data.values()) + pd.Timedelta(days=1)
    btc = data[("BTC/USDT:USDT", "1d")]
    daily = {L.plain(s): data[(s, "1d")] for s in syms}
    ltf = {L.plain(s): data[(s, "15m")] for s in syms}
    for name in ("V1_core_4hCE", "V6_stage1_h4", "V7_sector_heat"):
        cfg = replace(L.Config(), **L.VARIANTS[name], universe_mode="static")
        full = L.build_events(L.Dataset.from_frames(btc, daily, ltf), cfg, start, end)
        ok, n = True, 0
        for frac in (0.4, 0.7, 0.9):
            T = (start + (end - start) * frac).floor("15min")
            cut = lambda d, tf: d[d.index + pd.Timedelta(tf) <= T]
            ds_t = L.Dataset.from_frames(cut(btc, "1D"), {s: cut(d, "1D") for s, d in daily.items()},
                                         {s: cut(m, "15min") for s, m in ltf.items()})
            ev_t = L.build_events(ds_t, cfg, start, end)
            lim = (T - pd.Timedelta(minutes=30)).value
            f = lambda E: [(e["sym"], e["ts"], e["stage"], e["ep"], round(e["stop"], 10), round(e["risk_mult"], 6))
                           for e in E if e["ts"] < lim]
            a, b = f(full), f(ev_t)
            n += len(a)
            ok &= a == b
        check(f"no look-ahead [{name}]: truncated == full at 3 cut points", ok and n > 0, f"{n} signals compared")


# ---------------------------------------------------------------- G. archive layer + point-in-time universe
def vision_setup(now, cache):
    mv = mockvision.MockVision(MARKET, now=now)
    L.HTTP_GET = mv.get
    L._now = lambda: pd.Timestamp(now)
    shutil.rmtree(cache, ignore_errors=True)
    return mv


def test_vision():
    now = pd.Timestamp("2024-11-20 12:00", tz="UTC")
    mv = vision_setup(now, "cache_t_vis")
    cfg = replace(L.Config(), cache_dir="cache_t_vis", universe_n=8, vision_workers=4)
    vis = L.Vision(cfg)
    syms = vis.symbols()
    check("symbol list: delisted/new coins kept; stablecoin, index, stock and delivery contracts excluded",
          "DEADUSDT" in syms and "NEWUSDT" in syms and not {"USDCUSDT", "BTCDOMUSDT", "AMZNUSDT"} & set(syms)
          and not any("_" in s for s in syms), f"{len(syms)} symbols")
    months = vis.months_available("C04USDT", "klines", "1d")
    check("S3 listing pagination (4 keys per page) returns every monthly file", len(months) == 22, f"{len(months)} months")
    src = MARKET["C04USDT"]
    k = vis.klines("C04USDT", "15m", ["2023-06", "2024-02", "2024-11"])
    ref = src.loc[k.index]
    check("header-less (2023) and header (2024) files parse identically to the source",
          np.allclose(k[["open", "high", "low", "close"]].to_numpy(), ref[["open", "high", "low", "close"]].to_numpy())
          and set(k.index.strftime("%Y-%m")) == {"2023-06", "2024-02", "2024-11"})
    check("current month comes from daily files, up to yesterday",
          k.index[-1] == now.floor("D") - pd.Timedelta(minutes=15), str(k.index[-1]))
    dead = vis.klines("DEADUSDT", "1d", [L.mkey(m) for m in L.month_starts("2024-04-01", "2024-09-01")])
    check("delisted coin: archive's flat zero-trade rows are cut", dead.index[-1] < pd.Timestamp("2024-06-15", tz="UTC"),
          str(dead.index[-1].date()))
    f = vis.funding("C04USDT", ["2024-03"])
    check("funding archive: 8h settlements, times rounded, rates parsed",
          len(f) == 93 and f.index[0] == pd.Timestamp("2024-03-01", tz="UTC") and f.rate.between(0.0001, 0.00051).all())
    c0 = mv.calls
    with quiet():
        ds = L.build_pit_dataset(cfg, "2024-03-01", "2024-11-21")
    first_calls = mv.calls - c0
    u = {L.mkey(m): v for m, v in ds.universe.items()}
    check("PIT universe: NEW coin only after 120 days listed", "NEWUSDT" not in u["2024-06"] and "NEWUSDT" in u["2024-08"],
          f"2024-08: {u['2024-08'][:3]}")
    check("PIT universe: delisted coin drops out after it stops trading",
          "DEADUSDT" in sum(list(u.values())[:3], []) and all("DEADUSDT" not in u[m] for m in ("2024-07", "2024-08")))
    c1 = mv.calls
    with quiet():
        L.build_pit_dataset(cfg, "2024-03-01", "2024-11-21")
    check("second build is served from cache", mv.calls - c1 == 0, f"first build {first_calls} requests, second {mv.calls - c1}")
    return ds, cfg


# ---------------------------------------------------------------- H. end-to-end PIT: no look-ahead + experiment
def test_pit_causality_and_experiment(ds_full, cfg):
    start, end = "2024-03-01", "2024-11-21"
    for name in ("V1_core_4hCE", "V6_stage1_h4", "V7_sector_heat"):
        c = replace(cfg, **{k: v for k, v in L.VARIANTS[name].items()}, universe_n=8)
        full = L.build_events(ds_full, c, start, end)
        T = pd.Timestamp("2024-09-17 00:00", tz="UTC")
        vision_setup(T + pd.Timedelta(hours=9), "cache_t_vis_T")
        cT = replace(c, cache_dir="cache_t_vis_T")
        with quiet():
            dsT = L.build_pit_dataset(cT, start, end)
        evT = L.build_events(dsT, cT, start, end)
        lim = (T - pd.Timedelta(minutes=30)).value
        f = lambda E: [(e["sym"], e["ts"], e["stage"], round(e["stop"], 10)) for e in E if e["ts"] < lim]
        check(f"PIT end-to-end no look-ahead [{name}]: archive cut at {T.date()} gives the same signals",
              f(full) == f(evT) and len(f(full)) > 0, f"{len(f(full))} signals")
    vision_setup(pd.Timestamp("2024-11-20 12:00", tz="UTC"), "cache_t_vis_x")
    folds = [("2024-03-01", "2024-07-01", "holdout"), ("2024-07-01", "2024-11-21", "dev")]
    out = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(out):
        res, summary, gates = L.run_experiment(replace(cfg, cache_dir="cache_t_vis_x"), ds=ds_full, folds=folds,
                                               start=start, end=end)
    txt = out.getvalue()
    check("experiment runner: all 8 variants run, fold table + selection + gates printed",
          len(res) == 8 and ("pass/fail gates" in txt or "No variant passes" in txt),
          f"{time.perf_counter() - t:.0f}s, trades per variant: {[len(r['trades']) for r in res.values()]}")
    print("   " + "\n   ".join(l for l in txt.splitlines() if "ALL" in l or "selected" in l or "PASS" in l or "FAIL" in l))


# ---------------------------------------------------------------- I. live scan (ccxt mock)
def test_scan():
    data = mockex.synth()
    ev = L.build_events(L.Dataset.from_frames(data[("BTC/USDT:USDT", "1d")],
                                              {L.plain(s): d for (s, tf), d in data.items() if tf == "1d"},
                                              {L.plain(s): d for (s, tf), d in data.items() if tf == "15m"}),
                        replace(L.Config(), universe_mode="static"),
                        data[("BTC/USDT:USDT", "15m")].index[0] + pd.Timedelta(days=420),
                        data[("BTC/USDT:USDT", "15m")].index[-1])
    now = pd.Timestamp(ev[len(ev) // 2]["ts"], tz="UTC") + pd.Timedelta(minutes=16)
    mockex.MockExchange.load(data, int(now.value // 1_000_000))
    cfg = replace(L.Config(), cache_dir="cache_t_scan", universe_n=19)
    shutil.rmtree(cfg.cache_dir, ignore_errors=True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        L.scan(mockex.MockExchange(), cfg)
        L.scan(mockex.MockExchange(), cfg, ["C05USDT"], execute=True)
    txt = out.getvalue()
    check("scan runs with v2.1 sizing; --execute refuses for trailing exits",
          "season" in txt and "disabled for trailing exits" in txt or "no coin currently" in txt)
    print("   " + "\n   ".join(txt.strip().splitlines()[:5]))


def test_regression_v2():
    if importlib.util.find_spec("langlang_v2") is None:
        return
    import langlang_v2 as v2
    data = mockex.synth()
    syms = [s for s, tf in data if tf == "1d"]
    start = data[(syms[0], "15m")].index[0] + pd.Timedelta(days=420)
    end = max(v.index[-1] for v in data.values()) + pd.Timedelta(days=1)
    btc = data[(v2.BTC, "1d")]
    dd2 = {s: (data[(s, "1d")], data[(s, "15m")].loc[start - pd.Timedelta(days=2):]) for s in syms}
    tr2, tm2 = v2.run_backtest(btc, dd2, v2.Config(), start)
    cfg = replace(L.Config(), **L.VARIANTS["V0_v2_control"])
    cfg = replace(cfg, total_capital=30000, funding_source="flat", universe_mode="static")
    ds = L.Dataset.from_frames(btc, {L.plain(s): dd2[s][0] for s in syms}, {L.plain(s): dd2[s][1] for s in syms})
    t = time.perf_counter()
    tr21, acct = L.run_backtest(ds, cfg, start, end)
    same = len(tr2) == len(tr21) and all(
        L.plain(a["sym"]) == b["sym"] and a["entry_time"] == b["entry_time"] and np.isclose(a["pnl"], b["pnl"], rtol=1e-9)
        for a, b in zip(tr2, tr21))
    check(f"V0 control reproduces v2 trade-for-trade ({len(tr21)} trades)", same and abs(tm2.equity() - acct.equity()) < 1e-6,
          f"{time.perf_counter() - t:.2f}s")
    c1 = replace(L.Config(), universe_mode="static", funding_source="flat")
    t = time.perf_counter()
    tr, ac = L.run_backtest(ds, c1, start, end)
    mt = L.metrics(tr, ac, c1)
    print(f"   V1 on the same synthetic market: {mt['trades']} trades, avgR {mt['avg_R']}, maxDD {mt['max_DD']}, "
          f"{time.perf_counter() - t:.2f}s")


MARKET = None

if __name__ == "__main__":
    t0 = time.perf_counter()
    test_basics()
    test_sizing_account_cooldown()
    test_exits()
    test_regression_v2()
    test_causality_static()
    MARKET = mockvision.make_market()
    ds, cfg = test_vision()
    test_pit_causality_and_experiment(ds, cfg)
    test_scan()
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed ({time.perf_counter() - t0:.0f}s)")
