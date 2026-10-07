"""Offline tests for langlang.py (no network). Run:  python test_langlang.py
Uses a synthetic market + mock exchange (mockex.py). If langlang_v1.py is present, also runs regression/benchmark vs v1."""
import importlib.util
import math
import re
import shutil
import time

import numpy as np
import pandas as pd

import langlang as v2
import mockex

HAS_V1 = importlib.util.find_spec("langlang_v1") is not None
if HAS_V1:
    import langlang_v1 as v1

PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def setup(n_sym=20, now_offset=pd.Timedelta(days=1)):
    data = mockex.synth(n_sym=n_sym)
    end_ts = max(v.index[-1] for v in data.values())
    mockex.MockExchange.load(data, int((end_ts + now_offset).value // 1_000_000))
    return data, end_ts


def test_basics():
    for s in ["PEOPLEUSDT", "people/usdt", "PEOPLE/USDT:USDT", "PEOPLE-USDT"]:
        check(f"to_ccxt({s!r})", v2.to_ccxt(s) == "PEOPLE/USDT:USDT", v2.to_ccxt(s))
    check("to_ccxt('BTC/USDT') (v1 gave 'BTC//USDT:USDT')", v2.to_ccxt("BTC/USDT") == v2.BTC)
    check("tf_ms 5m/15m/4h/1d", [v2.tf_ms(x) for x in ("5m", "15m", "4h", "1d")] == [300_000, 900_000, 14_400_000, 86_400_000])
    cov = v2._merge_iv([[0, 10], [20, 30], [9, 12]])
    check("interval merge", cov == [[0, 12], [20, 30]], str(cov))
    check("interval missing", v2._missing_iv(cov, -5, 40) == [(-5, 0), (12, 20), (30, 40)])
    check("interval fully covered", v2._missing_iv(cov, 2, 11) == [])


def test_cache():
    data = mockex.synth(n_sym=2)
    end_ts = data[(v2.BTC, "15m")].index[-1]
    now = end_ts - pd.Timedelta(hours=3, minutes=7)
    mockex.MockExchange.load(data, int(now.value // 1_000_000))
    ex = mockex.MockExchange()
    cfg = v2.Config(cache_dir="cache_t_cache")
    shutil.rmtree(cfg.cache_dir, ignore_errors=True)
    calls = lambda: mockex.MockExchange.CALLS
    recent = now - pd.Timedelta(days=3)
    src = data[("C00/USDT:USDT", "15m")]
    c0 = calls(); df1 = v2.fetch_ohlcv(ex, cfg, "C00USDT", "15m", recent)
    last_closed = (now.floor("15min") - pd.Timedelta(minutes=15))
    check("only closed bars returned", df1.index[-1] == last_closed, f"last={df1.index[-1]}")
    check("returned data == source", np.allclose(df1.to_numpy(), src.loc[df1.index].to_numpy()))
    c0 = calls(); v2.fetch_ohlcv(ex, cfg, "C00USDT", "15m", recent)
    check("repeat query → 0 downloads", calls() - c0 == 0, f"{calls() - c0} calls")
    c0 = calls(); h = v2.fetch_ohlcv(ex, cfg, "C00USDT", "15m", "2023-05-13", "2023-05-23")
    check("historical query is [start,end)", h.index[0] == pd.Timestamp("2023-05-13", tz="UTC")
          and h.index[-1] == pd.Timestamp("2023-05-22 23:45", tz="UTC") and len(h) == 960, f"{len(h)} rows")
    c0 = calls(); df3 = v2.fetch_ohlcv(ex, cfg, "C00USDT", "15m", recent)
    check("scan after historical query → 0 downloads (v1: 53)", calls() - c0 == 0, f"{calls() - c0} calls")
    check("no data lost", len(df3) == len(df1))
    # clock moves forward 1h → only the tail is downloaded
    mockex.MockExchange.NOW += 3_600_000
    c0 = calls(); df4 = v2.fetch_ohlcv(ex, cfg, "C00USDT", "15m", recent)
    check("clock +1h → 1 incremental download, +4 bars", calls() - c0 == 1 and len(df4) == len(df1) + 4,
          f"{calls() - c0} calls, +{len(df4) - len(df1)} bars")
    # bad symbol fails fast (v1: 15 s of retries)
    t = time.perf_counter()
    got = v2.fetch_many(ex, v2.Config(cache_dir=cfg.cache_dir, workers=1), [("ZZZUSDT", "1d", "2024-01-01", None)])
    check("bad symbol → skipped fast", got[("ZZZ/USDT:USDT", "1d")] is None and time.perf_counter() - t < 1,
          f"{time.perf_counter() - t:.2f}s")


def test_stage_fix():
    idx = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
    c = np.r_[np.linspace(0.06, 0.026, 200), np.full(54, 0.026), 0.033, 0.044, 0.047, 0.05, 0.052, 0.055]
    c[200:254] += 0.002 * np.sin(np.arange(54))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.01; l = np.minimum(o, c) * 0.99
    h[-6], l[-6] = 0.034, 0.029; h[-5], l[-5] = 0.045, 0.032          # breakout, then a +33% wide candle
    d = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)
    got = v2.classify_stages(d, v2.Config()).stage.iloc[-7:].tolist()
    check("wide follow-through candle: stage 1 for 5 days then 2 (v1: 1→3→4)", got == [0, 1, 1, 1, 1, 1, 2], str(got))
    # a real large divergence must still be detected
    c2 = np.r_[c, np.linspace(0.055, 0.035, 12)]
    o2 = np.r_[c2[0], c2[:-1]]
    d2 = pd.DataFrame({"open": o2, "high": np.maximum(o2, c2) * 1.01, "low": np.minimum(o2, c2) * 0.99,
                       "close": c2, "volume": 1.0}, index=pd.date_range("2024-01-01", periods=len(c2), freq="D", tz="UTC"))
    check("real pullback → stage 3", 3 in v2.classify_stages(d2, v2.Config()).stage.iloc[-12:].tolist())


def test_pattern_equivalence(data):
    if not HAS_V1:
        return
    cfg1, cfg2 = v1.Config(), v2.Config()
    n_cmp = n_sig = 0; mism = []; pend_diff = pend_bad = 0
    for sym in ["C00/USDT:USDT", "C03/USDT:USDT", "C07/USDT:USDT"]:
        m = data[(sym, "15m")].iloc[-20000:]
        A1, L2 = v1.ltf_arrays(m, cfg1), v2.ltf_arrays(m, cfg2)
        for t in range(cfg2.pat_window, len(m)):
            a = v1.find_pattern(A1, t, cfg1); b = v2.find_pattern(L2, t, cfg2)
            n_cmp += 1
            if (a is None) != (b is None) or (a and (a["kind"] != b["kind"] or not np.isclose(a["stop"], b["stop"], rtol=1e-9))):
                mism.append((sym, t))
            n_sig += a is not None
            if t % 7 == 0:                                   # pending mode on a subsample
                pa = v1.find_pattern(A1, t, cfg1, pending=True); pb = v2.find_pattern(L2, t, cfg2, pending=True)
                if (pa is None) != (pb is None):
                    pend_diff += 1
                    pend_bad += not (pa is not None and pb is None)   # only allowed diff: v2 drops broken-down patterns
    check(f"find_pattern identical to v1 on {n_cmp:,} bars ({n_sig} signals)", not mism, f"{len(mism)} mismatches")
    check(f"pending mode: {pend_diff} differences, all are v1 reporting a downside-broken pattern", pend_bad == 0)


def test_causality(data, end_ts):
    """Signals computed on data truncated at T must equal signals from full data before T (no look-ahead)."""
    cfg = v2.Config()
    syms = [s for s, tf in data if tf == "1d"]
    start = data[(syms[0], "15m")].index[0] + pd.Timedelta(days=420)
    full_btc = data[(v2.BTC, "1d")]
    full = {s: (data[(s, "1d")], data[(s, "15m")].loc[start - pd.Timedelta(days=2):]) for s in syms}
    ev_full = v2.build_events(full_btc, full, cfg, start)
    ok_all = True; tested = 0
    for frac in (0.35, 0.6, 0.85):
        T = start + (end_ts - start) * frac
        T = T.floor("15min")
        cut = lambda d, tf: d[d.index + pd.Timedelta(tf) <= T]          # bars closed by T
        trunc = {s: (cut(d, "1D"), cut(m, "15min")) for s, (d, m) in full.items()}
        ev_t = v2.build_events(cut(full_btc, "1D"), trunc, cfg, start)
        lim = (T - pd.Timedelta(minutes=30)).value
        f = lambda evs: [(e["sym"], e["ts"], e["stage"], round(e["stop"], 12), round(e["tp1"], 12)) for e in evs if e["ts"] < lim]
        a, b = f(ev_full), f(ev_t)
        tested += len(a); ok_all &= a == b
    check(f"no look-ahead: truncated == full at 3 cut points ({tested} signals compared)", ok_all and tested > 0)
    return ev_full


def test_accounting():
    cfg = v2.Config()
    tm = v2.TrancheManager(cfg)
    tm.settle(-1000); check("loss refilled from reserve", tm.active == 10000 and tm.reserve == 19000)
    tm.settle(2500); check("profit withdrawn", tm.active == 10000 and tm.withdrawn == 2500)
    for _ in range(10):
        if not tm.halted:
            tm.settle(-tm.margin())                          # isolated: worst case = lose the whole margin
    check("halts only when all 3 tranches are gone", tm.halted and tm.margin() == 0, f"reserve={tm.reserve:.0f} active={tm.active:.0f}")
    tm = v2.TrancheManager(cfg); tm.settle(-1000); tm.settle(500)
    dd = v2.metrics([{"pnl": -1000, "R": -1}, {"pnl": 500, "R": .5}], tm, cfg)["最大回撤(平仓权益)"]
    check("max DD counts a first-trade loss (v1: 0.0)", abs(dd + 1000 / 30000) < 1e-4, str(dd))
    sz = v2.size_position("SOLUSDT", 100, 92, 10000, cfg)
    check("alt leverage ≤5, loss ≤35% of tranche, stop ≥2× closer than liq",
          sz["lev"] <= 5 and sz["risk"] <= 0.35 * 10000 + 1e-9 and (100 - 92) <= 0.5 * (100 - sz["liq"]), str(sz))
    check("BTC can use 10x on a tight stop", v2.size_position("BTCUSDT", 100, 99, 10000, cfg)["lev"] == 10)
    # gap through stop: isolated loss capped at margin
    idx = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    m = pd.DataFrame({"open": [10, 10.0, 6.0], "high": [10, 10.1, 6.1], "low": [10, 9.9, 5.5], "close": [10, 10.0, 6.0]}, index=idx)
    tm = v2.TrancheManager(cfg)
    tr = v2.simulate(v2._ltf_exec_arrays(m, cfg), dict(sym="C/USDT:USDT", t=0, stop=9.3, tp1=13, tp2=15, stage=1, kind="x"), tm, cfg)
    check("gap-through-stop loss capped at isolated margin (v1: -9114 on 5677 margin)",
          tr["why"] == "强平" and tr["pnl"] >= -(tr["margin"] * 1.01), f"pnl={tr['pnl']:.0f} margin={tr['margin']:.0f}")
    # funding charged only at settlement timestamps
    idx = pd.date_range("2024-01-01 06:00", periods=48, freq="15min", tz="UTC")       # 06:00–17:45 spans 08:00 and 16:00
    p = np.full(48, 10.0)
    m = pd.DataFrame({"open": p, "high": p * 1.001, "low": p * 0.999, "close": p}, index=idx)
    c2 = v2.Config(taker_fee=0, slippage=0, max_hold_bars=47)
    tm = v2.TrancheManager(c2)
    tr = v2.simulate(v2._ltf_exec_arrays(m, c2), dict(sym="C/USDT:USDT", t=0, stop=9, tp1=20, tp2=30, stage=1, kind="x"), tm, c2)
    expect = -2 * tr["margin"] * tr["lev"] * c2.funding_rate
    check("funding: exactly 2 settlements (08:00, 16:00 UTC)", abs(tr["pnl"] - expect) < 1e-6, f"{tr['pnl']:.4f} vs {expect:.4f}")


def benchmark(data, end_ts):
    syms = [s for s, tf in data if tf == "1d"]
    start = (data[(syms[0], "15m")].index[0] + pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    res = {}
    if HAS_V1:
        shutil.rmtree("cache_b1", ignore_errors=True)
        cfg1 = v1.Config(cache_dir="cache_b1")
        for run in ("cold", "warm"):
            t = time.perf_counter(); tr1, tm1 = v1.backtest(mockex.MockExchange(), cfg1, syms, start, end)
            res[f"v1 {run}"] = time.perf_counter() - t
    shutil.rmtree("cache_b2", ignore_errors=True)
    cfg2 = v2.Config(cache_dir="cache_b2", workers=4)
    for run in ("cold", "warm"):
        t = time.perf_counter(); tr2, tm2 = v2.backtest(mockex.MockExchange(), cfg2, syms, start, end)
        res[f"v2 {run}"] = time.perf_counter() - t
    btc = data[(v2.BTC, "1d")]
    dd = {s: (data[(s, "1d")], data[(s, "15m")].loc[pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=2):]) for s in syms}
    t = time.perf_counter(); v2.run_backtest(btc, dd, cfg2, start); res["v2 compute only"] = time.perf_counter() - t
    print("\n== benchmark: 20 symbols × 9 months of 15m (mock exchange, zero network latency) ==")
    for k, v in res.items():
        print(f"  {k:18s} {v:6.2f} s")
    if HAS_V1:
        print(f"  speed-up warm: {res['v1 warm'] / res['v2 warm']:.1f}×   cold: {res['v1 cold'] / res['v2 cold']:.1f}×")
        k1 = [(t["sym"], t["entry_time"], t["why"]) for t in tr1]; k2 = [(t["sym"], t["entry_time"], t["why"]) for t in tr2]
        print(f"  trades v1={len(tr1)}  v2={len(tr2)}  same entries: {sum(a == b for a, b in zip(k1, k2))}/{max(len(k1), len(k2))}")
    for k, v in v2.metrics(tr2, tm2, cfg2).items():
        print(f"   v2 {k}: {v}")


def test_smoke(data, end_ts):
    import contextlib, io
    # scan: put "now" inside a summer stretch of the synthetic market
    ev = test_causality.ev
    now = pd.Timestamp(ev[len(ev) // 2]["ts"], tz="UTC") + pd.Timedelta(minutes=16)
    mockex.MockExchange.NOW = int(now.value // 1_000_000)
    ex = mockex.MockExchange()
    cfg = v2.Config(cache_dir="cache_smoke", universe_top_n=19); shutil.rmtree(cfg.cache_dir, ignore_errors=True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        v2.scan(ex, cfg)
        v2.scan(ex, cfg, ["C05USDT", "C06USDT"])          # breadth must still use the whole universe
    txt = out.getvalue()
    seasons = re.findall(r"日线收盘\): (\S+)", txt)
    check("scan runs; season identical with/without --symbols", len(seasons) == 2 and seasons[0] == seasons[1], f"{seasons}")
    print("   " + "\n   ".join(txt.strip().splitlines()[:4]))
    # examples (PEOPLE aliased to a synthetic coin)
    for tf in ("1d", "15m", "1h"):
        mockex.MockExchange.DATA[("PEOPLE/USDT:USDT", tf)] = mockex.MockExchange.DATA[("C00/USDT:USDT", tf)]
    mockex.MockExchange.NOW = int((end_ts + pd.Timedelta(days=1)).value // 1_000_000)
    with contextlib.redirect_stdout(io.StringIO()):
        v2.examples(mockex.MockExchange(), cfg)
    check("examples runs", True)
    # order path: existing position → skip; failed protective order → flatten
    class Broker(mockex.MockExchange):
        def __init__(self, positions=(), fail=False):
            super().__init__(); self.orders = []; self.pos = list(positions); self.fail = fail
        def fetch_positions(self): return self.pos
        def fetch_balance(self): return {"free": {"USDT": 9000.0}}
        def set_margin_mode(self, *a): raise RuntimeError("No need to change margin type")
        def set_leverage(self, *a): pass
        def price_to_precision(self, s, x): return f"{x:.4f}"
        def amount_to_precision(self, s, x): return f"{math.floor(x * 10) / 10:.1f}"
        def create_order(self, s, typ, side, amt, price=None, params=None):
            if self.fail and params and "stopLossPrice" in params: raise RuntimeError("rejected")
            self.orders.append((typ, side, amt, dict(params or {})))
    plan = dict(sym="C01/USDT:USDT", entry=1.0, stop=0.95, tp1=1.2, tp2=1.4)
    b = Broker(positions=[{"symbol": "X", "contracts": 3}])
    with contextlib.redirect_stdout(io.StringIO()): v2.place_bracket(b, cfg, plan)
    check("live: refuses to open while a position exists", b.orders == [])
    b = Broker()
    with contextlib.redirect_stdout(io.StringIO()): v2.place_bracket(b, cfg, plan)
    tps = [o for o in b.orders if "takeProfitPrice" in o[3]]
    check("live: entry + SL + 2 TPs, sizes sum to position, prices rounded",
          len(b.orders) == 4 and abs(sum(o[2] for o in tps) - b.orders[0][2]) < 1e-9 and b.orders[1][3]["stopLossPrice"] == 0.95, str(b.orders[:2]))
    b = Broker(fail=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()): v2.place_bracket(b, cfg, plan)
    except RuntimeError:
        pass
    check("live: SL rejected → position flattened", b.orders[-1][3].get("reduceOnly") and b.orders[-1][1] == "sell")


if __name__ == "__main__":
    test_basics()
    test_cache()
    test_stage_fix()
    data, end_ts = setup()
    test_pattern_equivalence(data)
    test_causality.ev = test_causality(data, end_ts)
    test_accounting()
    test_smoke(data, end_ts)
    benchmark(data, end_ts)
    print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
