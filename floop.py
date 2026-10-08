#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floop.py — FLOOP Pro v7.1, ported to Python for honest backtesting (GitHub Codespaces) and paper trading (Azure).

V0 reproduces live v7.1 exactly: every 5m tick recomputes all indicators on the last 1000 closed 5m bars (300×15m,
200×1h, 200×4h), the frozen regime switch (2-tick confirmation), reversion inversion, fitted-GARCH brackets with pivot
stop-snap, Bayesian-quantile Kelly with 30 warm-up probes, correlation/exposure scalars, EV gate, 15× isolated,
30%-of-margin hard stop and the $200 daily loss limit. Exits are simulated on 1m MARK-price bars (as the live
STOP_MARKET/TAKE_PROFIT_MARKET use workingType=MARK_PRICE), stop checked before target inside a minute.

Commands (global options go before the command):
  python floop.py download                      # 5m klines + 1m mark klines + funding (one-time, cached)
  python floop.py backtest --start 2024-01-01 --end 2025-01-01 [--set variant=V2]
  python floop.py experiment                    # 8 pre-registered variants × folds × gates
  python floop.py paper                         # one paper tick (Azure cron/systemd every 5 min); no orders, no keys
  python floop.py golden live.json              # compare with the live bot's ?test=indicators output
Requires langlang.py (archive downloader) in the same folder.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from numba import njit, prange

import langlang as LL

# ================= 1. CONFIG =================
@dataclass
class FConfig:
    cache_dir: str = "./cache"
    vision_workers: int = 8
    symbols: str = "BTCUSDT,SOLUSDT"          # backtest symbols (USDT perps = long history proxies for the USDC pairs)
    live_symbols: str = "BTCUSDC,SOLUSDC,MSTRUSDT,QQQUSDT"
    variant: str = "V0"
    # ---- v7.1 constants (frozen in V0) ----
    window: int = 1000
    htf_window: int = 300
    mtf_window: int = 200
    leverage: float = 15.0
    hard_stop_pct: float = 0.30             # of margin
    daily_loss_limit: float = 200.0
    min_strength: int = 10
    adx_thresh: float = 25.0
    chop_thresh: float = 55.0
    cooldown_bars: int = 5
    kelly_fraction: float = 0.5
    kelly_lookback: int = 30
    min_trades_for_kelly: int = 10
    min_margin: float = 50.0
    max_margin: float = 600.0
    kelly_quantile: float = 0.25
    prior_strength: float = 8.0
    warmup_trades: int = 30
    min_trades_per_mode: int = 8
    mode_log_cap: int = 200
    hist_r_blend: float = 0.15
    max_stop_widen: float = 1.5
    corr_lookback: int = 288
    beta_lookback: int = 576
    max_factor_exposure: float = 2.0
    capital: float = 9000.0
    # live v7.1 only records a close into the $200 daily-loss ledger and the mode-labelled Kelly log when the
    # bot itself closes it (hard stop / time stop) or a bracket fills during the few seconds of a tick: main()
    # snapshots prevPositionMap at the START of the same tick, so bracket fills between ticks are never seen.
    ledger: str = "live"                      # live = reproduce that (V0 exact) | all = bug fixed (diagnostic)
    # v7.1 Kelly can freeze for good: once the rolling 30-trade sample says "no edge at q25", nothing trades, so the
    # sample never changes. unfreeze_hours > 0 = proposed fix: if Kelly blocks and the newest trade in the sample is
    # older than this, take a $50 probe anyway so the sample keeps updating (exploration capped ~1 probe/period).
    unfreeze_hours: float = 0.0               # 0 = off (v7.1 exact)
    # ---- variants ----
    sizing: str = "kelly"                     # kelly (v7.1) | fixed (V1)
    risk_pct: float = 0.005                   # V1
    max_gross_mult: float = 2.0               # V1 aggregate notional / equity
    trend_only: bool = False                  # V2
    mode_rule: str = "v71"                    # v71 | hysteresis (V3)
    persist_ticks: int = 12                   # V3
    trail_trend: bool = False                 # V4
    trail_mult: float = 3.0                   # V4 (in units of entry vol)
    time_stop: float = 0.0                    # V5: multiple of slMult*tpMult bars (0 = off)
    flow_filter: bool = False                 # V7
    universe: str = "fixed"                   # fixed | pit (V6)
    universe_n: int = 10
    # ---- execution / costs ----
    exit_tf: str = "1m"                       # 1m mark-price exits | 5m (fallback, stop-first)
    fee: float = 0.0                          # KOL: zero
    entry_slip: float = 0.0001
    tp_slip: float = 0.0001
    stop_slip: float = 0.0003
    mmr: float = 0.005


# Pre-registered BEFORE any real-data run. Amended (still before any real data was seen) after the offline suite
# showed that v7.1's Kelly freezes permanently once the rolling 30-trade sample's q25 win rate falls below break-even
# (no trades -> sample never updates). Variants built on V0's Kelly would inherit random freezes, so V2-V7 each add ONE
# change on top of V1 (same signals, fixed 0.5% risk, 2x) and every comparison isolates exactly one change.
FIXED = dict(sizing="fixed", leverage=2.0, hard_stop_pct=0.04)                  # hard stop stays a 2% price move
VARIANTS = OrderedDict([
    ("V0_v71_exact", dict()),                                                   # live v7.1, freeze included
    ("V1_fixed_risk", dict(FIXED)),
    ("V2_trend_only", dict(FIXED, trend_only=True)),
    ("V3_hysteresis", dict(FIXED, mode_rule="hysteresis")),
    ("V4_trail_trend", dict(FIXED, trail_trend=True)),
    ("V5_time_stop", dict(FIXED, time_stop=2.0)),
    ("V6_pit_top10", dict(FIXED, universe="pit")),
    ("V7_flow_pullback", dict(FIXED, flow_filter=True)),
])
SIMPLICITY = ["V0_v71_exact", "V1_fixed_risk", "V2_trend_only", "V5_time_stop", "V3_hysteresis", "V4_trail_trend",
              "V7_flow_pullback", "V6_pit_top10"]
NEIGHBOURS = {"V0_v71_exact": [("min_strength", 9), ("min_strength", 11), ("adx_thresh", 20.0), ("adx_thresh", 30.0)],
              "V1_fixed_risk": [("risk_pct", 0.0025), ("risk_pct", 0.01)],
              "V2_trend_only": [("min_strength", 9), ("min_strength", 11)],
              "V3_hysteresis": [("persist_ticks", 6), ("persist_ticks", 24)],
              "V4_trail_trend": [("trail_mult", 2.0), ("trail_mult", 4.0)],
              "V5_time_stop": [("time_stop", 1.0), ("time_stop", 3.0)],
              "V6_pit_top10": [("min_strength", 9), ("min_strength", 11)],
              "V7_flow_pullback": [("min_strength", 9), ("min_strength", 11)]}
FOLDS = LL.FOLDS
EXP_START, EXP_END = LL.EXP_START, LL.EXP_END
NS5 = 300 * 10**9
NS1 = 60 * 10**9
DAY = 86_400 * 10**9
TREND, REV, NEUTRAL = 1, 2, 0


def plist(s):
    return [LL.plain(x) for x in s.split(",") if x.strip()]


# ================= 2. DATA =================
KC = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
      "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _parse_full(raw: bytes, mark=False) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
    if len(df) and not LL._is_num(df.iat[0, 0]):
        df = df.iloc[1:]
    if df.empty:
        return pd.DataFrame()
    ot = pd.to_numeric(df.iloc[:, 0]).astype("int64")
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    idx = pd.to_datetime(ot.to_numpy(), unit=unit, utc=True).as_unit("ns")
    cols = {"open": 1, "high": 2, "low": 3, "close": 4}
    if not mark:
        cols.update(volume=5, count=8, tbv=9)
    return pd.DataFrame({k: pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy(dtype=float)
                         for k, i in cols.items()}, index=idx)


class FVision(LL.Vision):
    """adds 5m klines with taker-buy volume and 1m mark-price klines to the langlang archive layer"""
    PREFIX = {"klines": "klines", "mark": "markPriceKlines"}

    def avail(self, sym, kind, tf):
        path = os.path.join(self.root, "k", f"floop_{sym}_{kind}_{tf}.avail.json")
        hit = LL._load_json(path)
        if hit and time.time() - hit["t"] < 86_400:
            return set(hit["months"])
        _, keys = LL.s3_list(f"data/futures/um/monthly/{self.PREFIX[kind]}/{sym}/{tf}/")
        import re
        months = sorted({m.group(1) for k in keys if (m := re.search(r"-(\d{4}-\d{2})\.zip$", k))})
        LL._save_json({"t": time.time(), "months": months}, path)
        return set(months)

    def series(self, sym, kind, tf, months):
        avail = self.avail(sym, kind, tf)
        now = LL._now()
        recent = {LL.mkey(now), LL.mkey(now.floor("D").replace(day=1) - pd.Timedelta(days=1))}
        alive = bool(avail) and max(avail) >= LL.mkey(now - pd.Timedelta(days=75))
        months = sorted(m for m in months if m in avail or (alive and m in recent))
        pre, mark = self.PREFIX[kind], kind == "mark"

        def fm(m):
            if m in avail:
                raw = self._get_zip(f"data/futures/um/monthly/{pre}/{sym}/{tf}/{sym}-{tf}-{m}.zip", cache=False)
                return _parse_full(raw, mark) if raw else None
            parts, d0 = [], pd.Timestamp(m + "-01", tz="UTC")
            for d in pd.date_range(d0, d0 + pd.offsets.MonthEnd(0), freq="D"):
                if d >= now.floor("D"):
                    break
                raw = self._get_zip(f"data/futures/um/daily/{pre}/{sym}/{tf}/{sym}-{tf}-{d:%Y-%m-%d}.zip")
                if raw:
                    parts.append(_parse_full(raw, mark))
            return pd.concat(parts) if parts else None

        def trunc(df):
            if mark or "count" not in df:
                return df
            live = df.index[df["count"] > 0]
            return df.loc[: live.max()] if len(live) else df.iloc[:0]

        return self._combined(f"floop_{sym}_{kind}_{tf}", months, fm, trunc)


@dataclass
class SymData:
    sym: str
    m5: pd.DataFrame
    mark1: pd.DataFrame | None
    funding: pd.DataFrame | None
    member_months: set | None = None      # PIT universe (None = always tradable)


def load_symbols(cfg: FConfig, syms, start, end, member=None, verbose=True):
    """5m klines from 40 days before start (window + 4h warm-up), 1m mark klines and funding."""
    vis = FVision(replace(LL.Config(), cache_dir=cfg.cache_dir, vision_workers=cfg.vision_workers))
    m_ext = [LL.mkey(m) for m in LL.month_starts(LL._utc(start) - pd.Timedelta(days=40),
                                                 LL._utc(end) + pd.Timedelta(days=31))]
    say = print if verbose else (lambda *a, **k: None)

    def one(s):
        months = m_ext if member is None else sorted({mm for m in member.get(s, set()) for mm in
                                                       [LL.mkey(pd.Timestamp(m + "-01", tz="UTC") + pd.DateOffset(months=k))
                                                        for k in (-2, -1, 0, 1)]} & set(m_ext))
        try:
            m5 = vis.series(s, "klines", "5m", months)
            if m5 is None or m5.empty:
                return s, None
            mk = vis.series(s, "mark", "1m", months) if cfg.exit_tf == "1m" else None
            fu = vis.funding(s, months)
            return s, SymData(s, m5, mk, fu, None if member is None else set(member.get(s, set())))
        except Exception as e:                                     # noqa: BLE001
            say(f"   skip {s}: {type(e).__name__}: {str(e)[:100]}")
            return s, None

    t0 = time.time()
    with ThreadPoolExecutor(min(cfg.vision_workers, max(1, len(syms)))) as pool:
        out = {s: d for s, d in pool.map(one, syms) if d is not None}
    say(f"   data: {len(out)}/{len(syms)} symbols loaded in {time.time() - t0:.0f}s"
        + ("" if cfg.exit_tf != "1m" else
           f" (1m mark data for {sum(1 for d in out.values() if d.mark1 is not None and len(d.mark1))})"))
    return out


# ================= 3. NUMBA KERNELS — exact v7.1 window semantics =================
@njit(cache=True)
def _rma(vals, period, out):
    """calcRMA: seed = mean of the first run of `period` non-NaN values; NaN afterwards carries forward"""
    n = len(vals)
    for i in range(n):
        out[i] = np.nan
    run = 0
    seed = -1
    for i in range(n):
        if np.isnan(vals[i]):
            run = 0
        else:
            run += 1
            if run >= period:
                seed = i
                break
    if seed == -1:
        return
    s = 0.0
    for i in range(seed - period + 1, seed + 1):
        s += vals[i]
    out[seed] = s / period
    for i in range(seed + 1, n):
        if np.isnan(vals[i]):
            out[i] = out[i - 1]
        else:
            out[i] = (out[i - 1] * (period - 1) + vals[i]) / period


@njit(cache=True)
def _tr(h, l, c, a, n, out):
    out[0] = h[a] - l[a]
    for i in range(1, n):
        j = a + i
        x = h[j] - l[j]
        y = abs(h[j] - c[j - 1])
        z = abs(l[j] - c[j - 1])
        out[i] = max(x, max(y, z))


@njit(cache=True)
def _rf_last(c, a, n, atr, S, want_sig):
    """rangeFilter on window [a, a+n): returns (trend[n-1], trend[n-2], filt[n-1], sig[n-1], lastSigIdx)"""
    seed = -1
    for i in range(n):
        if not np.isnan(atr[i]):
            seed = i
            break
    if seed == -1:
        return 0, 0, np.nan, 0, 0
    filt = c[a + seed]
    trend = 0
    prev_trend = 0
    last_sig = 0
    sig = 0
    for i in range(seed + 1, n):
        rng = atr[i] * 1.6 * (S / 8.0)
        src = c[a + i]
        pf = filt
        if src > pf + rng:
            f = src - rng
        elif src < pf - rng:
            f = src + rng
        else:
            f = pf
        prev_trend = trend
        if f > pf:
            trend = 1
        elif f < pf:
            trend = -1
        filt = f
        sig = trend if trend != prev_trend else 0
        if want_sig and sig != 0 and i <= n - 2:
            last_sig = i
    return trend, prev_trend, filt, sig, last_sig


@njit(cache=True)
def _tick(o, h, l, c, a, n):
    """everything checkSignals/calcRegime read from the 5m window [a, a+n)"""
    tr = np.empty(n)
    atr = np.empty(n)
    _tr(h, l, c, a, n, tr)
    _rma(tr, 14, atr)
    e = a + n - 1
    price = c[e]
    t5, tp5, filt, sig, last_sig = _rf_last(c, a, n, atr, 5.0, True)
    tA = _rf_last(c, a, n, atr, 3.0, False)[0]
    tC = _rf_last(c, a, n, atr, 12.0, False)[0]
    tD = _rf_last(c, a, n, atr, 16.0, False)[0]
    # EMA 25 / 75 (SMA seed)
    ef = np.nan
    ef5 = np.nan
    es = np.nan
    for period in (25, 75):
        k = 2.0 / (period + 1)
        s = 0.0
        for i in range(period):
            s += c[a + i]
        v = s / period
        for i in range(period, n):
            v = c[a + i] * k + v * (1 - k)
            if period == 25 and i == n - 6:
                ef5 = v
        if period == 25:
            ef = v
        else:
            es = v
    # ADX 14
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        up = h[a + i] - h[a + i - 1]
        dn = l[a + i - 1] - l[a + i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    sp = np.empty(n)
    sm = np.empty(n)
    _rma(pdm, 14, sp)
    _rma(mdm, 14, sm)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = np.nan
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        dp = 100 * sp[i] / atr[i]
        dm = 100 * sm[i] / atr[i]
        ssum = dp + dm
        dx[i] = abs(dp - dm) / ssum * 100 if ssum > 0 else 0.0
    adxa = np.empty(n)
    _rma(dx, 14, adxa)
    adx = adxa[n - 1]
    # CHOP 14 (last bar)
    s = 0.0
    hi = -np.inf
    lo = np.inf
    for j in range(n - 14, n):
        s += tr[j]
        if h[a + j] > hi:
            hi = h[a + j]
        if l[a + j] < lo:
            lo = l[a + j]
    rngc = hi - lo
    chop = 100 * math.log10(s / rngc) / math.log10(14) if rngc > 0 else 50.0
    # ROC
    r5 = (c[e] - c[e - 5]) / c[e - 5] * 100
    r10 = (c[e] - c[e - 10]) / c[e - 10] * 100
    r20 = (c[e] - c[e - 20]) / c[e - 20] * 100
    # ATR percentile rank + ATR/price 65th pct (60 bars)
    val = atr[n - 1]
    cnt = 0
    tot = 0
    norms = np.empty(60)
    m = 0
    for i in range(max(0, n - 60), n):
        if not np.isnan(atr[i]):
            tot += 1
            if val >= atr[i]:
                cnt += 1
        x = atr[i] / c[a + i] * 100 if not np.isnan(atr[i]) else 0.0
        if x > 0:
            norms[m] = x
            m += 1
    rank = math.floor(cnt / tot * 100 + 0.5) if tot > 0 else 50.0
    atr_norm = val / price * 100 if not np.isnan(val) else 0.0
    if m > 0:
        srt = np.sort(norms[:m])
        p65 = srt[int(math.floor(m * 0.65))]
    else:
        p65 = atr_norm
    score_vol = (1 if rank < 80 else 0) + (1 if atr_norm < p65 else 0)
    # legacy GARCH volScore (frozen regime input)
    seed = 0.0
    for i in range(1, 51):
        r = math.log(c[a + i] / c[a + i - 1])
        seed += r * r
    pv = seed / 50
    sig_arr = np.empty(n)
    for i in range(n):
        sig_arr[i] = np.nan
    for i in range(51, n):
        r = math.log(c[a + i] / c[a + i - 1])
        if not np.isfinite(r):
            r = 0.0
        r2 = r * r
        nv = 1e-6 + 0.10 * r2 + 0.89 * pv
        sig_arr[i] = math.sqrt(nv) * c[a + i]
        pv = nv
    vs = 0.5
    ss = 0.0
    sc = 0
    for i in range(max(0, n - 60), n):
        if not np.isnan(sig_arr[i]) and sig_arr[i] > 0:
            ss += sig_arr[i]
            sc += 1
    if sc > 0 and sig_arr[n - 1] > 0:
        vs = min(1.0, max(0.0, (sig_arr[n - 1] / (ss / sc) - 0.5) / 1.0))
    # lag-1 autocorrelation of the last 20 returns
    rets = np.empty(20)
    k = 0
    for i in range(n - 20, n):
        r = math.log(c[a + i] / c[a + i - 1])
        if np.isfinite(r):
            rets[k] = r
            k += 1
    ac = 0.0
    if k >= 6:
        q = k - 1
        m0 = 0.0
        m1 = 0.0
        for i in range(q):
            m0 += rets[i]
        for i in range(1, k):
            m1 += rets[i]
        m0 /= q
        m1 /= q
        cv = 0.0
        v0 = 0.0
        v1 = 0.0
        for i in range(q):
            cv += (rets[i] - m0) * (rets[i + 1] - m1)
            v0 += (rets[i] - m0) ** 2
            v1 += (rets[i + 1] - m1) ** 2
        den = math.sqrt(v0 * v1)
        ac = cv / den if den > 0 else 0.0
    adx0 = 0.0 if np.isnan(adx) else adx
    chop0 = 50.0 if np.isnan(chop) else chop
    adxs = (max(15.0, min(40.0, adx0)) - 15) / 25
    chops = max(0.0, min(1.0, (61.8 - chop0) / 23.6))
    acs = max(0.0, min(1.0, (ac + 1) / 2))
    tprob = 0.30 * adxs + 0.25 * chops + 0.25 * acs + 0.20 * vs
    # dead bars (last 24)
    dead = 0
    tot = 0
    for i in range(max(1, n - 24), n):
        tot += 1
        if h[a + i] == l[a + i] or c[a + i] == c[a + i - 1]:
            dead += 1
    dfrac = dead / tot if tot > 0 else 1.0
    res = np.empty(25)
    vals = (float(t5), float(tp5), filt, float(sig), float(n - 1 - last_sig), float(tA), float(tC), float(tD),
            atr[n - 1], ef, ef5, es, adx0, chop0, r5, r10, r20, float(score_vol), tprob, dfrac, adxs, chops, acs, vs,
            sig_arr[n - 1])
    for j in range(25):
        res[j] = vals[j]
    return res


@njit(parallel=True, cache=True)
def window_features(o, h, l, c, W):
    N = len(c)
    out = np.full((N, 25), np.nan)
    for t in prange(W - 1, N):
        out[t, :] = _tick(o, h, l, c, t - W + 1, W)
    return out


@njit(parallel=True, cache=True)
def htf_trend(h, l, c, W):
    """range-filter trend of the last bar for every HTF window of W bars"""
    N = len(c)
    out = np.zeros(N, np.int8)
    for t in prange(W - 1, N):
        a = t - W + 1
        tr = np.empty(W)
        atr = np.empty(W)
        _tr(h, l, c, a, W, tr)
        _rma(tr, 14, atr)
        out[t] = _rf_last(c, a, W, atr, 5.0, False)[0]
    return out


@njit(cache=True)
def _garch_ll(r, n, omega, a, b, seed, warm):
    h = seed
    ll = 0.0
    used = 0
    for t in range(1, n):
        if t > warm:
            if not (h > 0) or not np.isfinite(h):
                return -np.inf
            ll += -0.5 * (math.log(h) + (r[t] * r[t]) / h)
            used += 1
        h = omega + a * (r[t] * r[t]) + b * h
    return ll if used > 0 else -np.inf


@njit(cache=True)
def garch_fit_window(c, a, n):
    """fitGarch on window [a, a+n): (sigma_next*close, converged, alpha, beta) — v7.1 grid QMLE, variance targeting"""
    r = np.empty(n)
    r[0] = np.nan
    for i in range(1, n):
        v = math.log(c[a + i] / c[a + i - 1])
        r[i] = v if np.isfinite(v) else 0.0
    s2 = 0.0
    for i in range(1, n):
        s2 += r[i] * r[i]
    sv = s2 / (n - 1)
    if not (sv > 0):
        return np.nan, False, np.nan, np.nan
    ss = 0.0
    for i in range(1, 51):
        ss += r[i] * r[i]
    seed = ss / 50 if ss > 0 else sv
    aLo, aHi, bLo, bHi = 0.01, 0.30, 0.50, 0.985
    bestA = np.nan
    bestB = np.nan
    bestLL = -np.inf
    interior = False
    G = 12
    for _ in range(3):
        bi = -1
        bj = -1
        pA = np.nan
        pB = np.nan
        pLL = -np.inf
        for i in range(G):
            av = aLo + (aHi - aLo) * i / (G - 1)
            for j in range(G):
                bv = bLo + (bHi - bLo) * j / (G - 1)
                if av + bv >= 0.999:
                    continue
                om = sv * (1 - av - bv)
                if not (om > 0):
                    continue
                ll = _garch_ll(r, n, om, av, bv, seed, 50)
                if ll > pLL:
                    pLL = ll
                    pA = av
                    pB = bv
                    bi = i
                    bj = j
        if not np.isfinite(pLL):
            break
        bestA = pA
        bestB = pB
        bestLL = pLL
        interior = bi > 0 and bi < G - 1 and bj > 0 and bj < G - 1
        aStep = (aHi - aLo) / (G - 1)
        bStep = (bHi - bLo) / (G - 1)
        aLo = max(0.01, bestA - aStep)
        aHi = min(0.30, bestA + aStep)
        bLo = max(0.50, bestB - bStep)
        bHi = min(0.985, bestB + bStep)
    if not np.isfinite(bestLL):
        return np.nan, False, np.nan, np.nan
    om = sv * (1 - bestA - bestB)
    hh = seed
    for i in range(1, n):
        hh = om + bestA * (r[i] * r[i]) + bestB * hh
    return math.sqrt(hh) * c[a + n - 1], interior, bestA, bestB


@njit(cache=True)
def modes_v71(tprob):
    """deriveMarketMode with a 2-entry raw history, every tick (1=trend, 2=reversion, 0=neutral)"""
    N = len(tprob)
    out = np.zeros(N, np.int8)
    h0 = -1
    h1 = -1                                    # previous raw modes (h0 = most recent), -1 = empty
    for t in range(N):
        p = tprob[t]
        if np.isnan(p):
            continue
        raw = TREND if p >= 0.58 else (REV if p <= 0.42 else NEUTRAL)
        run = 0
        for m in (raw, h0):                    # updatedRaw = [raw, h0] (slice to 2)
            if m == raw and m != NEUTRAL and m != -1:
                run += 1
            else:
                break
        last_nn = -1
        for m in (h0, h1):
            if m != NEUTRAL and m != -1:
                last_nn = m
                break
        if raw == NEUTRAL:
            conf = last_nn if last_nn != -1 else NEUTRAL
        elif run >= 2:
            conf = raw
        else:
            conf = last_nn if last_nn != -1 else raw
        out[t] = conf
        h1 = h0
        h0 = raw
    return out


@njit(cache=True)
def modes_hysteresis(tprob, persist):
    """V3: enter trend ≥0.58 / reversion ≤0.42, keep it until it crosses 0.50; confirm after `persist` ticks"""
    N = len(tprob)
    out = np.zeros(N, np.int8)
    raw_prev = NEUTRAL
    cand = NEUTRAL
    run = 0
    conf = NEUTRAL
    for t in range(N):
        p = tprob[t]
        if np.isnan(p):
            continue
        if p >= 0.58:
            raw = TREND
        elif p <= 0.42:
            raw = REV
        elif raw_prev == TREND and p >= 0.50:
            raw = TREND
        elif raw_prev == REV and p <= 0.50:
            raw = REV
        else:
            raw = NEUTRAL
        if raw == cand:
            run += 1
        else:
            cand = raw
            run = 1
        if run >= persist:
            conf = cand
        out[t] = conf
        raw_prev = raw
    return out


@njit(cache=True)
def scan_exit(t5, o5, h5, l5, c5, t1, o1, h1, l1, k0, j0, side, entry, sl, tp, use_tp, lev, hard,
              time_bars, trail_dist, stop_slip, tp_slip, mkt_slip, max_bars):
    """walk a position forward from 5m bar k0 (filled at its open). Stops/targets on 1m mark bars (stop first),
    the hard stop / time stop / trailing update on 5m closes (exit at the next 5m open).
    returns (exit_time_ns, exit_price, reason, bars_held, freed_at_ns); reasons 1=SL 2=TP 3=hard 4=time 5=trail 6=end"""
    N = len(t5)
    M = len(t1)
    j = j0
    stop = sl
    best = entry
    for k in range(k0, min(N, k0 + max_bars)):
        end = t5[k] + 300_000_000_000
        used = False
        while j < M and t1[j] < end:
            if t1[j] >= t5[k]:
                used = True
                if side > 0:
                    if l1[j] <= stop:
                        return t1[j], min(o1[j], stop) * (1 - stop_slip), 5 if stop != sl else 1, k - k0 + 1, t1[j]
                    if use_tp and h1[j] >= tp:
                        return t1[j], max(o1[j], tp) * (1 - tp_slip), 2, k - k0 + 1, t1[j]
                else:
                    if h1[j] >= stop:
                        return t1[j], max(o1[j], stop) * (1 + stop_slip), 5 if stop != sl else 1, k - k0 + 1, t1[j]
                    if use_tp and l1[j] <= tp:
                        return t1[j], min(o1[j], tp) * (1 + tp_slip), 2, k - k0 + 1, t1[j]
            j += 1
        if not used:                            # no 1m data for this bar: 5m bar, stop first
            if side > 0:
                if l5[k] <= stop:
                    return t5[k], min(o5[k], stop) * (1 - stop_slip), 5 if stop != sl else 1, k - k0 + 1, t5[k]
                if use_tp and h5[k] >= tp:
                    return t5[k], max(o5[k], tp) * (1 - tp_slip), 2, k - k0 + 1, t5[k]
            else:
                if h5[k] >= stop:
                    return t5[k], max(o5[k], stop) * (1 + stop_slip), 5 if stop != sl else 1, k - k0 + 1, t5[k]
                if use_tp and l5[k] <= tp:
                    return t5[k], min(o5[k], tp) * (1 + tp_slip), 2, k - k0 + 1, t5[k]
        held = k - k0 + 1
        nxt = k + 1 if k + 1 < N else k
        if (c5[k] - entry) / entry * lev * side <= -hard:
            return t5[nxt], o5[nxt] * (1 - side * mkt_slip), 3, held, end
        if time_bars > 0 and held >= time_bars:
            return t5[nxt], o5[nxt] * (1 - side * mkt_slip), 4, held, end
        if trail_dist > 0:
            if side > 0:
                best = max(best, c5[k])
                stop = max(stop, best - trail_dist)
            else:
                best = min(best, c5[k])
                stop = min(stop, best + trail_dist)
    k = min(N, k0 + max_bars) - 1
    return t5[k], c5[k], 6, k - k0 + 1, t5[k] + 300_000_000_000


# ================= 4. FEATURES PER SYMBOL =================
def _htf(m5: pd.DataFrame, minutes):
    key = m5.index.floor(f"{minutes}min")
    g = m5.groupby(key)
    return g["open"].first(), g["high"].max(), g["low"].min(), g["close"].last()


@dataclass
class Feat:
    sym: str
    t: np.ndarray                 # 5m open times (ns)
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    tbv: np.ndarray
    W: np.ndarray                 # window features (N×25)
    htf15: np.ndarray
    htf1h: np.ndarray
    htf4h: np.ndarray
    piv: np.ndarray               # N×9: pp r1 r2 r3 r4 s1 s2 s3 s4 (previous UTC day, classic)
    t1: np.ndarray
    o1: np.ndarray
    h1: np.ndarray
    l1: np.ndarray
    fund_t: np.ndarray
    fund_r: np.ndarray
    member: np.ndarray            # bool per tick (PIT universe)
    garch: dict = field(default_factory=dict)


def build_feat(sd: SymData, cfg: FConfig) -> Feat:
    m5 = sd.m5.sort_index()
    m5 = m5[~m5.index.duplicated(keep="last")]
    t = m5.index.as_unit("ns").asi8
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = m5["volume"].to_numpy(float) if "volume" in m5 else np.zeros(len(m5))
    tbv = m5["tbv"].to_numpy(float) if "tbv" in m5 else np.full(len(m5), np.nan)
    W = window_features(o, h, l, c, cfg.window) if len(c) >= cfg.window else np.full((len(c), 25), np.nan)
    tick = t + NS5                                         # tick time = bar close
    trends = []
    for minutes, win in ((15, cfg.htf_window), (60, cfg.mtf_window), (240, cfg.mtf_window)):
        ho, hh, hl, hc = _htf(m5, minutes)
        tr = htf_trend(hh.to_numpy(float), hl.to_numpy(float), hc.to_numpy(float), win) if len(hc) >= win \
            else np.zeros(len(hc), np.int8)
        close_t = hc.index.as_unit("ns").asi8 + minutes * NS1
        k = np.searchsorted(close_t, tick, side="right") - 1
        ok = k >= win - 1
        out = np.zeros(len(t), np.int8)
        out[ok] = tr[k[ok]]
        trends.append(out)
    day = m5.groupby(m5.index.floor("D")).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    dkey = day.index.as_unit("ns").asi8
    prev_day = (tick // DAY) * DAY - DAY
    di = np.searchsorted(dkey, prev_day)
    okd = (di < len(dkey)) & (dkey[np.clip(di, 0, len(dkey) - 1)] == prev_day)
    piv = np.full((len(t), 9), np.nan)
    dh, dl, dc = day.h.to_numpy(), day.l.to_numpy(), day.c.to_numpy()
    j = di[okd]
    ph, pl, pc = dh[j], dl[j], dc[j]
    pp = (ph + pl + pc) / 3
    rg = ph - pl
    piv[okd] = np.c_[pp, 2 * pp - pl, pp + rg, ph + 2 * (pp - pl), pp + rg * 3,
                     2 * pp - ph, pp - rg, pl - 2 * (ph - pp), pp - rg * 3]
    if sd.mark1 is not None and len(sd.mark1):
        mk = sd.mark1.sort_index()
        mk = mk[~mk.index.duplicated(keep="last")]
        t1 = mk.index.as_unit("ns").asi8
        o1, h1, l1 = (mk[k].to_numpy(float) for k in ("open", "high", "low"))
    else:
        t1 = np.zeros(0, np.int64); o1 = h1 = l1 = np.zeros(0)
    if sd.funding is not None and len(sd.funding):
        ft = sd.funding.index.as_unit("ns").asi8
        fr = sd.funding["rate"].to_numpy(float)
    else:
        ft = np.zeros(0, np.int64); fr = np.zeros(0)
    if sd.member_months is None:
        member = np.ones(len(t), bool)
    else:
        member = np.isin(pd.DatetimeIndex(tick, tz="UTC").strftime("%Y-%m"), list(sd.member_months))
    return Feat(sd.sym, t, o, h, l, c, v, tbv, W, trends[0], trends[1], trends[2], piv, t1, o1, h1, l1, ft, fr, member)


def pre_signals(F: Feat, cfg: FConfig):
    """everything in checkSignals that does not depend on the market mode or on trading state"""
    W = F.W
    price = F.c
    rf, rfp, filt, bars = W[:, 0], W[:, 1], W[:, 2], W[:, 4]
    atr, ef, es = W[:, 8], W[:, 9], W[:, 11]
    adx, chop, r5, r10, r20 = W[:, 12], W[:, 13], W[:, 14], W[:, 15], W[:, 16]
    with np.errstate(invalid="ignore"):
        cb, ca = ef > es, price > ef
        cr, cbl = ef < es, price < ef
        cond1 = np.where(rf == 1, cb, cr)
        cond2 = np.where(rf == 1, ca, cbl)
        mb = (r5 > 0) & (r10 > 0) & (r20 > 0)
        mr = (r5 < 0) & (r10 < 0) & (r20 < 0)
        mal = ((rf == 1) & mb) | ((rf == -1) & mr)
        mpa = ((rf == 1) & (r5 > 0)) | ((rf == -1) & (r5 < 0))
        adx_ok = adx >= cfg.adx_thresh
        chop_ok = chop <= cfg.chop_thresh
        chop_gate = adx_ok & chop_ok & (bars >= cfg.cooldown_bars)
        penalty = -(~adx_ok).astype(int) - (~chop_ok).astype(int)
        s_htf = (((rf == 1) & (F.htf15 == 1)) | ((rf == -1) & (F.htf15 != 1))).astype(int)
        s_mtf = (F.htf15 == rf).astype(int) + (F.htf1h == rf).astype(int) + (F.htf4h == rf).astype(int)
        s_sens = np.minimum((W[:, 5] == rf).astype(int) + 1 + 2 * (W[:, 6] == rf).astype(int) + (W[:, 7] == rf).astype(int), 4)
        s_ema = np.minimum(cond1.astype(int) + cond2.astype(int) + mal.astype(int) + mpa.astype(int), 4)
        total = np.clip(s_htf + s_mtf + s_sens + s_ema + np.nan_to_num(W[:, 17]).astype(int) + penalty, 0, 14)
        safe_atr = np.where(np.isnan(atr) | (atr <= 0), price * 0.005, atr)
        buf = safe_atr * 0.75
        P = F.piv
        res = np.where(P[:, [1, 2, 3, 4, 0]] > price[:, None], P[:, [1, 2, 3, 4, 0]], np.inf).min(axis=1)
        sup = np.where(P[:, [5, 6, 7, 8, 0]] < price[:, None], P[:, [5, 6, 7, 8, 0]], -np.inf).max(axis=1)
        long_clear = (res - price) >= buf
        short_clear = (price - sup) >= buf
        pb_bull = ~np.isnan(filt) & (price <= filt + safe_atr)
        pb_bear = ~np.isnan(filt) & (price >= filt - safe_atr)
        long_sig = (rf == 1) & (rfp == 1) & pb_bull & chop_gate & (s_mtf >= 2) & long_clear
        short_sig = (rf == -1) & (rfp == -1) & pb_bear & chop_gate & (s_mtf >= 2) & short_clear
        ema_up = ~np.isnan(ef) & ~np.isnan(es) & (ef > es)
        ema_dn = ~np.isnan(ef) & ~np.isnan(es) & (ef < es)
        dead = W[:, 19] > 0.30
    return dict(total=total, long=long_sig, short=short_sig, ema_up=ema_up, ema_dn=ema_dn, dead=dead,
                tprob=W[:, 18], valid=~np.isnan(W[:, 18]))


def mode_series(F: Feat, cfg: FConfig, pre):
    tp = np.where(pre["valid"], pre["tprob"], np.nan)
    return modes_hysteresis(tp, cfg.persist_ticks) if cfg.mode_rule == "hysteresis" else modes_v71(tp)


def directions(F: Feat, cfg: FConfig, pre, modes):
    gate = (modes != TREND) | (pre["long"] & pre["ema_up"]) | (pre["short"] & pre["ema_dn"])
    ok = (modes != NEUTRAL) & gate & ~pre["dead"] & pre["valid"]
    strong = pre["total"] >= cfg.min_strength
    d = np.where(ok & pre["long"] & strong, 1, np.where(ok & pre["short"] & strong, -1, 0)).astype(np.int8)
    if cfg.trend_only:
        d[modes == REV] = 0
    if cfg.flow_filter:                                   # V7: bet on the documented flow-driven bounce
        exch = np.where(modes == TREND, d, -d)
        ret = np.r_[0.0, np.diff(F.c)]
        with np.errstate(invalid="ignore", divide="ignore"):
            imb = np.where(F.v > 0, F.tbv / F.v - 0.5, np.nan)
        flow_driven = np.sign(imb) == np.sign(ret)
        d[~((np.sign(ret) == -exch) & flow_driven & (ret != 0))] = 0
    return d


def struct_r(price, side, piv, vol, sl_mult, tp_mult, cfg):
    """calcStructuralR (side = exchange side: +1 long / -1 short)"""
    ev = price * 0.005 if (np.isnan(vol) or vol <= 0) else vol
    vsd, tpd = ev * sl_mult, ev * tp_mult
    mx = vsd * cfg.max_stop_widen
    sld, snapped = vsd, False
    lv = [x for x in (piv[0], piv[5], piv[6], piv[7], piv[8], piv[1], piv[2], piv[3], piv[4]) if np.isfinite(x) and x > 0]
    cands = [d for d in ((price - x) if side > 0 else (x - price) for x in lv) if vsd < d <= mx]
    if cands:
        sld = min(min(cands) + ev * 0.10, mx)
        snapped = True
    risk_pct = sld / price
    stop = price - sld if side > 0 else price + sld
    tp = price + tpd if side > 0 else price - tpd
    reward_pct = abs(tp - price) / price
    raw = reward_pct / risk_pct if risk_pct > 0 else tp_mult / sl_mult
    R = max(0.8, min(5.0, raw))
    return dict(R=R, risk_pct=risk_pct, reward_pct=reward_pct, stop=stop, tp=tp, snapped=snapped)


def bracket(F: Feat, t, mode, sig_dir, cfg):
    if t not in F.garch:
        a = t - cfg.window + 1
        F.garch[t] = garch_fit_window(F.c, a, cfg.window) if a >= 0 else (np.nan, False, np.nan, np.nan)
    sig, conv, _, _ = F.garch[t]
    atr = F.W[t, 8]
    vol = sig if (conv and np.isfinite(sig) and sig > 0) else (0.0 if (np.isnan(atr) or atr <= 0) else atr)
    side = sig_dir if mode == TREND else -sig_dir
    slm, tpm = (1.5, 2.0) if mode == REV else (3.0, 4.0)
    b = struct_r(F.c[t], side, F.piv[t], vol, slm, tpm, cfg)
    b.update(side=side, sl_mult=slm, tp_mult=tpm, vol=vol if vol > 0 else F.c[t] * 0.005, garch=bool(conv))
    return b


# ================= 5. SIZING (v7.1 Kelly, or fixed fractional) =================
def realized_r(trades):
    w = sw = lo = sl = 0
    for t in trades:
        if not t["margin"] > 0:
            continue
        rom = t["pnl"] / t["margin"]
        if rom > 0:
            w += 1; sw += rom
        else:
            lo += 1; sl += abs(rom)
    if w == 0 or lo == 0:
        return np.nan
    aw, al = sw / w, sl / lo
    return min(aw / al, 10.0) if al > 0 else np.nan


def blend_r(sr, hr, cfg):
    if np.isnan(hr):
        return sr
    ch = max(0.8, min(5.0, hr))
    return max(0.8, min(5.0, (1 - cfg.hist_r_blend) * sr + cfg.hist_r_blend * ch))


def round2(x):
    return math.floor(x * 100 + 0.5) / 100


def kelly_margin(eng, T, mode, b, strength, corr_adj, cfg):
    """buildKellyState + calcKellyMargin + probe + correlation adjustment + EV gate. Returns (margin, info)."""
    from scipy.special import betaincinv, betainc
    pooled = eng.pooled(T)
    sub = [x for x in eng.mode_log(T) if x["mode"] == mode] if mode != NEUTRAL else []
    if mode != NEUTRAL and len(sub) >= cfg.min_trades_per_mode:
        trades, hist_r = sub, realized_r(sub)
    else:
        trades = pooled
        hist_r = realized_r(pooled) if len(pooled) >= cfg.min_trades_for_kelly else np.nan
    R = blend_r(b["R"], hist_r, cfg)
    null_w = 1 / (1 + R)
    wins = sum(1 for x in trades if x["pnl"] > 0)
    al = null_w * cfg.prior_strength + wins
    be = (1 - null_w) * cfg.prior_strength + (len(trades) - wins)
    q = float(betaincinv(al, be, cfg.kelly_quantile))
    raw_f = q - (1 - q) / R if R > 0 else 0.0
    half = raw_f * cfg.kelly_fraction
    bankroll = eng.bankroll(T)
    margin = 0.0
    if half > 0:
        mult = 1.0 if strength >= 11 else (0.85 if strength >= 9 else 0.70)
        margin = round2(bankroll * half * mult)
        margin = 0.0 if margin < cfg.min_margin else min(margin, cfg.max_margin)
    probe = len(trades) < cfg.warmup_trades
    if not probe and margin <= 0 and cfg.unfreeze_hours > 0 and trades:      # proposed fix for the Kelly freeze
        newest = max(x["exit_ns"] for x in trades)
        probe = T - newest > cfg.unfreeze_hours * 3600 * 10**9
    if probe:
        margin = cfg.min_margin
    if margin > 0 and corr_adj != 1.0:
        margin = round2(margin * corr_adj)
        margin = 0.0 if margin < cfg.min_margin else max(cfg.min_margin, min(cfg.max_margin, margin))
    if margin > 0 and not probe:                         # EV gate on the q25 win rate (fee & funding aware)
        notional = margin * cfg.leverage
        hold_h = max(b["sl_mult"] * b["tp_mult"] * 5 / 60, 1 / 60)
        fcost = b["side"] * eng.funding_rate(T, b["sym"]) * notional * (hold_h / 8)
        ev = q * notional * b["reward_pct"] - (1 - q) * notional * b["risk_pct"] - notional * cfg.fee * 2 - fcost
        if ev <= 0:
            margin = 0.0
    return margin, dict(R_blend=R, null_w=null_w, q=q, probe=probe, p_gt=float(1 - betainc(al, be, null_w)))


def fixed_notional(eng, T, b, cfg):
    eq = eng.cash_at(T)
    risk = cfg.risk_pct * eq
    notional = risk / (b["risk_pct"] + 2 * (cfg.fee + cfg.stop_slip))
    room = cfg.max_gross_mult * eq - eng.gross_open(T)
    notional = min(notional, max(0.0, room))
    return notional if notional > 10 else 0.0


# ================= 6. PORTFOLIO SIMULATION =================
def paired_returns(ca, cb, lookback):
    n = min(len(ca), len(cb), lookback + 1)
    if n < 3:
        return np.zeros(0), np.zeros(0)
    a, b = np.log(ca[-n:][1:] / ca[-n:][:-1]), np.log(cb[-n:][1:] / cb[-n:][:-1])
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    return a[ok], b[ok]


class Engine:
    """portfolio state. Lookups are binary searches over trades sorted by exit time (Binance userTrades view) and
    by record time (Deno-KV view: mode log and daily-loss ledger), so a 6-year run stays fast."""

    def __init__(self, feats: dict, cfg: FConfig):
        self.F, self.cfg = feats, cfg
        self.cash0 = cfg.capital
        self.open = {}                # sym -> position dict
        self.btc = next((s for s in feats if s.startswith("BTC")), None)
        self.load_closed([])

    def load_closed(self, trades):
        self.closed, self.exit_ns, self.cum = [], [], [0.0]
        self.rec, self.rec_ns, self.loss_cum = [], [], [0.0]
        self._add(sorted(trades, key=lambda x: x["exit_ns"]))

    def _add(self, batch):
        import bisect
        for x in batch:
            self.closed.append(x)
            self.exit_ns.append(x["exit_ns"])
            self.cum.append(self.cum[-1] + x["pnl"])
            if x["record_ns"] is None:                    # not visible to the daily ledger / mode log (live v7.1)
                continue
            k = bisect.bisect_right(self.rec_ns, x["record_ns"])
            self.rec.insert(k, x)
            self.rec_ns.insert(k, x["record_ns"])
            if k == len(self.rec) - 1:
                self.loss_cum.append(self.loss_cum[-1] + max(0.0, -x["pnl"]))
            else:                                     # out-of-order insert (rare): rebuild the prefix sums
                self.loss_cum = [0.0]
                for y in self.rec:
                    self.loss_cum.append(self.loss_cum[-1] + max(0.0, -y["pnl"]))

    # --- state views at tick T (ns, bar close) ---
    def pooled(self, T):
        import bisect
        k = bisect.bisect_left(self.exit_ns, T)
        return self.closed[max(0, k - self.cfg.kelly_lookback):k][::-1]

    def mode_log(self, T):
        import bisect
        k = bisect.bisect_left(self.rec_ns, T)
        return self.rec[max(0, k - self.cfg.mode_log_cap):k][::-1]

    def cash_at(self, T):
        import bisect
        return self.cash0 + self.cum[bisect.bisect_left(self.exit_ns, T)]

    def bankroll(self, T):
        return self.cash_at(T) - sum(p["margin"] for p in self.open.values())

    def gross_open(self, T):
        return sum(p["notional"] for p in self.open.values())

    def daily_loss(self, T):
        import bisect
        i0 = bisect.bisect_left(self.rec_ns, (T // DAY) * DAY)
        i1 = bisect.bisect_left(self.rec_ns, T)
        return self.loss_cum[i1] - self.loss_cum[i0]

    def funding_rate(self, T, sym):
        F = self.F[sym]
        k = np.searchsorted(F.fund_t, T, side="right") - 1
        return float(F.fund_r[k]) if k >= 0 else 0.0

    def idx(self, sym, T):
        return int(np.searchsorted(self.F[sym].t, T - NS5))

    def finalize(self, T):
        done = [s for s, p in self.open.items() if p["freed_ns"] < T]
        if done:
            self._add(sorted((self.open.pop(s) for s in done), key=lambda x: x["exit_ns"]))

    # --- correlation / exposure (v7.1) ---
    def scalars(self, T, firing):
        cfg = self.cfg
        if len(firing) >= 2:
            rs = []
            for a, b in itertools.combinations(firing, 2):
                ia, ib = self.idx(a, T), self.idx(b, T)
                x, y = paired_returns(self.F[a].c[:ia + 1], self.F[b].c[:ib + 1], cfg.corr_lookback)
                if len(x) >= 30 and x.std() > 0 and y.std() > 0:
                    rs.append(max(-1.0, min(1.0, float(np.corrcoef(x, y)[0, 1]))))
            n = len(firing)
            rho = np.mean(rs) if rs else 0.75
            eff = max(-1 / (n - 1) + 1e-6, rho)
            co = max(0.2, min(1.5, 1 / math.sqrt(n * (1 + (n - 1) * eff))))
        else:
            co = 1.0
        return co

    def beta(self, T, sym):
        if self.btc is None or sym == self.btc:
            return 1.0
        x, y = paired_returns(self.F[sym].c[:self.idx(sym, T) + 1], self.F[self.btc].c[:self.idx(self.btc, T) + 1],
                              self.cfg.beta_lookback)
        if len(x) < 30 or y.var() == 0:
            return 1.0
        return float(np.cov(x, y, bias=True)[0, 1] / y.var())

    # --- one position ---
    def open_position(self, T, sym, t, mode, b, margin, notional, info):
        cfg, F = self.cfg, self.F[sym]
        k0 = t + 1
        if k0 >= len(F.t) or F.t[k0] - F.t[t] > DAY:
            return None
        side = b["side"]
        fill = F.o[k0] * (1 + side * cfg.entry_slip)
        qty = notional / F.c[t]                            # size from the signal price (as live)
        j0 = int(np.searchsorted(F.t1, F.t[k0])) if len(F.t1) else 0
        use_1m = cfg.exit_tf == "1m" and len(F.t1) > 0
        trail = cfg.trail_mult * b["vol"] if (cfg.trail_trend and mode == TREND) else 0.0
        tbars = int(round(cfg.time_stop * b["sl_mult"] * b["tp_mult"])) if cfg.time_stop > 0 else 0
        lev = cfg.leverage
        ex_t, ex_px, why, held, freed = scan_exit(
            F.t, F.o, F.h, F.l, F.c, F.t1 if use_1m else np.zeros(0, np.int64), F.o1 if use_1m else np.zeros(0),
            F.h1 if use_1m else np.zeros(0), F.l1 if use_1m else np.zeros(0), k0, j0, side, fill, b["stop"], b["tp"],
            not (cfg.trail_trend and mode == TREND), lev, cfg.hard_stop_pct, tbars, trail,
            cfg.stop_slip, cfg.tp_slip, cfg.entry_slip, 288 * 60)
        gross = side * qty * (ex_px - fill)
        fees = cfg.fee * qty * (fill + ex_px)
        k = (F.fund_t > F.t[k0]) & (F.fund_t <= ex_t)
        fund = 0.0
        if k.any():
            ki = np.clip(np.searchsorted(F.t, F.fund_t[k], side="right") - 1, 0, len(F.c) - 1)
            fund = -side * float((F.fund_r[k] * qty * F.c[ki]).sum())
        pnl = gross - fees + fund
        if pnl < -(margin + cfg.fee * qty * fill):           # isolated: loss capped at margin
            pnl, why = -(margin + cfg.fee * qty * fill), 7
        risk = qty * abs(fill - b["stop"])
        if why in (3, 4):                                  # bot-side market close: recorded at that tick
            record = freed
        elif cfg.ledger == "all":                          # fixed bot: detected at the start of the next tick,
            record = int(math.ceil((ex_t + 1) / NS5)) * NS5 - 1  # before that tick's entry decisions
        else:                                              # live v7.1: never reaches the ledger/mode log
            record = None
        pos = dict(sym=sym, side=side, mode=mode, entry_ns=int(F.t[k0]), exit_ns=int(ex_t), freed_ns=int(freed),
                   record_ns=record, entry=fill, exit=float(ex_px), qty=qty, margin=margin, notional=qty * fill,
                   stop=b["stop"], tp=b["tp"], R_struct=b["R"], null_w=1 / (1 + b["R"]), pnl=float(pnl),
                   R=float(pnl / risk) if risk > 0 else 0.0, why=["", "SL", "TP", "hard", "time", "trail", "end", "liq"][why],
                   bars=int(held), probe=info.get("probe", False), garch=b["garch"], eq_before=self.cash_at(T))
        self.open[sym] = pos
        return pos


def run(feats: dict, cfg: FConfig, start, end, cands=None):
    """cands: {sym: (ticks, modes, dirs, strengths)} or None to compute from cfg"""
    eng = Engine(feats, cfg)
    order = list(feats)
    t_lo, t_hi = LL._utc(start).value, LL._utc(end).value
    events = defaultdict(list)
    for s in order:
        F = feats[s]
        if cands is None or s not in cands:
            pre = pre_signals(F, cfg)
            md = mode_series(F, cfg, pre)
            d = directions(F, cfg, pre, md)
            tot = pre["total"]
        else:
            md, d, tot = cands[s]
        T = F.t + NS5
        for i in np.flatnonzero((d != 0) & (T >= t_lo) & (T < t_hi) & F.member):
            if i + 1 < len(F.t):
                events[int(T[i])].append((s, int(i), int(md[i]), int(d[i]), int(tot[i])))
    for T in sorted(events):
        eng.finalize(T)
        todo = []
        for s, i, mode, d, strength in sorted(events[T], key=lambda e: order.index(e[0])):
            if s in eng.open:
                continue
            b = bracket(feats[s], i, mode, d, cfg)
            b["sym"] = s
            todo.append((s, i, mode, b, strength))
        if not todo:
            continue
        if cfg.sizing == "fixed":
            for s, i, mode, b, strength in todo:
                if eng.daily_loss(T) >= cfg.daily_loss_limit:
                    break
                n = fixed_notional(eng, T, b, cfg)
                if n > 0:
                    eng.open_position(T, s, i, mode, b, n / cfg.leverage, n, {})
            continue
        sized = []
        for s, i, mode, b, strength in todo:
            m, info = kelly_margin(eng, T, mode, b, strength, 1.0, cfg)
            if m > 0:
                sized.append([s, i, mode, b, strength, m, info])
        if not sized:
            continue
        co = eng.scalars(T, [x[0] for x in sized])
        bank = eng.bankroll(T)
        existing = sum(p["side"] * eng.beta(T, p["sym"]) * p["notional"] for p in eng.open.values())
        incoming = sum(x[3]["side"] * eng.beta(T, x[0]) * x[5] * cfg.leverage for x in sized)
        expo = 1.0
        cap = cfg.max_factor_exposure * bank
        if abs(existing + incoming) > cap and abs(incoming) > 0:
            expo = min(1.0, max(0.0, cap - abs(existing)) / abs(incoming))
        adj = co * expo
        for x in sized:
            s, i, mode, b, strength, m, info = x
            if adj != 1.0:
                m, info = kelly_margin(eng, T, mode, b, strength, adj, cfg)
                if m <= 0:
                    continue
            if eng.daily_loss(T) >= cfg.daily_loss_limit:
                continue
            eng.open_position(T, s, i, mode, b, m, m * cfg.leverage, info)
    eng.finalize(np.iinfo(np.int64).max)
    return eng.closed, eng


def metrics(trades, cfg: FConfig):
    if not trades:
        return {"trades": 0}
    R = np.array([t["R"] for t in trades])
    pnl = np.array([t["pnl"] for t in trades])
    eq = np.r_[cfg.capital, cfg.capital + np.cumsum(pnl)]
    w, lo = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    by_mode = {m: sum(1 for t in trades if t["mode"] == k) for m, k in (("trend", TREND), ("reversion", REV))}
    years = max((trades[-1]["exit_ns"] - trades[0]["entry_ns"]) / (365.25 * DAY), 1e-9)
    return {"trades": len(trades), "per_year": round(len(trades) / years, 1), "win_rate": round(float((pnl > 0).mean()), 3),
            "avg_R": round(float(R.mean()), 3), "PF": round(float(w / lo), 2) if lo > 0 else float("inf"),
            "pnl": round(float(pnl.sum()), 2), "max_DD": round(float((eq / np.maximum.accumulate(eq) - 1).min()), 4),
            "probes": sum(1 for t in trades if t["probe"]), "trend/rev": f"{by_mode['trend']}/{by_mode['reversion']}",
            "exits": dict(pd.Series([t["why"] for t in trades]).value_counts())}


def print_trades(trades, limit=40):
    print(f"{'entry (UTC)':<16} {'coin':<7} {'mode':<4} {'side':<5} {'R':>6} {'P&L':>8} {'exit':<5} {'min':>5}")
    for t in trades[-limit:]:
        print(f"{pd.Timestamp(t['entry_ns'], tz='UTC'):%Y-%m-%d %H:%M} {LL.base_of(t['sym'])[:7]:<7} "
              f"{'TRD' if t['mode'] == TREND else 'REV':<4} {'long' if t['side'] > 0 else 'short':<5} {t['R']:+6.2f} "
              f"{t['pnl']:+8.2f} {t['why']:<5} {(t['exit_ns'] - t['entry_ns']) / NS1:5.0f}")


# ================= 7. DATASETS FOR A RUN =================
def pit_members(cfg: FConfig, start, end, n):
    """month -> top-n USDT perps (langlang point-in-time rules); returns {sym: set('YYYY-MM')}"""
    lcfg = replace(LL.Config(), cache_dir=cfg.cache_dir, vision_workers=cfg.vision_workers)
    vis = LL.Vision(lcfg)
    syms = vis.symbols()
    d_months = [LL.mkey(m) for m in LL.month_starts(LL._utc(start) - pd.Timedelta(days=60), end)]
    with ThreadPoolExecutor(cfg.vision_workers) as pool:
        daily = {s: d for s, d in pool.map(lambda s: (s, vis.klines(s, "1d", d_months)), syms) if d is not None and len(d)}
    uni = LL.pit_universe(daily, LL.month_starts(start, end), n, lcfg.min_listing_days)
    member = defaultdict(set)
    for m, lst in uni.items():
        for s in lst:
            member[s].add(LL.mkey(m))
    return dict(member)


_FEAT_CACHE: dict = {}


def features_for(cfg: FConfig, start, end, verbose=True):
    key = (cfg.universe, cfg.symbols, cfg.universe_n, cfg.exit_tf, str(start), str(end), cfg.window)
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    if cfg.universe == "pit":
        member = pit_members(cfg, start, end, cfg.universe_n)
        syms = sorted(member)
        if not any(s.startswith("BTCUSDT") for s in syms):
            syms.append("BTCUSDT"); member["BTCUSDT"] = set()
        data = load_symbols(cfg, syms, start, end, member=member, verbose=verbose)
    else:
        data = load_symbols(cfg, plist(cfg.symbols), start, end, verbose=verbose)
    t0 = time.time()
    feats = {s: build_feat(d, cfg) for s, d in data.items() if len(d.m5) >= cfg.window + 10}
    if verbose:
        print(f"   features: {len(feats)} symbols in {time.time() - t0:.0f}s")
    _FEAT_CACHE[key] = feats
    return feats


# ================= 8. PRE-REGISTERED EXPERIMENT =================
def r_stats(trades, cfg):
    if not trades:
        return dict(n=0, win=np.nan, avgR=np.nan, pf=np.nan, pnl_pct=0.0, exTop2=0.0)
    R = np.array([t["R"] for t in trades])
    w, lo = R[R > 0].sum(), -R[R <= 0].sum()
    s = np.sort(R)
    return dict(n=len(R), win=float((R > 0).mean()), avgR=float(R.mean()), pf=float(w / lo) if lo > 0 else np.inf,
                pnl_pct=sum(t["pnl"] for t in trades) / cfg.capital, exTop2=float(s[:-2].sum()) if len(s) > 2 else 0.0)


def eq_dd(trades, cfg):
    if not trades:
        return 0.0
    eq = np.r_[cfg.capital, cfg.capital + np.cumsum([t["pnl"] for t in trades])]
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def in_fold(t, a, b):
    return LL._utc(a).value <= t["entry_ns"] < LL._utc(b).value


def null_test(trades):
    """win rate vs the bracket's own break-even W0=1/(1+R): Poisson-binomial z-test + Beta posterior"""
    from scipy.stats import beta as Bd, norm
    if len(trades) < 2:
        return np.nan, np.nan
    w = np.array([1.0 if t["pnl"] > 0 else 0.0 for t in trades])
    w0 = np.array([t["null_w"] for t in trades])
    var = (w0 * (1 - w0)).sum()
    z = (w - w0).sum() / math.sqrt(var) if var > 0 else 0.0
    post = 1 - Bd.cdf(w0.mean(), 1 + w.sum(), 1 + len(w) - w.sum())
    return float(1 - norm.cdf(z)), float(post)


def bootstrap_dd(trades, sims=4000, seed=0):
    r = np.array([t["pnl"] / max(t["eq_before"], 1e-9) for t in trades])
    if len(r) < 30:
        return dict(iid=np.nan, block=np.nan, worst=np.nan)
    years = max((trades[-1]["exit_ns"] - trades[0]["entry_ns"]) / (365.25 * DAY), 0.25)
    L = int(max(100, len(r) / years))
    rng = np.random.default_rng(seed)

    def dd(paths):
        eq = np.cumprod(1 + paths, axis=1)
        peak = np.maximum.accumulate(np.c_[np.ones(len(eq)), eq], axis=1)[:, 1:]
        return float(np.percentile((eq / peak - 1).min(axis=1), 5))

    iid = dd(rng.choice(r, size=(sims, L)))
    months = pd.Series([pd.Timestamp(t["entry_ns"], tz="UTC").strftime("%Y-%m") for t in trades])
    groups = [r[(months == m).to_numpy()] for m in months.unique()]
    paths = np.empty((min(sims, 2000), L))
    for k in range(len(paths)):
        seq = []
        while len(seq) < L:
            seq.extend(groups[rng.integers(len(groups))])
        paths[k] = seq[:L]
    blk = dd(paths)
    return dict(iid=iid, block=blk, worst=min(iid, blk))


def pbo_cscv(results, start, end, S=16):
    """probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu): S time blocks, all S/2 splits"""
    names = [n for n, r in results.items() if r["trades"]]
    if len(names) < 2:
        return np.nan
    edges = np.linspace(LL._utc(start).value, LL._utc(end).value, S + 1)
    M = np.zeros((S, len(names)))
    for j, n in enumerate(names):
        for t in results[n]["trades"]:
            b = min(S - 1, max(0, int(np.searchsorted(edges, t["entry_ns"], side="right") - 1)))
            M[b, j] += t["R"]
    lam = []
    for train in itertools.combinations(range(S), S // 2):
        test = [i for i in range(S) if i not in train]
        best = int(np.argmax(M[list(train)].sum(0)))
        oos = M[test].sum(0)
        rank = (oos < oos[best]).sum() + 1
        w = rank / (len(names) + 1)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def _log_trials(cfg, names):
    path = os.path.join(cfg.cache_dir, "floop_experiments_log.jsonl")
    os.makedirs(cfg.cache_dir, exist_ok=True)
    with open(path, "a") as f:
        for n in names:
            f.write(json.dumps({"t": pd.Timestamp.now(tz="UTC").isoformat(), "variant": n}) + "\n")
    with open(path) as f:
        return len({json.loads(x)["variant"] for x in f if x.strip()})


def _f(x, f="{:+.2f}"):
    return "    -" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def run_variant(base: FConfig, name, overrides, start, end, verbose=False):
    c = replace(base, **overrides)
    if c.universe == "pit" and "exit_tf" not in overrides:
        c = replace(c, exit_tf="5m")                     # V6: exits on 5m bars (stop first) to keep the download small
    feats = features_for(c, start, end, verbose=verbose)
    trades, _ = run(feats, c, start, end)
    return c, trades


def run_experiment(base: FConfig, folds=FOLDS, start=EXP_START, end=EXP_END, variants=VARIANTS, extra=True):
    n_trials = _log_trials(base, list(variants))
    res = OrderedDict()
    for name, ov in variants.items():
        t0 = time.time()
        c, tr = run_variant(base, name, ov, start, end, verbose=True)
        res[name] = dict(cfg=c, trades=tr)
        print(f"  {name:<17} {len(tr):5d} trades ({time.time() - t0:.0f}s)")
    print(f"\n{'variant':<17} {'fold':<8} {'n':>5} {'win':>4} {'avgR':>6} {'PF':>5} {'P&L%':>7} {'exTop2':>7}")
    summary = {}
    for name, r in res.items():
        c, tr = r["cfg"], r["trades"]
        fa, fp = [], []
        for a, b, tag in folds:
            st = r_stats([t for t in tr if in_fold(t, a, b)], c)
            fa.append(st["avgR"]); fp.append(st["pnl_pct"])
            print(f"{name[:17]:<17} {a[:7]:<8} {st['n']:5d} {_f(st['win'], '{:.2f}')[-4:]:>4} {_f(st['avgR']):>6} "
                  f"{_f(st['pf'], '{:.2f}'):>5} {st['pnl_pct'] * 100:+7.1f} {st['exTop2']:+7.1f}")
        hold = [t for t in tr if any(in_fold(t, a, b) for a, b, g in folds if g == "holdout")]
        sh, sa, dd = r_stats(hold, c), r_stats(tr, c), eq_dd(tr, c)
        for lab, st in (("HOLDOUT", sh), ("ALL", sa)):
            print(f"{name[:17]:<17} {lab:<8} {st['n']:5d} {_f(st['win'], '{:.2f}')[-4:]:>4} {_f(st['avgR']):>6} "
                  f"{_f(st['pf'], '{:.2f}'):>5} {st['pnl_pct'] * 100:+7.1f} {st['exTop2']:+7.1f}"
                  + (f"   maxDD {dd * 100:.1f}%" if lab == "ALL" else ""))
        fin = np.array([x for x in fa if np.isfinite(x)])
        summary[name] = dict(med=float(np.median(fin)) if len(fin) else np.nan, dd=dd, hold=sh, all=sa, fp=fp, hold_tr=hold)
    cands = [n for n in SIMPLICITY if n in summary and summary[n]["dd"] >= -0.20 and summary[n]["all"]["n"] > 0
             and np.isfinite(summary[n]["med"])]
    if not cands:
        print("\nNo variant passes the −20% drawdown filter. Verdict: FAIL — no edge worth trading.")
        return res, summary, None
    best = max(summary[n]["med"] for n in cands)
    chosen = next(n for n in SIMPLICITY if n in cands and summary[n]["med"] >= best - 0.02)
    print(f"\nselected: {chosen} (median-fold avg R {best:+.2f}; ties within 0.02R go to the simpler variant)")
    gates = check_gates(base, res, summary, chosen, folds, start, end, n_trials, extra)
    return res, summary, gates


def check_gates(base, res, summary, chosen, folds, start, end, n_trials, extra=True):
    c, tr = res[chosen]["cfg"], res[chosen]["trades"]
    S = summary[chosen]
    hold, alls = S["hold"], S["all"]
    g = OrderedDict()
    g["1 trades ≥600 (≥200 hold-out)"] = (alls["n"] >= 600 and hold["n"] >= 200, f"{alls['n']} / {hold['n']}")
    g["2 hold-out PF≥1.3 & avgR≥0.2; all PF≥1.5"] = (
        hold["n"] > 0 and hold["pf"] >= 1.3 and hold["avgR"] >= 0.2 and alls["pf"] >= 1.5,
        f"{_f(hold['pf'], '{:.2f}')}, {_f(hold['avgR'])}; {_f(alls['pf'], '{:.2f}')}")
    parts, ok2b = [], True
    for mname, mk in (("trend", TREND), ("rev", REV)):
        sub = [t for t in tr if t["mode"] == mk]
        if len(sub) >= 100:
            p, post = null_test(sub)
            ok2b &= p < 0.05 and post >= 0.95
            parts.append(f"{mname} n={len(sub)} W={np.mean([t['pnl'] > 0 for t in sub]):.3f} "
                         f"W0={np.mean([t['null_w'] for t in sub]):.3f} p={p:.3f}")
        elif sub:
            parts.append(f"{mname} n={len(sub)} (<100, untestable)")
    ok2b &= bool(parts) and any("p=" in x for x in parts)
    g["2b win rate beats break-even 1/(1+R) per mode"] = (ok2b, "; ".join(parts) or "no trades")
    R = np.sort([t["R"] for t in tr])
    k5 = max(1, int(round(0.05 * len(R)))) if len(R) else 0
    ex5 = R[:-k5] if len(R) > k5 else np.array([])
    pf5 = ex5[ex5 > 0].sum() / abs(ex5[ex5 <= 0].sum()) if (ex5 <= 0).any() and ex5[ex5 <= 0].sum() != 0 else np.nan
    by_y = defaultdict(list)
    for t in tr:
        by_y[pd.Timestamp(t["entry_ns"], tz="UTC").year].append(t["R"])
    ex2y = sum(sum(sorted(v)[:-2]) if len(v) > 2 else 0.0 for v in by_y.values())
    top = sorted(tr, key=lambda t: -t["R"])[:10]
    nsym = len({t["sym"] for t in tr})
    yrs = len({pd.Timestamp(t["entry_ns"], tz="UTC").year for t in top})
    cns = len({t["sym"] for t in top})
    g["3 outliers: PF ex-top5%≥1, ex-top2/yr, top10 spread"] = (
        np.isfinite(pf5) and pf5 >= 1.0 and ex2y >= -0.1 * len(tr) and yrs >= 3 and cns >= min(2, nsym),
        f"PF {_f(pf5, '{:.2f}')}, {ex2y:+.1f}R, top10 {cns} coins/{yrs} yrs")
    wf = min(S["fp"]) if S["fp"] else 0.0
    g["4 median fold avgR>0, no fold < −12%"] = (np.isfinite(S["med"]) and S["med"] > 0 and wf >= -0.12,
                                               f"{_f(S['med'])}, worst {wf * 100:+.1f}%")
    bs = bootstrap_dd(tr)
    g["5 bootstrap 95th-pct DD ≥ −25% (1-yr paths)"] = (np.isfinite(bs["worst"]) and bs["worst"] >= -0.25,
                                                      f"iid {_f(bs['iid'] * 100, '{:.1f}')}%, block {_f(bs['block'] * 100, '{:.1f}')}%")
    srs = [np.mean([t["R"] for t in r["trades"]]) / np.std([t["R"] for t in r["trades"]], ddof=1)
           for r in res.values() if len(r["trades"]) >= 10 and np.std([t["R"] for t in r["trades"]]) > 0]
    dsr = LL.deflated_sharpe([t["R"] for t in tr], srs, n_trials)
    g["6 deflated Sharpe ≥ 0.90"] = (np.isfinite(dsr) and dsr >= 0.90, f"{_f(dsr, '{:.2f}')} (N={n_trials})")
    pbo = pbo_cscv(res, start, end)
    g["6b probability of backtest overfitting ≤ 0.2"] = (np.isfinite(pbo) and pbo <= 0.2, _f(pbo, "{:.2f}"))
    if extra:
        cs = replace(c, fee=0.0005, entry_slip=c.entry_slip * 2, tp_slip=c.tp_slip * 2, stop_slip=c.stop_slip * 2)
        trc, _ = run(features_for(cs, start, end, verbose=False), cs, start, end)
        pfc = r_stats(trc, cs)["pf"]
        g["7 cost stress (0.05% fee/side, 2× slippage) PF≥1.2"] = (np.isfinite(pfc) and pfc >= 1.2, _f(pfc, "{:.2f}"))
        outs = []
        nb = NEIGHBOURS.get(chosen, [])
        _log_trials(base, [f"{chosen}:{k}={v}" for k, v in nb])
        for k, v in nb:
            cc = replace(c, **{k: v})
            trv, _ = run(features_for(cc, start, end, verbose=False), cc, start, end)
            outs.append((k, v, r_stats(trv, cc)["avgR"]))
        g["8 neighbours keep avgR>0"] = (all(np.isfinite(x) and x > 0 for _, _, x in outs),
                                         ", ".join(f"{k}={v}:{_f(x)}" for k, v, x in outs))
    print(f"\npass/fail gates for {chosen}:")
    for nm, (ok, detail) in g.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}: {detail}")
    print("  [ -- ] 9 three months of paper trading on Azure: manual")
    print("\nVerdict: " + ("all automated gates pass → paper-trade it next (gate 9). Not a guarantee of profit."
                          if all(ok for ok, _ in g.values()) else
                          "at least one gate FAILS → do not scale this live. A failed gate is the answer, "
                          "not a reason to tune until it passes."))
    return g


# ================= 9. AZURE PAPER RUNNER (public data only, no keys, no orders) =================
FAPI = "https://fapi.binance.com"


def _public_get_default(path, params):
    import requests
    for k in range(4):
        try:
            r = requests.get(FAPI + path, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(30)
        except requests.RequestException:
            time.sleep(2 ** k)
    raise IOError(f"GET {path} failed")


PUBLIC_GET = _public_get_default


def _klines(path, sym, interval, limit, extra=None):
    raw = PUBLIC_GET(path, dict(symbol=sym, interval=interval, limit=limit, **(extra or {})))
    df = pd.DataFrame([[r[0], *map(float, r[1:6]), float(r[9]) if len(r) > 9 else np.nan] for r in raw],
                      columns=["t", "open", "high", "low", "close", "volume", "tbv"])
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    return df


def live_window(sym, cfg, now_ms):
    """the same inputs the live bot reads at a tick: closed bars only"""
    def closed(df, minutes):
        return df[df.index + pd.Timedelta(minutes=minutes) <= pd.Timestamp(now_ms, unit="ms", tz="UTC")]
    m5 = closed(_klines("/fapi/v1/klines", sym, "5m", cfg.window + 1), 5).iloc[-cfg.window:]
    h = {}
    for iv, mins, n in (("15m", 15, cfg.htf_window), ("1h", 60, cfg.mtf_window), ("4h", 240, cfg.mtf_window)):
        h[iv] = closed(_klines("/fapi/v1/klines", sym, iv, n + 1), mins).iloc[-n:]
    d1 = _klines("/fapi/v1/klines", sym, "1d", 3)
    prev = d1.iloc[-2] if len(d1) >= 2 else d1.iloc[0]
    return m5, h, (float(prev.high), float(prev.low), float(prev.close))


def tick_signal(sym, cfg, m5, htf, prev_day, mode):
    """v7.1 decision for the latest closed bar from live-shaped windows"""
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    Wrow = _tick(o, h, l, c, 0, len(c))
    tr = [int(htf_trend(d.high.to_numpy(float), d.low.to_numpy(float), d.close.to_numpy(float), len(d))[-1])
          if len(d) >= 20 else 0 for d in (htf["15m"], htf["1h"], htf["4h"])]
    ph, pl, pc = prev_day
    pp = (ph + pl + pc) / 3
    rg = ph - pl
    piv = np.array([pp, 2 * pp - pl, pp + rg, ph + 2 * (pp - pl), pp + rg * 3, 2 * pp - ph, pp - rg, pl - 2 * (ph - pp), pp - rg * 3])
    n = len(c)
    Wm = np.full((n, 25), np.nan)
    Wm[-1] = Wrow
    F1 = Feat(sym, m5.index.as_unit("ns").asi8, o, h, l, c, m5.volume.to_numpy(float), m5.tbv.to_numpy(float), Wm,
              np.full(n, tr[0], np.int8), np.full(n, tr[1], np.int8), np.full(n, tr[2], np.int8), np.tile(piv, (n, 1)),
              np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, np.int64), np.zeros(0), np.ones(n, bool))
    pre = pre_signals(F1, cfg)
    modes = np.full(n, mode, np.int8)
    d = directions(F1, cfg, pre, modes)
    return F1, pre, int(d[-1]), Wrow


class PaperBook:
    """the state FLOOP keeps in Binance + Deno KV, as a JSON file"""

    def __init__(self, path, cfg):
        self.path, self.cfg = path, cfg
        self.s = LL._load_json(path, None) or dict(cash=cfg.capital, open={}, closed=[], hist={}, hyst={}, log=[])

    def save(self):
        LL._save_json(self.s, self.path)

    def pooled(self, T):
        return [x for x in self.s["closed"] if x["exit_ns"] < T][-self.cfg.kelly_lookback:][::-1]

    def mode_log(self, T):
        return [x for x in self.s["closed"] if x["exit_ns"] < T and x.get("recorded", True)][-self.cfg.mode_log_cap:][::-1]

    def bankroll(self, T):
        return self.s["cash"] - sum(p["margin"] for p in self.s["open"].values())

    def funding_rate(self, T, sym):
        return self.s.get("funding", {}).get(sym, 0.0)

    def daily_loss(self, T):
        d = T // DAY
        return sum(-x["pnl"] for x in self.s["closed"]
                   if x["exit_ns"] // DAY == d and x["pnl"] < 0 and x.get("recorded", True))


def notify(text):
    import requests
    tok, chat, bark = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("BARK_URL")
    try:
        if tok and chat:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data=dict(chat_id=chat, text=text), timeout=10)
        if bark:
            requests.get(bark.rstrip("/") + "/FLOOP/" + requests.utils.quote(text), timeout=10)
    except Exception as e:                                        # noqa: BLE001
        print("notify failed:", e)


def paper_tick(cfg: FConfig, state_path="paper_state.json"):
    now_ms = int(PUBLIC_GET("/fapi/v1/time", {})["serverTime"])
    T = (now_ms // 300_000) * 300_000 * 10**6                    # latest 5m close (ns)
    book = PaperBook(state_path, cfg)
    S = book.s
    msgs = []
    for sym in plist(cfg.live_symbols):
        try:
            m5, htf, prev = live_window(sym, cfg, now_ms)
        except Exception as e:                                    # noqa: BLE001
            print(f"{sym}: data error {e}")
            continue
        fr = PUBLIC_GET("/fapi/v1/fundingRate", dict(symbol=sym, limit=1))
        S.setdefault("funding", {})[sym] = float(fr[-1]["fundingRate"]) if fr else 0.0
        p = S["open"].get(sym)
        if p:                                                     # advance the paper position on 1m mark bars
            mk = _klines("/fapi/v1/markPriceKlines", sym, "1m", 1500, dict(startTime=p["checked_ms"]))
            mk = mk[mk.index < pd.Timestamp(now_ms, unit="ms", tz="UTC").floor("min")]
            why, px = None, None
            for ts, r in mk.iterrows():
                if p["side"] > 0 and r.low <= p["stop"]:
                    why, px = "SL", min(r.open, p["stop"]) * (1 - cfg.stop_slip)
                elif p["side"] < 0 and r.high >= p["stop"]:
                    why, px = "SL", max(r.open, p["stop"]) * (1 + cfg.stop_slip)
                elif p["side"] > 0 and r.high >= p["tp"]:
                    why, px = "TP", max(r.open, p["tp"]) * (1 - cfg.tp_slip)
                elif p["side"] < 0 and r.low <= p["tp"]:
                    why, px = "TP", min(r.open, p["tp"]) * (1 + cfg.tp_slip)
                if why:
                    break
                p["checked_ms"] = int(ts.value // 10**6) + 60_000
            last = float(m5.close.iloc[-1])
            if not why and (last - p["entry"]) / p["entry"] * cfg.leverage * p["side"] <= -cfg.hard_stop_pct:
                why, px = "hard", last
            p["bars"] = p.get("bars", 0) + 1
            if not why and p.get("time_bars", 0) and p["bars"] >= p["time_bars"]:
                why, px = "time", last                                # V5: bot-side market close
            if not why and p.get("trail", 0):                         # V4: ratchet the stop on 5m closes
                p["ext"] = max(p.get("ext", p["entry"]), last) if p["side"] > 0 else min(p.get("ext", p["entry"]), last)
                p["stop"] = max(p["stop"], p["ext"] - p["trail"]) if p["side"] > 0 else min(p["stop"], p["ext"] + p["trail"])
            if why:
                pnl = p["side"] * p["qty"] * (px - p["entry"])
                S["cash"] += pnl
                S["closed"].append(dict(sym=sym, mode=p["mode"], pnl=pnl, margin=p["margin"], exit_ns=T,
                                        R=pnl / (p["qty"] * abs(p["entry"] - p["stop0"] if "stop0" in p else p["entry"] - p["stop"])),
                                        null_w=p["null_w"], why=why,
                                        recorded=(why in ("hard", "time")) or cfg.ledger == "all"))
                del S["open"][sym]
                msgs.append(f"CLOSE {sym} {why} {pnl:+.2f} USDC")
        F1, pre, d, Wrow = tick_signal(sym, cfg, m5, htf, prev, NEUTRAL)
        tp = float(Wrow[18])
        if cfg.mode_rule == "hysteresis":
            hs = S["hyst"].setdefault(sym, dict(prev=0, cand=0, run=0, conf=0))
            raw = TREND if tp >= 0.58 else REV if tp <= 0.42 else (
                TREND if hs["prev"] == TREND and tp >= 0.5 else REV if hs["prev"] == REV and tp <= 0.5 else NEUTRAL)
            hs["run"] = hs["run"] + 1 if raw == hs["cand"] else 1
            hs["cand"] = raw if hs["run"] == 1 else hs["cand"]
            if hs["run"] >= cfg.persist_ticks:
                hs["conf"] = hs["cand"]
            hs["prev"] = raw
            mode = hs["conf"]
        else:
            h0, h1 = (S["hist"].get(sym, []) + [-1, -1])[:2]
            raw = TREND if tp >= 0.58 else (REV if tp <= 0.42 else NEUTRAL)
            run_len = 0
            for m in (raw, h0):
                if m == raw and m not in (NEUTRAL, -1):
                    run_len += 1
                else:
                    break
            last_nn = next((m for m in (h0, h1) if m not in (NEUTRAL, -1)), -1)
            mode = (last_nn if last_nn != -1 else NEUTRAL) if raw == NEUTRAL else (
                raw if run_len >= 2 else (last_nn if last_nn != -1 else raw))
            S["hist"][sym] = [raw, h0]
        _, _, d, _ = tick_signal(sym, cfg, m5, htf, prev, mode)
        if cfg.trend_only and mode == REV:
            d = 0
        if d == 0 or sym in S["open"]:
            continue
        t = len(F1.c) - 1
        b = bracket(F1, t, mode, d, cfg)
        b["sym"] = sym
        if cfg.sizing == "fixed":
            notional = cfg.risk_pct * S["cash"] / (b["risk_pct"] + 2 * (cfg.fee + cfg.stop_slip))
            margin = notional / cfg.leverage
        else:
            margin, info = kelly_margin(book, T, mode, b, int(pre["total"][-1]), 1.0, cfg)
            notional = margin * cfg.leverage
        if margin <= 0 or book.daily_loss(T) >= cfg.daily_loss_limit:
            continue
        price = float(F1.c[-1])
        trail = cfg.trail_mult * b["vol"] if (cfg.trail_trend and mode == TREND) else 0.0
        S["open"][sym] = dict(side=b["side"], entry=price, qty=notional / price, margin=margin, stop=b["stop"],
                              stop0=b["stop"], tp=(b["tp"] if not trail else (1e18 if b["side"] > 0 else -1e18)),
                              mode=mode, null_w=1 / (1 + b["R"]), opened_ns=T, checked_ms=now_ms, trail=trail,
                              time_bars=int(round(cfg.time_stop * b["sl_mult"] * b["tp_mult"])) if cfg.time_stop > 0 else 0)
        msgs.append(f"OPEN {sym} {'LONG' if b['side'] > 0 else 'SHORT'} {'trend' if mode == TREND else 'reversion'} "
                    f"@{price:g} SL {b['stop']:.6g} TP {b['tp']:.6g} margin ${margin:.0f}")
    S["log"] = (S["log"] + [f"{pd.Timestamp(now_ms, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} " + m for m in msgs])[-500:]
    book.save()
    for m in msgs:
        print(m)
        notify("FLOOP paper: " + m)
    if not msgs:
        print(f"{pd.Timestamp(now_ms, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} no change; open: {list(S['open'])}; "
              f"cash {S['cash']:.2f}")


def golden_live(cfg: FConfig, path):
    """compare with the live bot's ?test=indicators JSON (run both within the same 5-minute bar)"""
    live = json.load(open(path))
    now_ms = int(PUBLIC_GET("/fapi/v1/time", {})["serverTime"])
    rows = []
    for sym, L in live.items():
        m5, htf, prev = live_window(sym, cfg, now_ms)
        _, pre, _, W = tick_signal(sym, cfg, m5, htf, prev, NEUTRAL)
        ind = L.get("indicators", {})
        pairs = [("close", float(m5.close.iloc[-1]), ind.get("close")), ("rfTrend", W[0], ind.get("rfTrend")),
                 ("rfFilt", W[2], ind.get("rfFilt")), ("atr", W[8], ind.get("atr")), ("adx", W[12], ind.get("adx")),
                 ("chop", W[13], ind.get("chop")), ("strength", pre["total"][-1], ind.get("strength")),
                 ("trendProb", W[18], L.get("regime", {}).get("trendProb")),
                 ("garchSigma", garch_fit_window(m5.close.to_numpy(float), 0, len(m5))[0],
                  L.get("garchFitted", {}).get("lastSigma"))]
        for k, a, b in pairs:
            ok = b is not None and abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
            rows.append((sym, k, a, b, ok))
    bad = [r for r in rows if not r[4]]
    for r in rows:
        print(f"{r[0]:<9} {r[1]:<11} python={r[2]!s:<22} live={r[3]!s:<22} {'OK' if r[4] else 'DIFF'}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} match" + ("" if not bad else
          " — if 'close' differs, the two were run in different 5-minute bars: rerun both"))


# ================= 10. CLI =================
def main(argv=None):
    ap = argparse.ArgumentParser(description="FLOOP Pro v7.1 Python port")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download")
    p.add_argument("--start", default=EXP_START)
    p.add_argument("--end", default=EXP_END)
    p.add_argument("--pit", action="store_true", help="also the V6 point-in-time top-10 universe")
    p = sub.add_parser("backtest")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--trades", action="store_true")
    sub.add_parser("experiment")
    p = sub.add_parser("paper")
    p.add_argument("--state", default="paper_state.json")
    p = sub.add_parser("golden")
    p.add_argument("live_json")
    a = ap.parse_args(argv)
    cfg = LL.apply_overrides(FConfig(), a.set)
    if cfg.variant != "V0" and cfg.variant in {k.split("_")[0] for k in VARIANTS}:
        name = next(k for k in VARIANTS if k.startswith(cfg.variant + "_"))
        cfg = LL.apply_overrides(replace(cfg, **VARIANTS[name]), a.set)
    if a.cmd == "download":
        features_for(cfg, a.start, a.end)
        if a.pit:
            features_for(replace(cfg, universe="pit", exit_tf="5m"), a.start, a.end)
    elif a.cmd == "backtest":
        c = replace(cfg, exit_tf="5m") if cfg.universe == "pit" else cfg
        trades, _ = run(features_for(c, a.start, a.end), c, a.start, a.end)
        if trades and (a.trades or len(trades) <= 40):
            print_trades(trades, limit=10**9 if a.trades else 40)
        for k, v in metrics(trades, c).items():
            print(f"  {k:<11} {v}")
    elif a.cmd == "experiment":
        run_experiment(cfg)
    elif a.cmd == "paper":
        paper_tick(cfg, a.state)
    elif a.cmd == "golden":
        golden_live(cfg, a.live_json)


if __name__ == "__main__":
    main()
