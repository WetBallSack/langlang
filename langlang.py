#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""langlang.py v2 —— Bit浪浪交易系统量化版（六阶段 / 介入三步 / 分仓 / 只做夏天）

  python langlang.py examples
  python langlang.py backtest --start 2024-01-01 --end 2025-01-01 --top 40 --workers 4
  python langlang.py scan [--symbols PEOPLEUSDT,SOLUSDT] [--capital 9000] [--execute] [--testnet]

约定：时间区间一律 [start, end)，UTC，按K线开盘时间；只使用已收盘K线。
v2 缓存格式与 v1 不兼容（v1 缓存里可能混有未收盘K线），请使用新的 cache 目录或删除旧目录。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

try:
    import ccxt
    _RETRYABLE: tuple = (ccxt.NetworkError,)          # 只重试网络类错误；BadSymbol 等直接抛出
except ImportError:                                  # 允许离线测试
    ccxt = None
    _RETRYABLE = (ConnectionError, TimeoutError)

try:
    import pyarrow  # noqa: F401
    _FMT = "parquet"
except ImportError:
    _FMT = "pkl"                                     # 比 CSV 快一个数量级，且保留 tz/dtype


# ================= 1. CONFIG =================
@dataclass
class Config:
    cache_dir: str = "./cache"
    ltf: str = "15m"
    ltf_warmup_days: int = 2            # 15m 指标预热天数，预热期内不开仓
    universe_top_n: int = 60
    min_quote_vol: float = 2e7
    workers: int = 4                    # 并行下载线程数（总请求速率仍受限）
    # --- 季节 ---
    ema_fast: int = 20; ema_slow: int = 50
    breadth_summer: float = 0.60
    breadth_winter: float = 0.30
    newhigh_summer: float = 0.15
    btc_autumn_dd: float = 0.10
    allow_late_spring: bool = False
    require_rs: bool = True
    # --- 阶段（日线）---
    base_min_bars: int = 20
    base_max_width: float = 0.60
    prior_decline: float = 0.40
    decline_lookback: int = 180
    stage1_bars: int = 5
    large_div_min_bars: int = 8
    large_div_depth: float = 0.50
    stage6_drop: float = 0.35
    # --- 空间 ---
    pivot_left: int = 3; pivot_right: int = 3
    res_cluster_pct: float = 0.015
    tp_front_run: float = 0.003
    min_room_pct: float = 0.12
    min_rr: float = 3.0
    use_round_numbers: bool = True
    use_volume_profile: bool = True; vp_bins: int = 60
    # --- 15m 拐点 ---
    pat_window: int = 96; pat_min_bars: int = 24; ltf_pivot: int = 3
    slope_tol_atr: float = 0.05
    contraction_max: float = 0.65
    breakout_buf_atr: float = 0.10
    vol_mult: float = 1.3; vol_lookback: int = 20
    stop_buf_atr: float = 0.25
    max_stop_pct: float = 0.08
    box_fallback: bool = True; box_bars: int = 32; box_max_atr: float = 4.0
    require_squeeze: bool = False
    max_attempts: int = 3
    # --- 仓位 ---
    total_capital: float = 30000; n_tranches: int = 3
    lev_btc: int = 10; lev_alt: int = 5
    max_loss_frac_of_tranche: float = 0.35
    size_scale_by_stop: bool = True; ref_stop_pct: float = 0.04
    tp1_frac: float = 0.5; be_after_tp1: bool = True
    mmr: float = 0.01; liq_buffer: float = 0.5
    # --- 成本 ---
    taker_fee: float = 0.0005; slippage: float = 0.0005
    funding_rate: float = 0.0001        # 每次结算，多头支付；建议换成历史资金费
    funding_interval_h: int = 8         # 部分合约是 4h 结算
    max_hold_bars: int = 96 * 10


PRIORITY = {1: 0, 4: 1, 2: 2}           # 1 > 4 > 2
BTC = "BTC/USDT:USDT"
STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "PYUSD", "RLUSD", "USD1", "XUSD"}
EXCLUDE_FROM_BREADTH: set[str] = set()  # 若币安把美股/商品永续也标成 COIN，在这里手动加 base（如 "AMZN"）
_TF_UNIT = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def tf_ms(tf: str) -> int:
    m = re.fullmatch(r"(\d+)([mhd])", tf)
    if not m:
        raise ValueError(f"unsupported timeframe {tf!r} (use e.g. 5m/15m/1h/4h/1d)")
    return int(m.group(1)) * _TF_UNIT[m.group(2)]


def _ms(x) -> int:
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 1_000_000)


def _ts(ms: int) -> pd.Timestamp:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


def to_ccxt(sym: str) -> str:
    """PEOPLEUSDT / PEOPLE/USDT / PEOPLE/USDT:USDT / people-usdt -> PEOPLE/USDT:USDT"""
    s = sym.upper().split(":")[0].replace("/", "").replace("-", "")
    return (s[:-4] if s.endswith("USDT") else s) + "/USDT:USDT"


# ================= 2. DATA：ccxt 币安U本位永续 + 区间覆盖缓存 =================
def make_exchange(private=False, testnet=False):
    kw = {"enableRateLimit": True}
    if private:
        kw.update(apiKey=os.getenv("BINANCE_KEY"), secret=os.getenv("BINANCE_SECRET"))
    ex = ccxt.binanceusdm(kw)
    if testnet:
        ex.enable_demo_trading(True)     # ccxt ≥4.5：币安合约测试环境走 demo trading
    return ex


_COLS = ["open", "high", "low", "close", "volume"]
_locks: defaultdict = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in _COLS}, index=pd.DatetimeIndex([], tz="UTC"))


def _to_df(rows) -> pd.DataFrame:
    if not rows:
        return _empty()
    arr = np.asarray(rows, dtype=float)
    idx = pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True).as_unit("ns")
    return pd.DataFrame(arr[:, 1:6], index=idx, columns=_COLS)


def _base(cfg, sym, tf):
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return os.path.join(cfg.cache_dir, f"{sym.split(':')[0].replace('/', '')}_{tf}")


def _read_cache(base):
    p = f"{base}.{_FMT}"
    if not os.path.exists(p):
        return None, []
    df = pd.read_parquet(p) if _FMT == "parquet" else pd.read_pickle(p)
    cov = []
    if os.path.exists(base + ".cov.json"):
        with open(base + ".cov.json") as f:
            cov = json.load(f)
    return df, cov


def _write_cache(base, df, cov):
    p = f"{base}.{_FMT}"
    tmp = p + ".tmp"
    df.to_parquet(tmp) if _FMT == "parquet" else df.to_pickle(tmp)
    os.replace(tmp, p)                                   # 原子替换，进程中断不损坏缓存
    with open(base + ".cov.json.tmp", "w") as f:
        json.dump(cov, f)
    os.replace(base + ".cov.json.tmp", base + ".cov.json")


def _merge_iv(iv):
    out = []
    for a, b in sorted(map(list, iv)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _missing_iv(cov, a, b):
    """[a,b) 减去已覆盖区间（cov 已排序合并）"""
    out, cur = [], a
    for x, y in cov:
        if y <= cur:
            continue
        if x >= b:
            break
        if x > cur:
            out.append((cur, x))
        cur = max(cur, y)
        if cur >= b:
            break
    if cur < b:
        out.append((cur, b))
    return out


def _download(ex, sym, tf, a, b, step):
    """分页下载开盘时间 ∈ [a,b) 的K线；每页1000根（权重5，性价比最高）。"""
    rows, since = [], a
    while since < b:
        for k in range(5):
            try:
                batch = ex.fetch_ohlcv(sym, tf, since=since, limit=1000, params={"until": b - 1})
                break
            except _RETRYABLE:
                if k == 4:
                    raise
                time.sleep(2 ** k)
        if not batch:
            break
        rows += [r for r in batch if a <= r[0] < b]
        since = batch[-1][0] + step
        if len(batch) < 1000:
            break
    return rows


def fetch_ohlcv(ex, cfg, symbol, tf, start, end=None) -> pd.DataFrame:
    """返回开盘时间 ∈ [start, end) 的已收盘K线。缓存记录“已下载过的时间区间”，
    只下载缺口；任何查询都不会覆盖/丢失已有数据。"""
    sym = to_ccxt(symbol)
    step = tf_ms(tf)
    base = _base(cfg, sym, tf)
    closed_until = ex.milliseconds() // step * step      # 开盘时间早于此的K线都已收盘
    a = _ms(start) // step * step
    b = min(_ms(end), closed_until) if end is not None else closed_until
    with _locks_guard:
        lock = _locks[base]
    with lock:
        df, cov = _read_cache(base)
        miss = _missing_iv(cov, a, b) if a < b else []
        if miss:
            new = _to_df([r for x, y in miss for r in _download(ex, sym, tf, x, y, step)])
            df = new if df is None else pd.concat([df, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            _write_cache(base, df, _merge_iv(cov + [[x, y] for x, y in miss]))
    if df is None or df.empty or a >= b:
        return _empty()
    return df.loc[(df.index >= _ts(a)) & (df.index < _ts(b))]


_tls = threading.local()


def _thread_exchange(base_ex, workers):
    ex = getattr(_tls, "ex", None)
    if ex is None or getattr(_tls, "owner", None) is not base_ex:
        ex = base_ex.__class__({"enableRateLimit": True})
        # 每个线程限速放慢，使总速率约为单实例的 1.5 倍（≈1800 权重/分钟 < 币安 2400）
        ex.rateLimit = int(math.ceil(base_ex.rateLimit * workers / 1.5))
        if base_ex.markets:
            ex.set_markets(base_ex.markets, getattr(base_ex, "currencies", None))
        _tls.ex, _tls.owner = ex, base_ex
    return ex


def fetch_many(ex, cfg, jobs):
    """jobs: [(symbol, tf, start, end)] -> {(ccxt_symbol, tf): DataFrame | None}；自动去重、并行。"""
    jobs = list({(to_ccxt(s), tf): (to_ccxt(s), tf, a, b) for s, tf, a, b in jobs}.values())

    def run(job):
        s, tf, a, b = job
        e = _thread_exchange(ex, cfg.workers) if cfg.workers > 1 else ex
        try:
            return (s, tf), fetch_ohlcv(e, cfg, s, tf, a, b)
        except Exception as err:                         # noqa: BLE001
            print(f"跳过 {s} {tf}: {type(err).__name__}: {str(err)[:120]}")
            return (s, tf), None

    if cfg.workers <= 1 or len(jobs) <= 1:
        return dict(map(run, jobs))
    if not ex.markets:
        ex.load_markets()
    with ThreadPoolExecutor(cfg.workers) as pool:
        return dict(pool.map(run, jobs))


def top_universe(ex, cfg, coins_only=True):
    ex.load_markets()
    tk = ex.fetch_tickers()
    out = []
    for s, m in ex.markets.items():
        if not (m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")):
            continue
        info = m.get("info") or {}
        if m.get("base") in STABLES or (coins_only and (
                m.get("base") in EXCLUDE_FROM_BREADTH
                or info.get("contractType", "PERPETUAL") != "PERPETUAL"
                or info.get("underlyingType", "COIN") != "COIN")):
            continue
        qv = float((tk.get(s) or {}).get("quoteVolume") or 0)
        if qv >= cfg.min_quote_vol:
            out.append((s, qv))
    return [s for s, _ in sorted(out, key=lambda x: -x[1])[: cfg.universe_top_n]]


def align_htf(df_htf, htf, ltf_index, ltf):
    """高周期K线在其收盘时刻才可见；用低周期K线的收盘时刻对齐（等价 freqtrade 的 merge_informative_pair）。"""
    x = df_htf.copy()
    x.index = x.index + pd.Timedelta(milliseconds=tf_ms(htf))
    return x.reindex(ltf_index + pd.Timedelta(milliseconds=tf_ms(ltf)), method="ffill").set_axis(ltf_index)


# ================= 3. INDICATORS =================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def pivots(df, left, right):
    """分形摆动点；位置 i 的点要到 i+right 才确认，使用时只取 i <= t-right。"""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df); w = left + right + 1
    ph = np.full(n, np.nan); pl = np.full(n, np.nan)
    if n >= w:
        c = np.arange(left, n - right)
        ph[c] = np.where(h[c] == swv(h, w).max(1), h[c], np.nan)
        pl[c] = np.where(l[c] == swv(l, w).min(1), l[c], np.nan)
    return ph, pl


def squeeze_on(df, n=20, bb=2.0, kc=1.5):
    m = df.close.rolling(n).mean(); sd = df.close.rolling(n).std(); e = ema(df.close, n); a = atr(df, n)
    return ((m + bb * sd) < (e + kc * a)) & ((m - bb * sd) > (e - kc * a))


def _fit(x, y):
    """2–3 点最小二乘直线（比 np.polyfit 快约 20 倍，结果一致）"""
    x = x.astype(float); xm, ym = x.mean(), y.mean(); dx = x - xm
    b = (dx * (y - ym)).sum() / (dx * dx).sum()
    return b, ym - b * xm


def ltf_arrays(m, cfg):
    """15m 全部预计算：pivot 索引、成交量均值、箱体高低点（逐根只做 O(1)/O(log n) 查找）"""
    ph, pl = pivots(m, cfg.ltf_pivot, cfg.ltf_pivot)
    h, l, v = m.high.to_numpy(), m.low.to_numpy(), m.volume.to_numpy()
    B = cfg.box_bars
    return dict(h=h, l=l, c=m.close.to_numpy(), v=v, a=atr(m).to_numpy(), ph=ph, pl=pl,
                hp=np.flatnonzero(np.isfinite(ph)), lp=np.flatnonzero(np.isfinite(pl)),
                vmean=pd.Series(v).rolling(cfg.vol_lookback).mean().shift(1).to_numpy(),
                box_hh=pd.Series(h).rolling(B).max().shift(1).to_numpy(),
                box_ll=pd.Series(l).rolling(B).min().shift(1).to_numpy(),
                sq=squeeze_on(m).to_numpy() if cfg.require_squeeze else None)


# ================= 4. MARKET SEASON =================
def breadth_table(closes, cfg):
    v = closes.notna()
    n = v.sum(axis=1).replace(0, np.nan)
    above = closes > closes.ewm(span=cfg.ema_fast, adjust=False).mean()
    newhigh = closes >= closes.rolling(30, min_periods=30).max()
    return pd.DataFrame({"breadth": (above & v).sum(axis=1) / n, "newhigh": (newhigh & v).sum(axis=1) / n})


def classify_season(btc, cfg, br=None, k=1):
    """k = 每天K线根数（日线 1，1h 为 24）。无宽度数据时退化为纯 BTC 版。"""
    cs = btc.close
    c = cs.to_numpy()
    es_s = ema(cs, cfg.ema_slow * k)
    ef, es = ema(cs, cfg.ema_fast * k).to_numpy(), es_s.to_numpy()
    slope = es_s.diff(5 * k).to_numpy()
    dd = (1 - cs / btc.high.rolling(30 * k, min_periods=1).max()).to_numpy()
    if br is None:
        b = nh = np.full(len(c), np.nan)
    else:
        br = br.reindex(btc.index).ffill()
        b, nh = br.breadth.to_numpy(), br.newhigh.to_numpy()
    out = np.empty(len(c), dtype=object)
    prev = "冬"
    for t in range(len(c)):
        hb = b[t] == b[t]
        up = c[t] > ef[t] > es[t] and slope[t] > 0
        if c[t] < es[t] and slope[t] <= 0 and (not hb or b[t] < cfg.breadth_winter):
            s = "冬"
        elif up and dd[t] < cfg.btc_autumn_dd and (not hb or (b[t] >= cfg.breadth_summer and nh[t] >= cfg.newhigh_summer)):
            s = "夏"
        elif prev in ("夏", "秋") and (dd[t] >= cfg.btc_autumn_dd or (hb and b[t] < cfg.breadth_summer)):
            s = "秋"
        elif c[t] > es[t] or (hb and b[t] >= cfg.breadth_winter):
            s = "春"
        else:
            s = "冬"
        out[t] = prev = s
    return pd.Series(out, index=btc.index, name="season")


# ================= 5. STAGE CLASSIFIER（日线状态机，无未来函数）=================
def classify_stages(d, cfg):
    n = len(d)
    h, l, c = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    e50 = ema(d.close, 50).to_numpy()
    B = cfg.base_min_bars
    hh_prev = pd.Series(h).rolling(B).max().shift(1).to_numpy()      # = h[i-B:i].max()
    ll_prev = pd.Series(l).rolling(B).min().shift(1).to_numpy()
    out = np.zeros(n, np.int8); epi = np.zeros(n, np.int32)
    bh = np.full(n, np.nan); bl = np.full(n, np.nan)
    st = ep = 0
    base_hi = base_lo = imp_hi = leg0 = s2_hi = div_lo = top = np.nan
    bo = hi_i = 0
    low_after = math.inf          # 冲击高点“之后”的最低价（v1 把高点当天的下影线也算成回撤）
    for i in range(n):
        if st in (0, 6):
            if i > B:
                hh, ll = hh_prev[i], ll_prev[i]
                j0 = max(0, i - cfg.decline_lookback)
                if c[i] > hh and i - B > j0 and (hh - ll) / ll <= cfg.base_max_width \
                        and ll <= h[j0:i - B].max() * (1 - cfg.prior_decline):
                    st, ep = 1, ep + 1
                    base_hi, base_lo, leg0 = hh, ll, ll
                    bo = hi_i = i; imp_hi = h[i]; low_after = math.inf
        elif st in (1, 2, 4):
            if h[i] >= imp_hi:
                imp_hi, hi_i, low_after = h[i], i, math.inf
            else:
                low_after = min(low_after, l[i])
            depth = (imp_hi - low_after) / max(imp_hi - leg0, 1e-12) if low_after < math.inf else 0.0
            if st == 1 and i - bo >= cfg.stage1_bars:
                st = 2
            if (st in (1, 2) and c[i] < base_lo) or (st == 4 and c[i] < leg0):
                st = 6 if st == 4 else 0
            elif i - hi_i >= cfg.large_div_min_bars or depth > cfg.large_div_depth:
                if st in (1, 2):
                    st, s2_hi, div_lo = 3, imp_hi, low_after
                else:
                    st, top = 5, imp_hi
        elif st == 3:
            div_lo = min(div_lo, l[i])
            if c[i] > s2_hi:
                st, ep = 4, ep + 1
                leg0, imp_hi, hi_i, low_after = div_lo, h[i], i, math.inf
            elif c[i] < e50[i] and c[i] <= s2_hi * (1 - cfg.stage6_drop):
                st = 6
        elif st == 5:
            top = max(top, h[i])
            if c[i] < e50[i] and c[i] <= top * (1 - cfg.stage6_drop):
                st = 6
        out[i], epi[i], bh[i], bl[i] = st, ep, base_hi, base_lo
    return pd.DataFrame({"stage": out, "episode": epi, "base_hi": bh, "base_lo": bl}, index=d.index)


# ================= 6. ROOM TO RESISTANCE =================
def daily_pack(d, cfg):
    """日线阻力相关数组一次性预计算"""
    ph = pivots(d, cfg.pivot_left, cfg.pivot_right)[0]
    idx = np.flatnonzero(np.isfinite(ph))
    tp = ((d.high + d.low + d.close) / 3).to_numpy()
    return dict(ph_idx=idx, ph_val=ph[idx], ath=d.high.cummax().to_numpy(), tp=tp, qv=tp * d.volume.to_numpy())


def round_levels(p):
    e = math.floor(math.log10(p))
    return [m * 10 ** e for m in (1, 2, 5, 10, 20)]


def hvn_levels(tp, w, bins):
    if len(tp) < 20:
        return []
    hist, edges = np.histogram(tp, bins=bins, weights=w)
    ctr = (edges[:-1] + edges[1:]) / 2
    mid = hist[1:-1]
    return ctr[1:-1][(mid > hist[:-2]) & (mid >= hist[2:]) & (mid > hist.mean())].tolist()


def resistance_levels(D, di, price, cfg):
    k = np.searchsorted(D["ph_idx"], di - cfg.pivot_right, side="right")   # 只用已确认的日线高点
    lv = D["ph_val"][:k].tolist()
    lv.append(float(D["ath"][di]))
    if cfg.use_round_numbers:
        lv += round_levels(price)
    if cfg.use_volume_profile:
        j = max(0, di - 180)
        lv += hvn_levels(D["tp"][j:di + 1], D["qv"][j:di + 1], cfg.vp_bins)
    out = []
    for x in sorted(x for x in lv if x > price * 1.005):
        if not out or x > out[-1] * (1 + cfg.res_cluster_pct):
            out.append(x)
    return out


def build_plan(sym, D, di, stage, ep, rs, bhi, blo, sig, cfg):
    entry, stop = sig["entry"], sig["stop"]
    if di < 0 or stop <= 0 or stop >= entry or (entry - stop) / entry > cfg.max_stop_pct:
        return None
    lv = resistance_levels(D, di, entry, cfg)
    risk = entry - stop
    fallback = entry + max(bhi - blo if np.isfinite(bhi) else 0, cfg.min_rr * 1.5 * risk)  # 无阻力→测量目标
    f = 1 - cfg.tp_front_run
    tp1 = lv[0] * f if lv else fallback
    tp2 = lv[1] * f if len(lv) > 1 else tp1 + (tp1 - entry)
    room, rr = (tp1 - entry) / entry, (tp1 - entry) / risk
    if room < cfg.min_room_pct or rr < cfg.min_rr:
        return None
    return dict(sym=sym, stage=int(stage), ep=int(ep), rs=float(rs), entry=entry, stop=stop, tp1=tp1, tp2=tp2,
                room=room, rr=rr, kind=sig["kind"], status=sig.get("status", "已突破"))


# ================= 7. 15m 拐点：收敛三角/楔形/窄箱体突破 =================
def find_pattern(L, t, cfg, pending=False):
    l, c, v = L["l"], L["c"], L["v"]
    A = L["a"][t]
    if t < cfg.pat_window or not A > 0:                  # 同时挡掉 NaN
        return None
    r, s = cfg.ltf_pivot, t - cfg.pat_window
    hp, lp = L["hp"], L["lp"]
    hi = hp[np.searchsorted(hp, s):np.searchsorted(hp, t - r, side="right")]
    lo = lp[np.searchsorted(lp, s):np.searchsorted(lp, t - r, side="right")]
    cand = None
    if len(hi) >= 2 and len(lo) >= 2:
        hx, lx = hi[-3:], lo[-3:]
        bh, ah = _fit(hx, L["ph"][hx])
        bl, al = _fit(lx, L["pl"][lx])
        x0 = int(min(hx[0], lx[0]))
        w0 = (bh - bl) * x0 + (ah - al)
        w1 = (bh - bl) * (t - 1) + (ah - al)
        if (t - x0 >= cfg.pat_min_bars and bh <= cfg.slope_tol_atr * A and bl >= -cfg.slope_tol_atr * A
                and w0 > 0 and w1 > 0 and w1 / w0 <= cfg.contraction_max):
            xs = np.arange(x0, t)
            cx = c[x0:t]
            if np.all(cx <= bh * xs + ah + 0.5 * A) and np.all(cx >= bl * xs + al - 0.5 * A):
                cand = ("收敛三角/楔形", bh * t + ah, bh * (t - 1) + ah, bh * (t + 1) + ah,
                        l[lx[-1]:t + 1].min(), bl * t + al)
    if cand is None and cfg.box_fallback:
        hh, ll = L["box_hh"][t], L["box_ll"][t]
        if hh - ll <= cfg.box_max_atr * A and (not cfg.require_squeeze or L["sq"][t - 1]):
            cand = ("窄幅箱体", hh, hh, hh, ll, ll)
    if cand is None:
        return None
    kind, line, prev, nxt, stop_ref, lower = cand
    buf = cfg.breakout_buf_atr * A
    stop = float(stop_ref - cfg.stop_buf_atr * A)
    if c[t] > line + buf and c[t - 1] <= prev + buf and v[t] >= cfg.vol_mult * v[t - cfg.vol_lookback:t].mean():
        return dict(kind=kind, status="已突破", entry=float(c[t]), stop=stop)
    if pending and lower - 0.5 * A <= c[t] <= line + buf:   # v1 会把已经向下破位的形态也报成“待突破”
        return dict(kind=kind, status="待突破(挂stop单)", entry=float(nxt + buf), stop=stop)
    return None


# ================= 8. 仓位：杠杆/强平检查 + 分仓记账 =================
def liq_price(entry, lev, mmr):
    """逐仓多头强平价（币安公式，忽略维持保证金速算额）"""
    return entry * (1 - 1 / lev) / (1 - mmr)


def size_position(sym, entry, stop, margin, cfg):
    cap = cfg.lev_btc if to_ccxt(sym) == BTC else cfg.lev_alt      # 美股永续按山寨处理
    sp = (entry - stop) / entry
    lev = cap
    while lev > 1 and sp * lev > cfg.max_loss_frac_of_tranche:
        lev -= 1
    while lev > 1 and (1 - liq_price(1.0, lev, cfg.mmr)) * cfg.liq_buffer < sp:   # 止损必须远在强平之前
        lev -= 1
    scale = min(1.0, max(0.3, cfg.ref_stop_pct / sp)) if cfg.size_scale_by_stop else 1.0
    mg = margin * scale
    notional = mg * lev
    return dict(lev=lev, margin=mg, notional=notional, qty=notional / entry,
                liq=liq_price(entry, lev, cfg.mmr), risk=notional * sp)


class TrancheManager:
    """3份；每笔固定用一份保证金；亏了从储备补齐，赚了把超出一份的部分提走；全部亏完停机。"""

    def __init__(self, cfg):
        self.unit = cfg.total_capital / cfg.n_tranches
        self.active, self.reserve = self.unit, cfg.total_capital - self.unit
        self.withdrawn = self.refills = 0.0
        self.halted = False

    def margin(self):
        return 0.0 if self.halted else min(self.active, self.unit)

    def settle(self, pnl):
        self.active += pnl
        if self.active > self.unit:
            self.withdrawn += self.active - self.unit
            self.active = self.unit
        else:
            take = min(self.unit - self.active, self.reserve)
            self.reserve -= take
            self.active += take
            self.refills += take
        if self.reserve <= 0 and self.active < 0.05 * self.unit:
            self.halted = True

    def equity(self):
        return self.active + self.reserve + self.withdrawn


# ================= 9. BACKTEST ENGINE =================
def _ltf_exec_arrays(m, cfg):
    tsv = m.index.as_unit("ns").asi8
    fund = tsv % (cfg.funding_interval_h * 3_600_000_000_000) == 0     # 资金费结算时刻（UTC 00/08/16）
    return (m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), m.close.to_numpy(), fund, tsv)


def simulate(X, plan, tm, cfg):
    o, h, l, c, fund, tsv = X
    i0 = plan["t"] + 1
    entry = o[i0] * (1 + cfg.slippage)
    stop = plan["stop"]
    if entry <= stop or entry >= plan["tp1"]:
        return None
    sz = size_position(plan["sym"], entry, stop, tm.margin(), cfg)
    qty = rem = sz["qty"]
    risk = qty * (entry - stop)
    pnl = -entry * qty * cfg.taker_fee
    tp1_done, why, last, fee, slip = False, "时间止损", entry, cfg.taker_fee, cfg.slippage
    end = min(len(o), i0 + cfg.max_hold_bars)
    i = i0
    for i in range(i0, end):
        if i > i0 and fund[i]:                            # 只在结算时刻持仓才付资金费
            pnl -= rem * o[i] * cfg.funding_rate
        if l[i] <= stop:                                  # 同根K线先算止损（保守）
            last = min(o[i], stop) * (1 - slip)
            pnl += rem * (last - entry) - rem * last * fee
            rem = 0.0
            why = "保本/止损" if tp1_done else "止损"
            break
        if not tp1_done and h[i] >= plan["tp1"]:
            q = qty * cfg.tp1_frac
            last = max(o[i], plan["tp1"]) * (1 - slip)     # TAKE_PROFIT_MARKET 也有滑点
            pnl += q * (last - entry) - q * last * fee
            rem -= q
            tp1_done = True
            if cfg.be_after_tp1:
                stop = max(stop, entry)
            if rem <= 1e-12:
                why = "止盈1"
                break
        if tp1_done and h[i] >= plan["tp2"]:
            last = max(o[i], plan["tp2"]) * (1 - slip)
            pnl += rem * (last - entry) - rem * last * fee
            rem = 0.0
            why = "止盈2"
            break
    if rem > 0:
        last = c[i] * (1 - slip)
        pnl += rem * (last - entry) - rem * last * fee
        if i == len(o) - 1 and end == len(o):
            why = "数据结束"
    cap = -(sz["margin"] + entry * qty * fee)             # 逐仓最多亏掉保证金（+开仓手续费）
    if pnl < cap:
        pnl, why = cap, "强平"
    tm.settle(pnl)
    return dict(sym=plan["sym"], stage=plan["stage"], kind=plan["kind"], entry_time=_ts(tsv[i0] // 1_000_000),
                exit_time=_ts(tsv[i] // 1_000_000), exit_ts=int(tsv[i]), entry=entry, stop=plan["stop"],
                tp1=plan["tp1"], tp2=plan["tp2"], exit=last, lev=sz["lev"], margin=sz["margin"], pnl=pnl,
                R=pnl / risk, why=why, equity=tm.equity(), reserve=tm.reserve)


def build_events(btc, data, cfg, start):
    """一次性生成全部候选信号（与持仓状态无关），按时间排序。
    先用向量化条件（季节/阶段/RS/放量）过滤，只在极少数K线上跑形态识别。"""
    season = classify_season(btc, cfg, breadth_table(pd.DataFrame({s: d.close for s, (d, _) in data.items()}), cfg))
    ok = ["夏"] + (["春"] if cfg.allow_late_spring else [])
    btc_r = btc.close / btc.close.shift(30)
    t_start = _ts(_ms(start))
    events = []
    for s, (d, m) in data.items():
        stg = classify_stages(d, cfg)
        daily = stg.assign(rs=(d.close / d.close.shift(30)) / btc_r.reindex(d.index) - 1,
                           season=season.reindex(d.index).ffill(), di=np.arange(len(d)))
        al = align_htf(daily, "1d", m.index, cfg.ltf)
        stage = al.stage.fillna(0).astype(int).to_numpy()
        ep = al.episode.fillna(0).astype(int).to_numpy()
        di = al.di.fillna(-1).astype(int).to_numpy()
        rs = al.rs.to_numpy(dtype=float)
        bhi, blo = al.base_hi.to_numpy(dtype=float), al.base_lo.to_numpy(dtype=float)
        L = ltf_arrays(m, cfg)
        n = len(m)
        tt = np.arange(n)
        with np.errstate(invalid="ignore"):
            mask = (al.season.isin(ok).to_numpy() & np.isin(stage, list(PRIORITY))
                    & ((rs > 0) if cfg.require_rs else True)
                    & (tt >= cfg.pat_window) & (tt + 1 < n) & np.asarray(m.index >= t_start)
                    & (L["v"] >= cfg.vol_mult * L["vmean"] * (1 - 1e-9)))   # 放量是突破的必要条件
        if not mask.any():
            continue
        D = daily_pack(d, cfg)
        tsv = m.index.as_unit("ns").asi8
        for t in np.flatnonzero(mask):
            sig = find_pattern(L, t, cfg)
            if sig is None:
                continue
            plan = build_plan(s, D, di[t], stage[t], ep[t], rs[t], bhi[t], blo[t], sig, cfg)
            if plan:
                plan.update(t=int(t), ts=int(tsv[t]), key=(s, int(ep[t]), int(stage[t])))
                events.append(plan)
    events.sort(key=lambda p: p["ts"])                    # 稳定排序：同一时刻保持品种顺序
    return events


def run_backtest(btc, data, cfg, start):
    events = build_events(btc, data, cfg, start)
    X = {s: _ltf_exec_arrays(m, cfg) for s, (_, m) in data.items()}
    tm = TrancheManager(cfg)
    trades, attempts, busy = [], {}, -1
    for ts, grp in itertools.groupby(events, key=lambda p: p["ts"]):
        if tm.halted:
            break
        if ts <= busy:                                    # 一次只占一份：持仓期间不开新仓
            continue
        cands = [p for p in grp if attempts.get(p["key"], 0) < cfg.max_attempts]
        if not cands:
            continue
        plan = min(cands, key=lambda x: (PRIORITY[x["stage"]], -x["rs"]))
        attempts[plan["key"]] = attempts.get(plan["key"], 0) + 1
        tr = simulate(X[plan["sym"]], plan, tm, cfg)
        if tr:
            trades.append(tr)
            busy = tr["exit_ts"]
    return trades, tm


def warm(start, days=420):
    return pd.Timestamp(start) - pd.Timedelta(days=days)


def backtest(ex, cfg, symbols, start, end):
    symbols = list(dict.fromkeys(to_ccxt(s) for s in symbols))
    ltf_start = pd.Timestamp(start) - pd.Timedelta(days=cfg.ltf_warmup_days)
    jobs = [(s, "1d", warm(start), end) for s in [BTC] + symbols] + [(s, cfg.ltf, ltf_start, end) for s in symbols]
    got = fetch_many(ex, cfg, jobs)
    btc = got.get((BTC, "1d"))
    if btc is None or btc.empty:
        raise RuntimeError("BTC 日线缺失，无法判断季节")
    data = {}
    for s in symbols:
        d, m = got.get((s, "1d")), got.get((s, cfg.ltf))
        if d is not None and m is not None and len(d) > cfg.base_min_bars + 30 and len(m) > cfg.pat_window + 30:
            data[s] = (d, m)
    return run_backtest(btc, data, cfg, start)


def metrics(trades, tm, cfg):
    if not trades:
        return {"笔数": 0}
    df = pd.DataFrame(trades)
    pn = df.pnl
    eq = pd.concat([pd.Series([cfg.total_capital]), cfg.total_capital + pn.cumsum()], ignore_index=True)
    gl = -pn[pn < 0].sum()
    return {"笔数": len(df), "胜率": round((pn > 0).mean(), 4), "平均R": round(df.R.mean(), 3),
            "PF": round(pn[pn > 0].sum() / gl, 3) if gl > 0 else float("inf"),
            "最大回撤(平仓权益)": round((eq / eq.cummax() - 1).min(), 4), "累计提走利润": round(tm.withdrawn, 2),
            "储备补仓总额": round(tm.refills, 2), "消耗份额数": round(tm.refills / tm.unit, 2),
            "剩余储备": round(tm.reserve, 2), "停机": tm.halted, "总权益": round(tm.equity(), 2)}


# ================= 10. LIVE SCANNER（默认只出信号）=================
def place_bracket(ex, cfg, plan):
    """市价开多 + 止损 + 两档止盈（ccxt 会把 USDⓈ-M 条件单路由到币安 algo 下单接口）。
    TP1 成交后把止损移到保本需要另行监控，本函数不做。先用 --testnet 验证。"""
    s = to_ccxt(plan["sym"])
    ex.load_markets()
    open_pos = [p["symbol"] for p in ex.fetch_positions() if abs(float(p.get("contracts") or 0)) > 0]
    if open_pos:
        print("已有持仓，跳过（一次只占一份）:", open_pos)
        return
    free = float(ex.fetch_balance().get("free", {}).get("USDT") or 0)
    sz = size_position(s, plan["entry"], plan["stop"], min(TrancheManager(cfg).unit, free * 0.95), cfg)
    px = lambda x: float(ex.price_to_precision(s, x))
    amt = lambda x: float(ex.amount_to_precision(s, x))
    q = amt(sz["qty"]); half = amt(q * cfg.tp1_frac); rest = amt(q - half)   # 先算好精度，失败则不开仓
    try:
        ex.set_margin_mode("isolated", s)
    except Exception as e:                                # noqa: BLE001  已经是逐仓时币安会报错，可忽略
        print("margin mode:", str(e)[:80])
    ex.set_leverage(sz["lev"], s)
    ex.create_order(s, "market", "buy", q)
    try:
        ex.create_order(s, "market", "sell", q, None, {"stopLossPrice": px(plan["stop"]), "reduceOnly": True})
        ex.create_order(s, "market", "sell", half, None, {"takeProfitPrice": px(plan["tp1"]), "reduceOnly": True})
        ex.create_order(s, "market", "sell", rest, None, {"takeProfitPrice": px(plan["tp2"]), "reduceOnly": True})
    except Exception:
        print("保护单下单失败 → 立即市价平仓")
        ex.create_order(s, "market", "sell", q, None, {"reduceOnly": True})
        raise
    print(f"已下单 {s}: {q} @市价, {sz['lev']}x, 止损 {px(plan['stop'])}, TP1 {px(plan['tp1'])}, TP2 {px(plan['tp2'])}")


def scan(ex, cfg, symbols=None, execute=False, testnet=False):
    ex.load_markets()
    universe = top_universe(ex, cfg)                      # 宽度永远按大盘算，不能只按 --symbols 的几个币
    targets = [to_ccxt(s) for s in symbols] if symbols else universe
    now = _ts(ex.milliseconds())                          # 与数据层同一时钟
    daily = fetch_many(ex, cfg, [(s, "1d", now - pd.Timedelta(days=420), None) for s in [BTC] + universe + targets])
    btc = daily.get((BTC, "1d"))
    if btc is None or btc.empty:
        raise RuntimeError("BTC 日线缺失")
    closes = pd.DataFrame({s: daily[(s, "1d")].close for s in universe if daily.get((s, "1d")) is not None})
    season = classify_season(btc, cfg, breadth_table(closes, cfg))
    cur = season.iloc[-1]
    ok = cur == "夏" or (cfg.allow_late_spring and cur == "春")
    print(f"市场季节(截至 {season.index[-1].date()} UTC 日线收盘): {cur} → {'可以做' if ok else '不做，只观察'}")
    btc_r = btc.close / btc.close.shift(30)
    pre = {}
    for s in targets:                                     # 先用日线筛：阶段 + 龙头，再只给幸存者拉 15m
        d = daily.get((s, "1d"))
        if d is None or len(d) < 60 or d.index[-1] < btc.index[-1]:   # 数据不全/已停更的跳过
            continue
        stg = classify_stages(d, cfg)
        st = int(stg.stage.iloc[-1])
        rs = float((d.close / d.close.shift(30)).iloc[-1] / btc_r.reindex(d.index).iloc[-1] - 1)
        if st in PRIORITY and (not cfg.require_rs or rs > 0):
            pre[s] = (d, stg, st, rs)
    ltf = fetch_many(ex, cfg, [(s, cfg.ltf, now - pd.Timedelta(days=3), None) for s in pre])
    tm = TrancheManager(cfg)
    rows = []
    for s, (d, stg, st, rs) in pre.items():
        m = ltf.get((s, cfg.ltf))
        if m is None or len(m) <= cfg.pat_window + 1:
            continue
        sig = find_pattern(ltf_arrays(m, cfg), len(m) - 1, cfg, pending=True)
        if not sig:
            continue
        plan = build_plan(s, daily_pack(d, cfg), len(d) - 1, st, stg.episode.iloc[-1], rs,
                          stg.base_hi.iloc[-1], stg.base_lo.iloc[-1], sig, cfg)
        if not plan:
            continue
        sz = size_position(s, plan["entry"], plan["stop"], tm.margin(), cfg)
        P = lambda x: float(ex.price_to_precision(s, x))
        try:
            qty = float(ex.amount_to_precision(s, sz["qty"]))
        except Exception:                                 # noqa: BLE001  低于最小下单量
            qty = float("nan")
        rows.append({"品种": s.split("/")[0] + "USDT", "阶段": st, "形态": plan["kind"], "状态": plan["status"],
                     "入场": P(plan["entry"]), "止损": P(plan["stop"]),
                     "止损%": round((plan["entry"] - plan["stop"]) / plan["entry"] * 100, 2),
                     "TP1(50%)": P(plan["tp1"]), "TP2": P(plan["tp2"]), "空间%": round(plan["room"] * 100, 1),
                     "RR": round(plan["rr"], 2), "杠杆": f'{sz["lev"]}x', "保证金U": round(sz["margin"], 0),
                     "名义U": round(sz["notional"], 0), "数量": qty, "强平估算": P(sz["liq"]),
                     "RS vs BTC%": round(rs * 100, 1), "_plan": plan})
    if not rows:
        print("当前没有满足 阶段+空间+拐点 的标的。")
        return
    rows.sort(key=lambda r: (PRIORITY[r["阶段"]], -r["RS vs BTC%"]))
    print(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_string(index=False))
    if execute:
        if not ok:
            print("非夏天，不下单。")
        elif os.getenv("LANGLANG_LIVE") != "1":
            print("未设置 LANGLANG_LIVE=1，不下单。")
        else:
            r = next((r for r in rows if r["状态"] == "已突破"), None)     # 一次只开一个币、一份
            if r:
                place_bracket(make_exchange(private=True, testnet=testnet), cfg, r["_plan"])


# ================= 11. 图中例子复现 =================
def examples(ex, cfg):
    d = fetch_ohlcv(ex, cfg, "PEOPLEUSDT", "1d", "2023-06-01", "2024-06-01")
    v = d.join(classify_stages(d, cfg)).assign(chg=lambda x: (x.close.pct_change() * 100).round(2))
    print("== PEOPLE 日线阶段 2024-02-05 ~ 2024-05-21 ==")
    print(v.loc["2024-02-05":"2024-05-21", ["open", "high", "low", "close", "chg", "stage", "base_hi", "base_lo"]].to_string())
    m = fetch_ohlcv(ex, cfg, "PEOPLEUSDT", cfg.ltf, "2024-05-13", "2024-05-23")
    L = ltf_arrays(m, cfg)
    print("== PEOPLE 15m 收敛突破 2024-05-15 ~ 05-22 ==")
    for t in np.flatnonzero(np.asarray(m.index >= pd.Timestamp("2024-05-15", tz="UTC"))):
        s = find_pattern(L, t, cfg)
        if s:
            print(m.index[t], s["kind"], "入场", round(s["entry"], 5), "止损", round(s["stop"], 5))
    b = fetch_ohlcv(ex, cfg, "BTCUSDT", "1h", "2023-10-01", "2024-03-30")
    se = classify_season(b, cfg, None, k=24).loc["2024-01-17":"2024-03-29"]
    print("== BTC 1h 季节(纯BTC版，按日取众数) ==")
    print(se.resample("1D").agg(lambda x: x.mode().iat[0] if len(x) else None).to_string())


# ================= 12. CLI =================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--capital", type=float, default=None, help="分仓总资金（默认 30000）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--start", default="2024-01-01")
    b.add_argument("--end", default="2025-01-01", help="不含当天")
    b.add_argument("--symbols", default="")
    b.add_argument("--top", type=int, default=40)
    s = sub.add_parser("scan")
    s.add_argument("--symbols", default="")
    s.add_argument("--execute", action="store_true")
    s.add_argument("--testnet", action="store_true")
    sub.add_parser("examples")
    a = ap.parse_args()
    cfg = Config(workers=a.workers)
    if a.capital:
        cfg.total_capital = a.capital
    ex = make_exchange()
    if a.cmd == "examples":
        examples(ex, cfg)
    elif a.cmd == "scan":
        scan(ex, cfg, [x for x in a.symbols.split(",") if x] or None, a.execute, a.testnet)
    else:
        cfg.universe_top_n = a.top
        syms = [x for x in a.symbols.split(",") if x] or top_universe(ex, cfg)   # 当前成交额榜 = 幸存者偏差
        trades, tm = backtest(ex, cfg, syms, a.start, a.end)
        if trades:
            print(pd.DataFrame(trades).drop(columns=["tp2", "exit_ts"]).to_string(index=False))
        for k, val in metrics(trades, tm, cfg).items():
            print(f"{k}: {val}")
