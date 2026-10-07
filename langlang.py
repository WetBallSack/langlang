#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""langlang.py v2.1 —— Bit浪浪 stage-breakout system, refined.

What changed vs v2 (see the refinement plan):
  * risk-based sizing: 1% of trading capital at the stop, 2x default; the tranche is only a margin ceiling
  * trailing exits (4h chandelier by default, armed at +1R, no fixed TP, no breakeven); v2 exits kept as exit_mode=fixed
  * minimum stop distance max(2×ATR14(1h), 1.5%), per-coin 48h cooldown, ≤2 losses per stage episode
  * point-in-time universe from data.binance.vision (includes delisted coins) + real historical funding
  * optional: stage-1 trigger on 4h closes, sector/cluster heat filter
  * pre-registered experiment runner with walk-forward folds, bootstrap drawdowns, deflated Sharpe and pass/fail gates

Commands (ranges are [start, end), UTC, by bar open time; only closed bars are used):
  python langlang.py download   --start 2021-01-01 --end 2026-10-01        # one-time data build (~20–40 min)
  python langlang.py backtest   --start 2024-01-01 --end 2025-01-01 [--set exit_mode=donchian]
  python langlang.py experiment                                            # frozen variants × folds × gates
  python langlang.py scan       [--symbols SOLUSDT,PEOPLEUSDT]               # live signals, no orders by default
  python langlang.py examples
Global options go BEFORE the command: --capital 9000 --workers 4 --set key=value (repeatable)
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import os
import re
import threading
import time
import typing
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

try:
    import ccxt
    _RETRYABLE: tuple = (ccxt.NetworkError,)
except ImportError:                                  # offline tests
    ccxt = None
    _RETRYABLE = (ConnectionError, TimeoutError)

try:
    import pyarrow  # noqa: F401
    _FMT = "parquet"
except ImportError:
    _FMT = "pkl"


# ================= 1. CONFIG =================
@dataclass
class Config:
    cache_dir: str = "./cache"
    ltf: str = "15m"
    ltf_warmup_days: int = 2
    workers: int = 4                     # ccxt download threads
    vision_workers: int = 8              # data.binance.vision download threads
    # --- universe ---
    universe_mode: str = "pit"           # pit = data.binance.vision incl. delisted | static = today's top-N via ccxt
    universe_n: int = 60
    min_listing_days: int = 120
    min_quote_vol: float = 2e7           # static mode only
    # --- season ---
    ema_fast: int = 20
    ema_slow: int = 50
    breadth_summer: float = 0.60
    breadth_winter: float = 0.30
    newhigh_summer: float = 0.15
    btc_autumn_dd: float = 0.10
    allow_late_spring: bool = False
    require_rs: bool = True
    # --- stages (daily) ---
    base_min_bars: int = 20
    base_max_width: float = 0.60
    prior_decline: float = 0.40
    decline_lookback: int = 180
    stage1_bars: int = 5
    large_div_min_bars: int = 8
    large_div_depth: float = 0.50
    stage6_drop: float = 0.35
    stage1_mode: str = "d20"             # d20 = daily close > 20-day high | h4 = first 4h close above it + volume
    stage1_h4_vol_mult: float = 2.0
    # --- room to resistance ---
    pivot_left: int = 3
    pivot_right: int = 3
    res_cluster_pct: float = 0.015
    tp_front_run: float = 0.003
    min_room_pct: float = 0.12
    min_rr: float = 2.0                  # v2: 3.0
    round_mode: str = "next"             # next = only the next 1-2-5 level counts | grid = v2 (all levels)
    blue_sky_pass: bool = True           # no real resistance above → room filter passes
    use_round_numbers: bool = True
    use_volume_profile: bool = True
    vp_bins: int = 60
    # --- 15m pattern ---
    pat_window: int = 96
    pat_min_bars: int = 24
    ltf_pivot: int = 3
    slope_tol_atr: float = 0.05
    contraction_max: float = 0.65
    breakout_buf_atr: float = 0.10
    vol_mult: float = 1.3
    vol_lookback: int = 20
    stop_buf_atr: float = 0.25
    box_fallback: bool = True
    box_bars: int = 32
    box_max_atr: float = 4.0
    require_squeeze: bool = False
    max_attempts: int = 0                # v2: 3 per (coin, episode, stage); 0 = off
    # --- stops / cooldown ---
    min_stop_atr1h: float = 2.0
    min_stop_pct: float = 0.015
    max_stop_pct: float = 0.08
    cooldown_bars: int = 192             # 48h of 15m bars after a losing exit
    max_losses_per_episode: int = 2      # 0 = off
    # --- sizing / account ---
    sizing: str = "risk"                 # risk | tranche (v2)
    total_capital: float = 9000.0
    n_tranches: int = 3
    risk_pct: float = 0.01
    lev_default: float = 2.0
    lev_btc: int = 10
    lev_alt: int = 5
    halt_frac: float = 0.75
    port_loss_streak: int = 3
    port_streak_risk_mult: float = 0.5
    max_loss_frac_of_tranche: float = 0.35   # v2 tranche sizing only
    size_scale_by_stop: bool = True          # v2 tranche sizing only
    ref_stop_pct: float = 0.04               # v2 tranche sizing only
    mmr: float = 0.01
    liq_buffer: float = 0.5
    # --- exits ---
    exit_mode: str = "chandelier"        # chandelier | donchian | partial | fixed
    trail_tf_min: int = 240
    trail_atr_n: int = 14
    trail_mult: float = 3.0
    trail_activate_R: float = 1.0
    donchian_days: int = 10
    partial_frac: float = 0.33
    partial_R: float = 2.0               # partial target when there is no real resistance
    tp1_frac: float = 0.5                # fixed mode
    be_after_tp1: bool = True            # fixed mode
    time_stop_bars: int = 960            # fixed mode (v2: 10 days)
    stale_bars: int = 480
    stale_R: float = 1.0
    max_hold_bars: int = 5760
    # --- sector heat (optional) ---
    heat_mode: str = "off"               # off | expand
    heat_n_clusters: int = 8
    heat_risk_mult: float = 0.5
    # --- costs ---
    taker_fee: float = 0.0005
    slippage: float = 0.0005
    funding_source: str = "real"         # real (archive, pit mode) | flat
    funding_rate: float = 0.0001
    funding_interval_h: int = 8


PRIORITY = {1: 0, 4: 1, 2: 2}            # 1 > 4 > 2
BTC = "BTC/USDT:USDT"
DAY_NS = 86_400 * 10**9
STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "PYUSD", "RLUSD", "USD1", "XUSD", "EUR"}
INDEX_PERPS = {"BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL"}
# Not crypto-native (stocks / metals / ETFs / gold tokens): excluded from universe and breadth. Edit as needed.
EXCLUDE_BASES = {"XAU", "XAG", "XPT", "XPD", "PAXG", "AMZN", "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "META",
                 "COIN", "MSTR", "HOOD", "KORU", "SKHY", "AMAT", "SNDK", "SPY", "QQQ", "CRCL"}
_TF_UNIT = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


# ================= 2. UTILITIES =================
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


def _utc(x) -> pd.Timestamp:
    return _ts(_ms(x))


def to_ccxt(sym: str) -> str:
    """PEOPLEUSDT / PEOPLE/USDT / PEOPLE/USDT:USDT / people-usdt -> PEOPLE/USDT:USDT"""
    s = sym.upper().split(":")[0].replace("/", "").replace("-", "")
    return (s[:-4] if s.endswith("USDT") else s) + "/USDT:USDT"


def plain(sym: str) -> str:
    """-> PEOPLEUSDT"""
    return to_ccxt(sym).split("/")[0] + "USDT"


def base_of(sym: str) -> str:
    return to_ccxt(sym).split("/")[0]


def month_starts(start, end):
    """month-start Timestamps m with start's month <= m < end"""
    a = _utc(start).floor("D").replace(day=1)
    return list(pd.date_range(a, _utc(end) - pd.Timedelta(milliseconds=1), freq="MS"))


def mkey(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def apply_overrides(cfg: Config, pairs) -> Config:
    """--set key=value (repeatable) → new Config"""
    hints = typing.get_type_hints(type(cfg))
    for kv in pairs or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip()
        if k not in hints:
            raise SystemExit(f"unknown Config key {k!r}. Valid keys:\n  " + " ".join(sorted(hints)))
        t = hints[k]
        if t is bool:
            val = v.strip().lower() in ("1", "true", "yes", "on")
        elif t is int:
            val = int(float(v))
        elif t is float:
            val = float(v)
        else:
            val = v
        cfg = replace(cfg, **{k: val})
    return cfg


# ================= 3. DATA (live / static): ccxt Binance USDT-M + interval cache (from v2) =================
def make_exchange(private=False, testnet=False):
    kw = {"enableRateLimit": True}
    if private:
        kw.update(apiKey=os.getenv("BINANCE_KEY"), secret=os.getenv("BINANCE_SECRET"))
    ex = ccxt.binanceusdm(kw)
    if testnet:
        ex.enable_demo_trading(True)
    return ex


_COLS = ["open", "high", "low", "close", "volume"]
_locks: defaultdict = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _lock(key):
    with _locks_guard:
        return _locks[key]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in _COLS}, index=pd.DatetimeIndex([], tz="UTC"))


def _to_df(rows) -> pd.DataFrame:
    if not rows:
        return _empty()
    arr = np.asarray(rows, dtype=float)
    idx = pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True).as_unit("ns")
    return pd.DataFrame(arr[:, 1:6], index=idx, columns=_COLS)


def _save(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp) if _FMT == "parquet" else df.to_pickle(tmp)
    os.replace(tmp, path)


def _load(path):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path) if _FMT == "parquet" else pd.read_pickle(path)


def _save_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _base(cfg, sym, tf):
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return os.path.join(cfg.cache_dir, f"{sym.split(':')[0].replace('/', '')}_{tf}")


def _merge_iv(iv):
    out = []
    for a, b in sorted(map(list, iv)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _missing_iv(cov, a, b):
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
    """Closed bars with open time in [start, end). Interval-tracked cache: only gaps are downloaded."""
    sym = to_ccxt(symbol)
    step = tf_ms(tf)
    base = _base(cfg, sym, tf)
    closed_until = ex.milliseconds() // step * step
    a = _ms(start) // step * step
    b = min(_ms(end), closed_until) if end is not None else closed_until
    with _lock(base):
        df = _load(f"{base}.{_FMT}")
        cov = _load_json(base + ".cov.json", []) if df is not None else []
        miss = _missing_iv(cov, a, b) if a < b else []
        if miss:
            new = _to_df([r for x, y in miss for r in _download(ex, sym, tf, x, y, step)])
            df = new if df is None else pd.concat([df, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            _save(df, f"{base}.{_FMT}")
            _save_json(_merge_iv(cov + [[x, y] for x, y in miss]), base + ".cov.json")
    if df is None or df.empty or a >= b:
        return _empty()
    return df.loc[(df.index >= _ts(a)) & (df.index < _ts(b))]


_tls = threading.local()


def _thread_exchange(base_ex, workers):
    ex = getattr(_tls, "ex", None)
    if ex is None or getattr(_tls, "owner", None) is not base_ex:
        ex = base_ex.__class__({"enableRateLimit": True})
        ex.rateLimit = int(math.ceil(base_ex.rateLimit * workers / 1.5))   # total ≈1800 weight/min < 2400
        if base_ex.markets:
            ex.set_markets(base_ex.markets, getattr(base_ex, "currencies", None))
        _tls.ex, _tls.owner = ex, base_ex
    return ex


def fetch_many(ex, cfg, jobs):
    """jobs: [(symbol, tf, start, end)] -> {(ccxt_symbol, tf): DataFrame | None}; deduplicated, parallel."""
    jobs = list({(to_ccxt(s), tf): (to_ccxt(s), tf, a, b) for s, tf, a, b in jobs}.values())

    def run(job):
        s, tf, a, b = job
        e = _thread_exchange(ex, cfg.workers) if cfg.workers > 1 else ex
        try:
            return (s, tf), fetch_ohlcv(e, cfg, s, tf, a, b)
        except Exception as err:                         # noqa: BLE001
            print(f"skip {s} {tf}: {type(err).__name__}: {str(err)[:120]}")
            return (s, tf), None

    if cfg.workers <= 1 or len(jobs) <= 1:
        return dict(map(run, jobs))
    if not ex.markets:
        ex.load_markets()
    with ThreadPoolExecutor(cfg.workers) as pool:
        return dict(pool.map(run, jobs))


def top_universe(ex, cfg, coins_only=True):
    """today's top-N USDT-M perps by 24h quote volume (live scanning / static backtests)"""
    ex.load_markets()
    tk = ex.fetch_tickers()
    out = []
    for s, m in ex.markets.items():
        if not (m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")):
            continue
        info = m.get("info") or {}
        b = m.get("base")
        if b in STABLES or b in INDEX_PERPS or (coins_only and (
                b in EXCLUDE_BASES or info.get("contractType", "PERPETUAL") != "PERPETUAL"
                or info.get("underlyingType", "COIN") != "COIN")):
            continue
        qv = float((tk.get(s) or {}).get("quoteVolume") or 0)
        if qv >= cfg.min_quote_vol:
            out.append((s, qv))
    return [s for s, _ in sorted(out, key=lambda x: -x[1])[: cfg.universe_n]]


def align_htf(df_htf, htf, ltf_index, ltf):
    """Higher-timeframe bars become visible at their close; aligned to lower-timeframe bar closes."""
    x = df_htf.copy()
    x.index = x.index + pd.Timedelta(milliseconds=tf_ms(htf))
    return x.reindex(ltf_index + pd.Timedelta(milliseconds=tf_ms(ltf)), method="ffill").set_axis(ltf_index)


# ================= 4. DATA (backtests): data.binance.vision archive, incl. delisted symbols =================
VISION_WEB = "https://data.binance.vision"
VISION_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _http_get_default(url, params=None, timeout=60):
    import requests
    last = None
    for k in range(5):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = (r.status_code, r.content)
                time.sleep(2 ** k)
                continue
            return r.status_code, r.content
        except requests.RequestException:
            if k == 4:
                raise
            time.sleep(2 ** k)
    return last


HTTP_GET = _http_get_default          # (url, params) -> (status, bytes); replaced in offline tests


def _now() -> pd.Timestamp:            # replaced in offline tests
    return pd.Timestamp.now(tz="UTC")


def s3_list(prefix, delimiter="/"):
    """List data.binance.vision via its S3 bucket (V1 API, paginated). Returns (prefixes, keys)."""
    prefixes, keys, marker = [], [], ""
    while True:
        st, body = HTTP_GET(VISION_S3, {"prefix": prefix, "delimiter": delimiter, "marker": marker})
        if st != 200:
            raise IOError(f"S3 listing {prefix!r}: HTTP {st}")
        root = ET.fromstring(body)
        ns = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""
        q = (lambda t: f"{{{ns}}}{t}") if ns else (lambda t: t)
        pp = [e.findtext(q("Prefix")) for e in root.findall(q("CommonPrefixes"))]
        kk = [e.findtext(q("Key")) for e in root.findall(q("Contents"))]
        prefixes += pp
        keys += kk
        if (root.findtext(q("IsTruncated")) or "false").lower() != "true" or not (pp or kk):
            break
        nxt = root.findtext(q("NextMarker")) or max(pp + kk)
        if nxt == marker:
            break
        marker = nxt
    return prefixes, keys


def _is_num(x) -> bool:
    try:
        float(str(x).strip())
        return True
    except ValueError:
        return False


def _parse_klines(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
    if len(df) and not _is_num(df.iat[0, 0]):          # older USD-M files have no header row
        df = df.iloc[1:]
    if df.empty:
        return pd.DataFrame(columns=_COLS + ["quote_volume", "count"])
    df = df.iloc[:, :12]
    df.columns = KCOLS[: df.shape[1]]
    ot = pd.to_numeric(df["open_time"]).astype("int64")
    unit = "us" if ot.iloc[0] > 10**14 else "ms"
    idx = pd.to_datetime(ot.to_numpy(), unit=unit, utc=True).as_unit("ns")
    out = pd.DataFrame({c: pd.to_numeric(df[c]).to_numpy(dtype=float)
                        for c in _COLS + ["quote_volume", "count"]}, index=idx)
    return out


def _parse_funding(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
    cols = [0, 1, 2]                                         # calc_time, funding_interval_hours, last_funding_rate
    if len(df) and not _is_num(df.iat[0, 0]):
        names = [str(x).strip().lower() for x in df.iloc[0]]
        pick = lambda key, d: next((i for i, n in enumerate(names) if key in n), d)
        cols = [pick("time", 0), pick("interval", 1), pick("rate", 2)]
        df = df.iloc[1:]
    if df.empty:
        return pd.DataFrame(columns=["rate", "interval_h"])
    t = pd.to_numeric(df.iloc[:, cols[0]]).astype("int64")
    idx = pd.to_datetime(t.to_numpy(), unit="ms", utc=True).as_unit("ns").round("1min")
    return pd.DataFrame({"rate": pd.to_numeric(df.iloc[:, cols[2]]).to_numpy(dtype=float),
                         "interval_h": pd.to_numeric(df.iloc[:, cols[1]]).to_numpy(dtype=float)}, index=idx)


class Vision:
    """data.binance.vision downloader with a local cache. Delisted symbols are kept in the archive;
    their series are truncated after the last bar with trades (the archive carries flat rows after settlement)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = os.path.join(cfg.cache_dir, "vision")
        os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "k"), exist_ok=True)

    # ---- listings ----
    def symbols(self, refresh=False):
        path = os.path.join(self.root, "symbols.json")
        hit = _load_json(path)
        if hit and not refresh and time.time() - hit["t"] < 86_400:
            return hit["symbols"]
        pre, _ = s3_list("data/futures/um/monthly/klines/")
        syms = []
        for p in pre:
            s = p.rstrip("/").split("/")[-1]
            if not s.endswith("USDT") or "_" in s:          # delivery contracts carry an underscore
                continue
            b = s[:-4]
            if b in STABLES or b in INDEX_PERPS or b in EXCLUDE_BASES:
                continue
            syms.append(s)
        syms = sorted(set(syms))
        _save_json({"t": time.time(), "symbols": syms}, path)
        return syms

    def months_available(self, sym, kind="klines", tf="1d"):
        path = os.path.join(self.root, "k", f"{sym}_{kind}_{tf}.avail.json")
        hit = _load_json(path)
        if hit and time.time() - hit["t"] < 86_400:
            return set(hit["months"])
        prefix = (f"data/futures/um/monthly/klines/{sym}/{tf}/" if kind == "klines"
                  else f"data/futures/um/monthly/fundingRate/{sym}/")
        _, keys = s3_list(prefix)
        months = sorted({m.group(1) for k in keys if (m := re.search(r"-(\d{4}-\d{2})\.zip$", k))})
        _save_json({"t": time.time(), "months": months}, path)
        return set(months)

    # ---- files ----
    def _get_zip(self, path, cache=True):
        local = os.path.join(self.root, "raw", path)
        if os.path.exists(local):
            with open(local, "rb") as f:
                body = f.read()
        else:
            st, body = HTTP_GET(f"{VISION_WEB}/{path}")
            if st == 404:
                return None
            if st != 200:
                raise IOError(f"{path}: HTTP {st}")
            if cache:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                with open(local + ".tmp", "wb") as f:
                    f.write(body)
                os.replace(local + ".tmp", local)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            return z.read(z.namelist()[0])

    def _month_klines(self, sym, tf, month, monthly_ok):
        if monthly_ok:
            raw = self._get_zip(f"data/futures/um/monthly/klines/{sym}/{tf}/{sym}-{tf}-{month}.zip", cache=False)
            return _parse_klines(raw) if raw else None
        # month not (yet) published as a monthly file → daily files (published the next day)
        parts, d0 = [], pd.Timestamp(month + "-01", tz="UTC")
        today = _now().floor("D")
        for d in pd.date_range(d0, d0 + pd.offsets.MonthEnd(0), freq="D"):
            if d >= today:
                break
            ds = d.strftime("%Y-%m-%d")
            raw = self._get_zip(f"data/futures/um/daily/klines/{sym}/{tf}/{sym}-{tf}-{ds}.zip", cache=True)
            if raw:
                parts.append(_parse_klines(raw))
        return pd.concat(parts) if parts else None

    def _month_funding(self, sym, month):
        raw = self._get_zip(f"data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{month}.zip", cache=False)
        return _parse_funding(raw) if raw else None

    # ---- combined, cached series ----
    def _combined(self, name, months, fetch_month, finalize=None):
        """one cached file per (symbol, kind). Completed months are fetched once; the current and previous
        month (possibly incomplete) are re-fetched at most every 6 hours."""
        path = os.path.join(self.root, "k", name + "." + _FMT)
        meta_path = os.path.join(self.root, "k", name + ".meta.json")
        with _lock(path):
            df = _load(path)
            meta = _load_json(meta_path, {"done": [], "recent": {}}) if df is not None else {"done": [], "recent": {}}
            done, recent = set(meta["done"]), dict(meta["recent"])
            now = _now()
            cur, prev = mkey(now), mkey(now.floor("D").replace(day=1) - pd.Timedelta(days=1))
            fresh = lambda m: m in recent and time.time() - recent[m] < 6 * 3600
            want = [m for m in months if m not in done and not fresh(m)]
            if want:
                parts = [p for m in want if (p := fetch_month(m)) is not None and len(p)]
                if parts:
                    df = pd.concat(([df] if df is not None else []) + parts)
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                if df is not None and finalize is not None:
                    df = finalize(df)
                if df is not None:
                    _save(df, path)
                for m in want:
                    if m in (cur, prev):
                        recent[m] = time.time()
                    else:
                        done.add(m)
                        recent.pop(m, None)
                _save_json({"done": sorted(done), "recent": recent}, meta_path)
        if df is None or df.empty:
            return None
        keep = np.isin(df.index.strftime("%Y-%m"), list(months))
        return df.loc[keep]

    def klines(self, sym, tf, months):
        avail = self.months_available(sym, "klines", tf)
        now = _now()
        recent = {mkey(now), mkey(now.floor("D").replace(day=1) - pd.Timedelta(days=1))}
        alive = bool(avail) and max(avail) >= mkey(now - pd.Timedelta(days=75))
        # daily-file fallback only for symbols that are still trading (avoids 404 storms on delisted coins)
        months = sorted(m for m in months if m in avail or (alive and m in recent))

        def fm(m):
            return self._month_klines(sym, tf, m, m in avail)

        def trunc(df):                                       # cut carried-forward rows after delisting
            live = df.index[df["count"] > 0]
            return df.loc[: live.max()] if len(live) else df.iloc[:0]

        return self._combined(f"{sym}_{tf}", months, fm, trunc)

    def funding(self, sym, months):
        avail = self.months_available(sym, "funding")
        months = sorted(m for m in months if m in avail)
        if not months:
            return None
        return self._combined(f"{sym}_funding", months, lambda m: self._month_funding(sym, m))


def pit_universe(daily: dict, months, n, min_days=120):
    """{month_start: [top-n symbols]} ranked by median daily quote volume over the prior 30 days,
    listed ≥ min_days before the month, and trading during the last 7 days before it."""
    qv = pd.DataFrame({s: d["quote_volume"] for s, d in daily.items() if "quote_volume" in d})
    cnt = pd.DataFrame({s: d["count"] for s, d in daily.items() if "count" in d})
    first = {s: d.index.min() for s, d in daily.items() if len(d)}
    out = {}
    for m in months:
        win = qv.loc[(qv.index >= m - pd.Timedelta(days=30)) & (qv.index < m)]
        last7 = cnt.loc[(cnt.index >= m - pd.Timedelta(days=7)) & (cnt.index < m)]
        if win.empty:
            out[m] = []
            continue
        alive = (last7.fillna(0) > 0).any()
        old = pd.Series({s: (m - first[s]).days >= min_days for s in qv.columns})
        med = win.median()
        ok = med.index[(alive.reindex(med.index, fill_value=False)) & old.reindex(med.index, fill_value=False)
                       & med.notna()]
        out[m] = med[ok].sort_values(ascending=False, kind="stable").index[:n].tolist()
    return out


class Dataset:
    """Everything a backtest needs. universe=None → static (every symbol is always a member)."""

    def __init__(self, daily: dict, btc: pd.DataFrame, ltf_loader, universe=None, funding_loader=None,
                 lru=24):
        self.daily, self.btc, self.universe = daily, btc, universe
        self._ltf_loader, self._funding_loader = ltf_loader, funding_loader
        self._cache = OrderedDict()
        self._lru = lru
        self._months = sorted(universe) if universe else []

    @classmethod
    def from_frames(cls, btc, daily: dict, ltf: dict, universe=None):
        return cls(daily, btc, lambda s: ltf.get(s), universe)

    def members(self, m, n):
        if self.universe is None:
            return list(self.daily)
        return self.universe.get(m, [])[:n]

    def tradable(self, n, start, end):
        if self.universe is None:
            return [s for s in self.daily]
        out = []
        for m in self._months:
            if _utc(start).floor("D").replace(day=1) <= m < _utc(end):
                out += [s for s in self.universe[m][:n] if s not in out]
        return out

    def member_months(self, sym, n):
        """sorted array of month-start ns where sym is in the top-n (None = always)"""
        if self.universe is None:
            return None
        return np.array([m.value for m in self._months if sym in self.universe[m][:n]], dtype=np.int64)

    def member_mask_daily(self, closes: pd.DataFrame, n):
        if self.universe is None:
            return None
        mask = pd.DataFrame(False, index=closes.index, columns=closes.columns)
        months = self._months
        for i, m in enumerate(months):
            nxt = months[i + 1] if i + 1 < len(months) else m + pd.offsets.MonthBegin(1)
            rows = (closes.index >= m) & (closes.index < nxt)
            cols = [s for s in self.universe[m][:n] if s in mask.columns]
            if rows.any() and cols:
                mask.loc[rows, cols] = True
        return mask

    def ltf(self, sym):
        if sym in self._cache:
            self._cache.move_to_end(sym)
            return self._cache[sym]
        m = self._ltf_loader(sym)
        self._cache[sym] = m
        if len(self._cache) > self._lru:
            self._cache.popitem(last=False)
        return m

    def funding(self, sym):
        return self._funding_loader(sym) if self._funding_loader else None


def build_pit_dataset(cfg: Config, start, end, n_max=None, symbols_only=None, verbose=True):
    """Download (cached) daily data for every USDT-M perp ever listed, build the point-in-time universe,
    then 15m klines + funding for every symbol-month that can matter. Returns a Dataset."""
    n_max = max(n_max or cfg.universe_n, cfg.universe_n)
    vis = Vision(cfg)
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    syms = vis.symbols()
    if "BTCUSDT" not in syms:
        syms.append("BTCUSDT")
    d_months = [mkey(m) for m in month_starts(_utc(start) - pd.Timedelta(days=460), end)]
    say(f"[1/3] daily klines for {len(syms)} symbols ({d_months[0]} → {d_months[-1]}) …")

    def load_daily(s):
        try:
            return s, vis.klines(s, "1d", d_months)
        except Exception as e:                             # noqa: BLE001
            say(f"   skip {s}: {type(e).__name__}: {str(e)[:100]}")
            return s, None

    with ThreadPoolExecutor(cfg.vision_workers) as pool:
        daily = {s: d for s, d in pool.map(load_daily, syms) if d is not None and len(d)}
    btc = daily.get("BTCUSDT")
    if btc is None:
        raise RuntimeError("BTCUSDT daily data missing")
    months = month_starts(start, end)
    universe = pit_universe(daily, months, n_max, cfg.min_listing_days)
    say(f"      done in {time.time() - t0:.0f}s; universe: {len(months)} months, "
        f"{len({s for v in universe.values() for s in v})} distinct symbols")
    # which 15m months matter: member months, the month before (warm-up) and two after (open trades)
    need = defaultdict(set)
    now_m = mkey(_now())
    for m, members in universe.items():
        for s in members:
            for k in (-1, 0, 1, 2):
                mm = mkey(m + pd.DateOffset(months=k))
                if mm <= now_m:
                    need[s].add(mm)
    if symbols_only:
        keep = {plain(s) for s in symbols_only}
        need = {s: v for s, v in need.items() if s in keep}
    n_files = sum(len(v) for v in need.values())
    say(f"[2/3] 15m klines: {len(need)} symbols, {n_files} symbol-months (cached files are skipped) …")
    t1 = time.time()

    def load_ltf(item):
        s, mm = item
        try:
            vis.klines(s, cfg.ltf, sorted(mm))
            if cfg.funding_source == "real":
                vis.funding(s, sorted(mm))
        except Exception as e:                             # noqa: BLE001
            say(f"   skip {s} 15m: {type(e).__name__}: {str(e)[:100]}")

    with ThreadPoolExecutor(cfg.vision_workers) as pool:
        list(pool.map(load_ltf, need.items()))
    say(f"      done in {time.time() - t1:.0f}s")
    say("[3/3] dataset ready")
    need_sorted = {s: sorted(v) for s, v in need.items()}
    _save_json({mkey(m): v for m, v in universe.items()},
               os.path.join(vis.root, f"universe_top{n_max}_{mkey(months[0])}_{mkey(months[-1])}.json"))

    def ltf_loader(s):
        return vis.klines(s, cfg.ltf, need_sorted.get(s, [])) if s in need_sorted else None

    def funding_loader(s):
        if cfg.funding_source != "real" or s not in need_sorted:
            return None
        return vis.funding(s, need_sorted[s])

    return Dataset(daily, btc, ltf_loader, universe, funding_loader)


def build_static_dataset(ex, cfg: Config, symbols, start, end):
    """v2-style: today's symbols via ccxt (survivorship-biased; quick checks only)."""
    syms = [plain(s) for s in symbols]
    ltf_start = _utc(start) - pd.Timedelta(days=cfg.ltf_warmup_days)
    jobs = [(s, "1d", _utc(start) - pd.Timedelta(days=420), end) for s in ["BTCUSDT"] + syms]
    jobs += [(s, cfg.ltf, ltf_start, _utc(end) + pd.Timedelta(days=62)) for s in syms]
    got = fetch_many(ex, cfg, jobs)
    btc = got.get((BTC, "1d"))
    if btc is None or btc.empty:
        raise RuntimeError("BTC daily data missing")
    daily, ltf = {}, {}
    for s in syms:
        d, m = got.get((to_ccxt(s), "1d")), got.get((to_ccxt(s), cfg.ltf))
        if d is not None and m is not None and len(d) > cfg.base_min_bars + 30 and len(m) > cfg.pat_window + 30:
            daily[s], ltf[s] = d, m
    return Dataset.from_frames(btc, daily, ltf)


# ================= 5. INDICATORS =================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def pivots(df, left, right):
    """fractal swing points; the point at i is confirmed at i+right → use only i <= t-right"""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    w = left + right + 1
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    if n >= w:
        c = np.arange(left, n - right)
        ph[c] = np.where(h[c] == swv(h, w).max(1), h[c], np.nan)
        pl[c] = np.where(l[c] == swv(l, w).min(1), l[c], np.nan)
    return ph, pl


def squeeze_on(df, n=20, bb=2.0, kc=1.5):
    m = df.close.rolling(n).mean(); sd = df.close.rolling(n).std(); e = ema(df.close, n); a = atr(df, n)
    return ((m + bb * sd) < (e + kc * a)) & ((m - bb * sd) > (e - kc * a))


def _fit(x, y):
    x = x.astype(float); xm, ym = x.mean(), y.mean(); dx = x - xm
    b = (dx * (y - ym)).sum() / (dx * dx).sum()
    return b, ym - b * xm


def htf_bars(o, h, l, c, v, ts_ns, minutes):
    """Aggregate 15m arrays into higher-timeframe bars aligned to UTC boundaries.
    ends[k] = index of the last 15m bar of HTF bar k (the bar is complete at that bar's close)."""
    key = ts_ns // (minutes * 60 * 10**9)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)] - 1
    return dict(o=o[starts], h=np.maximum.reduceat(h, starts), l=np.minimum.reduceat(l, starts), c=c[ends],
                v=np.add.reduceat(v, starts) if v is not None else None, starts=starts, ends=ends)


def ewm_atr(H, L, C, n=14):
    pc = np.r_[C[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def htf_to_ltf(values, ends, n):
    """value of the latest COMPLETE higher-timeframe bar at each 15m index (NaN before the first)"""
    last = np.searchsorted(ends, np.arange(n), side="right") - 1
    out = np.full(n, np.nan)
    ok = last >= 0
    out[ok] = values[last[ok]]
    return out


def ltf_arrays(m, cfg):
    """15m precomputation: pivots, volume mean, box, ATR(15m) and ATR14 of completed 1h bars"""
    ph, pl = pivots(m, cfg.ltf_pivot, cfg.ltf_pivot)
    h, l, c, v = m.high.to_numpy(), m.low.to_numpy(), m.close.to_numpy(), m.volume.to_numpy()
    o = m.open.to_numpy()
    B = cfg.box_bars
    tsv = m.index.as_unit("ns").asi8
    hb = htf_bars(o, h, l, c, None, tsv, 60)
    a1h = htf_to_ltf(ewm_atr(hb["h"], hb["l"], hb["c"], 14), hb["ends"], len(m))
    return dict(h=h, l=l, c=c, v=v, a=atr(m).to_numpy(), ph=ph, pl=pl, ts=tsv,
                hp=np.flatnonzero(np.isfinite(ph)), lp=np.flatnonzero(np.isfinite(pl)),
                vmean=pd.Series(v).rolling(cfg.vol_lookback).mean().shift(1).to_numpy(),
                box_hh=pd.Series(h).rolling(B).max().shift(1).to_numpy(),
                box_ll=pd.Series(l).rolling(B).min().shift(1).to_numpy(),
                atr1h=a1h, sq=squeeze_on(m).to_numpy() if cfg.require_squeeze else None)


# ================= 6. MARKET SEASON =================
def breadth_table(closes, cfg, mask=None):
    v = closes.notna()
    if mask is not None:
        v = v & mask.reindex(index=closes.index, columns=closes.columns, fill_value=False).astype(bool)
    n = v.sum(axis=1).replace(0, np.nan)
    above = closes > closes.ewm(span=cfg.ema_fast, adjust=False).mean()
    newhigh = closes >= closes.rolling(30, min_periods=30).max()
    return pd.DataFrame({"breadth": (above & v).sum(axis=1) / n, "newhigh": (newhigh & v).sum(axis=1) / n})


def classify_season(btc, cfg, br=None, k=1):
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


# ================= 7. STAGE CLASSIFIER (daily state machine, no look-ahead) =================
def classify_stages(d, cfg):
    """stage/episode/base per day, plus cand_* = stage-1 breakout conditions for day i known at its OPEN
    (used by the 4h trigger: all inputs come from bars < i)."""
    n = len(d)
    h, l, c = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    e50 = ema(d.close, 50).to_numpy()
    B = cfg.base_min_bars
    hh_prev = pd.Series(h).rolling(B).max().shift(1).to_numpy()
    ll_prev = pd.Series(l).rolling(B).min().shift(1).to_numpy()
    out = np.zeros(n, np.int8); epi = np.zeros(n, np.int32)
    bh = np.full(n, np.nan); bl = np.full(n, np.nan)
    cand = np.zeros(n, bool)
    st = ep = 0
    base_hi = base_lo = imp_hi = leg0 = s2_hi = div_lo = top = np.nan
    bo = hi_i = 0
    low_after = math.inf
    for i in range(n):
        if st in (0, 6) and i > B:
            hh, ll = hh_prev[i], ll_prev[i]
            j0 = max(0, i - cfg.decline_lookback)
            cand[i] = (i - B > j0 and (hh - ll) / ll <= cfg.base_max_width
                       and ll <= h[j0:i - B].max() * (1 - cfg.prior_decline))
            if cand[i] and c[i] > hh:
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
    return pd.DataFrame({"stage": out, "episode": epi, "base_hi": bh, "base_lo": bl,
                         "cand": cand, "cand_hh": hh_prev, "cand_ll": ll_prev}, index=d.index)


# ================= 8. SECTOR / CLUSTER HEAT (optional, needs scipy) =================
def heat_table(ds: Dataset, n, cfg, start, end):
    """daily bool table: coin is 'hot' (its correlation cluster is hot AND the coin itself is hot).
    Clusters are recomputed monthly from the prior 90 days of BTC-residual returns (point-in-time)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    syms = [s for s in ds.tradable(n, start, end) if s in ds.daily]
    closes = pd.DataFrame({s: ds.daily[s].close for s in syms})
    qv = pd.DataFrame({s: ds.daily[s]["quote_volume"] if "quote_volume" in ds.daily[s] else
                       ds.daily[s].volume * ds.daily[s].close for s in syms})
    btc = ds.btc.close.reindex(closes.index)
    r = np.log(closes).diff()
    rb = np.log(btc).diff()
    mean_rb = rb.rolling(60).mean()
    cov = r.mul(rb, axis=0).rolling(60).mean() - r.rolling(60).mean().mul(mean_rb, axis=0)
    beta = cov.div(rb.rolling(60).var(ddof=0), axis=0)
    res = r - beta.shift(1).mul(rb, axis=0)
    above = closes > closes.ewm(span=20, adjust=False).mean()
    hi20 = closes >= closes.rolling(20).max()
    r14 = closes.pct_change(14, fill_method=None).sub(btc.pct_change(14, fill_method=None), axis=0)
    coin_hot = ((qv.rolling(7).mean() >= 1.5 * qv.rolling(60).mean())
                & ((closes >= closes.rolling(30).max()).astype(float).rolling(5).max() > 0))
    hot = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    months = [m for m in (sorted(ds.universe) if ds.universe else month_starts(start, end))]
    for i, m in enumerate(months):
        nxt = months[i + 1] if i + 1 < len(months) else m + pd.offsets.MonthBegin(1)
        mem = [s for s in ds.members(m, n) if s in res.columns]
        win = res.loc[(res.index >= m - pd.Timedelta(days=90)) & (res.index < m), mem]
        win = win.dropna(axis=1, thresh=60)
        if win.shape[1] < 6:
            continue
        rho = win.corr().fillna(0).clip(-1, 1).to_numpy()
        dist = np.sqrt(np.maximum(2 * (1 - rho), 0))
        np.fill_diagonal(dist, 0)
        lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                       t=cfg.heat_n_clusters, criterion="maxclust")
        rows = (closes.index >= m) & (closes.index < nxt)
        for cl in np.unique(lab):
            grp = list(win.columns[lab == cl])
            if len(grp) < 3:
                continue
            ok = ((above.loc[rows, grp].mean(axis=1) >= 0.6) & (hi20.loc[rows, grp].mean(axis=1) >= 0.25)
                  & (r14.loc[rows, grp].median(axis=1) > 0))
            hot.loc[rows, grp] = coin_hot.loc[rows, grp].to_numpy() & ok.to_numpy()[:, None]
    return hot


# ================= 9. ROOM TO RESISTANCE =================
def daily_pack(d, cfg):
    ph = pivots(d, cfg.pivot_left, cfg.pivot_right)[0]
    idx = np.flatnonzero(np.isfinite(ph))
    tp = ((d.high + d.low + d.close) / 3).to_numpy()
    return dict(ph_idx=idx, ph_val=ph[idx], ath=d.high.cummax().to_numpy(), tp=tp, qv=tp * d.volume.to_numpy())


def round_levels(p):
    e = math.floor(math.log10(p))
    return [m * 10 ** e for m in (1, 2, 5, 10, 20)]


def next_round(p):
    e = math.floor(math.log10(p))
    for k in (e, e + 1):
        for m in (1, 2, 5):
            x = m * 10 ** k
            if x > p * 1.005:
                return x
    return 10 ** (e + 2)


def hvn_levels(tp, w, bins):
    if len(tp) < 20:
        return []
    hist, edges = np.histogram(tp, bins=bins, weights=w)
    ctr = (edges[:-1] + edges[1:]) / 2
    mid = hist[1:-1]
    return ctr[1:-1][(mid > hist[:-2]) & (mid >= hist[2:]) & (mid > hist.mean())].tolist()


def resistance_levels(D, di, price, cfg):
    """(clustered levels above price, real levels above price). Real = swing highs, ATH, volume nodes."""
    k = np.searchsorted(D["ph_idx"], di - cfg.pivot_right, side="right")
    real = D["ph_val"][:k].tolist()
    real.append(float(D["ath"][di]))
    if cfg.use_volume_profile:
        j = max(0, di - 180)
        real += hvn_levels(D["tp"][j:di + 1], D["qv"][j:di + 1], cfg.vp_bins)
    rounds = []
    if cfg.use_round_numbers:
        rounds = round_levels(price) if cfg.round_mode == "grid" else [next_round(price)]
    lv = real + rounds
    out = []
    for x in sorted(x for x in lv if x > price * 1.005):
        if not out or x > out[-1] * (1 + cfg.res_cluster_pct):
            out.append(x)
    real_above = sorted(x for x in real if x > price * 1.005)
    return out, real_above


def build_plan(sym, D, di, stage, ep, rs, bhi, blo, sig, cfg, atr1h=np.nan):
    entry, stop = sig["entry"], sig["stop"]
    if cfg.min_stop_atr1h > 0 or cfg.min_stop_pct > 0:      # minimum stop: outside 1h noise
        if cfg.min_stop_atr1h > 0 and not np.isfinite(atr1h):
            return None
        d_min = max(cfg.min_stop_atr1h * atr1h if cfg.min_stop_atr1h > 0 else 0.0, cfg.min_stop_pct * entry)
        stop = min(stop, entry - d_min)
    if di < 0 or stop <= 0 or stop >= entry or (entry - stop) / entry > cfg.max_stop_pct:
        return None
    lv, real = resistance_levels(D, di, entry, cfg)
    risk = entry - stop
    fallback = entry + max(bhi - blo if np.isfinite(bhi) else 0, cfg.min_rr * 1.5 * risk)
    f = 1 - cfg.tp_front_run
    tp1 = lv[0] * f if lv else fallback
    tp2 = lv[1] * f if len(lv) > 1 else tp1 + (tp1 - entry)
    room, rr = (tp1 - entry) / entry, (tp1 - entry) / risk
    blue = cfg.blue_sky_pass and not real
    if not blue and (room < cfg.min_room_pct or rr < cfg.min_rr):
        return None
    return dict(sym=sym, stage=int(stage), ep=int(ep), rs=float(rs), entry=entry, stop=stop, tp1=tp1, tp2=tp2,
                tp_real=real[0] * f if real else None, room=room, rr=rr, blue=blue, kind=sig["kind"],
                status=sig.get("status", "已突破"))


# ================= 10. 15m TRIGGER: converging triangle/wedge or narrow box breakout =================
def find_pattern(L, t, cfg, pending=False):
    l, c, v = L["l"], L["c"], L["v"]
    A = L["a"][t]
    if t < cfg.pat_window or not A > 0:
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
                        l[lx[-1]:t + 1].min(), bl * t + al, x0)
    if cand is None and cfg.box_fallback:
        hh, ll = L["box_hh"][t], L["box_ll"][t]
        if hh - ll <= cfg.box_max_atr * A and (not cfg.require_squeeze or L["sq"][t - 1]):
            cand = ("窄幅箱体", hh, hh, hh, ll, ll, t - cfg.box_bars)
    if cand is None:
        return None
    kind, line, prev, nxt, stop_ref, lower, i_start = cand
    buf = cfg.breakout_buf_atr * A
    stop = float(stop_ref - cfg.stop_buf_atr * A)
    if c[t] > line + buf and c[t - 1] <= prev + buf and v[t] >= cfg.vol_mult * v[t - cfg.vol_lookback:t].mean():
        return dict(kind=kind, status="已突破", entry=float(c[t]), stop=stop, i_start=int(i_start))
    if pending and lower - 0.5 * A <= c[t] <= line + buf:
        return dict(kind=kind, status="待突破(挂stop单)", entry=float(nxt + buf), stop=stop, i_start=int(i_start))
    return None


# ================= 11. SIZING, ACCOUNT, COOLDOWN =================
def liq_price(entry, lev, mmr):
    """isolated long liquidation price (Binance formula without the maintenance-amount term)"""
    return entry * (1 - 1 / lev) / (1 - mmr)


def lev_cap(sym, cfg):
    return cfg.lev_btc if to_ccxt(sym) == BTC else cfg.lev_alt


def size_tranche(sym, entry, stop, margin, cfg):
    """v2 sizing: one full tranche of margin at the leverage cap (kept for the control variant)"""
    sp = (entry - stop) / entry
    lev = lev_cap(sym, cfg)
    while lev > 1 and sp * lev > cfg.max_loss_frac_of_tranche:
        lev -= 1
    while lev > 1 and (1 - liq_price(1.0, lev, cfg.mmr)) * cfg.liq_buffer < sp:
        lev -= 1
    scale = min(1.0, max(0.3, cfg.ref_stop_pct / sp)) if cfg.size_scale_by_stop else 1.0
    mg = margin * scale
    notional = mg * lev
    return dict(lev=lev, margin=mg, notional=notional, qty=notional / entry,
                liq=liq_price(entry, lev, cfg.mmr), risk=notional * sp)


def size_risk(sym, entry, stop, risk_usd, margin_cap, cfg):
    """v2.1 sizing: risk_usd lost at the stop (incl. round-trip costs). 2x by default; leverage only rises
    if the margin would exceed one tranche; the stop must sit well before the liquidation price."""
    sp = (entry - stop) / entry
    if sp <= 0 or risk_usd <= 0 or margin_cap <= 0:
        return None
    cap = lev_cap(sym, cfg)
    notional = risk_usd / (sp + 2 * (cfg.taker_fee + cfg.slippage))
    lev = float(min(cfg.lev_default, cap))
    if notional / lev > margin_cap:
        lev = float(min(cap, math.ceil(notional / margin_cap)))
        notional = min(notional, margin_cap * lev)
    while lev > 1 and (entry - stop) > cfg.liq_buffer * (entry - liq_price(entry, lev, cfg.mmr)):
        lev -= 1
        notional = min(notional, margin_cap * lev)
    if lev > 1 and (entry - stop) > cfg.liq_buffer * (entry - liq_price(entry, lev, cfg.mmr)):
        return None
    return dict(lev=lev, margin=notional / lev, notional=notional, qty=notional / entry,
                liq=liq_price(entry, lev, cfg.mmr) if lev > 1 else 0.0, risk=notional * sp)


class Account:
    """3-tranche bookkeeping: losses are refilled from the reserve, profits above one tranche are swept to a
    vault. sizing=risk: risk per trade = risk_pct × (active + reserve), never the vault; halt when trading
    capital < halt_frac × start. sizing=tranche: v2 rules (halt when all three tranches are gone)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = cfg.total_capital
        self.unit = cfg.total_capital / cfg.n_tranches
        self.active, self.reserve = self.unit, cfg.total_capital - self.unit
        self.vault = self.refills = 0.0
        self.loss_streak = 0
        self._dead = False

    @property
    def halted(self):
        if self.cfg.sizing == "tranche":
            return self._dead
        return self.risk_base() < self.cfg.halt_frac * self.base

    def margin(self):
        return 0.0 if self.halted else min(self.active, self.unit)

    def risk_base(self):
        return self.active + self.reserve

    def equity(self):
        return self.active + self.reserve + self.vault

    def risk_usd(self, mult=1.0):
        k = (self.cfg.port_streak_risk_mult
             if self.cfg.port_loss_streak > 0 and self.loss_streak >= self.cfg.port_loss_streak else 1.0)
        return self.cfg.risk_pct * self.risk_base() * k * mult

    def settle(self, pnl):
        self.active += pnl
        self.loss_streak = self.loss_streak + 1 if pnl < 0 else 0
        if self.active > self.unit:
            self.vault += self.active - self.unit
            self.active = self.unit
        else:
            take = min(self.unit - self.active, self.reserve)
            self.reserve -= take
            self.active += take
            self.refills += take
        if self.reserve <= 0 and self.active < 0.05 * self.unit:
            self._dead = True


class Cooldown:
    """after a losing exit on a coin: no new entry for cooldown_bars, and the next entry's structure must
    start after the loss; at most max_losses_per_episode losses per (coin, stage episode)"""

    def __init__(self, cfg: Config, bar_ns=15 * 60 * 10**9):
        self.cfg, self.bar_ns = cfg, bar_ns
        self.last_loss = {}
        self.ep_losses = defaultdict(int)

    def allowed(self, sym, ep, sig_ts, start_ts):
        if self.cfg.max_losses_per_episode > 0 and self.ep_losses[(sym, ep)] >= self.cfg.max_losses_per_episode:
            return False
        if self.cfg.cooldown_bars <= 0:
            return True
        ll = self.last_loss.get(sym)
        if ll is None:
            return True
        return sig_ts >= ll + self.cfg.cooldown_bars * self.bar_ns and start_ts > ll

    def record(self, tr):
        if tr["pnl"] < 0:
            self.last_loss[tr["sym"]] = tr["exit_ts"]
            self.ep_losses[(tr["sym"], tr["ep"])] += 1


# ================= 12. EXITS / TRADE SIMULATION =================
class ExecData:
    """15m execution arrays for one symbol + lazily computed exit context"""

    def __init__(self, m: pd.DataFrame, cfg: Config, funding: pd.DataFrame | None = None):
        self.o, self.h, self.l, self.c = (m[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        self.ts = m.index.as_unit("ns").asi8
        self.fund = np.zeros(len(m))
        if cfg.funding_source == "real" and funding is not None and len(funding):
            ft = funding.index.as_unit("ns").asi8
            j = np.searchsorted(self.ts, ft, side="right") - 1
            ok = (j >= 0) & (j < len(self.ts)) & (self.ts[np.clip(j, 0, None)] + 15 * 60 * 10**9 > ft)
            np.add.at(self.fund, j[ok], funding["rate"].to_numpy()[ok])
        else:                                               # flat rate at settlement timestamps
            self.fund[self.ts % (cfg.funding_interval_h * 3_600 * 10**9) == 0] = cfg.funding_rate
        self._ctx = {}

    def ctx(self, cfg):
        key = (cfg.trail_tf_min, cfg.trail_atr_n, cfg.donchian_days)
        if key not in self._ctx:
            n = len(self.c)
            tb = htf_bars(self.o, self.h, self.l, self.c, None, self.ts, cfg.trail_tf_min)
            trail_close = np.zeros(n, bool)
            trail_close[tb["ends"]] = True
            trail_atr = htf_to_ltf(ewm_atr(tb["h"], tb["l"], tb["c"], cfg.trail_atr_n), tb["ends"], n)
            db = htf_bars(self.o, self.h, self.l, self.c, None, self.ts, 1440)
            d_close = np.zeros(n, bool)
            d_close[db["ends"]] = True
            don = pd.Series(db["l"]).rolling(cfg.donchian_days, min_periods=cfg.donchian_days).min().to_numpy()
            self._ctx[key] = dict(trail_close=trail_close, trail_atr=trail_atr, d_close=d_close,
                                  don_low=htf_to_ltf(don, db["ends"], n))
        return self._ctx[key]


def simulate(X: ExecData, plan, acct: Account, cfg: Config):
    o, h, l, c, fund, tsv = X.o, X.h, X.l, X.c, X.fund, X.ts
    i_sig = int(np.searchsorted(tsv, plan["ts"]))
    if i_sig >= len(tsv) or tsv[i_sig] != plan["ts"]:
        return None
    i0 = i_sig + 1
    if i0 >= len(o) or tsv[i0] - tsv[i_sig] > DAY_NS:
        return None
    fee, slip, mode = cfg.taker_fee, cfg.slippage, cfg.exit_mode
    entry = o[i0] * (1 + slip)
    stop0 = plan["stop"]
    if entry <= stop0 or (mode == "fixed" and entry >= plan["tp1"]):
        return None
    if cfg.sizing == "tranche":
        sz = size_tranche(plan["sym"], entry, stop0, acct.margin(), cfg)
    else:
        sz = size_risk(plan["sym"], entry, stop0, acct.risk_usd(plan.get("risk_mult", 1.0)), acct.margin(), cfg)
    if sz is None or sz["qty"] <= 0:
        return None
    qty = rem = sz["qty"]
    R = entry - stop0
    pnl = -entry * qty * fee
    stop, hh, took, last = stop0, entry, False, entry
    tp1, tp2 = plan["tp1"], plan["tp2"]
    if mode == "partial":
        tp_part = plan["tp_real"] if plan.get("tp_real") and plan["tp_real"] > entry else entry + cfg.partial_R * R
    ctx = X.ctx(cfg) if mode in ("chandelier", "partial", "donchian") else None
    end = min(len(o), i0 + (cfg.time_stop_bars if mode == "fixed" else cfg.max_hold_bars))
    reason = None
    i = i0
    for i in range(i0, end):
        if i > i0 and tsv[i] - tsv[i - 1] > DAY_NS:      # data gap (symbol left the dataset): exit before it
            i -= 1
            reason = "gap"
            break
        if i > i0 and fund[i] != 0:                      # funding paid at settlement (positive: longs pay)
            pnl -= rem * o[i] * fund[i]
        if l[i] <= stop:                                 # stop first when both are inside one bar
            last = min(o[i], stop) * (1 - slip)
            pnl += rem * (last - entry) - rem * last * fee
            rem = 0.0
            reason = ("be-stop" if took else "stop") if mode == "fixed" else ("trail" if stop > stop0 else "stop")
            break
        if mode == "fixed":                              # v2 exits: 50% TP1 → breakeven → TP2, 10-day time stop
            if not took and h[i] >= tp1:
                q = qty * cfg.tp1_frac
                last = max(o[i], tp1) * (1 - slip)
                pnl += q * (last - entry) - q * last * fee
                rem -= q
                took = True
                if cfg.be_after_tp1:
                    stop = max(stop, entry)
                if rem <= 1e-12:
                    reason = "tp1"
                    break
            if took and h[i] >= tp2:
                last = max(o[i], tp2) * (1 - slip)
                pnl += rem * (last - entry) - rem * last * fee
                rem = 0.0
                reason = "tp2"
                break
            hh = max(hh, h[i])
            continue
        if mode == "partial" and not took and h[i] >= tp_part:
            q = qty * cfg.partial_frac
            px = max(o[i], tp_part) * (1 - slip)
            pnl += q * (px - entry) - q * px * fee
            rem -= q
            took = True
        hh = max(hh, h[i])
        if hh >= entry + cfg.trail_activate_R * R:       # trail only after +1R; no breakeven move
            if mode in ("chandelier", "partial"):
                if ctx["trail_close"][i] and np.isfinite(ctx["trail_atr"][i]):
                    stop = max(stop, hh - cfg.trail_mult * ctx["trail_atr"][i])
            elif ctx["d_close"][i] and np.isfinite(ctx["don_low"][i]):
                stop = max(stop, ctx["don_low"][i])
        if i - i0 + 1 >= cfg.stale_bars and hh < entry + cfg.stale_R * R:
            reason = "stale"
            break
    if reason is None:                                   # ran out of bars without an exit signal
        reason = "end" if end == len(o) else ("time" if mode == "fixed" else "maxhold")
    if rem > 0:
        last = c[i] * (1 - slip)
        pnl += rem * (last - entry) - rem * last * fee
    capv = -(sz["margin"] + entry * qty * fee)           # isolated: can't lose more than the margin
    if pnl < capv:
        pnl, reason = capv, "liq"
    acct.settle(pnl)
    risk_usd = qty * R
    return dict(sym=plan["sym"], stage=plan["stage"], ep=plan["ep"], kind=plan["kind"],
                entry_time=pd.Timestamp(tsv[i0], tz="UTC"), exit_time=pd.Timestamp(tsv[i], tz="UTC"),
                exit_ts=int(tsv[i]), entry=entry, stop=stop0, tp1=tp1, exit=last, lev=sz["lev"],
                margin=sz["margin"], notional=sz["notional"], pnl=pnl, R=pnl / risk_usd if risk_usd else 0.0,
                mfe_R=(hh - entry) / R, why=reason, equity=acct.equity(), reserve=acct.reserve,
                hold_d=(tsv[i] - tsv[i0]) / DAY_NS)


# ================= 13. SIGNALS → BACKTEST =================
def season_for(ds: Dataset, cfg: Config, n):
    cache = ds.__dict__.setdefault("_season_cache", {})
    key = (n, cfg.ema_fast, cfg.ema_slow, cfg.breadth_summer, cfg.breadth_winter, cfg.newhigh_summer,
           cfg.btc_autumn_dd)
    if key not in cache:
        syms = list(ds.daily) if ds.universe is None else sorted({s for m in ds.universe for s in ds.members(m, n)
                                                                  if s in ds.daily})
        closes = pd.DataFrame({s: ds.daily[s].close for s in syms})
        mask = ds.member_mask_daily(closes, n)
        cache[key] = classify_season(ds.btc, cfg, breadth_table(closes, cfg, mask))
    return cache[key]


def _member_on_bars(ds: Dataset, sym, n, tsv):
    mm = ds.member_months(sym, n)
    if mm is None:
        return np.ones(len(tsv), bool)
    if len(mm) == 0:
        return np.zeros(len(tsv), bool)
    months = np.array([m.value for m in ds._months], dtype=np.int64)
    k = np.searchsorted(months, tsv, side="right") - 1
    ok = k >= 0
    out = np.zeros(len(tsv), bool)
    out[ok] = np.isin(months[k[ok]], mm)
    return out


def _stage1_h4(m: pd.DataFrame, stg: pd.DataFrame, vis_stage, cfg):
    """early stage 1: first 4h close above the prior 20-day high (with a volume surge) on a day whose
    open-time conditions allow a stage-1 breakout. Returns bool per 15m bar + that day's base hi/lo."""
    n = len(m)
    tsv = m.index.as_unit("ns").asi8
    b4 = htf_bars(m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), m.close.to_numpy(),
                  m.volume.to_numpy(), tsv, 240)
    vmean = pd.Series(b4["v"]).rolling(30).mean().shift(1).to_numpy()
    day_of_bar = (tsv // DAY_NS) * DAY_NS
    d_idx = stg.index.as_unit("ns").asi8
    pos = np.searchsorted(d_idx, day_of_bar)
    valid = (pos < len(d_idx)) & (d_idx[np.clip(pos, 0, len(d_idx) - 1)] == day_of_bar)
    cand = np.zeros(n, bool)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    p = np.clip(pos, 0, len(d_idx) - 1)
    cand[valid] = stg["cand"].to_numpy()[p[valid]]
    hh[valid] = stg["cand_hh"].to_numpy()[p[valid]]
    ll[valid] = stg["cand_ll"].to_numpy()[p[valid]]
    early = np.zeros(n, bool)
    for k in range(len(b4["ends"])):
        e = b4["ends"][k]
        if cand[e] and vis_stage[e] in (0, 6) and b4["c"][k] > hh[e] and b4["v"][k] >= cfg.stage1_h4_vol_mult * vmean[k]:
            same_day = day_of_bar == day_of_bar[e]
            early[e:][same_day[e:]] = True
    return early, hh, ll


def build_events(ds: Dataset, cfg: Config, start, end, symbols=None, verbose=False):
    """All candidate entries (independent of account state), sorted by signal time."""
    n = cfg.universe_n
    season = season_for(ds, cfg, n)
    ok_seasons = ["夏"] + (["春"] if cfg.allow_late_spring else [])
    heat = heat_table(ds, n, cfg, start, end) if cfg.heat_mode == "expand" else None
    btc_r = ds.btc.close / ds.btc.close.shift(30)
    t_lo, t_hi = _utc(start).value, _utc(end).value
    syms = ds.tradable(n, start, end)
    if symbols:
        keep = {plain(s) for s in symbols}
        syms = [s for s in syms if s in keep]
    events = []
    for s in syms:
        d = ds.daily.get(s)
        m = ds.ltf(s)
        if d is None or m is None or len(d) < cfg.base_min_bars + 30 or len(m) < cfg.pat_window + 30:
            continue
        m = m.loc[(m.index >= _utc(start) - pd.Timedelta(days=cfg.ltf_warmup_days)) & (m.index < _utc(end))]
        if len(m) < cfg.pat_window + 30:
            continue
        stg = classify_stages(d, cfg)
        daily = stg[["stage", "episode", "base_hi", "base_lo"]].assign(
            rs=(d.close / d.close.shift(30)) / btc_r.reindex(d.index) - 1,
            season=season.reindex(d.index).ffill(), di=np.arange(len(d)))
        if heat is not None and s in heat.columns:
            daily["hot"] = heat[s].reindex(d.index, fill_value=False).astype(float)
        al = align_htf(daily, "1d", m.index, cfg.ltf)
        stage = al.stage.fillna(0).astype(int).to_numpy()
        ep = al.episode.fillna(0).astype(int).to_numpy()
        di = al.di.fillna(-1).astype(int).to_numpy()
        rs = al.rs.to_numpy(dtype=float)
        bhi, blo = al.base_hi.to_numpy(dtype=float), al.base_lo.to_numpy(dtype=float)
        seas = al.season.to_numpy()
        risk_mult = np.where(np.isin(seas, ok_seasons), 1.0, 0.0)
        if heat is not None and "hot" in al:
            soft = np.isin(seas, ["春", "秋"]) & (al.hot.fillna(0).to_numpy() > 0)
            risk_mult = np.where((risk_mult == 0) & soft, cfg.heat_risk_mult, risk_mult)
        if cfg.stage1_mode == "h4":
            early, ehh, ell = _stage1_h4(m, stg, stage, cfg)
            stage = np.where(early, 1, stage)
            ep = np.where(early, ep + 1, ep)
            bhi = np.where(early, ehh, bhi)
            blo = np.where(early, ell, blo)
        L = ltf_arrays(m, cfg)
        tsv = L["ts"]
        nb = len(m)
        tt = np.arange(nb)
        member = _member_on_bars(ds, s, n, tsv)
        with np.errstate(invalid="ignore"):
            mask = ((risk_mult > 0) & np.isin(stage, list(PRIORITY)) & member
                    & ((rs > 0) if cfg.require_rs else True)
                    & (tt >= cfg.pat_window) & (tt + 1 < nb) & (tsv >= t_lo) & (tsv < t_hi)
                    & (L["v"] >= cfg.vol_mult * L["vmean"] * (1 - 1e-9)))
        if not mask.any():
            continue
        D = daily_pack(d, cfg)
        for t in np.flatnonzero(mask):
            sig = find_pattern(L, t, cfg)
            if sig is None:
                continue
            plan = build_plan(s, D, di[t], stage[t], ep[t], rs[t], bhi[t], blo[t], sig, cfg, atr1h=L["atr1h"][t])
            if plan:
                plan.update(ts=int(tsv[t]), start_ts=int(tsv[max(sig["i_start"], 0)]),
                            key=(s, int(ep[t]), int(stage[t])), risk_mult=float(risk_mult[t]))
                events.append(plan)
        if verbose:
            print(f"   {s:<14} {sum(1 for e in events if e['sym'] == s):4d} signals")
    events.sort(key=lambda p: p["ts"])
    return events


def run_backtest(ds: Dataset, cfg: Config, start, end, symbols=None, events=None):
    """Replay signals: one position at a time; priority stage 1 > 4 > 2, then relative strength."""
    if events is None:
        events = build_events(ds, cfg, start, end, symbols)
    acct = Account(cfg)
    cd = Cooldown(cfg, bar_ns=tf_ms(cfg.ltf) * 10**6)
    exec_cache = OrderedDict()
    trades, attempts, busy = [], {}, -1
    for ts, grp in itertools.groupby(events, key=lambda p: p["ts"]):
        if acct.halted:
            break
        if ts <= busy:
            continue
        cands = [p for p in grp
                 if (cfg.max_attempts <= 0 or attempts.get(p["key"], 0) < cfg.max_attempts)
                 and cd.allowed(p["sym"], p["ep"], ts, p["start_ts"])]
        if not cands:
            continue
        plan = min(cands, key=lambda x: (PRIORITY[x["stage"]], -x["rs"]))
        attempts[plan["key"]] = attempts.get(plan["key"], 0) + 1
        s = plan["sym"]
        if s not in exec_cache:
            exec_cache[s] = ExecData(ds.ltf(s), cfg, ds.funding(s))
            if len(exec_cache) > 16:
                exec_cache.popitem(last=False)
        tr = simulate(exec_cache[s], plan, acct, cfg)
        if tr:
            trades.append(tr)
            busy = tr["exit_ts"]
            cd.record(tr)
    return trades, acct


def metrics(trades, acct: Account, cfg: Config):
    if not trades:
        return {"trades": 0}
    df = pd.DataFrame(trades)
    pn = df.pnl
    eq = pd.concat([pd.Series([cfg.total_capital]), cfg.total_capital + pn.cumsum()], ignore_index=True)
    gl = -pn[pn < 0].sum()
    R = df.R.sort_values()
    return {"trades": len(df), "win_rate": round((pn > 0).mean(), 3), "avg_R": round(df.R.mean(), 3),
            "PF": round(pn[pn > 0].sum() / gl, 2) if gl > 0 else float("inf"),
            "sum_R": round(df.R.sum(), 2), "sum_R_ex_top2": round(R.iloc[:-2].sum(), 2) if len(R) > 2 else 0.0,
            "max_DD": round((eq / eq.cummax() - 1).min(), 4),
            "return": round(acct.equity() / cfg.total_capital - 1, 4),
            "final_equity": round(acct.equity(), 2), "vault": round(acct.vault, 2),
            "avg_hold_d": round(df.hold_d.mean(), 1), "halted": acct.halted}


SHORT_KIND = {"收敛三角/楔形": "tri", "窄幅箱体": "box"}


def print_trades(trades):
    """compact list that fits an iPad terminal"""
    print(f"{'entry (UTC)':<16} {'coin':<10} st kind {'R':>6} {'P&L':>7} {'exit':<7} {'days':>5}")
    for t in trades:
        print(f"{t['entry_time'].strftime('%Y-%m-%d %H:%M'):<16} {base_of(t['sym'])[:10]:<10} "
              f"{t['stage']:>2} {SHORT_KIND.get(t['kind'], t['kind'])[:4]:<4} {t['R']:+6.2f} {t['pnl']:+7.0f} "
              f"{t['why']:<7} {t['hold_d']:5.1f}")


def print_metrics(mt):
    for k, v in mt.items():
        print(f"  {k:<14} {v}")


# ================= 14. PRE-REGISTERED EXPERIMENT =================
# FROZEN before looking at results. Do not edit or add variants after a run — every variant ever run counts
# toward the deflated-Sharpe trial count (logged in cache/experiments_log.jsonl).
VARIANTS = OrderedDict([
    ("V0_v2_control", dict(sizing="tranche", exit_mode="fixed", min_stop_atr1h=0.0, min_stop_pct=0.0,
                           cooldown_bars=0, max_losses_per_episode=0, max_attempts=3, universe_n=40,
                           be_after_tp1=True, round_mode="grid", blue_sky_pass=False, min_rr=3.0)),
    ("V1_core_4hCE", dict()),
    ("V2_trail_1h", dict(trail_tf_min=60)),
    ("V3_donchian_d10", dict(exit_mode="donchian")),
    ("V4_partial_4hCE", dict(exit_mode="partial")),
    ("V5_fixed_newrisk", dict(exit_mode="fixed", be_after_tp1=True)),
    ("V6_stage1_h4", dict(stage1_mode="h4")),
    ("V7_sector_heat", dict(heat_mode="expand")),
])
SIMPLICITY = ["V1_core_4hCE", "V3_donchian_d10", "V2_trail_1h", "V4_partial_4hCE", "V5_fixed_newrisk",
              "V6_stage1_h4", "V7_sector_heat"]
FOLDS = [("2021-01-01", "2022-01-01", "holdout"), ("2022-01-01", "2023-01-01", "holdout"),
         ("2023-01-01", "2024-01-01", "holdout"), ("2024-01-01", "2025-01-01", "dev"),
         ("2025-01-01", "2025-10-01", "holdout"), ("2025-10-01", "2026-10-01", "dev")]
EXP_START, EXP_END = "2021-01-01", "2026-10-01"

SIM_ONLY = {"sizing", "total_capital", "n_tranches", "risk_pct", "lev_default", "lev_btc", "lev_alt", "halt_frac",
            "port_loss_streak", "port_streak_risk_mult", "max_loss_frac_of_tranche", "size_scale_by_stop",
            "ref_stop_pct", "mmr", "liq_buffer", "exit_mode", "trail_tf_min", "trail_atr_n", "trail_mult",
            "trail_activate_R", "donchian_days", "partial_frac", "partial_R", "tp1_frac", "be_after_tp1",
            "time_stop_bars", "stale_bars", "stale_R", "max_hold_bars", "taker_fee", "slippage", "funding_source",
            "funding_rate", "funding_interval_h", "cooldown_bars", "max_losses_per_episode", "max_attempts",
            "workers", "vision_workers", "cache_dir"}


def event_key(cfg: Config):
    return tuple((k, v) for k, v in asdict(cfg).items() if k not in SIM_ONLY)


def r_stats(trades, cfg: Config):
    """trade statistics in R plus P&L as a fraction of starting capital"""
    if not trades:
        return dict(n=0, win=np.nan, avgR=np.nan, pf=np.nan, sumR=0.0, exTop2=0.0, pnl_pct=0.0)
    R = np.array([t["R"] for t in trades], float)
    pnl = np.array([t["pnl"] for t in trades], float)
    w, lo = R[R > 0], R[R <= 0]
    s = np.sort(R)
    return dict(n=len(R), win=len(w) / len(R), avgR=R.mean(),
                pf=w.sum() / abs(lo.sum()) if lo.size and lo.sum() != 0 else np.inf,
                sumR=R.sum(), exTop2=s[:-2].sum() if len(s) > 2 else 0.0, pnl_pct=pnl.sum() / cfg.total_capital)


def equity_dd(trades, cfg):
    if not trades:
        return 0.0
    eq = cfg.total_capital + np.cumsum([t["pnl"] for t in trades])
    eq = np.r_[cfg.total_capital, eq]
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def in_range(t, a, b):
    return _utc(a) <= t["entry_time"] < _utc(b)


def bootstrap_dd(trades, risk_pct, n_trades=100, sims=20000, seed=0, block="month"):
    """95th-percentile drawdown of 100-trade paths. iid resampling and month-block resampling
    (keeps loss clustering); the worse of the two is reported."""
    R = np.array([t["R"] for t in trades], float)
    if len(R) < 10:
        return dict(iid=np.nan, block=np.nan, worst=np.nan)
    rng = np.random.default_rng(seed)

    def dd_of(paths):
        eq = np.cumprod(1 + paths * risk_pct, axis=1)
        peak = np.maximum.accumulate(np.c_[np.ones(len(eq)), eq], axis=1)[:, 1:]
        return np.percentile((eq / peak - 1).min(axis=1), 5)

    iid = dd_of(rng.choice(R, size=(sims, n_trades), replace=True))
    months = pd.Series([t["entry_time"].strftime("%Y-%m") for t in trades])
    groups = [R[(months == m).to_numpy()] for m in months.unique()]
    paths = np.empty((2000, n_trades))
    for k in range(2000):
        seq = []
        while len(seq) < n_trades:
            seq.extend(groups[rng.integers(len(groups))])
        paths[k] = seq[:n_trades]
    blk = dd_of(paths)
    return dict(iid=float(iid), block=float(blk), worst=float(min(iid, blk)))


def deflated_sharpe(R_best, trial_srs, n_trials):
    """Bailey & López de Prado (2014) on per-trade R (approximation of the return-series DSR)"""
    from scipy.stats import kurtosis, norm, skew
    R = np.asarray(R_best, float)
    if len(R) < 10 or len(trial_srs) < 2 or R.std(ddof=1) == 0:
        return np.nan
    sr = R.mean() / R.std(ddof=1)
    N = max(n_trials, len(trial_srs))
    v = np.var(trial_srs, ddof=1)
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    g3, g4 = skew(R), kurtosis(R, fisher=False)
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if den <= 0:
        return np.nan
    return float(norm.cdf((sr - sr0) * np.sqrt(len(R) - 1) / np.sqrt(den)))


def _log_trials(cfg, names):
    path = os.path.join(cfg.cache_dir, "experiments_log.jsonl")
    os.makedirs(cfg.cache_dir, exist_ok=True)
    with open(path, "a") as f:
        for nm in names:
            f.write(json.dumps({"t": pd.Timestamp.now(tz="UTC").isoformat(), "variant": nm}) + "\n")
    with open(path) as f:
        return len({json.loads(x)["variant"] for x in f if x.strip()})


def _fmt(x, f="{:+.2f}"):
    return "   -" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def run_experiment(cfg: Config, ds: Dataset = None, folds=FOLDS, start=EXP_START, end=EXP_END, variants=VARIANTS,
                   extra_checks=True):
    if ds is None:
        ds = build_pit_dataset(cfg, start, end, n_max=max([cfg.universe_n] + [v.get("universe_n", 0)
                                                                                for v in variants.values()]))
    n_trials = _log_trials(cfg, list(variants))
    ev_cache, res = {}, OrderedDict()
    for name, ov in variants.items():
        c = replace(cfg, **ov)
        k = event_key(c)
        t0 = time.time()
        if k not in ev_cache:
            ev_cache[k] = build_events(ds, c, start, end)
        trades, acct = run_backtest(ds, c, start, end, events=ev_cache[k])
        res[name] = dict(cfg=c, trades=trades, acct=acct)
        print(f"  {name:<17} {len(trades):4d} trades  ({time.time() - t0:.0f}s)")
    # ---- fold table ----
    print(f"\n{'variant':<17} {'fold':<8} {'n':>4} {'win':>4} {'avgR':>6} {'PF':>5} {'P&L%':>6} {'exTop2':>7}")
    summary = {}
    for name, r in res.items():
        c, tr = r["cfg"], r["trades"]
        fold_avg, fold_pnl = [], []
        for a, b, tag in folds:
            st = r_stats([t for t in tr if in_range(t, a, b)], c)
            fold_avg.append(st["avgR"]); fold_pnl.append(st["pnl_pct"])
            print(f"{name[:17]:<17} {a[:7]:<8} {st['n']:4d} {_fmt(st['win'], '{:.2f}')[-4:]:>4} "
                  f"{_fmt(st['avgR']):>6} {_fmt(st['pf'], '{:.2f}'):>5} {st['pnl_pct'] * 100:+6.1f} {st['exTop2']:+7.1f}")
        hold = [t for t in tr if any(in_range(t, a, b) for a, b, g in folds if g == "holdout")]
        sh, sa = r_stats(hold, c), r_stats(tr, c)
        dd = equity_dd(tr, c)
        print(f"{name[:17]:<17} {'HOLDOUT':<8} {sh['n']:4d} {_fmt(sh['win'], '{:.2f}')[-4:]:>4} "
              f"{_fmt(sh['avgR']):>6} {_fmt(sh['pf'], '{:.2f}'):>5} {sh['pnl_pct'] * 100:+6.1f} {sh['exTop2']:+7.1f}")
        print(f"{name[:17]:<17} {'ALL':<8} {sa['n']:4d} {_fmt(sa['win'], '{:.2f}')[-4:]:>4} "
              f"{_fmt(sa['avgR']):>6} {_fmt(sa['pf'], '{:.2f}'):>5} {sa['pnl_pct'] * 100:+6.1f} {sa['exTop2']:+7.1f}"
              f"   maxDD {dd * 100:.1f}%")
        fa = np.array([x for x in fold_avg if np.isfinite(x)])
        summary[name] = dict(median_fold_R=float(np.median(fa)) if len(fa) else np.nan, dd=dd, hold=sh, all=sa,
                             fold_pnl=fold_pnl, fold_avg=fold_avg)
    # ---- selection rule (pre-registered) ----
    cands = [nm for nm in SIMPLICITY if nm in summary and summary[nm]["dd"] >= -0.20 and summary[nm]["all"]["n"] > 0
             and np.isfinite(summary[nm]["median_fold_R"])]
    if not cands:
        print("\nNo variant passes the −20% drawdown filter. Verdict: FAIL — do not trade this system.")
        return res, summary, None
    best = max(summary[nm]["median_fold_R"] for nm in cands)
    chosen = next(nm for nm in SIMPLICITY if nm in cands and summary[nm]["median_fold_R"] >= best - 0.02)
    print(f"\nselected: {chosen} (highest median-fold avg R {best:+.2f}; ties within 0.02R go to the simpler variant)")
    gates = check_gates(ds, cfg, res, summary, chosen, folds, start, end, n_trials, ev_cache, extra_checks)
    return res, summary, gates


def check_gates(ds, base_cfg, res, summary, chosen, folds, start, end, n_trials, ev_cache, extra_checks=True):
    c, tr = res[chosen]["cfg"], res[chosen]["trades"]
    S = summary[chosen]
    hold, alls = S["hold"], S["all"]
    g = OrderedDict()
    g["1 trades ≥60 (≥30 hold-out)"] = (alls["n"] >= 60 and hold["n"] >= 30, f"{alls['n']} / {hold['n']}")
    g["2 hold-out PF≥1.3, avgR≥0.2; all PF≥1.5"] = (
        hold["n"] > 0 and hold["pf"] >= 1.3 and hold["avgR"] >= 0.2 and alls["pf"] >= 1.5,
        f"{_fmt(hold['pf'], '{:.2f}')}, {_fmt(hold['avgR'])}; {_fmt(alls['pf'], '{:.2f}')}")
    R = np.sort([t["R"] for t in tr])
    k5 = max(1, int(round(0.05 * len(R)))) if len(R) else 0
    ex5 = R[:-k5] if len(R) > k5 else np.array([])
    pf_ex5 = ex5[ex5 > 0].sum() / abs(ex5[ex5 <= 0].sum()) if (ex5 <= 0).any() and ex5[ex5 <= 0].sum() != 0 else np.nan
    by_year = defaultdict(list)
    for t in tr:
        by_year[t["entry_time"].year].append(t["R"])
    ex2y = sum(sum(sorted(v)[:-2]) if len(v) > 2 else 0.0 for v in by_year.values())
    top10 = sorted(tr, key=lambda t: -t["R"])[:10]
    coins, years = len({t["sym"] for t in top10}), len({t["entry_time"].year for t in top10})
    g["3 outliers: PF ex-top5%≥1, ex-top2/yr, top10 spread"] = (
        np.isfinite(pf_ex5) and pf_ex5 >= 1.0 and ex2y >= -0.1 * len(tr) and coins >= 6 and years >= 3,
        f"PF {_fmt(pf_ex5, '{:.2f}')}, {ex2y:+.1f}R, {coins} coins/{years} yrs")
    worst_fold = min(S["fold_pnl"]) if S["fold_pnl"] else 0.0
    g["4 median fold avgR>0, no fold < −12%"] = (
        np.isfinite(S["median_fold_R"]) and S["median_fold_R"] > 0 and worst_fold >= -0.12,
        f"{_fmt(S['median_fold_R'])}, worst {worst_fold * 100:+.1f}%")
    bs = bootstrap_dd(tr, c.risk_pct)
    g["5 bootstrap 95th-pct DD ≥ −25%"] = (np.isfinite(bs["worst"]) and bs["worst"] >= -0.25,
                                           f"iid {_fmt(bs['iid'] * 100, '{:.1f}')}%, "
                                           f"block {_fmt(bs['block'] * 100, '{:.1f}')}%")
    srs = [np.mean([t["R"] for t in r["trades"]]) / np.std([t["R"] for t in r["trades"]], ddof=1)
           for r in res.values() if len(r["trades"]) >= 10 and np.std([t["R"] for t in r["trades"]]) > 0]
    dsr = deflated_sharpe([t["R"] for t in tr], srs, n_trials)
    g["6 deflated Sharpe ≥ 0.90"] = (np.isfinite(dsr) and dsr >= 0.90, f"{_fmt(dsr, '{:.2f}')} (N={n_trials})")
    if extra_checks:
        ck = replace(c, taker_fee=0.001, slippage=0.001)
        trc, _ = run_backtest(ds, ck, start, end, events=ev_cache.get(event_key(ck)) or build_events(ds, ck, start, end))
        pf_c = r_stats(trc, ck)["pf"]
        g["7 cost stress (0.10%+0.10%) PF≥1.2"] = (np.isfinite(pf_c) and pf_c >= 1.2, _fmt(pf_c, "{:.2f}"))
        if c.exit_mode == "donchian":
            cliffs = [("donchian_days", 7), ("donchian_days", 15)]
        else:
            cliffs = [("trail_mult", 2.5), ("trail_mult", 3.5)]
        cliffs += [("min_stop_atr1h", 1.5), ("min_stop_atr1h", 2.5)]
        outs = []
        for k, v in cliffs:
            cc = replace(c, **{k: v})
            ek = event_key(cc)
            if ek not in ev_cache:
                ev_cache[ek] = build_events(ds, cc, start, end)
            trv, _ = run_backtest(ds, cc, start, end, events=ev_cache[ek])
            outs.append((k, v, r_stats(trv, cc)["avgR"]))
        ok = all(np.isfinite(x) and x > 0 for _, _, x in outs)
        g["8 cliff check: neighbours keep avgR>0"] = (ok, ", ".join(f"{k}={v}:{_fmt(x)}" for k, v, x in outs))
    print(f"\npass/fail gates for {chosen}:")
    for name, (ok, detail) in g.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("  [ -- ] 9 forward paper / micro-live 3 months at 0.25–0.5% risk: manual")
    if all(ok for ok, _ in g.values()):
        print("\nVerdict: all automated gates pass → paper/micro-live next (gate 9). Not a guarantee of profit.")
    else:
        print("\nVerdict: at least one gate FAILS → do not trade this live. A failed gate is the answer, "
              "not a reason to tune until it passes.")
    return g


# ================= 15. LIVE SCANNER (signals only by default) =================
def scan(ex, cfg: Config, symbols=None, execute=False, testnet=False):
    ex.load_markets()
    universe = [plain(s) for s in top_universe(ex, cfg)]
    targets = [plain(s) for s in symbols] if symbols else universe
    now = _ts(ex.milliseconds())
    got = fetch_many(ex, cfg, [(s, "1d", now - pd.Timedelta(days=420), None) for s in ["BTCUSDT"] + universe + targets])
    daily = {plain(k[0]): v for k, v in got.items() if v is not None and len(v)}
    btc = daily.get("BTCUSDT")
    if btc is None:
        raise RuntimeError("BTC daily data missing")
    closes = pd.DataFrame({s: daily[s].close for s in universe if s in daily})
    season = classify_season(btc, cfg, breadth_table(closes, cfg))
    cur = season.iloc[-1]
    hot = None
    if cfg.heat_mode == "expand":
        ds = Dataset.from_frames(btc, {s: daily[s] for s in universe if s in daily}, {})
        hot = heat_table(ds, cfg.universe_n, cfg, now - pd.Timedelta(days=200), now + pd.Timedelta(days=1)).iloc[-1]
    print(f"season (as of {season.index[-1].date()} UTC daily close): {cur} → "
          f"{'trade' if cur == '夏' else ('hot-sector trades only, half risk' if hot is not None else 'no trades')}")
    btc_r = btc.close / btc.close.shift(30)
    pre = {}
    for s in targets:
        d = daily.get(s)
        if d is None or len(d) < 60 or d.index[-1] < btc.index[-1]:
            continue
        stg = classify_stages(d, cfg)
        st = int(stg.stage.iloc[-1])
        rs = float((d.close / d.close.shift(30)).iloc[-1] / btc_r.reindex(d.index).iloc[-1] - 1)
        mult = 1.0 if cur == "夏" or (cfg.allow_late_spring and cur == "春") else (
            cfg.heat_risk_mult if hot is not None and cur in ("春", "秋") and bool(hot.get(s, False)) else 0.0)
        if st in PRIORITY and (not cfg.require_rs or rs > 0):
            pre[s] = (d, stg, st, rs, mult)
    ltf = fetch_many(ex, cfg, [(s, cfg.ltf, now - pd.Timedelta(days=4), None) for s in pre])
    acct = Account(cfg)
    rows = []
    for s, (d, stg, st, rs, mult) in pre.items():
        m = ltf.get((to_ccxt(s), cfg.ltf))
        if m is None or len(m) <= cfg.pat_window + 1:
            continue
        L = ltf_arrays(m, cfg)
        t = len(m) - 1
        sig = find_pattern(L, t, cfg, pending=True)
        if not sig:
            continue
        plan = build_plan(s, daily_pack(d, cfg), len(d) - 1, st, stg.episode.iloc[-1], rs,
                          stg.base_hi.iloc[-1], stg.base_lo.iloc[-1], sig, cfg, atr1h=L["atr1h"][t])
        if not plan:
            continue
        if cfg.sizing == "tranche":
            sz = size_tranche(s, plan["entry"], plan["stop"], acct.margin(), cfg)
        else:
            sz = size_risk(s, plan["entry"], plan["stop"], acct.risk_usd(mult if mult > 0 else 1.0), acct.margin(), cfg)
        if sz is None:
            continue
        P = lambda x: float(ex.price_to_precision(to_ccxt(s), x))
        rows.append(dict(coin=base_of(s), st=st, kind=SHORT_KIND.get(plan["kind"], "?"),
                         status="BREAKOUT" if plan["status"] == "已突破" else "pending",
                         entry=P(plan["entry"]), stop=P(plan["stop"]),
                         stop_pct=round((plan["entry"] - plan["stop"]) / plan["entry"] * 100, 2),
                         tp1=P(plan["tp1"]), rr="blue sky" if plan["blue"] else f"{plan['rr']:.1f}",
                         lev=f"{sz['lev']:.0f}x", margin=round(sz["margin"]), notional=round(sz["notional"]),
                         risk=round(sz["risk"]), liq=P(sz["liq"]) if sz["liq"] else 0.0, rs=round(rs * 100, 1),
                         tradeable=mult > 0, _plan=plan, _sz=sz))
    if not rows:
        print("no coin currently passes stage + room + trigger.")
        return
    rows.sort(key=lambda r: (not r["tradeable"], PRIORITY[r["st"]], -r["rs"]))
    print(f"{'coin':<8} st {'kind':<4} {'status':<8} {'entry':>10} {'stop':>10} {'stop%':>5} "
          f"{'TP1(ref)':>10} {'RR':>8} {'lev':>3} {'risk$':>5} {'notional':>8}")
    for r in rows:
        flag = "" if r["tradeable"] else "  (season: watch only)"
        print(f"{r['coin']:<8} {r['st']:>2} {r['kind']:<4} {r['status']:<8} {r['entry']:>10g} {r['stop']:>10g} "
              f"{r['stop_pct']:>5.2f} {r['tp1']:>10g} {r['rr']:>8} {r['lev']:>3} {r['risk']:>5} {r['notional']:>8}{flag}")
    if cfg.exit_mode != "fixed":
        print(f"\nexit rule: {cfg.exit_mode} trailing stop (arm at +{cfg.trail_activate_R:g}R, "
              f"{cfg.trail_mult:g}×ATR on {cfg.trail_tf_min // 60}h closes). TP1 is a reference level only.")
    if execute:
        if cfg.exit_mode != "fixed":
            print("--execute is disabled for trailing exits: the stop must be ratcheted on every "
                  f"{cfg.trail_tf_min // 60}h close and there is no position manager yet. Validate first (experiment).")
        elif os.getenv("LANGLANG_LIVE") != "1":
            print("LANGLANG_LIVE=1 not set → no orders.")
        else:
            r = next((r for r in rows if r["tradeable"] and r["status"] == "BREAKOUT"), None)
            if r:
                place_bracket(make_exchange(private=True, testnet=testnet), cfg, r["_plan"], r["_sz"])


def place_bracket(ex, cfg, plan, sz):
    """fixed-exit mode only: market entry + stop + two take-profits (ccxt routes USDⓈ-M conditionals to the
    algo-order endpoint). Test with --testnet first."""
    s = to_ccxt(plan["sym"])
    ex.load_markets()
    open_pos = [p["symbol"] for p in ex.fetch_positions() if abs(float(p.get("contracts") or 0)) > 0]
    if open_pos:
        print("position already open, skipping (one position at a time):", open_pos)
        return
    free = float(ex.fetch_balance().get("free", {}).get("USDT") or 0)
    if sz["margin"] > free * 0.95:
        print(f"not enough free USDT ({free:.0f}) for margin {sz['margin']:.0f}")
        return
    px = lambda x: float(ex.price_to_precision(s, x))
    amt = lambda x: float(ex.amount_to_precision(s, x))
    q = amt(sz["qty"]); half = amt(q * cfg.tp1_frac); rest = amt(q - half)
    try:
        ex.set_margin_mode("isolated", s)
    except Exception as e:                                # noqa: BLE001
        print("margin mode:", str(e)[:80])
    ex.set_leverage(int(math.ceil(sz["lev"])), s)
    ex.create_order(s, "market", "buy", q)
    try:
        ex.create_order(s, "market", "sell", q, None, {"stopLossPrice": px(plan["stop"]), "reduceOnly": True})
        ex.create_order(s, "market", "sell", half, None, {"takeProfitPrice": px(plan["tp1"]), "reduceOnly": True})
        ex.create_order(s, "market", "sell", rest, None, {"takeProfitPrice": px(plan["tp2"]), "reduceOnly": True})
    except Exception:
        print("protective order failed → closing the position at market")
        ex.create_order(s, "market", "sell", q, None, {"reduceOnly": True})
        raise
    print(f"order placed {s}: {q} @market, {sz['lev']:.0f}x, stop {px(plan['stop'])}, "
          f"TP1 {px(plan['tp1'])}, TP2 {px(plan['tp2'])}")


# ================= 16. INFOGRAPHIC EXAMPLES =================
def examples(ex, cfg):
    d = fetch_ohlcv(ex, cfg, "PEOPLEUSDT", "1d", "2023-06-01", "2024-06-01")
    v = d.join(classify_stages(d, cfg)[["stage", "base_hi", "base_lo"]]).assign(
        chg=lambda x: (x.close.pct_change() * 100).round(2))
    print("== PEOPLE daily stages 2024-02-05 ~ 2024-05-21 ==")
    print(v.loc["2024-02-05":"2024-05-21", ["open", "high", "low", "close", "chg", "stage", "base_hi", "base_lo"]].to_string())
    m = fetch_ohlcv(ex, cfg, "PEOPLEUSDT", cfg.ltf, "2024-05-13", "2024-05-23")
    L = ltf_arrays(m, cfg)
    print("== PEOPLE 15m breakouts 2024-05-15 ~ 05-22 ==")
    for t in np.flatnonzero(np.asarray(m.index >= pd.Timestamp("2024-05-15", tz="UTC"))):
        s = find_pattern(L, t, cfg)
        if s:
            print(m.index[t], s["kind"], "entry", round(s["entry"], 5), "stop", round(s["stop"], 5))
    b = fetch_ohlcv(ex, cfg, "BTCUSDT", "1h", "2023-10-01", "2024-03-30")
    se = classify_season(b, cfg, None, k=24).loc["2024-01-17":"2024-03-29"]
    print("== BTC 1h season (BTC-only, daily mode) ==")
    print(se.resample("1D").agg(lambda x: x.mode().iat[0] if len(x) else None).to_string())


# ================= 17. CLI =================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Bit浪浪 stage-breakout system v2.1")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--capital", type=float, default=None, help="starting capital (default 9000)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any Config field")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="build the point-in-time dataset from data.binance.vision")
    p.add_argument("--start", default=EXP_START)
    p.add_argument("--end", default=EXP_END)
    p = sub.add_parser("backtest")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-01-01", help="exclusive")
    p.add_argument("--symbols", default="", help="restrict tradable coins (breadth still uses the universe)")
    p.add_argument("--source", choices=["pit", "static"], default=None)
    p.add_argument("--trades", action="store_true", help="print every trade")
    sub.add_parser("experiment", help="pre-registered variants × walk-forward folds × pass/fail gates")
    p = sub.add_parser("scan")
    p.add_argument("--symbols", default="")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--testnet", action="store_true")
    sub.add_parser("examples")
    a = ap.parse_args(argv)
    cfg = apply_overrides(Config(), a.set)
    if a.workers:
        cfg = replace(cfg, workers=a.workers)
    if a.capital:
        cfg = replace(cfg, total_capital=a.capital)
    syms = [x for x in getattr(a, "symbols", "").split(",") if x]
    if a.cmd == "download":
        build_pit_dataset(cfg, a.start, a.end, n_max=max(60, cfg.universe_n))
    elif a.cmd == "backtest":
        source = a.source or cfg.universe_mode
        if source == "pit":
            ds = build_pit_dataset(cfg, a.start, a.end, symbols_only=syms or None)
        else:
            ex = make_exchange()
            ds = build_static_dataset(ex, cfg, syms or top_universe(ex, cfg), a.start, a.end)
        trades, acct = run_backtest(ds, cfg, a.start, a.end, symbols=syms or None)
        if trades and (a.trades or len(trades) <= 40):
            print_trades(trades)
        print_metrics(metrics(trades, acct, cfg))
    elif a.cmd == "experiment":
        run_experiment(cfg)
    elif a.cmd == "scan":
        scan(make_exchange(), cfg, syms or None, a.execute, a.testnet)
    elif a.cmd == "examples":
        examples(make_exchange(), cfg)


if __name__ == "__main__":
    main()
