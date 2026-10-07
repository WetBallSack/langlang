"""Offline stand-in for ccxt.binanceusdm: serves synthetic OHLCV with Binance-like paging semantics."""
import threading
import numpy as np
import pandas as pd

STEP = {"15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}

def synth(n_sym=20, days=690, seed=7, t0="2023-01-01"):
    rng = np.random.default_rng(seed)
    n = days * 96
    idx = pd.date_range(t0, periods=n, freq="15min", tz="UTC")
    day = np.arange(n) / 96.0
    def drift_path(shift, amp):
        # piecewise daily log-drift: decline, base, rally, chop, second leg, decline
        knots = np.array([0, 230, 330, 430, 500, 570, 690]) + shift
        rates = np.array([-0.0045, 0.0, 0.009, 0.0, 0.008, -0.006]) * amp
        seg = np.clip(np.searchsorted(knots, day, side="right") - 1, 0, len(rates) - 1)
        return rates[seg] / 96.0
    mkt = drift_path(0, 1.0)
    out = {}
    names = ["BTC"] + [f"C{i:02d}" for i in range(n_sym - 1)]
    for k, name in enumerate(names):
        if name == "BTC":
            mu, sig = mkt * 0.6, 0.025 / np.sqrt(96)
        else:
            mu = 0.5 * mkt + drift_path(rng.integers(-40, 60), rng.uniform(0.8, 1.6))
            sig = rng.uniform(0.035, 0.06) / np.sqrt(96)
        r = mu + sig * rng.standard_t(4, n) / np.sqrt(2)
        c = rng.uniform(0.02, 50) * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        wig = np.abs(rng.normal(0, sig, (2, n)))
        h = np.maximum(o, c) * np.exp(wig[0]); l = np.minimum(o, c) * np.exp(-wig[1])
        v = rng.lognormal(10, 0.5, n) * (1 + 25 * np.abs(r - mu) / sig / 10) * (1 + 3 * (mu > 0))
        m = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
        sym = f"{name}/USDT:USDT"
        out[(sym, "15m")] = m
        out[(sym, "1h")] = m.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        out[(sym, "1d")] = m.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out

class MockExchange:
    DATA: dict = {}
    NOW: int = 0
    CALLS = 0
    _lock = threading.Lock()
    rateLimit = 0

    def __init__(self, config=None):
        self.markets = {s: {"symbol": s, "swap": True, "linear": True, "quote": "USDT", "active": True,
                            "base": s.split("/")[0], "info": {"contractType": "PERPETUAL", "underlyingType": "COIN"}}
                        for s, tf in self.DATA if tf == "1d"}
        self.currencies = {}

    @classmethod
    def load(cls, data, now_ms):
        cls.DATA = {k: (v.index.as_unit("ns").asi8 // 1_000_000, v.to_numpy()) for k, v in data.items()}
        cls.NOW = now_ms; cls.CALLS = 0

    def milliseconds(self): return self.NOW
    def price_to_precision(self, s, x): return f"{float(f'{x:.5g}')}"
    def amount_to_precision(self, s, x): return f"{int(x * 1000) / 1000}"
    def load_markets(self, reload=False): return self.markets
    def set_markets(self, markets, currencies=None): self.markets = markets
    def fetch_tickers(self): return {s: {"quoteVolume": 1e9} for s in self.markets}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500, params=None):
        with MockExchange._lock: MockExchange.CALLS += 1
        if (symbol, timeframe) not in self.DATA: raise KeyError(f"BadSymbol {symbol}")
        ts, vals = self.DATA[(symbol, timeframe)]
        until = (params or {}).get("until")
        i = np.searchsorted(ts, since or 0, side="left")
        j = min(i + min(limit, 1500), np.searchsorted(ts, self.NOW, side="right"))
        if until is not None: j = min(j, np.searchsorted(ts, until, side="right"))
        return [[int(ts[k]), *map(float, vals[k])] for k in range(i, j)]
