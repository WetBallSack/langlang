"""mockfeeds.py — offline data.binance.vision stand-in serving 1m/5m/15m/1h/1d klines (with taker volume),
1m mark-price klines and funding, built from synthetic 1m frames. Used by test_vegas.py and test_kwrev.py."""
import numpy as np
import pandas as pd

import test_floop as TF

FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}
STEP = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


class Feeds(TF.FMock):
    KINDS = (("klines", "1m"), ("klines", "5m"), ("klines", "15m"), ("klines", "1d"), ("markPriceKlines", "1m"))

    def _index(self):
        keys, cur = [], self.now.strftime("%Y-%m")
        for s, df in self.m1.items():
            df = df.loc[df.index < self.now.floor("D")]
            months = sorted(set(df.index.strftime("%Y-%m")))
            for kind, tf in self.KINDS:
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
        if "/fundingRate/" in path:
            return super()._build(path)
        name = path.rsplit("/", 1)[1][:-4]
        sym, tf, period = name.split("-", 2)
        df = self.m1[sym]
        lo = pd.Timestamp(period + ("-01" if len(period) == 7 else ""), tz="UTC")
        hi = lo + (pd.offsets.MonthBegin(1) if len(period) == 7 else pd.Timedelta(days=1))
        df = df.loc[(df.index >= lo) & (df.index < hi)]
        mark = "/markPriceKlines/" in path
        if mark and "mark_bias" in df:
            df = df.assign(open=df.open * (1 + df.mark_bias), high=df.high * (1 + df.mark_bias),
                           low=df.low * (1 + df.mark_bias), close=df.close * (1 + df.mark_bias))
        if not mark and tf != "1m":
            g = df.groupby(df.index.floor(FREQ[tf]))
            df = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                               "volume": g.volume.sum(), "tbv": g.tbv.sum()})
        step = STEP[tf]
        rows = []
        for t, r in zip(df.index, df.itertuples(index=False)):
            ot = t.value // 1_000_000
            v, tb = (0.0, 0.0) if mark else (r.volume, r.tbv)
            rows.append(f"{ot},{r.open:.10g},{r.high:.10g},{r.low:.10g},{r.close:.10g},{v:.6f},{ot + step - 1},"
                        f"{v * r.close:.4f},{0 if mark else 100},{tb:.6f},{tb * r.close:.4f},0\n")
        header = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                  "taker_buy_quote_volume,ignore\n") if period[:7] >= self.header_from else ""
        return self._zip(name + ".csv", header + "".join(rows))


def install(m1, now, cache):
    import shutil
    import langlang as LL
    mv = Feeds(m1, now)
    LL.HTTP_GET = mv.get
    LL._now = lambda: pd.Timestamp(now)
    shutil.rmtree(cache, ignore_errors=True)
    return mv


def gbm1m(start, days, seed, p0=100.0, vol=0.0008, drift=0.0):
    """driftless (or drifting) 1m geometric random walk with symmetric wicks and taker volume"""
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    r = drift + vol * rng.standard_normal(n)
    c = p0 * np.exp(np.cumsum(r))
    o = np.r_[p0, c[:-1]]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    v = rng.lognormal(2, 0.5, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v,
                         "tbv": v * rng.uniform(0.3, 0.7, n)}, index=idx)
