"""Offline stand-in for data.binance.vision: paginated S3 XML listing + zipped kline / funding CSVs.
Mimics archive quirks: header-less files before `header_from`, flat zero-trade rows after a delisting,
the current month only as daily files, and decoy symbols (stablecoin, index perp, stock perp, delivery)."""
import io
import threading
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

import mockex

WEB = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def make_market(n_sym=14, days=690, seed=7, t0="2023-01-01", delist_at="2024-06-15", new_at="2024-03-01"):
    raw = mockex.synth(n_sym=n_sym, days=days, seed=seed, t0=t0)
    m15 = {s.split("/")[0] + "USDT": df.copy() for (s, tf), df in raw.items() if tf == "15m"}
    for df in m15.values():
        df["count"] = 100.0
    # a coin that gets delisted: the archive keeps flat, zero-trade rows after settlement
    dead = m15["C01USDT"].copy()
    cut = pd.Timestamp(delist_at, tz="UTC")
    last = dead.loc[dead.index < cut, "close"].iloc[-1]
    post = (dead.index >= cut) & (dead.index < cut + pd.Timedelta(days=40))
    dead.loc[post, ["open", "high", "low", "close"]] = last
    dead.loc[post, ["volume", "count"]] = 0.0
    m15["DEADUSDT"] = dead.loc[dead.index < cut + pd.Timedelta(days=40)]
    # a newly listed coin (must wait min_listing_days before joining the universe)
    m15["NEWUSDT"] = m15["C02USDT"].loc[m15["C02USDT"].index >= pd.Timestamp(new_at, tz="UTC")].copy() * 1.0
    m15["NEWUSDT"]["volume"] *= 50                                     # huge volume: would rank #1 if allowed
    # decoys that must never enter the universe
    for decoy in ("USDCUSDT", "BTCDOMUSDT", "AMZNUSDT"):
        m15[decoy] = m15["C03USDT"].copy()
        m15[decoy]["volume"] *= 100
    return m15


class MockVision:
    def __init__(self, m15: dict, now, page=4, header_from="2024-01"):
        self.m15, self.now, self.page, self.header_from = m15, pd.Timestamp(now), page, header_from
        self.calls = 0
        self._lock = threading.Lock()
        self._zips = {}
        self.keys = self._index()

    # ---------- virtual file tree ----------
    def _month_published(self, m):          # monthly file exists for completed months
        return pd.Timestamp(m + "-01", tz="UTC") + pd.offsets.MonthBegin(1) <= self.now.floor("D")

    def _index(self):
        keys = []
        cur = self.now.strftime("%Y-%m")
        for s, df in self.m15.items():
            df = df.loc[df.index < self.now.floor("D")]
            months = sorted(set(df.index.strftime("%Y-%m")))
            for tf in ("15m", "1d"):
                for m in months:
                    if m != cur and self._month_published(m):
                        p = f"data/futures/um/monthly/klines/{s}/{tf}/{s}-{tf}-{m}.zip"
                        keys += [p, p + ".CHECKSUM"]
                days = sorted(set(df.index[df.index >= pd.Timestamp(cur + "-01", tz="UTC")].strftime("%Y-%m-%d")))
                for d in days:
                    p = f"data/futures/um/daily/klines/{s}/{tf}/{s}-{tf}-{d}.zip"
                    keys += [p, p + ".CHECKSUM"]
            for m in months:
                if m != cur and self._month_published(m):
                    p = f"data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"
                    keys += [p, p + ".CHECKSUM"]
        keys.append("data/futures/um/monthly/klines/BTCUSDT_240329/1d/BTCUSDT_240329-1d-2024-01.zip")
        return sorted(set(keys))

    # ---------- HTTP ----------
    def get(self, url, params=None):
        with self._lock:
            self.calls += 1
        if url == S3:
            return 200, self._list(params["prefix"], params.get("delimiter", ""), params.get("marker", ""))
        path = url[len(WEB) + 1:]
        if path not in self.keys:
            return 404, b"<Error>NoSuchKey</Error>"
        with self._lock:
            if path not in self._zips:
                self._zips[path] = self._build(path)
        return 200, self._zips[path]

    def _list(self, prefix, delim, marker):
        entries = set()
        for k in self.keys:
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            if delim and delim in rest:
                entries.add(("P", prefix + rest.split(delim)[0] + delim))
            else:
                entries.add(("K", k))
        ordered = sorted(entries, key=lambda e: e[1])
        ordered = [e for e in ordered if e[1] > marker]
        page, more = ordered[: self.page], len(ordered) > self.page
        ns = "http://s3.amazonaws.com/doc/2006-03-01/"
        parts = [f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{ns}">',
                 f"<Name>data.binance.vision</Name><Prefix>{escape(prefix)}</Prefix><Marker>{escape(marker)}</Marker>",
                 f"<MaxKeys>{self.page}</MaxKeys><Delimiter>{escape(delim)}</Delimiter>",
                 f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"]
        if more:
            parts.append(f"<NextMarker>{escape(page[-1][1])}</NextMarker>")
        for kind, v in page:
            if kind == "K":
                parts.append(f"<Contents><Key>{escape(v)}</Key><Size>1</Size></Contents>")
            else:
                parts.append(f"<CommonPrefixes><Prefix>{escape(v)}</Prefix></CommonPrefixes>")
        parts.append("</ListBucketResult>")
        return "".join(parts).encode()

    def _build(self, path):
        name = path.rsplit("/", 1)[1][:-4]
        if path.endswith(".CHECKSUM"):
            return b"0" * 64
        if "/fundingRate/" in path:
            sym, _, m = name.split("-", 2)
            idx = pd.date_range(m + "-01", pd.Timestamp(m + "-01") + pd.offsets.MonthBegin(1), freq="8h",
                                inclusive="left", tz="UTC")
            rate = 0.0001 + 0.0004 * np.sin(np.arange(len(idx)) / 9.0) ** 2        # hot-coin-like positive funding
            csv = "calc_time,funding_interval_hours,last_funding_rate\n" + "".join(
                f"{t.value // 1_000_000 + 7},8,{r:.8f}\n" for t, r in zip(idx, rate))
            return self._zip(name + ".csv", csv)
        sym, tf, period = name.split("-", 2)
        df = self.m15[sym]
        if tf == "1d":
            df = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                        "volume": "sum", "count": "sum"}).dropna()
        lo = pd.Timestamp(period + ("-01" if len(period) == 7 else ""), tz="UTC")
        hi = lo + (pd.offsets.MonthBegin(1) if len(period) == 7 else pd.Timedelta(days=1))
        df = df.loc[(df.index >= lo) & (df.index < hi)]
        step = 900_000 if tf == "15m" else 86_400_000
        rows = []
        for t, r in zip(df.index, df.itertuples(index=False)):
            ot = t.value // 1_000_000
            qv = r.volume * r.close
            rows.append(f"{ot},{r.open:.10g},{r.high:.10g},{r.low:.10g},{r.close:.10g},{r.volume:.6f},{ot + step - 1},"
                        f"{qv:.6f},{int(r.count)},{r.volume / 2:.6f},{qv / 2:.6f},0\n")
        header = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                  "taker_buy_quote_volume,ignore\n") if period[:7] >= self.header_from else ""
        return self._zip(name + ".csv", header + "".join(rows))

    @staticmethod
    def _zip(fname, text):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(fname, text)
        return buf.getvalue()
