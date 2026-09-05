"""yfinance を使った株価・ファンダメンタルズの取得。

株価は素直に使える。ファンダメンタルズには次の制約があることを
承知の上で使うこと (README にも記載):

    * `.info` は「いま現在」の値しか返さない。過去の PER は取れないので、
      毎日実行して fundamentals テーブルに積み上げることで履歴を作る。
    * 銘柄ごとに 1 リクエスト必要で、株価取得よりかなり遅い。
    * 小型株ではキーが欠損することがある。欠損は None のまま入れる。
"""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd

from .base import normalize_price_frame

_RETRY_SLEEP = (1.0, 3.0, 7.0)


def _import_yfinance():
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - 環境依存
        raise RuntimeError(
            "yfinance が入っていません。`pip install -r app/requirements.txt` を実行してください。"
        ) from e
    return yf


def _clean(value: Any) -> float | None:
    """数値以外・NaN・infinity を None に潰す。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class YFinanceProvider:
    """PriceProvider と FundamentalsProvider の両方を兼ねる。"""

    name = "yfinance"

    def __init__(self, retries: int = 2):
        self.retries = retries

    # --- 株価 ---------------------------------------------------------------

    def fetch_prices(
        self, ticker: str, start: str, end: str | None = None
    ) -> pd.DataFrame | None:
        yf = _import_yfinance()
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                raw = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
                return normalize_price_frame(raw)
            except Exception as e:  # ネットワーク・レート制限
                last_error = e
                if attempt < self.retries:
                    time.sleep(_RETRY_SLEEP[min(attempt, len(_RETRY_SLEEP) - 1)])

        print(f"  [WARN] price fetch failed {ticker}: {last_error}")
        return None

    # --- ファンダメンタルズ --------------------------------------------------

    def fetch_fundamentals(
        self, ticker: str, *, close: float | None = None, deep: bool = False
    ) -> dict[str, float | None] | None:
        yf = _import_yfinance()
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception as e:
            print(f"  [WARN] fundamentals fetch failed {ticker}: {e}")
            return None

        if not info:
            return None

        out: dict[str, float | None] = {}

        # 時価総額: 円 -> 億円
        cap = _clean(info.get("marketCap"))
        out["market_cap"] = cap / 1e8 if cap else None

        out["per"] = _clean(info.get("trailingPE"))
        out["pbr"] = _clean(info.get("priceToBook"))
        out["eps"] = _clean(info.get("trailingEps"))
        out["bps"] = _clean(info.get("bookValue"))

        # ROE: yfinance は小数 (0.12) で返す -> %
        roe = _clean(info.get("returnOnEquity"))
        out["roe"] = roe * 100 if roe is not None else None

        out["dividend_yield"] = self._dividend_yield(info, close)
        out["equity_ratio"] = self._equity_ratio(t) if deep else None

        return out

    @staticmethod
    def _dividend_yield(info: dict, close: float | None) -> float | None:
        """配当利回り(%)。

        `info["dividendYield"]` は yfinance のバージョンによって小数だったり
        パーセントだったりして判別できない。年間配当額と終値が揃っているなら
        自分で割る方が確実なので、そちらを優先する。
        """
        rate = _clean(info.get("dividendRate")) or _clean(
            info.get("trailingAnnualDividendRate")
        )
        price = close or _clean(info.get("currentPrice")) or _clean(
            info.get("previousClose")
        )
        if rate is not None and price:
            return rate / price * 100

        raw = _clean(info.get("dividendYield"))
        if raw is None:
            return None
        # 日本株の利回りが 30% を超えることはまずないので、
        # 大きい値はすでに % 表記とみなす。
        return raw if raw > 1.0 else raw * 100

    @staticmethod
    def _equity_ratio(ticker_obj: Any) -> float | None:
        """自己資本比率(%)。`.info` に無いので財務諸表から計算する。"""
        try:
            bs = ticker_obj.balance_sheet
        except Exception:
            return None
        if bs is None or getattr(bs, "empty", True):
            return None

        def pick(*labels: str) -> float | None:
            for label in labels:
                if label in bs.index:
                    series = bs.loc[label].dropna()
                    if len(series):
                        return _clean(series.iloc[0])
            return None

        equity = pick(
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Common Stock Equity",
        )
        assets = pick("Total Assets")
        if equity is None or not assets:
            return None
        return equity / assets * 100
