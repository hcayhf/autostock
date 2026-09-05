"""データ取得元のインターフェース。

株価とファンダメンタルズを別プロトコルに分けてあるのは、
両者で使いたいデータ元が違うため。株価は yfinance で十分だが、
ファンダメンタルズは yfinance だと「現在値スナップショット」しか
取れず履歴が作れない。将来 J-Quants API や EDINET のアダプタを
足すときは FundamentalsProvider だけを差し替えれば済む。

新しいプロバイダの追加手順:
    1. このモジュールの Protocol を満たすクラスを書く
    2. providers/__init__.py の PRICE_PROVIDERS / FUNDAMENTALS_PROVIDERS に登録
    3. CLI から --price-provider / --fundamentals-provider で選ぶ
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

#: fetch_prices が返す DataFrame の列。index は tz-naive な DatetimeIndex。
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@runtime_checkable
class PriceProvider(Protocol):
    """日足 OHLCV の取得元。"""

    name: str

    def fetch_prices(
        self, ticker: str, start: str, end: str | None = None
    ) -> pd.DataFrame | None:
        """1 銘柄の日足を返す。取得できなければ None。

        返す DataFrame は PRICE_COLUMNS の列を持ち、配当・分割調整済みで、
        index は日付 (tz 情報なし) であること。
        """
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """財務指標の取得元。"""

    name: str

    def fetch_fundamentals(
        self, ticker: str, *, close: float | None = None, deep: bool = False
    ) -> dict[str, float | None] | None:
        """1 銘柄のファンダメンタルズを返す。取得できなければ None。

        Args:
            ticker: ティッカー (例 "7203.T")
            close:  直近終値。配当利回りを自前で計算するのに使う。
            deep:   True なら財務諸表も取りに行く (自己資本比率など)。遅い。

        Returns:
            fields.FUNDAMENTAL_KEYS のサブセットをキーに持つ dict。
            値の単位は fields.py の定義に従う (時価総額=億円, 利回り=% など)。
        """
        ...


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """プロバイダごとに揺れる列名・tz を PRICE_COLUMNS に揃える。"""
    if df is None or len(df) == 0:
        return None

    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        return None

    df = df[list(PRICE_COLUMNS)].copy()

    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    df.index.name = "date"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    # 終値が無い行 (休場・未約定) は指標計算を壊すので落とす
    df = df[df["close"].notna() & (df["close"] > 0)]
    return df if len(df) else None
