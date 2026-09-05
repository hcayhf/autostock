"""オフライン用の合成データプロバイダ。

外部ネットワークに出られない環境 (CI、サンドボックス、機内) でも
「取得 → 指標計算 → 検索」の一連の流れを最後まで動かせるようにするためのもの。
ティッカーから決定論的に系列を生成するので、同じ引数なら常に同じ値になる。

**これは実在の株価ではない。投資判断に使わないこと。**
"""

from __future__ import annotations

import zlib
from datetime import date

import numpy as np
import pandas as pd

_MARKETS = ("プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）")
_SECTORS33 = (
    "水産・農林業", "建設業", "食料品", "化学", "医薬品", "鉄鋼", "機械",
    "電気機器", "輸送用機器", "精密機器", "情報・通信業", "卸売業",
    "小売業", "銀行業", "証券、商品先物取引業", "保険業", "不動産業", "サービス業",
)
_SECTORS17 = (
    "食品", "エネルギー資源", "建設・資材", "素材・化学", "医薬品",
    "自動車・輸送機", "鉄鋼・非鉄", "機械", "電機・精密", "情報通信・サービスその他",
    "電力・ガス", "運輸・物流", "商社・卸売", "小売", "銀行", "金融（除く銀行）", "不動産",
)
_SCALES = ("TOPIX Core30", "TOPIX Large70", "TOPIX Mid400", "TOPIX Small 1", "TOPIX Small 2", "-")


def _rng(*parts: str | int) -> np.random.Generator:
    """引数から決定論的に乱数生成器を作る。"""
    seed = zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))
    return np.random.default_rng(seed)


class SyntheticProvider:
    """PriceProvider / FundamentalsProvider の両方を満たすダミー実装。"""

    name = "synthetic"

    def fetch_prices(
        self, ticker: str, start: str, end: str | None = None
    ) -> pd.DataFrame | None:
        end = end or date.today().isoformat()
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) == 0:
            return None

        rng = _rng("price", ticker)
        n = len(dates)

        # 銘柄ごとにドリフトとボラティリティを散らす。
        # 一部の銘柄がはっきり上昇/下落トレンドを持つようにして、
        # スクリーニング条件が意味のある差を返すようにしている。
        drift = rng.normal(0.00015, 0.00025)
        vol = float(np.clip(rng.gamma(shape=4.0, scale=0.004), 0.006, 0.055))

        shocks = rng.normal(drift, vol, size=n)
        # 緩やかなレジーム変化 (数か月周期のサイクル) を重ねる
        cycle = 0.0009 * np.sin(np.linspace(0, rng.uniform(3, 12) * np.pi, n))
        close = float(rng.uniform(300, 9000)) * np.exp(np.cumsum(shocks + cycle))

        intraday = np.abs(rng.normal(0, vol * 0.7, size=n))
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = low + (high - low) * rng.uniform(0, 1, size=n)

        base_volume = float(rng.uniform(3e4, 6e6))
        volume = base_volume * np.exp(rng.normal(0, 0.45, size=n))

        return pd.DataFrame(
            {
                "open": np.round(open_, 1),
                "high": np.round(high, 1),
                "low": np.round(low, 1),
                "close": np.round(close, 1),
                "volume": np.round(volume, 0),
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )

    def fetch_fundamentals(
        self, ticker: str, *, close: float | None = None, deep: bool = False
    ) -> dict[str, float | None] | None:
        rng = _rng("fundamentals", ticker)
        price = close if close else float(rng.uniform(300, 9000))

        eps = price / float(np.clip(rng.gamma(6.0, 3.0), 3.0, 90.0))
        bps = price / float(np.clip(rng.gamma(4.0, 0.45), 0.25, 8.0))
        shares = float(rng.uniform(5e6, 1.5e9))

        # 1割ほどは赤字企業として PER を欠損させ、欠損の扱いを検証できるようにする
        loss_making = rng.random() < 0.10

        return {
            "market_cap": price * shares / 1e8,
            "per": None if loss_making else price / eps,
            "pbr": price / bps,
            "dividend_yield": float(np.clip(rng.gamma(2.0, 1.1), 0.0, 9.0)),
            "roe": None if loss_making else float(np.clip(rng.normal(9.0, 6.0), -25.0, 45.0)),
            "equity_ratio": float(np.clip(rng.normal(52.0, 18.0), 3.0, 96.0)),
            "eps": None if loss_making else eps,
            "bps": bps,
        }


def synthetic_universe(n: int = 200) -> pd.DataFrame:
    """data_j.xls が無い環境用のダミー銘柄マスタ。"""
    rng = _rng("universe", n)
    rows = []
    for i in range(n):
        code = f"{1300 + i * 7:04d}"
        s33 = _SECTORS33[i % len(_SECTORS33)]
        rows.append(
            {
                "code": code,
                "name": f"サンプル商事{code}",
                "market": _MARKETS[int(rng.integers(0, len(_MARKETS)))],
                "sector33_code": f"{(i % len(_SECTORS33)) + 1:04d}",
                "sector33": s33,
                "sector17_code": f"{(i % len(_SECTORS17)) + 1:02d}",
                "sector17": _SECTORS17[i % len(_SECTORS17)],
                "scale_code": str(int(rng.integers(1, 8))),
                "scale": _SCALES[int(rng.integers(0, len(_SCALES)))],
                "ticker": f"{code}.T",
            }
        )
    return pd.DataFrame(rows)
