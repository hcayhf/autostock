"""データ取得プロバイダのレジストリ。"""

from __future__ import annotations

from .base import FundamentalsProvider, PriceProvider, normalize_price_frame
from .synthetic import SyntheticProvider, synthetic_universe
from .yfinance_provider import YFinanceProvider

PRICE_PROVIDERS: dict[str, type] = {
    "yfinance": YFinanceProvider,
    "synthetic": SyntheticProvider,
}

FUNDAMENTALS_PROVIDERS: dict[str, type] = {
    "yfinance": YFinanceProvider,
    "synthetic": SyntheticProvider,
}

DEFAULT_PROVIDER = "yfinance"


def get_price_provider(name: str = DEFAULT_PROVIDER) -> PriceProvider:
    try:
        return PRICE_PROVIDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown price provider: {name!r} "
            f"(available: {', '.join(sorted(PRICE_PROVIDERS))})"
        ) from None


def get_fundamentals_provider(name: str = DEFAULT_PROVIDER) -> FundamentalsProvider:
    try:
        return FUNDAMENTALS_PROVIDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown fundamentals provider: {name!r} "
            f"(available: {', '.join(sorted(FUNDAMENTALS_PROVIDERS))})"
        ) from None


__all__ = [
    "FundamentalsProvider",
    "PriceProvider",
    "SyntheticProvider",
    "YFinanceProvider",
    "get_fundamentals_provider",
    "get_price_provider",
    "normalize_price_frame",
    "synthetic_universe",
]
