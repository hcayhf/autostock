import numpy as np
import pandas as pd
import pytest

from autostock_app.fields import INDICATOR_KEYS
from autostock_app.indicators import compute_indicators


def make_prices(closes, volumes=None):
    n = len(closes)
    index = pd.bdate_range("2020-01-01", periods=n)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1000.0) if volumes is None else np.asarray(volumes, float),
        },
        index=index,
    )


def test_returns_are_percent_not_ratio():
    # 20 営業日で +10% になる系列
    closes = np.linspace(100, 110, 21)
    out = compute_indicators(make_prices(closes))
    assert out["ret_20d"].iloc[-1] == pytest.approx(10.0, abs=1e-9)


def test_all_declared_columns_are_produced():
    out = compute_indicators(make_prices(np.linspace(100, 200, 300)))
    assert list(out.columns) == list(INDICATOR_KEYS)


def test_rsi_is_100_when_price_only_rises():
    out = compute_indicators(make_prices(np.linspace(100, 300, 120)))
    assert out["rsi_14"].iloc[-1] == pytest.approx(100.0)


def test_rsi_stays_in_range():
    rng = np.random.default_rng(0)
    closes = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    rsi = compute_indicators(make_prices(closes))["rsi_14"].dropna()
    assert len(rsi) > 0
    assert rsi.between(0, 100).all()


def test_turnover_is_in_millions_of_yen():
    # 終値 1000 円 × 出来高 1,000,000 株 = 10 億円 = 1000 百万円
    out = compute_indicators(make_prices([1000.0] * 40, [1_000_000.0] * 40))
    assert out["turnover_20d"].iloc[-1] == pytest.approx(1000.0)


def test_sma_deviation_sign_follows_trend():
    up = compute_indicators(make_prices(np.linspace(100, 400, 300)))
    down = compute_indicators(make_prices(np.linspace(400, 100, 300)))
    assert up["sma_dev_200"].iloc[-1] > 0
    assert down["sma_dev_200"].iloc[-1] < 0


def test_max_drawdown_is_never_positive():
    rng = np.random.default_rng(7)
    closes = 1000 * np.exp(np.cumsum(rng.normal(0, 0.015, 700)))
    dd = compute_indicators(make_prices(closes))["max_drawdown_250d"].dropna()
    assert len(dd) > 0
    assert (dd <= 1e-9).all()


def test_max_drawdown_remembers_a_crash_that_already_recovered():
    """high_250d_pct と max_drawdown_250d が別物であることの確認。

    高値まで戻った銘柄は「高値からの位置」はほぼ 0 になるが、
    途中で 30% 落ちた事実は最大ドローダウンに残る。
    """
    closes = np.concatenate([
        np.full(60, 100.0),            # 助走 (ローリング窓を埋める)
        np.linspace(100, 200, 100),    # 上昇
        np.linspace(200, 140, 30),     # -30% の急落
        np.linspace(140, 200, 100),    # 高値まで戻る
    ])
    row = compute_indicators(make_prices(closes)).iloc[-1]
    assert row["high_250d_pct"] == pytest.approx(0.0, abs=1.5)
    assert row["max_drawdown_250d"] == pytest.approx(-30.0, abs=1.5)


def test_high_pct_is_zero_at_a_new_high():
    out = compute_indicators(make_prices(np.linspace(100, 500, 300)))
    # 終値が高値を更新し続けるので、高値からの位置はほぼ 0
    assert out["high_250d_pct"].iloc[-1] == pytest.approx(0.0, abs=1.0)


def test_empty_input_returns_empty_frame_with_columns():
    out = compute_indicators(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == list(INDICATOR_KEYS)


def test_short_history_leaves_long_windows_null():
    out = compute_indicators(make_prices(np.linspace(100, 110, 30)))
    assert out["ret_250d"].isna().all()
    assert out["ret_20d"].notna().any()


def test_no_infinities_survive():
    # 出来高 0 やゼロ除算になりうる系列を混ぜる
    closes = np.concatenate([np.full(30, 100.0), np.full(30, 100.0)])
    out = compute_indicators(make_prices(closes, volumes=np.zeros(60)))
    assert not np.isinf(out.select_dtypes("number").to_numpy(dtype=float, na_value=0.0)).any()
