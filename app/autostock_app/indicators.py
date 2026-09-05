"""日足 OHLCV からテクニカル指標を計算する。

`prepare.py` の指標をベースにしつつ、中長期保有を前提にした指標を足している:

    * 52週(250営業日)高値・安値からの位置と最大下落率
    * 200日移動平均乖離率 — 長期トレンドの向き
    * 20日平均売買代金 — 流動性フィルタ。板が薄い銘柄を除外するのに要る

比率はすべて % で返す (0.0532 ではなく 5.32)。単位は fields.py の定義に一致させること。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import INDICATOR_KEYS

#: 最も長い窓。これ未満の履歴しかない銘柄は主要指標が NaN になる。
LONGEST_WINDOW = 250


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # 下落が一度も無い期間は avg_loss=0 で NaN になるが、実質 RSI=100
    return rsi.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line - signal_line


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / width
    bandwidth = width / sma.replace(0.0, np.nan)
    return pct_b, bandwidth


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """1 銘柄の OHLCV から指標を計算する。

    Args:
        prices: date を index に持ち open/high/low/close/volume 列を持つ DataFrame。

    Returns:
        date を index、fields.INDICATOR_KEYS を列に持つ DataFrame。
        履歴が足りない期間は NaN のまま返す (呼び出し側で落とす)。
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=list(INDICATOR_KEYS))

    prices = prices.sort_index()
    close = prices["close"].astype("float64")
    high = prices["high"].astype("float64")
    low = prices["low"].astype("float64")
    volume = prices["volume"].astype("float64")

    out = pd.DataFrame(index=prices.index)

    # --- 株価・流動性 -------------------------------------------------------
    out["close"] = close
    out["volume"] = volume
    # 売買代金 (円) の 20 日平均 -> 百万円
    out["turnover_20d"] = (close * volume).rolling(20).mean() / 1e6

    # --- モメンタム ---------------------------------------------------------
    for days in (5, 20, 60, 120, 250):
        out[f"ret_{days}d"] = close.pct_change(days) * 100

    # --- トレンド (移動平均乖離率) ------------------------------------------
    for days in (20, 60, 200):
        sma = close.rolling(days).mean()
        out[f"sma_dev_{days}"] = (close / sma.replace(0.0, np.nan) - 1.0) * 100

    # --- 高値・安値からの位置 -----------------------------------------------
    for days in (60, 250):
        rolling_high = high.rolling(days).max().replace(0.0, np.nan)
        rolling_low = low.rolling(days).min().replace(0.0, np.nan)
        out[f"high_{days}d_pct"] = (close / rolling_high - 1.0) * 100
        out[f"low_{days}d_pct"] = (close / rolling_low - 1.0) * 100

    # 直近52週で経験した最大ドローダウン。
    # high_250d_pct が「いま高値からどれだけ離れているか」なのに対し、
    # こちらは「その期間で最悪どこまで落ちたか」で、値持ちの良さを測る。
    # 内側の rolling に min_periods を付けないと 250 行揃うまで NaN になり、
    # 外側と合わせて実質 500 営業日ぶんの履歴が要る指標になってしまう。
    running_peak = close.rolling(LONGEST_WINDOW, min_periods=1).max()
    drawdown = close / running_peak - 1.0
    out["max_drawdown_250d"] = (
        drawdown.rolling(LONGEST_WINDOW, min_periods=60).min() * 100
    )

    # --- オシレーター -------------------------------------------------------
    out["rsi_14"] = _rsi(close, 14)
    out["macd_hist"] = _macd_hist(close) / close.replace(0.0, np.nan) * 100

    pct_b, bandwidth = _bollinger(close)
    out["bb_pct_b"] = pct_b * 100  # 0〜100 に正規化 (下限バンド=0, 上限バンド=100)
    out["bb_bandwidth"] = bandwidth * 100

    # --- リスク -------------------------------------------------------------
    daily_return = close.pct_change()
    for days in (20, 60):
        out[f"volatility_{days}d"] = daily_return.rolling(days).std() * 100

    # --- 出来高 -------------------------------------------------------------
    vol_5d = volume.rolling(5).mean()
    vol_20d = volume.rolling(20).mean()
    out["volume_ratio_5_20"] = (vol_5d / vol_20d.replace(0.0, np.nan) - 1.0) * 100

    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "date"

    missing = set(INDICATOR_KEYS) - set(out.columns)
    if missing:  # fields.py と実装がずれたら早期に落とす
        raise RuntimeError(f"indicator columns missing: {sorted(missing)}")
    return out[list(INDICATOR_KEYS)]
