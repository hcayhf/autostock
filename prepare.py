"""
Stock Prediction - AutoResearch 方式の日本株スクリーニング
prepare.py: データ取得・特徴量生成・評価関数 (FIXED - AIが変更しないファイル)

Usage:
    python prepare.py                    # 少数銘柄でテスト (10銘柄)
    python prepare.py --num-stocks 50    # 50銘柄でテスト
    python prepare.py --all              # 全銘柄ダウンロード
    python prepare.py --update           # キャッシュを最新日まで更新
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "ohlcv"
UNIVERSE_FILE = PROJECT_DIR / "data_j.xls"

# 対象市場
TARGET_MARKETS = ["プライム（内国株式）", "スタンダード（内国株式）"]

# データ期間
DATA_START = "2015-01-01"

# ターゲット変数のデフォルト
DEFAULT_FORWARD_DAYS = 40

# 評価のデフォルト
DEFAULT_TOP_N = 20

# ベンチマーク
TOPIX_TICKER = "1306.T"  # TOPIX連動ETF
NIKKEI_TICKER = "^N225"  # 日経225

# 訓練/テスト期間
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# ---------------------------------------------------------------------------
# Universe (銘柄リスト)
# ---------------------------------------------------------------------------

def load_universe() -> pd.DataFrame:
    """
    data_j.xls から東証プライム・スタンダードの内国株式を読み込む。

    Returns:
        DataFrame with columns: code, name, market, sector33_code, sector33,
                                sector17_code, sector17, scale_code, scale
    """
    df = pd.read_excel(UNIVERSE_FILE)
    # プライム＋スタンダードの内国株式のみ
    mask = df["市場・商品区分"].isin(TARGET_MARKETS)
    df = df[mask].copy()

    # カラム名を英語に変換
    df = df.rename(columns={
        "コード": "code",
        "銘柄名": "name",
        "市場・商品区分": "market",
        "33業種コード": "sector33_code",
        "33業種区分": "sector33",
        "17業種コード": "sector17_code",
        "17業種区分": "sector17",
        "規模コード": "scale_code",
        "規模区分": "scale",
    })

    # コードを文字列に（ゼロ埋め4桁）
    df["code"] = df["code"].astype(str).str.zfill(4)
    # yfinance用ティッカー
    df["ticker"] = df["code"] + ".T"

    # 不要列を削除してインデックスをリセット
    df = df[["code", "name", "market", "sector33_code", "sector33",
             "sector17_code", "sector17", "scale_code", "scale", "ticker"]]
    df = df.reset_index(drop=True)

    print(f"Universe: {len(df)} stocks loaded "
          f"(Prime: {(df['market'] == TARGET_MARKETS[0]).sum()}, "
          f"Standard: {(df['market'] == TARGET_MARKETS[1]).sum()})")
    return df


# ---------------------------------------------------------------------------
# Data Download & Cache
# ---------------------------------------------------------------------------

def _ticker_cache_path(ticker: str) -> Path:
    """キャッシュファイルパスを返す。"""
    safe_name = ticker.replace("^", "_").replace(".", "_")
    return CACHE_DIR / f"{safe_name}.csv"


def _download_single(ticker: str, start: str, end: str = None) -> pd.DataFrame | None:
    """
    yfinance で1銘柄の日足データを取得する。
    失敗時は None を返す。
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            return None
        # カラム名を正規化
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        print(f"  [WARN] Failed to download {ticker}: {e}")
        return None


def download_stock_data(
    tickers: list[str],
    start: str = DATA_START,
    end: str = None,
    force: bool = False,
    update: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    複数銘柄の日足データをダウンロードしてキャッシュする。

    Args:
        tickers: ティッカーリスト (例: ["7203.T", "6758.T"])
        start: 開始日
        end: 終了日 (None = 今日)
        force: True ならキャッシュを無視して再ダウンロード
        update: True なら既存キャッシュの最終日から更新

    Returns:
        dict[ticker, DataFrame]
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    download_count = 0
    cache_count = 0
    fail_count = 0

    for i, ticker in enumerate(tickers):
        cache_path = _ticker_cache_path(ticker)

        # キャッシュが存在し、force でない場合
        if cache_path.exists() and not force:
            if update:
                # 既存キャッシュの最終日以降を追加ダウンロード
                existing = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                last_date = existing.index.max()
                new_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                if new_start < (end or datetime.now().strftime("%Y-%m-%d")):
                    new_data = _download_single(ticker, start=new_start, end=end)
                    if new_data is not None and not new_data.empty:
                        combined = pd.concat([existing, new_data])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        combined.to_csv(cache_path)
                        result[ticker] = combined
                        download_count += 1
                    else:
                        result[ticker] = existing
                        cache_count += 1
                else:
                    result[ticker] = existing
                    cache_count += 1
            else:
                result[ticker] = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                cache_count += 1
            continue

        # ダウンロード
        df = _download_single(ticker, start=start, end=end)
        if df is not None and not df.empty:
            df.to_csv(cache_path)
            result[ticker] = df
            download_count += 1
        else:
            fail_count += 1

        # 進捗表示（50銘柄ごと）
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} "
                  f"(downloaded: {download_count}, cached: {cache_count}, failed: {fail_count})")

        # レート制限対策（ダウンロードした場合のみ）
        if df is not None:
            time.sleep(0.1)

    print(f"Data: {len(result)} stocks ready "
          f"(downloaded: {download_count}, cached: {cache_count}, failed: {fail_count})")
    return result


def load_cached_data(tickers: list[str] = None) -> dict[str, pd.DataFrame]:
    """
    キャッシュ済みのデータを読み込む。tickers が None なら全キャッシュを読む。
    """
    if tickers is None:
        csv_files = list(CACHE_DIR.glob("*.csv"))
        result = {}
        for f in csv_files:
            ticker = f.stem.replace("_T", ".T").replace("_N225", "^N225")
            df = pd.read_csv(f, index_col="Date", parse_dates=True)
            result[ticker] = df
        return result

    result = {}
    for ticker in tickers:
        cache_path = _ticker_cache_path(ticker)
        if cache_path.exists():
            result[ticker] = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
    return result


# ---------------------------------------------------------------------------
# Feature Engineering (特徴量生成)
# ---------------------------------------------------------------------------

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) を計算する。"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD (Moving Average Convergence Divergence) を計算する。"""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def _calc_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    """ボリンジャーバンド: %B (位置) と帯幅を返す。"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # %B: 現在値が下限バンドから上限バンドのどこにいるか (0〜1が正常)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    # 帯幅: ボラティリティの指標
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return pct_b, bandwidth


def compute_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    1銘柄のOHLCVから特徴量を計算する。

    Args:
        df: OHLCV DataFrame (Date index, Open/High/Low/Close/Volume columns)

    Returns:
        特徴量DataFrame (Date index)
    """
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    features = pd.DataFrame(index=df.index)

    # --- モメンタム ---
    features["ret_5d"] = close.pct_change(5)
    features["ret_20d"] = close.pct_change(20)
    features["ret_60d"] = close.pct_change(60)

    # --- 出来高 ---
    vol_5d = volume.rolling(5).mean()
    vol_20d = volume.rolling(20).mean()
    features["volume_ratio_5_20"] = (vol_5d / vol_20d.replace(0, np.nan)) - 1

    # --- RSI ---
    features["rsi_14"] = _calc_rsi(close, period=14)

    # --- MACD ---
    macd_line, signal_line, macd_hist = _calc_macd(close)
    features["macd"] = macd_line / close.replace(0, np.nan)  # 株価で正規化
    features["macd_signal"] = signal_line / close.replace(0, np.nan)
    features["macd_hist"] = macd_hist / close.replace(0, np.nan)

    # --- ボリンジャーバンド ---
    pct_b, bandwidth = _calc_bollinger(close)
    features["bb_pct_b"] = pct_b
    features["bb_bandwidth"] = bandwidth

    # --- ボラティリティ ---
    features["volatility_20d"] = close.pct_change().rolling(20).std()
    features["volatility_60d"] = close.pct_change().rolling(60).std()

    # --- 高値/安値からの位置 ---
    features["high_20d_pct"] = close / high.rolling(20).max().replace(0, np.nan) - 1
    features["low_20d_pct"] = close / low.rolling(20).min().replace(0, np.nan) - 1
    features["high_60d_pct"] = close / high.rolling(60).max().replace(0, np.nan) - 1
    features["low_60d_pct"] = close / low.rolling(60).min().replace(0, np.nan) - 1

    # --- 移動平均乖離率 ---
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    sma_60 = close.rolling(60).mean()
    features["sma_dev_5"] = (close - sma_5) / sma_5.replace(0, np.nan)
    features["sma_dev_20"] = (close - sma_20) / sma_20.replace(0, np.nan)
    features["sma_dev_60"] = (close - sma_60) / sma_60.replace(0, np.nan)

    return features


def load_features(
    start_date: str = None,
    end_date: str = None,
    tickers: list[str] = None,
) -> pd.DataFrame:
    """
    全銘柄の特徴量を読み込む。キャッシュ済みOHLCVから計算。

    Args:
        start_date: 開始日 (None = 全期間)
        end_date: 終了日 (None = 全期間)
        tickers: 対象ティッカー (None = 全キャッシュ)

    Returns:
        MultiIndex DataFrame: (Date, ticker) -> features
    """
    stock_data = load_cached_data(tickers)
    if not stock_data:
        raise ValueError("No cached data found. Run `python prepare.py` first.")

    # ベンチマーク読み込み（相対強度の計算用）
    bench_data = {}
    for bench_ticker in [TOPIX_TICKER, NIKKEI_TICKER]:
        bench_path = _ticker_cache_path(bench_ticker)
        if bench_path.exists():
            bench_data[bench_ticker] = pd.read_csv(
                bench_path, index_col="Date", parse_dates=True
            )

    all_features = []

    for ticker, ohlcv in stock_data.items():
        # ベンチマーク銘柄はスキップ
        if ticker in [TOPIX_TICKER, NIKKEI_TICKER]:
            continue

        try:
            feat = compute_features_single(ohlcv)

            # --- 市場相対強度 ---
            close = ohlcv["Close"]
            for bench_name, bench_df in bench_data.items():
                if bench_df is not None and not bench_df.empty:
                    bench_close = bench_df["Close"].reindex(close.index, method="ffill")
                    label = "topix" if "1306" in bench_name else "nikkei"
                    for period in [20, 60]:
                        stock_ret = close.pct_change(period)
                        bench_ret = bench_close.pct_change(period)
                        feat[f"rel_{label}_{period}d"] = stock_ret - bench_ret

            feat["ticker"] = ticker

            # 期間フィルタ
            if start_date:
                feat = feat[feat.index >= start_date]
            if end_date:
                feat = feat[feat.index <= end_date]

            if not feat.empty:
                all_features.append(feat)
        except Exception as e:
            print(f"  [WARN] Feature computation failed for {ticker}: {e}")
            continue

    if not all_features:
        raise ValueError("No features computed. Check your data.")

    result = pd.concat(all_features)
    result = result.set_index("ticker", append=True)
    result.index.names = ["Date", "ticker"]

    # NaN が多すぎる行を除去（特徴量の60%以上がNaN）
    feature_cols = [c for c in result.columns]
    nan_ratio = result[feature_cols].isna().sum(axis=1) / len(feature_cols)
    result = result[nan_ratio < 0.6]

    print(f"Features: {result.shape[0]} rows x {result.shape[1]} columns "
          f"({result.index.get_level_values('ticker').nunique()} stocks)")
    return result


def _compute_sector_relative_strength(
    features: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    セクター内相対強度を追加する。
    load_features の後に呼ぶ。

    Args:
        features: load_features() の戻り値
        universe: load_universe() の戻り値

    Returns:
        セクター相対強度列が追加された features
    """
    # ticker → sector33 のマッピング
    ticker_sector = universe.set_index("ticker")["sector33"].to_dict()

    # セクター列を追加
    tickers = features.index.get_level_values("ticker")
    features["sector33"] = tickers.map(ticker_sector)

    # セクター内の平均リターンとの差
    for ret_col in ["ret_5d", "ret_20d", "ret_60d"]:
        if ret_col in features.columns:
            sector_mean = features.groupby(
                [features.index.get_level_values("Date"), "sector33"]
            )[ret_col].transform("mean")
            features[f"{ret_col}_sector_rel"] = features[ret_col] - sector_mean

    # sector33列は不要なので削除
    features = features.drop(columns=["sector33"])
    return features


# ---------------------------------------------------------------------------
# Target Variable (ターゲット変数)
# ---------------------------------------------------------------------------

def load_targets(
    start_date: str = None,
    end_date: str = None,
    forward_days: int = DEFAULT_FORWARD_DAYS,
    tickers: list[str] = None,
) -> pd.DataFrame:
    """
    N営業日後のリターンをターゲット変数として計算する。

    Args:
        start_date: 開始日
        end_date: 終了日
        forward_days: 何営業日後のリターンか
        tickers: 対象ティッカー (None = 全キャッシュ)

    Returns:
        DataFrame: (Date, ticker) -> forward_return
    """
    stock_data = load_cached_data(tickers)
    if not stock_data:
        raise ValueError("No cached data found. Run `python prepare.py` first.")

    all_targets = []

    for ticker, ohlcv in stock_data.items():
        if ticker in [TOPIX_TICKER, NIKKEI_TICKER]:
            continue

        try:
            close = ohlcv["Close"]
            # N営業日後のリターン (shift でマイナス方向 = 未来のデータ)
            future_close = close.shift(-forward_days)
            forward_return = (future_close / close) - 1

            target_df = pd.DataFrame({
                "forward_return": forward_return,
                "ticker": ticker,
            }, index=ohlcv.index)

            # 期間フィルタ
            if start_date:
                target_df = target_df[target_df.index >= start_date]
            if end_date:
                target_df = target_df[target_df.index <= end_date]

            # NaN (未来データがない直近部分) を除去
            target_df = target_df.dropna(subset=["forward_return"])

            if not target_df.empty:
                all_targets.append(target_df)
        except Exception as e:
            print(f"  [WARN] Target computation failed for {ticker}: {e}")
            continue

    if not all_targets:
        raise ValueError("No targets computed. Check your data.")

    result = pd.concat(all_targets)
    result = result.set_index("ticker", append=True)
    result.index.names = ["Date", "ticker"]

    print(f"Targets: {result.shape[0]} rows "
          f"({result.index.get_level_values('ticker').nunique()} stocks, "
          f"forward_days={forward_days})")
    return result


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def get_train_test_split(
    forward_days: int = DEFAULT_FORWARD_DAYS,
    tickers: list[str] = None,
    include_sector_rel: bool = True,
) -> dict:
    """
    訓練/テストデータを分割して返す。

    Args:
        forward_days: ターゲット変数の先読み日数
        tickers: 対象ティッカー (None = 全キャッシュ)
        include_sector_rel: セクター相対強度を含めるか

    Returns:
        dict with keys:
            X_train, y_train, X_test, y_test,
            feature_names, train_index, test_index
    """
    # 特徴量
    features_train = load_features(TRAIN_START, TRAIN_END, tickers)
    features_test = load_features(TEST_START, TEST_END, tickers)

    # セクター相対強度
    if include_sector_rel:
        try:
            universe = load_universe()
            features_train = _compute_sector_relative_strength(features_train, universe)
            features_test = _compute_sector_relative_strength(features_test, universe)
        except Exception as e:
            print(f"  [WARN] Sector relative strength failed: {e}")

    # ターゲット
    targets_train = load_targets(TRAIN_START, TRAIN_END, forward_days, tickers)
    targets_test = load_targets(TEST_START, TEST_END, forward_days, tickers)

    # 共通インデックスで結合
    train_idx = features_train.index.intersection(targets_train.index)
    test_idx = features_test.index.intersection(targets_test.index)

    X_train = features_train.loc[train_idx]
    y_train = targets_train.loc[train_idx, "forward_return"]
    X_test = features_test.loc[test_idx]
    y_test = targets_test.loc[test_idx, "forward_return"]

    feature_names = list(X_train.columns)

    print(f"\nTrain/Test Split:")
    print(f"  Train: {X_train.shape[0]} rows ({TRAIN_START} ~ {TRAIN_END})")
    print(f"  Test:  {X_test.shape[0]} rows ({TEST_START} ~ {TEST_END})")
    print(f"  Features: {len(feature_names)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "train_index": train_idx,
        "test_index": test_idx,
    }


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(
    predictions: pd.Series,
    actuals: pd.Series,
    top_n: int = DEFAULT_TOP_N,
    benchmark_returns: pd.Series = None,
) -> dict:
    """
    スコアリングモデルの評価関数。

    Args:
        predictions: モデルのスコア (index = (Date, ticker))
        actuals: 実際のN日後リターン (index = (Date, ticker))
        top_n: 上位何銘柄を評価するか
        benchmark_returns: ベンチマーク（TOPIX）のリターン

    Returns:
        dict: 評価指標
            - top_n_return: 上位N銘柄の平均リターン
            - excess_return: TOPIX対比の超過リターン (benchmark_returns指定時)
            - sharpe_ratio: Sharpe Ratio
            - hit_rate: 上位N銘柄のうちプラスリターンの割合
            - bottom_n_return: 下位N銘柄の平均リターン (参考)
            - all_return: 全銘柄平均リターン (ベースライン)
            - long_short_return: 上位 - 下位の差
    """
    # predictions と actuals の共通インデックス
    common_idx = predictions.index.intersection(actuals.index)
    if len(common_idx) == 0:
        raise ValueError("No common index between predictions and actuals.")

    preds = predictions.loc[common_idx]
    acts = actuals.loc[common_idx]

    # 日付ごとに評価
    dates = preds.index.get_level_values("Date").unique()
    daily_metrics = []

    for date in dates:
        try:
            day_preds = preds.xs(date, level="Date")
            day_acts = acts.xs(date, level="Date")
        except KeyError:
            continue

        if len(day_preds) < top_n * 2:
            continue

        # 上位N銘柄
        top_tickers = day_preds.nlargest(top_n).index
        top_return = day_acts.loc[top_tickers].mean()

        # 下位N銘柄
        bottom_tickers = day_preds.nsmallest(top_n).index
        bottom_return = day_acts.loc[bottom_tickers].mean()

        # 全銘柄平均
        all_return = day_acts.mean()

        # ヒット率
        hit_rate = (day_acts.loc[top_tickers] > 0).mean()

        daily_metrics.append({
            "date": date,
            "top_return": top_return,
            "bottom_return": bottom_return,
            "all_return": all_return,
            "hit_rate": hit_rate,
            "long_short": top_return - bottom_return,
        })

    if not daily_metrics:
        return {
            "top_n_return": np.nan,
            "excess_return": np.nan,
            "sharpe_ratio": np.nan,
            "hit_rate": np.nan,
            "bottom_n_return": np.nan,
            "all_return": np.nan,
            "long_short_return": np.nan,
            "num_eval_dates": 0,
        }

    metrics_df = pd.DataFrame(daily_metrics)

    top_n_return = metrics_df["top_return"].mean()
    bottom_n_return = metrics_df["bottom_return"].mean()
    all_return = metrics_df["all_return"].mean()
    hit_rate = metrics_df["hit_rate"].mean()
    long_short_return = metrics_df["long_short"].mean()

    # Sharpe Ratio (日次の上位N銘柄リターンから計算)
    top_returns = metrics_df["top_return"]
    sharpe = (top_returns.mean() / top_returns.std()) if top_returns.std() > 0 else 0.0

    # 超過リターン
    excess_return = top_n_return - all_return

    # ベンチマーク対比 (指定時)
    if benchmark_returns is not None:
        bench_mean = benchmark_returns.mean()
        excess_vs_bench = top_n_return - bench_mean
    else:
        excess_vs_bench = excess_return

    result = {
        "top_n_return": float(top_n_return),
        "excess_return": float(excess_vs_bench),
        "sharpe_ratio": float(sharpe),
        "hit_rate": float(hit_rate),
        "bottom_n_return": float(bottom_n_return),
        "all_return": float(all_return),
        "long_short_return": float(long_short_return),
        "num_eval_dates": len(daily_metrics),
    }

    return result


def print_evaluation(metrics: dict, top_n: int = DEFAULT_TOP_N):
    """評価結果を見やすく表示する。"""
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  Top-{top_n} Avg Return:    {metrics['top_n_return']:.4f} "
          f"({metrics['top_n_return'] * 100:.2f}%)")
    print(f"  Excess Return:           {metrics['excess_return']:.4f} "
          f"({metrics['excess_return'] * 100:.2f}%)")
    print(f"  Sharpe Ratio:            {metrics['sharpe_ratio']:.4f}")
    print(f"  Hit Rate:                {metrics['hit_rate']:.4f} "
          f"({metrics['hit_rate'] * 100:.1f}%)")
    print(f"  Bottom-{top_n} Avg Return: {metrics['bottom_n_return']:.4f} "
          f"({metrics['bottom_n_return'] * 100:.2f}%)")
    print(f"  All Stocks Avg Return:   {metrics['all_return']:.4f} "
          f"({metrics['all_return'] * 100:.2f}%)")
    print(f"  Long-Short Return:       {metrics['long_short_return']:.4f} "
          f"({metrics['long_short_return'] * 100:.2f}%)")
    print(f"  Eval Dates:              {metrics['num_eval_dates']}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main (テスト用)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Japanese stock data for AutoResearch"
    )
    parser.add_argument(
        "--num-stocks", type=int, default=10,
        help="Number of stocks to download for testing (default: 10)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all stocks in the universe"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Update existing cache to latest date"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download (ignore cache)"
    )
    args = parser.parse_args()

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print()

    # Step 1: 銘柄リスト読み込み
    print("=" * 50)
    print("Step 1: Loading universe")
    print("=" * 50)
    universe = load_universe()
    print()

    # Step 2: 対象銘柄を選択
    if args.all:
        target_tickers = universe["ticker"].tolist()
    else:
        # テスト用: 代表銘柄を優先的に選択（大型株）
        large_caps = universe[universe["scale_code"].astype(str).isin(["1", "2"])]
        if len(large_caps) >= args.num_stocks:
            target_tickers = large_caps["ticker"].head(args.num_stocks).tolist()
        else:
            target_tickers = universe["ticker"].head(args.num_stocks).tolist()
    print(f"Target: {len(target_tickers)} stocks")
    print(f"Tickers: {target_tickers[:10]}{'...' if len(target_tickers) > 10 else ''}")
    print()

    # ベンチマークも追加
    benchmark_tickers = [TOPIX_TICKER, NIKKEI_TICKER]
    all_tickers = target_tickers + benchmark_tickers

    # Step 3: データダウンロード
    print("=" * 50)
    print("Step 2: Downloading stock data")
    print("=" * 50)
    stock_data = download_stock_data(
        all_tickers,
        start=DATA_START,
        force=args.force,
        update=args.update,
    )
    print()

    # Step 4: 特徴量テスト
    print("=" * 50)
    print("Step 3: Computing features (test)")
    print("=" * 50)
    features = load_features(tickers=target_tickers)
    print(f"Feature columns: {list(features.columns)}")
    print(f"Sample:\n{features.head(3)}")
    print()

    # Step 5: ターゲット変数テスト
    print("=" * 50)
    print("Step 4: Computing targets (test)")
    print("=" * 50)
    targets = load_targets(forward_days=DEFAULT_FORWARD_DAYS, tickers=target_tickers)
    print(f"Target stats:\n{targets['forward_return'].describe()}")
    print()

    # Step 6: 評価関数テスト (ランダムスコアで)
    print("=" * 50)
    print("Step 5: Evaluation test (random baseline)")
    print("=" * 50)
    common_idx = features.index.intersection(targets.index)
    if len(common_idx) > 0:
        random_scores = pd.Series(
            np.random.randn(len(common_idx)),
            index=common_idx,
            name="score"
        )
        actual_returns = targets.loc[common_idx, "forward_return"]
        # テスト時は銘柄数に合わせて top_n を調整
        n_stocks = len(target_tickers)
        test_top_n = min(DEFAULT_TOP_N, max(1, n_stocks // 3))
        metrics = evaluate(random_scores, actual_returns, top_n=test_top_n)
        print_evaluation(metrics, top_n=test_top_n)
    else:
        print("  [WARN] No common index for evaluation test.")

    print()
    print("Done! Data is ready for training.")


if __name__ == "__main__":
    main()
