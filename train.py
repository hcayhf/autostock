"""
Stock Prediction - AutoResearch 方式の日本株スクリーニング
train.py: スコアリングモデル（LightGBM）+ 訓練ループ (AIが編集するファイル)

Usage: python train.py
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from prepare import (
    get_train_test_split,
    load_universe,
    evaluate,
    print_evaluation,
    DEFAULT_TOP_N,
    DEFAULT_FORWARD_DAYS,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyperparameters (AIが編集するセクション)
# ---------------------------------------------------------------------------

FORWARD_DAYS = DEFAULT_FORWARD_DAYS  # ターゲット: N営業日後リターン
TOP_N = DEFAULT_TOP_N                # 上位何銘柄を評価するか
INCLUDE_SECTOR_REL = True            # セクター相対強度を含めるか
PRIME_ONLY = True                    # メモリ制約: プライム大型・中型のみ使用

# LightGBM ハイパーパラメータ
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 7,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# Feature Selection (AIが編集するセクション)
# ---------------------------------------------------------------------------

# None = 全特徴量を使用。リストで指定すると選択的に使用
# 上位7特徴量（imp>=2550）
SELECTED_FEATURES = [
    "volatility_60d", "low_60d_pct", "volatility_20d", "ret_60d_sector_rel",
    "high_60d_pct", "ret_60d", "macd",
]

# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, feature_names):
    """
    LightGBMモデルを訓練する。

    Args:
        X_train: 訓練特徴量 (DataFrame, MultiIndex: Date x ticker)
        y_train: 訓練ターゲット (Series)
        feature_names: 特徴量名リスト

    Returns:
        trained model (lgb.Booster)
    """
    # 特徴量選択
    if SELECTED_FEATURES is not None:
        use_features = [f for f in SELECTED_FEATURES if f in feature_names]
    else:
        use_features = feature_names

    # 特徴量をクロスセクショナルランク化（日付ごと、外れ値耐性UP）
    X_ranked = X_train[use_features].groupby(level="Date").rank(pct=True)
    X = X_ranked.values
    # ターゲットをクロスセクショナルランク（日付ごと）に変換
    y = y_train.groupby(level="Date").rank(pct=True)

    # NaN を処理（LightGBMはNaN対応だがinf対策）
    X = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)

    # 訓練/検証を時系列で分割（訓練期間の最後20%を検証に使用）
    dates = X_train.index.get_level_values("Date")
    unique_dates = dates.unique().sort_values()
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_mask = dates < split_date
    val_mask = dates >= split_date

    dtrain = lgb.Dataset(X[train_mask], label=y[train_mask], feature_name=use_features)
    dval = lgb.Dataset(X[val_mask], label=y[val_mask], feature_name=use_features, reference=dtrain)

    print(f"Training LightGBM...")
    print(f"  Features: {len(use_features)}")
    print(f"  Train samples: {train_mask.sum()}")
    print(f"  Val samples: {val_mask.sum()}")
    print(f"  Split date: {split_date}")

    callbacks = [
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=callbacks,
    )

    print(f"  Rounds trained: {model.num_trees()}")

    return model, use_features


# ---------------------------------------------------------------------------
# Prediction & Scoring
# ---------------------------------------------------------------------------

def predict_scores(model, X, feature_names):
    """
    モデルでスコアを予測する。

    Args:
        model: 訓練済みLightGBMモデル
        X: 特徴量DataFrame (MultiIndex: Date x ticker)
        feature_names: 使用した特徴量名リスト

    Returns:
        pd.Series: スコア (index = X.index)
    """
    X_ranked = X[feature_names].groupby(level="Date").rank(pct=True)
    X_vals = X_ranked.values
    X_vals = np.nan_to_num(X_vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
    scores = model.predict(X_vals)
    return pd.Series(scores, index=X.index, name="score")


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def print_feature_importance(model, feature_names, top_k=15):
    """特徴量重要度を表示する。"""
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    print(f"\nTop-{top_k} Feature Importance (gain):")
    for i, row in feat_imp.head(top_k).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 60)
    print("  Stock Prediction - AutoResearch Train")
    print("=" * 60)
    print(f"  Forward days: {FORWARD_DAYS}")
    print(f"  Top N: {TOP_N}")
    print()

    # --- Step 1: データ取得 ---
    print("Step 1: Loading train/test data...")
    tickers = None
    if PRIME_ONLY:
        universe = load_universe()
        # メモリ制約: Large70 + Core30 + Mid400 のみ（約500銘柄）
        large_mid_scales = ["TOPIX Large70", "TOPIX Core30", "TOPIX Mid400"]
        prime_mask = universe["market"] == "プライム（内国株式）"
        scale_mask = universe["scale"].isin(large_mid_scales)
        tickers = universe[prime_mask & scale_mask]["ticker"].tolist()
        print(f"  Using Prime Large+Mid only: {len(tickers)} tickers")
    data = get_train_test_split(
        forward_days=FORWARD_DAYS,
        tickers=tickers,
        include_sector_rel=INCLUDE_SECTOR_REL,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    print()

    # --- Step 2: モデル訓練 ---
    print("Step 2: Training model...")
    model, used_features = train_model(X_train, y_train, feature_names)
    print()

    # --- Step 3: 特徴量重要度 ---
    print("Step 3: Feature importance")
    print_feature_importance(model, used_features)
    print()

    # --- Step 4: テストデータで予測 ---
    print("Step 4: Predicting on test data...")
    test_scores = predict_scores(model, X_test, used_features)
    print(f"  Test predictions: {len(test_scores)}")
    print()

    # --- Step 5: 評価 ---
    print("Step 5: Evaluation on test period")
    # 銘柄数に応じてtop_nを調整（evaluate は top_n*2 銘柄が必要）
    n_test_stocks = X_test.index.get_level_values("ticker").nunique()
    effective_top_n = min(TOP_N, max(1, n_test_stocks // 3))
    if effective_top_n != TOP_N:
        print(f"  [NOTE] Adjusted top_n from {TOP_N} to {effective_top_n} "
              f"(only {n_test_stocks} stocks available)")

    metrics = evaluate(
        predictions=test_scores,
        actuals=y_test,
        top_n=effective_top_n,
    )
    print_evaluation(metrics, top_n=effective_top_n)

    # --- Step 6: 訓練データでの評価（参考・過学習チェック） ---
    print("\n[Reference] Evaluation on train period:")
    train_scores = predict_scores(model, X_train, used_features)
    train_metrics = evaluate(
        predictions=train_scores,
        actuals=y_train,
        top_n=effective_top_n,
    )
    print_evaluation(train_metrics, top_n=effective_top_n)

    # --- 結果出力（AutoResearchループ用） ---
    t_end = time.time()
    elapsed = t_end - t_start

    print(f"\nElapsed time: {elapsed:.1f}s")

    print("\n=== RESULT ===")
    print(f"top20_return: {metrics['top_n_return']:.4f}")
    print(f"excess_return: {metrics['excess_return']:.4f}")
    print(f"sharpe_ratio: {metrics['sharpe_ratio']:.4f}")
    print("===")


if __name__ == "__main__":
    main()
