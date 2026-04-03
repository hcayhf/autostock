# AutoStock — AIが自律実験を繰り返す日本株スクリーニングモデル

Andrej Karpathy の [AutoResearch](https://github.com/karpathy/autoresearch) を日本株予測に転用した実験プロジェクト。
Claude Code（AI）が `train.py` を自律編集しながら約180回の実験を繰り返し、モデルを最適化した。

**実験結果: top20_return 6.76% → 10.95%（+62%改善）**

## 実験の仕組み

AIが以下のループを自律で繰り返す：

```
train.py を修正（ハイパーパラメータ・特徴量・アーキテクチャ変更）
  ↓
python3 train.py を実行
  ↓
top20_return が改善した？
  ├─ Yes → git commit（改善を記録）
  └─ No  → git checkout（変更を破棄）
  ↓
繰り返す
```

## ファイル構成

| ファイル | 役割 |
|---------|------|
| `prepare.py` | データ取得（yfinance）/ 特徴量生成 / 評価関数（AIが変更しない） |
| `train.py` | LightGBMモデル / ハイパーパラメータ / 訓練ループ（AIが編集） |
| `program.md` | 実験ルール・探索方向性（人間が記述） |
| `plot_results.py` | 実験経過グラフの生成 |

## セットアップ

```bash
pip install -r requirements.txt
```

銘柄一覧ファイル（`data_j.xls`）は [JPX公式サイト](https://www.jpx.co.jp/markets/statistics-equities/misc/01.html) から取得し、プロジェクトルートに配置してください。

株価データは初回実行時に yfinance 経由で自動ダウンロードされます：

```bash
python prepare.py --all   # 全銘柄ダウンロード（時間がかかります）
python train.py           # 学習・評価
```

## 実験結果

| 指標 | ベースライン | 最終ベスト |
|------|------------|----------|
| top20_return | 6.76% | **10.95%** |
| excess_return | — | **7.09%** |
| Sharpe ratio | 0.60 | **0.84** |

訓練期間: 2015-01-01 〜 2023-12-31 / テスト期間: 2024-01-01 〜 2025-12-31

### 主なブレークスルー

1. **Early Stopping の除去** — バリデーション期間のレジーム変化で即終了していた問題を修正
2. **クロスセクショナルランク変換** — 特徴量を日付ごとのパーセンタイルに変換
3. **特徴量の絞り込み** — 26個 → 5個（60日系のみ）
4. **num_leaves=4** — 極端に浅い木が最適（金融データのノイジーさに対応）
5. **マルチクリップ加重アンサンブル** — 異なるclip値のモデルを重み付き平均

### 最終モデル設定

```python
SELECTED_FEATURES = [
    "volatility_60d", "low_60d_pct", "volatility_20d",
    "high_60d_pct", "ret_60d",
]
ENSEMBLE_CLIPS   = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
ENSEMBLE_WEIGHTS = [1, 1, 2, 3, 1, 1]

LGB_PARAMS = {"num_leaves": 4, "learning_rate": 0.01, "seed": 42, ...}
NUM_BOOST_ROUND = 500  # Early Stopping なし
```

## 参考

- [AutoResearch](https://github.com/karpathy/autoresearch) — Andrej Karpathy
- [実験についてのブログ記事](#) — 詳細な考察・失敗事例はこちら

## 免責事項

本ソフトウェアは研究・教育目的で作成されたものであり、投資助言を目的としたものではありません。本ソフトウェアの利用によって生じた損害について、作者は一切の責任を負いません。投資判断はご自身の責任のもとで行ってください。

## License

MIT
