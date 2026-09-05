# AutoStock スクリーナー

日本株の日足を毎日取り込み、テクニカル指標とファンダメンタルズを組み合わせて
銘柄を検索するアプリケーション。

リポジトリ直下の `prepare.py` / `train.py`（AutoResearch 方式の実験コード）とは
独立していて、そちらのファイルには一切手を入れていない。

```
取り込み: JPX 銘柄一覧 → 日足 OHLCV → テクニカル指標 → ファンダメンタルズ → 検索スナップショット
検索:     スナップショット 1 テーブルへのクエリのみ（Web UI / REST API / CLI）
```

---

## 5 分で動かす（ネットワーク不要）

合成データで一通り触れる。**実在の株価ではない**ので投資判断には使わないこと。

```bash
pip install -r app/requirements.txt
cd app
python -m autostock_app.cli --db ../data/demo.db demo-seed --stocks 150
python -m autostock_app.cli --db ../data/demo.db serve      # http://127.0.0.1:8000
```

---

## 実データで動かす

### 1. 銘柄一覧を用意する

[JPX の「東証上場銘柄一覧」](https://www.jpx.co.jp/markets/statistics-equities/misc/01.html)
から `data_j.xls` をダウンロードし、リポジトリ直下に置く。

### 2. 取り込む

```bash
cd app
python -m autostock_app.cli ingest universe   # 銘柄マスタ
python -m autostock_app.cli ingest daily      # 株価 → 指標 → 財務 → スナップショット
```

初回は全銘柄 × 約 10 年分を取りに行くので **数十分から数時間** かかる。
まず動作確認したいときは `--limit 50` を付ける。

2 回目以降は各銘柄の保存済み最終日の翌日からしか取得しないので、
日次運用なら通常 10 分程度で終わる。

### 3. 検索する

```bash
# Web UI
python -m autostock_app.cli serve

# CLI
python -m autostock_app.cli screen --max per=15 --min dividend_yield=3
python -m autostock_app.cli screen --preset pullback
```

### 4. 毎日回す

```cron
0 19 * * 1-5  cd /path/to/autostock/app && python -m autostock_app.cli ingest daily >> ../data/ingest.log 2>&1
```

---

## データソースについて（重要）

### 株価は問題ない

yfinance の日足は配当・分割調整済みで、2015 年以降を全銘柄ぶん取得できる。

### ファンダメンタルズには履歴がない

yfinance の `.info` が返すのは **「いま現在」のスナップショットだけ**で、
過去の PER や PBR は取得できない。この制約から次のことが言える。

| やりたいこと | できるか |
|---|---|
| 今 PER15 倍以下・配当利回り 3% 以上の銘柄を探す | できる |
| 過去 5 年で PER が最低水準の銘柄を探す | **できない**（履歴がない） |
| 「2024 年 6 月時点でこの条件だと何が出たか」を検証する | **できない** |

緩和策として、`fundamentals` テーブルは取得日 (`as_of`) をキーに含めていて、
日次で回すと **実行した日から先の履歴が自前で貯まっていく**。
過去に遡ることはできないが、運用を続ければ検証できる範囲は広がる。

自己資本比率だけは `.info` に無いため、`--deep` を付けたときだけ
財務諸表を追加取得して計算する（銘柄ごとにもう 1 リクエスト増えるので遅い）。

### 履歴が本当に必要になったら

`providers/base.py` の `FundamentalsProvider` プロトコルを満たすクラスを足し、
`providers/__init__.py` の `FUNDAMENTALS_PROVIDERS` に登録すれば
`--fundamentals-provider` で切り替えられる。株価側の実装には影響しない。

有力な移行先は [J-Quants API](https://jpx-jquants.com/)（JPX 公式・無料プランあり）。
財務諸表と PER/PBR の履歴が正式に提供されている。

---

## コマンド

| コマンド | 説明 |
|---|---|
| `init-db` | DB とテーブルを作る |
| `status` | 各テーブルの行数・データの鮮度・直近の取り込み履歴 |
| `fields` | 検索できる指標の一覧（単位と説明つき） |
| `presets` | プリセット条件の一覧 |
| `ingest universe` | JPX の銘柄一覧を取り込む |
| `ingest prices` | 日足を差分取得する（`--full` で全期間取り直し） |
| `ingest indicators` | 株価から指標を再計算する |
| `ingest fundamentals` | 財務指標を取得日つきで追記する（`--deep` で自己資本比率も） |
| `ingest snapshot` | 検索スナップショットを作り直す |
| `ingest daily` | 上記を通しで実行する（日次運用はこれ 1 本） |
| `demo-seed` | 合成データでデモ用 DB を作る |
| `screen` | 銘柄を検索する |
| `serve` | Web UI と API を起動する |

共通オプション: `--db`（DB のパス）、`--limit`（先頭 N 銘柄のみ）、`--codes`（銘柄指定）。

### 検索の指定方法

```bash
# 数値レンジ。--min / --max は繰り返せる
python -m autostock_app.cli screen \
  --max per=15 --max pbr=1.5 --min dividend_yield=3 --min turnover_20d=100 \
  --sort dividend_yield --limit 30

# カテゴリで絞る
python -m autostock_app.cli screen --sector 情報・通信業 医薬品 --market "プライム（内国株式）"

# JSON で受け取る（他のツールに繋ぐとき）
python -m autostock_app.cli screen --preset value_dividend --json
```

指定できる指標名は `python -m autostock_app.cli fields` で確認できる。

---

## API

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/api/health` | 稼働確認とデータ件数 |
| GET | `/api/meta` | 指標定義・カテゴリ選択肢・プリセット・データ鮮度 |
| POST | `/api/screen` | スクリーニング |
| GET | `/api/stocks/{code}` | 個別銘柄の詳細と株価推移 |

```bash
curl -X POST http://127.0.0.1:8000/api/screen \
  -H 'Content-Type: application/json' \
  -d '{"ranges":[{"field":"per","max":15},{"field":"roe","min":10}],
       "sort_by":"roe","limit":20}'
```

`ranges` の `include_null` を `true` にすると、その指標が欠損している銘柄も残る
（既定は除外。赤字で PER が無い銘柄などが該当する）。

---

## 指標について

数値は **検索窓にそのまま打ち込む単位** で格納している。
比率は %（`0.0532` ではなく `5.32`）、時価総額は億円、売買代金は百万円。
SQL を直接叩いても直感どおりの値が出る。

中長期保有を前提に、実験用コードの指標へ次を追加してある。

| 指標 | 用途 |
|---|---|
| `sma_dev_200` | 200 日線乖離率。長期トレンドの向き |
| `high_250d_pct` / `low_250d_pct` | 52 週高値・安値からの位置 |
| `max_drawdown_250d` | 直近 52 週で経験した最大の下落幅。値持ちの良さ |
| `ret_120d` / `ret_250d` | 中期・年間の騰落率 |
| `turnover_20d` | 20 日平均売買代金。板の薄い銘柄を落とす流動性フィルタ |

`high_250d_pct` が「いま高値からどれだけ離れているか」なのに対し、
`max_drawdown_250d` は「その 1 年で最悪どこまで落ちたか」で、両者は別物。
高値まで戻った銘柄は前者がほぼ 0 でも、後者には下落の履歴が残る。

---

## 構成

```
app/
  autostock_app/
    fields.py       検索できるカラムの単一の定義元（DB・API・UI が全部これを見る）
    config.py       パスと設定（環境変数で上書き可）
    db.py           SQLite スキーマ
    universe.py     JPX 銘柄一覧の読み込み
    indicators.py   テクニカル指標の計算
    ingest.py       取り込みパイプライン
    screener.py     検索クエリの組み立てとプリセット
    api.py          FastAPI
    cli.py          コマンドライン
    demo.py         合成データの投入
    providers/      データ取得元（yfinance / synthetic）
    web/            Web UI（依存ライブラリなしの HTML + CSS + JS）
  tests/
```

### 設計上の要点

**検索スナップショット** — 銘柄マスタ・最新指標・最新ファンダを 1 テーブルに畳んでいる。
株価テーブルが 1,000 万行を超えても、検索は銘柄数（数千行）のスキャンで済む。

**カラム定義の一元化** — `fields.py` に 1 行足せば、DB スキーマ・検索の許可リスト・
API のメタ情報・UI のプルダウンにすべて反映される。

**SQL の安全性** — カラム名は `fields.py` のホワイトリストで検証してからでないと
SQL に埋め込まない。値は必ずバインドする。未知のカラム名は API では 400 になる。

### 環境変数

| 変数 | 既定値 | 用途 |
|---|---|---|
| `AUTOSTOCK_DB` | `data/autostock.db` | SQLite のパス |
| `AUTOSTOCK_DATA_DIR` | `data/` | データ置き場 |
| `AUTOSTOCK_UNIVERSE_FILE` | `data_j.xls` | 銘柄一覧 |
| `AUTOSTOCK_DATA_START` | `2015-01-01` | 株価の取得開始日 |
| `AUTOSTOCK_REQUEST_SLEEP` | `0.12` | 銘柄あたりの待ち時間（秒） |
| `AUTOSTOCK_MAX_STALE_DAYS` | `30` | 株価がこれ以上古い銘柄は検索対象から外す |

---

## テスト

```bash
cd app && python -m pytest
```

外部ネットワークには一切出ず、合成プロバイダだけで完結する。

---

## 免責事項

本ソフトウェアは調査・学習目的で作成されたものであり、投資助言を目的としたものではない。
表示される数値は取得元データをそのまま計算したもので、正確性を保証しない。
投資判断はご自身の責任のもとで行うこと。
