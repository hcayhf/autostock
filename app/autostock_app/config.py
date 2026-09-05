"""設定値とパス解決。

環境変数で上書きできるようにしておき、テストでは一時ディレクトリを差し込む。
"""

from __future__ import annotations

import os
from pathlib import Path

# app/autostock_app/config.py -> app/ -> リポジトリルート
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


#: SQLite ファイルの置き場所。既存の実験用キャッシュと同じ data/ 配下にまとめる。
DATA_DIR = _env_path("AUTOSTOCK_DATA_DIR", PROJECT_DIR / "data")
DB_PATH = _env_path("AUTOSTOCK_DB", DATA_DIR / "autostock.db")

#: JPX が配布する銘柄一覧 (東証上場銘柄一覧 data_j.xls)。
UNIVERSE_FILE = _env_path("AUTOSTOCK_UNIVERSE_FILE", PROJECT_DIR / "data_j.xls")

#: 取り込み対象とする市場区分。
TARGET_MARKETS = ("プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）")

#: 株価をどこまで遡って取得するか。
DATA_START = os.environ.get("AUTOSTOCK_DATA_START", "2015-01-01")

#: yfinance を連打しないための銘柄あたりの待ち時間(秒)。
REQUEST_SLEEP = float(os.environ.get("AUTOSTOCK_REQUEST_SLEEP", "0.12"))

#: 検索スナップショットから除外する「株価が古すぎる」銘柄の判定日数。
#: 全銘柄の最新営業日からこの日数以上遅れている銘柄 (上場廃止など) は載せない。
MAX_STALE_DAYS = int(os.environ.get("AUTOSTOCK_MAX_STALE_DAYS", "30"))
