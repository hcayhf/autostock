"""SQLite スキーマと接続管理。

銘柄ごとの CSV では 4000 銘柄 × 10 年分を横断検索するのが重いので、
1 ファイルの SQLite にまとめて「最新日スナップショット」を別テーブルに
持たせている。検索はそのスナップショット 1 枚だけを見るため、
銘柄数が増えても応答が悪化しない。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import config
from .fields import FUNDAMENTAL_KEYS, INDICATOR_KEYS

SCHEMA_VERSION = 1


def _real_columns(keys: tuple[str, ...], indent: str = "    ") -> str:
    return "".join(f"{indent}{k} REAL,\n" for k in keys)


SCHEMA_SQL = f"""
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 銘柄マスタ (JPX の東証上場銘柄一覧由来)
CREATE TABLE IF NOT EXISTS stocks (
    code          TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    market        TEXT,
    sector33_code TEXT,
    sector33      TEXT,
    sector17_code TEXT,
    sector17      TEXT,
    scale_code    TEXT,
    scale         TEXT,
    ticker        TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stocks_sector33 ON stocks(sector33);
CREATE INDEX IF NOT EXISTS idx_stocks_market   ON stocks(market);

-- 日足 OHLCV (配当・分割調整済み)
CREATE TABLE IF NOT EXISTS prices (
    code   TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    PRIMARY KEY (code, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

-- 日次テクニカル指標
CREATE TABLE IF NOT EXISTS indicators (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
{_real_columns(INDICATOR_KEYS)}    PRIMARY KEY (code, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date);

-- ファンダメンタルズ。取得日ごとに 1 行残すので、
-- 毎日走らせればその日から先の履歴が自前で貯まっていく。
CREATE TABLE IF NOT EXISTS fundamentals (
    code   TEXT NOT NULL,
    as_of  TEXT NOT NULL,
{_real_columns(FUNDAMENTAL_KEYS)}    source TEXT NOT NULL,
    PRIMARY KEY (code, as_of)
) WITHOUT ROWID;

-- 検索用スナップショット (銘柄マスタ + 最新指標 + 最新ファンダ)。
-- 検索はこの 1 テーブルへのクエリだけで完結する。
CREATE TABLE IF NOT EXISTS screen_snapshot (
    code          TEXT PRIMARY KEY,
    name          TEXT,
    market        TEXT,
    sector33      TEXT,
    sector17      TEXT,
    scale         TEXT,
    price_date    TEXT,
    fundamental_date TEXT,
{_real_columns(INDICATOR_KEYS)}{_real_columns(FUNDAMENTAL_KEYS)}    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshot_sector33 ON screen_snapshot(sector33);
CREATE INDEX IF NOT EXISTS idx_snapshot_market   ON screen_snapshot(market);

-- 取り込み履歴 (いつ何件取れたか / どこで落ちたか)
CREATE TABLE IF NOT EXISTS ingest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    rows        INTEGER DEFAULT 0,
    detail      TEXT
);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """接続を開いて必要ならスキーマを作る。"""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT INTO schema_info(key, value) VALUES('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


@contextmanager
def session(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """初期化済みの接続を貸し出すコンテキストマネージャ。"""
    conn = connect(db_path)
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


# --- 取り込み履歴 ------------------------------------------------------------


def start_run(conn: sqlite3.Connection, kind: str) -> int:
    cur = conn.execute(
        "INSERT INTO ingest_runs(kind, started_at, status) "
        "VALUES(?, datetime('now'), 'running')",
        (kind,),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    rows: int = 0,
    detail: str | None = None,
) -> None:
    conn.execute(
        "UPDATE ingest_runs SET finished_at = datetime('now'), status = ?, "
        "rows = ?, detail = ? WHERE id = ?",
        (status, rows, detail, run_id),
    )
    conn.commit()


def table_count(conn: sqlite3.Connection, table: str) -> int:
    # table は呼び出し側の定数のみ。ユーザー入力を渡さないこと。
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
