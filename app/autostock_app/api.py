"""FastAPI アプリ。

    GET  /                    Web UI
    GET  /api/health          稼働確認
    GET  /api/meta            検索フォームを組み立てるためのメタ情報
    POST /api/screen          スクリーニング実行
    GET  /api/stocks/{code}   個別銘柄の詳細と株価推移

SQLite は接続をスレッド間で共有できないので、リクエストごとに開いて閉じる。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field as PField

from . import config, db
from .fields import CATEGORICAL_KEYS, GROUP_LABELS
from .screener import (
    DEFAULT_COLUMNS,
    PRESETS,
    distinct_values,
    field_metadata,
    query_from_dict,
    screen,
    snapshot_status,
)

WEB_DIR = Path(__file__).resolve().parent / "web"


# --- リクエストモデル --------------------------------------------------------


class RangeIn(BaseModel):
    field: str
    min: float | None = None
    max: float | None = None
    include_null: bool = False


class ScreenIn(BaseModel):
    text: str | None = None
    categories: dict[str, list[str]] = PField(default_factory=dict)
    ranges: list[RangeIn] = PField(default_factory=list)
    sort_by: str = "turnover_20d"
    sort_desc: bool = True
    limit: int = PField(default=50, ge=1, le=500)
    offset: int = PField(default=0, ge=0)
    columns: list[str] | None = None


# --- アプリ ------------------------------------------------------------------


def create_app(db_path: Path | str | None = None) -> FastAPI:
    resolved = Path(db_path) if db_path is not None else config.DB_PATH

    app = FastAPI(
        title="AutoStock Screener",
        description="日本株の日次データを指標で検索する",
        version="0.1.0",
    )
    app.state.db_path = resolved

    def get_conn() -> Iterator[sqlite3.Connection]:
        conn = db.connect(app.state.db_path)
        try:
            db.init_db(conn)
            yield conn
        finally:
            conn.close()

    @app.get("/api/health")
    def health(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        status = snapshot_status(conn)
        return {
            "ok": True,
            "db": str(app.state.db_path),
            "stocks": status.get("stocks", 0),
            "price_date": status.get("price_date"),
        }

    @app.get("/api/meta")
    def meta(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, Any]:
        status = snapshot_status(conn)
        return {
            "fields": field_metadata(),
            "groups": GROUP_LABELS,
            "default_columns": list(DEFAULT_COLUMNS),
            "categories": {k: distinct_values(conn, k) for k in CATEGORICAL_KEYS},
            "presets": list(PRESETS),
            "status": status,
            "empty": not status.get("stocks"),
        }

    @app.post("/api/screen")
    def run_screen(
        payload: ScreenIn, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict[str, Any]:
        try:
            query = query_from_dict(payload.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return screen(conn, query)

    @app.get("/api/stocks/{code}")
    def stock_detail(
        code: str,
        days: int = Query(default=250, ge=20, le=2500),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict[str, Any]:
        stock = conn.execute("SELECT * FROM stocks WHERE code = ?", (code,)).fetchone()
        if stock is None:
            raise HTTPException(status_code=404, detail=f"unknown code: {code}")

        snapshot = conn.execute(
            "SELECT * FROM screen_snapshot WHERE code = ?", (code,)
        ).fetchone()

        prices = conn.execute(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE code = ? ORDER BY date DESC LIMIT ?",
            (code, days),
        ).fetchall()

        history = conn.execute(
            "SELECT * FROM fundamentals WHERE code = ? ORDER BY as_of DESC LIMIT 60",
            (code,),
        ).fetchall()

        return {
            "stock": dict(stock),
            "snapshot": dict(snapshot) if snapshot else None,
            "prices": [dict(r) for r in reversed(prices)],
            "fundamentals_history": [dict(r) for r in history],
        }

    # --- Web UI -------------------------------------------------------------

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> Response:
            # ブラウザが必ず取りに来るので、404 をログに垂れ流さないよう空で返す
            return Response(status_code=204)

    return app


#: `uvicorn autostock_app.api:app` で起動できるようにしておく。
app = create_app()
