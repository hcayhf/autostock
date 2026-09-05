"""テスト共通のフィクスチャ。

ネットワークには一切出ず、合成プロバイダだけで完結させる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autostock_app import db, ingest  # noqa: E402
from autostock_app.providers import SyntheticProvider, synthetic_universe  # noqa: E402
from autostock_app.universe import save_universe  # noqa: E402

N_STOCKS = 12
START = "2023-01-01"


@pytest.fixture
def empty_db(tmp_path):
    """スキーマだけ作った空の DB。"""
    conn = db.connect(tmp_path / "test.db")
    db.init_db(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def seeded_db_path(tmp_path_factory):
    """合成データを流し込んだ DB を 1 度だけ作って使い回す。"""
    path = tmp_path_factory.mktemp("db") / "seeded.db"
    provider = SyntheticProvider()
    conn = db.connect(path)
    db.init_db(conn)
    save_universe(conn, synthetic_universe(N_STOCKS))
    ingest.ingest_prices(conn, provider, start=START, full=True, sleep=0)
    ingest.rebuild_indicators(conn, full=True)
    ingest.ingest_fundamentals(conn, provider, sleep=0)
    ingest.rebuild_snapshot(conn)
    conn.close()
    return path


@pytest.fixture
def seeded_db(seeded_db_path):
    conn = db.connect(seeded_db_path)
    yield conn
    conn.close()
