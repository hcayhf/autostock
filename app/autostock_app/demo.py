"""合成データでデモ用 DB を作る。

外部ネットワークに出られない環境でも UI と検索を触れるようにするためのもの。
中身は実在の株価ではないので、投資判断には使わないこと。
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from . import ingest
from .providers import SyntheticProvider, synthetic_universe
from .universe import save_universe


def seed_demo(
    conn: sqlite3.Connection,
    *,
    n_stocks: int = 150,
    start: str = "2021-01-01",
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """ダミー銘柄・株価・指標・ファンダを一式作る。"""
    provider = SyntheticProvider()

    universe = synthetic_universe(n_stocks)
    progress(f"universe: {save_universe(conn, universe)} 銘柄 (合成データ)")

    result = {
        "prices": ingest.ingest_prices(
            conn, provider, start=start, full=True, sleep=0, progress=progress
        ),
        "indicators": ingest.rebuild_indicators(conn, full=True, progress=progress),
        "fundamentals": ingest.ingest_fundamentals(
            conn, provider, sleep=0, progress=progress
        ),
    }
    result["snapshot"] = ingest.rebuild_snapshot(conn, progress=progress)
    return result
