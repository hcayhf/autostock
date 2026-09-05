"""銘柄マスタ (ユニバース) の読み込みと保存。

JPX が配布する「東証上場銘柄一覧」(data_j.xls) が入力。
https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config

_COLUMN_MAP = {
    "コード": "code",
    "銘柄名": "name",
    "市場・商品区分": "market",
    "33業種コード": "sector33_code",
    "33業種区分": "sector33",
    "17業種コード": "sector17_code",
    "17業種区分": "sector17",
    "規模コード": "scale_code",
    "規模区分": "scale",
}

_COLUMNS = (
    "code", "name", "market", "sector33_code", "sector33",
    "sector17_code", "sector17", "scale_code", "scale", "ticker",
)


def load_universe_file(
    path: Path | str | None = None,
    markets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """data_j.xls を読んで銘柄マスタの DataFrame にする。"""
    path = Path(path) if path is not None else config.UNIVERSE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"銘柄一覧ファイルが見つかりません: {path}\n"
            "JPX (https://www.jpx.co.jp/markets/statistics-equities/misc/01.html) から "
            "data_j.xls を取得して配置するか、--universe-file でパスを指定してください。"
        )

    df = pd.read_excel(path)
    missing = [c for c in _COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"data_j.xls に想定の列がありません: {missing}")

    df = df.rename(columns=_COLUMN_MAP)

    markets = markets if markets is not None else config.TARGET_MARKETS
    df = df[df["market"].isin(markets)].copy()

    df["code"] = df["code"].astype(str).str.strip().str.zfill(4)
    for col in ("sector33_code", "sector17_code", "scale_code"):
        df[col] = df[col].astype(str).str.strip()
    df["ticker"] = df["code"] + ".T"

    # ETF や REIT が紛れ込んでも困るので 4 桁数字コードだけ残す
    df = df[df["code"].str.fullmatch(r"\d{4}")]
    df = df.drop_duplicates(subset="code", keep="first")
    return df[list(_COLUMNS)].reset_index(drop=True)


def save_universe(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """銘柄マスタを upsert する。上場廃止銘柄は消さず残す (過去データが引ける)。"""
    rows = [tuple(r) for r in df[list(_COLUMNS)].astype(object).where(df.notna(), None).values]
    conn.executemany(
        f"""
        INSERT INTO stocks ({", ".join(_COLUMNS)}, updated_at)
        VALUES ({", ".join("?" * len(_COLUMNS))}, datetime('now'))
        ON CONFLICT(code) DO UPDATE SET
            {", ".join(f"{c} = excluded.{c}" for c in _COLUMNS if c != "code")},
            updated_at = excluded.updated_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_universe(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[sqlite3.Row]:
    """DB に入っている銘柄を code 順で返す。"""
    sql = "SELECT code, name, ticker FROM stocks ORDER BY code"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql))
