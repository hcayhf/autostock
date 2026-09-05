"""検索 (スクリーニング) の問い合わせ組み立て。

screen_snapshot テーブル 1 枚に対するクエリだけで完結する。
カラム名は fields.py のホワイトリストで検証してから SQL に埋め込むので、
利用者入力が識別子として展開されることはない。値は必ずバインドする。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field as dc_field
from typing import Any

from .fields import (
    ALL_FIELDS,
    CATEGORICAL_KEYS,
    FIELDS_BY_KEY,
    NUMERIC_KEYS,
    TEXT_SEARCH_KEYS,
    field_or_raise,
)

TABLE = "screen_snapshot"

#: 検索条件を指定しなかったときに表示する列。
DEFAULT_COLUMNS: tuple[str, ...] = (
    "code", "name", "market", "sector33", "close", "turnover_20d",
    "per", "pbr", "dividend_yield", "roe", "market_cap",
    "rsi_14", "ret_20d", "ret_250d", "sma_dev_200",
)

DEFAULT_SORT = "turnover_20d"
MAX_LIMIT = 500


@dataclass(frozen=True)
class Range:
    """1 カラムに対する数値レンジ条件。min / max はどちらも省略可。"""

    field: str
    min: float | None = None
    max: float | None = None
    #: True にすると値が NULL の銘柄も通す (赤字で PER が無い銘柄を残したいときなど)
    include_null: bool = False

    def __post_init__(self) -> None:
        f = field_or_raise(self.field)
        if not f.numeric:
            raise ValueError(f"field {self.field!r} is not numeric")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.field}: min ({self.min}) > max ({self.max})")


@dataclass
class ScreenQuery:
    """検索条件。"""

    text: str | None = None
    categories: dict[str, list[str]] = dc_field(default_factory=dict)
    ranges: list[Range] = dc_field(default_factory=list)
    sort_by: str = DEFAULT_SORT
    sort_desc: bool = True
    limit: int = 50
    offset: int = 0
    columns: list[str] | None = None

    def resolved_columns(self) -> list[str]:
        cols = list(self.columns) if self.columns else list(DEFAULT_COLUMNS)
        for key in cols:
            field_or_raise(key)
        # code と name は結果の識別に要るので必ず先頭に置く
        for key in ("name", "code"):
            if key in cols:
                cols.remove(key)
            cols.insert(0, key)
        # 条件に使った列が表示されないと結果を検証できないので足す
        for r in self.ranges:
            if r.field not in cols:
                cols.append(r.field)
        if self.sort_by not in cols:
            cols.append(self.sort_by)
        return cols


def _build_where(query: ScreenQuery) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if query.text:
        needle = f"%{query.text.strip()}%"
        ors = " OR ".join(f"{k} LIKE ?" for k in TEXT_SEARCH_KEYS)
        clauses.append(f"({ors})")
        params.extend([needle] * len(TEXT_SEARCH_KEYS))

    for key, values in query.categories.items():
        if key not in CATEGORICAL_KEYS:
            raise ValueError(f"field {key!r} is not filterable as a category")
        values = [v for v in values if v]
        if not values:
            continue
        clauses.append(f"{key} IN ({', '.join('?' * len(values))})")
        params.extend(values)

    for r in query.ranges:
        conditions: list[str] = []
        if r.min is not None:
            conditions.append(f"{r.field} >= ?")
            params.append(r.min)
        if r.max is not None:
            conditions.append(f"{r.field} <= ?")
            params.append(r.max)
        if not conditions:
            continue
        clause = " AND ".join(conditions)
        # 既定では NULL を除外する (一般的なスクリーナと同じ挙動)。
        clauses.append(f"({clause} OR {r.field} IS NULL)" if r.include_null else f"({clause})")

    where = " AND ".join(clauses) if clauses else "1 = 1"
    return where, params


def build_sql(query: ScreenQuery) -> tuple[str, list[Any], str, list[Any], list[str]]:
    """(本体SQL, 本体params, 件数SQL, 件数params, 列名) を返す。"""
    columns = query.resolved_columns()
    sort_key = field_or_raise(query.sort_by).key
    where, params = _build_where(query)

    limit = max(1, min(int(query.limit), MAX_LIMIT))
    offset = max(0, int(query.offset))
    direction = "DESC" if query.sort_desc else "ASC"

    # 値が無い銘柄が上位を占めると使い物にならないので、NULL は常に最後。
    sql = (
        f"SELECT {', '.join(columns)}, price_date, fundamental_date FROM {TABLE} "
        f"WHERE {where} "
        f"ORDER BY {sort_key} IS NULL, {sort_key} {direction}, code ASC "
        f"LIMIT ? OFFSET ?"
    )
    count_sql = f"SELECT COUNT(*) FROM {TABLE} WHERE {where}"
    return sql, [*params, limit, offset], count_sql, list(params), columns


def screen(conn: sqlite3.Connection, query: ScreenQuery) -> dict[str, Any]:
    """検索を実行して結果を返す。"""
    sql, params, count_sql, count_params, columns = build_sql(query)
    total = int(conn.execute(count_sql, count_params).fetchone()[0])
    rows = [dict(r) for r in conn.execute(sql, params)]
    return {
        "total": total,
        "count": len(rows),
        "limit": query.limit,
        "offset": query.offset,
        "columns": columns,
        "rows": rows,
    }


def distinct_values(conn: sqlite3.Connection, key: str) -> list[str]:
    """カテゴリ列の選択肢を返す (UI のプルダウン用)。"""
    if key not in CATEGORICAL_KEYS:
        raise ValueError(f"field {key!r} is not a category")
    return [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT {key} FROM {TABLE} WHERE {key} IS NOT NULL AND {key} != '' "
            f"ORDER BY {key}"
        )
    ]


def snapshot_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """データの鮮度。UI に「いつ時点のデータか」を出すために使う。"""
    row = conn.execute(
        f"SELECT COUNT(*) AS stocks, MAX(price_date) AS price_date, "
        f"MAX(fundamental_date) AS fundamental_date, MAX(updated_at) AS updated_at "
        f"FROM {TABLE}"
    ).fetchone()
    return dict(row) if row else {}


# --- プリセット --------------------------------------------------------------
# 「何を検索すればいいか分からない」状態を避けるための出発点。
# そのまま使うのではなく、条件をいじる土台として置いている。

PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "value_dividend",
        "name": "割安・高配当",
        "description": "PER15倍以下・PBR1.5倍以下・配当利回り3%以上。オーソドックスなバリュー条件。",
        "ranges": [
            {"field": "per", "min": 0, "max": 15},
            {"field": "pbr", "max": 1.5},
            {"field": "dividend_yield", "min": 3.0},
            {"field": "turnover_20d", "min": 100},
        ],
        "sort_by": "dividend_yield",
        "sort_desc": True,
    },
    {
        "id": "quality_growth",
        "name": "高ROE・長期上昇トレンド",
        "description": "ROE12%以上かつ200日線より上。稼ぐ力があって株価も評価されている銘柄。",
        "ranges": [
            {"field": "roe", "min": 12.0},
            {"field": "sma_dev_200", "min": 0},
            {"field": "ret_250d", "min": 0},
            {"field": "turnover_20d", "min": 100},
        ],
        "sort_by": "roe",
        "sort_desc": True,
    },
    {
        "id": "pullback",
        "name": "押し目候補",
        "description": "長期は上昇トレンドだが直近で売られてRSIが低い銘柄。",
        "ranges": [
            {"field": "sma_dev_200", "min": 0},
            {"field": "rsi_14", "max": 40},
            {"field": "ret_20d", "max": 0},
            {"field": "turnover_20d", "min": 100},
        ],
        "sort_by": "rsi_14",
        "sort_desc": False,
    },
    {
        "id": "net_net",
        "name": "低PBR・財務健全",
        "description": "PBR1倍割れかつ自己資本比率50%以上。資産バリュー狙い。",
        "ranges": [
            {"field": "pbr", "min": 0, "max": 1.0},
            {"field": "equity_ratio", "min": 50.0},
            {"field": "turnover_20d", "min": 50},
        ],
        "sort_by": "pbr",
        "sort_desc": False,
    },
    {
        "id": "near_high",
        "name": "52週高値圏",
        "description": "52週高値から5%以内。強い銘柄をそのまま買う順張り。",
        "ranges": [
            {"field": "high_250d_pct", "min": -5.0},
            {"field": "turnover_20d", "min": 100},
        ],
        "sort_by": "ret_250d",
        "sort_desc": True,
    },
)

PRESETS_BY_ID = {p["id"]: p for p in PRESETS}


def query_from_dict(payload: dict[str, Any]) -> ScreenQuery:
    """API / CLI から来た dict を ScreenQuery にする (検証込み)。"""
    ranges = [
        Range(
            field=r["field"],
            min=_opt_float(r.get("min")),
            max=_opt_float(r.get("max")),
            include_null=bool(r.get("include_null", False)),
        )
        for r in payload.get("ranges", [])
    ]
    sort_by = payload.get("sort_by") or DEFAULT_SORT
    field_or_raise(sort_by)

    categories = {}
    for key, values in (payload.get("categories") or {}).items():
        if key not in CATEGORICAL_KEYS:
            # 実行時ではなく入力の検証時点で弾く (API が 400 を返せるように)
            raise ValueError(f"field {key!r} is not filterable as a category")
        categories[key] = list(values)

    for key in payload.get("columns") or []:
        field_or_raise(key)

    return ScreenQuery(
        text=payload.get("text") or None,
        categories=categories,
        ranges=ranges,
        sort_by=sort_by,
        sort_desc=bool(payload.get("sort_desc", True)),
        limit=int(payload.get("limit", 50)),
        offset=int(payload.get("offset", 0)),
        columns=payload.get("columns"),
    )


def _opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def field_metadata() -> list[dict[str, Any]]:
    """UI が検索フォームを組み立てるためのカラム定義。"""
    return [
        {
            "key": f.key,
            "label": f.label,
            "group": f.group,
            "dtype": f.dtype,
            "unit": f.unit,
            "desc": f.desc,
            "primary": f.primary,
            "signed": f.signed,
            "filterable": f.numeric or f.key in CATEGORICAL_KEYS,
            "sortable": f.key in NUMERIC_KEYS,
        }
        for f in ALL_FIELDS
    ]


__all__ = [
    "DEFAULT_COLUMNS",
    "PRESETS",
    "PRESETS_BY_ID",
    "Range",
    "ScreenQuery",
    "build_sql",
    "distinct_values",
    "field_metadata",
    "query_from_dict",
    "screen",
    "snapshot_status",
]
