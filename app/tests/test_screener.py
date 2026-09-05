import pytest

from autostock_app.screener import (
    Range,
    ScreenQuery,
    distinct_values,
    field_metadata,
    query_from_dict,
    screen,
    snapshot_status,
)


def test_snapshot_is_populated(seeded_db):
    status = snapshot_status(seeded_db)
    assert status["stocks"] > 0
    assert status["price_date"]


def test_range_filter_is_applied(seeded_db):
    result = screen(seeded_db, ScreenQuery(ranges=[Range("rsi_14", max=50)], limit=100))
    assert result["total"] > 0
    assert all(row["rsi_14"] <= 50 for row in result["rows"])


def test_min_and_max_together(seeded_db):
    q = ScreenQuery(ranges=[Range("pbr", min=0.5, max=2.0)], limit=100)
    for row in screen(seeded_db, q)["rows"]:
        assert 0.5 <= row["pbr"] <= 2.0


def test_null_values_are_excluded_by_default(seeded_db):
    nulls = seeded_db.execute(
        "SELECT COUNT(*) FROM screen_snapshot WHERE per IS NULL"
    ).fetchone()[0]
    total = seeded_db.execute("SELECT COUNT(*) FROM screen_snapshot").fetchone()[0]
    excluded = screen(seeded_db, ScreenQuery(ranges=[Range("per", max=1e9)]))
    included = screen(seeded_db, ScreenQuery(ranges=[Range("per", max=1e9, include_null=True)]))
    assert excluded["total"] == total - nulls
    assert included["total"] == total


def test_nulls_are_sorted_last(seeded_db):
    for descending in (True, False):
        rows = screen(
            seeded_db, ScreenQuery(sort_by="per", sort_desc=descending, limit=500)
        )["rows"]
        values = [r["per"] for r in rows]
        first_null = next((i for i, v in enumerate(values) if v is None), len(values))
        assert all(v is None for v in values[first_null:])


def test_sort_direction(seeded_db):
    desc = [r["market_cap"] for r in screen(
        seeded_db, ScreenQuery(sort_by="market_cap", sort_desc=True, limit=10)
    )["rows"] if r["market_cap"] is not None]
    assert desc == sorted(desc, reverse=True)


def test_category_filter(seeded_db):
    markets = distinct_values(seeded_db, "market")
    target = markets[0]
    result = screen(seeded_db, ScreenQuery(categories={"market": [target]}, limit=100))
    assert result["total"] > 0
    assert all(row["market"] == target for row in result["rows"])


def test_text_search_matches_code_and_name(seeded_db):
    any_code = seeded_db.execute("SELECT code FROM screen_snapshot LIMIT 1").fetchone()[0]
    result = screen(seeded_db, ScreenQuery(text=any_code))
    assert result["total"] >= 1
    assert any(row["code"] == any_code for row in result["rows"])


def test_pagination_does_not_overlap(seeded_db):
    first = screen(seeded_db, ScreenQuery(limit=5, offset=0))
    second = screen(seeded_db, ScreenQuery(limit=5, offset=5))
    assert first["total"] == second["total"]
    assert not ({r["code"] for r in first["rows"]} & {r["code"] for r in second["rows"]})


def test_filtered_and_sorted_columns_are_always_shown(seeded_db):
    q = ScreenQuery(ranges=[Range("bb_pct_b", min=0)], sort_by="volatility_20d", columns=["code"])
    result = screen(seeded_db, q)
    assert "bb_pct_b" in result["columns"]
    assert "volatility_20d" in result["columns"]


def test_limit_is_capped(seeded_db):
    result = screen(seeded_db, ScreenQuery(limit=100000))
    assert result["count"] <= 500


@pytest.mark.parametrize("payload", [
    {"ranges": [{"field": "close; DROP TABLE stocks", "max": 1}]},
    {"ranges": [{"field": "code", "max": 1}]},          # 文字列カラムはレンジ不可
    {"sort_by": "1=1"},
    {"categories": {"close": ["x"]}},                    # 数値カラムはカテゴリ不可
    {"columns": ["__proto__"]},
])
def test_invalid_input_is_rejected(seeded_db, payload):
    with pytest.raises(ValueError):
        screen(seeded_db, query_from_dict(payload))


def test_inverted_range_is_rejected():
    with pytest.raises(ValueError):
        Range("per", min=20, max=10)


def test_field_metadata_covers_every_column(seeded_db):
    meta = {f["key"] for f in field_metadata()}
    columns = {r[1] for r in seeded_db.execute("PRAGMA table_info(screen_snapshot)")}
    bookkeeping = {"price_date", "fundamental_date", "updated_at"}
    assert (columns - bookkeeping) <= meta
