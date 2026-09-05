from datetime import date

import pandas as pd
import pytest

from autostock_app import db, ingest
from autostock_app.providers import SyntheticProvider, synthetic_universe
from autostock_app.providers.base import normalize_price_frame
from autostock_app.universe import save_universe

START = "2023-06-01"


@pytest.fixture
def small_db(empty_db):
    save_universe(empty_db, synthetic_universe(4))
    return empty_db


def test_universe_is_upserted_not_duplicated(small_db):
    save_universe(small_db, synthetic_universe(4))
    assert db.table_count(small_db, "stocks") == 4


def test_prices_are_stored(small_db):
    rows = ingest.ingest_prices(small_db, SyntheticProvider(), start=START, sleep=0)
    assert rows > 0
    assert db.table_count(small_db, "prices") == rows


def test_second_run_does_not_duplicate_prices(small_db):
    provider = SyntheticProvider()
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)
    before = db.table_count(small_db, "prices")
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)
    assert db.table_count(small_db, "prices") == before


def test_incremental_run_only_fetches_new_days(small_db, monkeypatch):
    provider = SyntheticProvider()
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)

    requested: list[str] = []
    original = provider.fetch_prices

    def spy(ticker, start, end=None):
        requested.append(start)
        return original(ticker, start, end)

    monkeypatch.setattr(provider, "fetch_prices", spy)
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)

    last = small_db.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    # 2 回目は全期間ではなく、保存済みの最終日より後だけを要求する
    assert requested and all(s > last for s in requested)


def test_full_flag_refetches_from_the_beginning(small_db, monkeypatch):
    provider = SyntheticProvider()
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)
    requested: list[str] = []
    original = provider.fetch_prices
    monkeypatch.setattr(
        provider, "fetch_prices",
        lambda ticker, start, end=None: (requested.append(start), original(ticker, start, end))[1],
    )
    ingest.ingest_prices(small_db, provider, start=START, full=True, sleep=0)
    assert set(requested) == {START}


def test_indicators_match_the_price_rows_they_can_cover(small_db):
    ingest.ingest_prices(small_db, SyntheticProvider(), start=START, sleep=0)
    written = ingest.rebuild_indicators(small_db)
    assert written > 0
    assert db.table_count(small_db, "indicators") == written


def test_fundamentals_accumulate_one_row_per_day(small_db):
    provider = SyntheticProvider()
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)
    ingest.ingest_fundamentals(small_db, provider, as_of="2024-01-04", sleep=0)
    ingest.ingest_fundamentals(small_db, provider, as_of="2024-01-05", sleep=0)
    assert db.table_count(small_db, "fundamentals") == 8

    # 同じ日を 2 回流しても増えない (上書き)
    ingest.ingest_fundamentals(small_db, provider, as_of="2024-01-05", sleep=0)
    assert db.table_count(small_db, "fundamentals") == 8


def test_snapshot_uses_the_latest_rows(small_db):
    provider = SyntheticProvider()
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)
    ingest.rebuild_indicators(small_db)
    ingest.ingest_fundamentals(small_db, provider, as_of="2024-01-04", sleep=0)
    ingest.ingest_fundamentals(small_db, provider, as_of=date.today().isoformat(), sleep=0)
    ingest.rebuild_snapshot(small_db)

    latest_price = small_db.execute("SELECT MAX(date) FROM indicators").fetchone()[0]
    latest_fund = small_db.execute("SELECT MAX(as_of) FROM fundamentals").fetchone()[0]
    rows = small_db.execute(
        "SELECT price_date, fundamental_date FROM screen_snapshot"
    ).fetchall()
    assert rows
    assert all(r["price_date"] == latest_price for r in rows)
    assert all(r["fundamental_date"] == latest_fund for r in rows)


def test_snapshot_drops_stocks_with_stale_prices(small_db):
    provider = SyntheticProvider()
    ingest.ingest_prices(small_db, provider, start=START, sleep=0)
    ingest.rebuild_indicators(small_db)

    # 1 銘柄だけ 1 年前で止まっている状態を作る (上場廃止を模擬)
    stale = small_db.execute("SELECT code FROM stocks ORDER BY code LIMIT 1").fetchone()[0]
    cutoff = small_db.execute("SELECT date(MAX(date), '-365 day') FROM indicators").fetchone()[0]
    small_db.execute("DELETE FROM indicators WHERE code = ? AND date > ?", (stale, cutoff))
    small_db.commit()

    ingest.rebuild_snapshot(small_db, max_stale_days=30)
    codes = {r[0] for r in small_db.execute("SELECT code FROM screen_snapshot")}
    assert stale not in codes
    assert len(codes) == 3

    ingest.rebuild_snapshot(small_db, max_stale_days=None)
    assert db.table_count(small_db, "screen_snapshot") == 4


def test_snapshot_keeps_stocks_without_fundamentals(small_db):
    ingest.ingest_prices(small_db, SyntheticProvider(), start=START, sleep=0)
    ingest.rebuild_indicators(small_db)
    ingest.rebuild_snapshot(small_db)
    assert db.table_count(small_db, "screen_snapshot") == 4
    assert all(
        r["per"] is None for r in small_db.execute("SELECT per FROM screen_snapshot")
    )


def test_run_daily_records_the_run(small_db):
    provider = SyntheticProvider()
    ingest.run_daily(small_db, provider, provider)
    run = small_db.execute("SELECT * FROM ingest_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert run["kind"] == "daily"
    assert run["status"] == "ok"
    assert run["finished_at"]


def test_run_daily_marks_failures(small_db):
    class Broken:
        name = "broken"

        def fetch_prices(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ingest.run_daily(small_db, Broken())
    run = small_db.execute("SELECT * FROM ingest_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert run["status"] == "failed"
    assert "boom" in run["detail"]


# --- プロバイダ層 -----------------------------------------------------------


def test_normalize_strips_timezone_and_sorts():
    index = pd.to_datetime(["2024-01-03", "2024-01-02"]).tz_localize("Asia/Tokyo")
    frame = pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [10, 20]},
        index=index,
    )
    out = normalize_price_frame(frame)
    assert out.index.tz is None
    assert list(out.index) == sorted(out.index)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_rejects_frames_missing_columns():
    frame = pd.DataFrame({"Close": [1]}, index=pd.to_datetime(["2024-01-02"]))
    assert normalize_price_frame(frame) is None


def test_normalize_drops_rows_without_a_close():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame(
        {"Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [100, None], "Volume": [1, 1]},
        index=index,
    )
    out = normalize_price_frame(frame)
    assert len(out) == 1


def test_normalize_handles_empty():
    assert normalize_price_frame(pd.DataFrame()) is None
    assert normalize_price_frame(None) is None
