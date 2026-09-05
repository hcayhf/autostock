"""取り込みパイプライン。

    ユニバース -> 日足株価 -> 指標 -> ファンダメンタルズ -> 検索スナップショット

株価は差分取得する。各銘柄について DB にある最終日の翌日から取りに行くので、
毎日回す運用ではリクエストが 1 銘柄あたり数日分で済む。
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Callable, Sequence

import pandas as pd

from . import config, db
from .fields import FUNDAMENTAL_KEYS, INDICATOR_KEYS
from .indicators import compute_indicators
from .providers import FundamentalsProvider, PriceProvider

ProgressFn = Callable[[str], None]

#: 引数を省いたのか、明示的に None (= フィルタ無効) を渡したのかを区別するための番兵。
_UNSET: Any = object()


def _noop(_: str) -> None:
    pass


def _codes_and_tickers(
    conn: sqlite3.Connection, codes: Sequence[str] | None, limit: int | None
) -> list[tuple[str, str]]:
    sql = "SELECT code, ticker FROM stocks"
    params: list = []
    if codes:
        sql += f" WHERE code IN ({', '.join('?' * len(codes))})"
        params.extend(codes)
    sql += " ORDER BY code"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [(r["code"], r["ticker"]) for r in conn.execute(sql, params)]


# --- 株価 --------------------------------------------------------------------


def _last_price_dates(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        r["code"]: r["last_date"]
        for r in conn.execute("SELECT code, MAX(date) AS last_date FROM prices GROUP BY code")
    }


def _save_prices(conn: sqlite3.Connection, code: str, frame: pd.DataFrame) -> int:
    rows = [
        (
            code,
            idx.strftime("%Y-%m-%d"),
            _f(r.open), _f(r.high), _f(r.low), _f(r.close), _f(r.volume),
        )
        for idx, r in zip(frame.index, frame.itertuples(index=False))
    ]
    conn.executemany(
        "INSERT INTO prices(code, date, open, high, low, close, volume) "
        "VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(code, date) DO UPDATE SET "
        "open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, volume=excluded.volume",
        rows,
    )
    return len(rows)


def _f(value) -> float | None:
    """NaN を SQLite の NULL にする。"""
    if value is None or pd.isna(value):
        return None
    return float(value)


def ingest_prices(
    conn: sqlite3.Connection,
    provider: PriceProvider,
    *,
    codes: Sequence[str] | None = None,
    limit: int | None = None,
    start: str | None = None,
    full: bool = False,
    sleep: float | None = None,
    progress: ProgressFn = _noop,
) -> int:
    """日足を取り込む。既存分がある銘柄はその翌日から差分取得する。"""
    import time

    start = start or config.DATA_START
    sleep = config.REQUEST_SLEEP if sleep is None else sleep
    targets = _codes_and_tickers(conn, codes, limit)
    last_dates = {} if full else _last_price_dates(conn)
    today = date.today().isoformat()

    total_rows = 0
    fetched = skipped = failed = 0

    for i, (code, ticker) in enumerate(targets, start=1):
        fetch_start = start
        if (last := last_dates.get(code)) is not None:
            next_day = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).date()
            if next_day.isoformat() > today:
                skipped += 1
                continue
            fetch_start = next_day.isoformat()

        frame = provider.fetch_prices(ticker, start=fetch_start)
        if frame is None or frame.empty:
            # 差分取得で新しい足が無いのは正常 (休場明けなど)
            if last is None:
                failed += 1
            else:
                skipped += 1
        else:
            total_rows += _save_prices(conn, code, frame)
            fetched += 1
            if sleep:
                time.sleep(sleep)

        if i % 100 == 0:
            conn.commit()
            progress(
                f"  prices {i}/{len(targets)} "
                f"(取得 {fetched} / スキップ {skipped} / 失敗 {failed} / {total_rows} 行)"
            )

    conn.commit()
    progress(
        f"prices: {total_rows} 行 "
        f"(取得 {fetched} 銘柄 / スキップ {skipped} / 失敗 {failed})"
    )
    return total_rows


# --- 指標 --------------------------------------------------------------------


def rebuild_indicators(
    conn: sqlite3.Connection,
    *,
    codes: Sequence[str] | None = None,
    limit: int | None = None,
    full: bool = False,
    progress: ProgressFn = _noop,
) -> int:
    """株価から指標を再計算して書き込む。

    ローリング窓の計算に過去が必要なので株価は常に全期間読むが、
    書き込みは既存の最終指標日以降だけに絞る (full=True で全期間書き直し)。
    """
    targets = _codes_and_tickers(conn, codes, limit)
    last_indicator = {} if full else {
        r["code"]: r["last_date"]
        for r in conn.execute("SELECT code, MAX(date) AS last_date FROM indicators GROUP BY code")
    }

    columns = ("code", "date", *INDICATOR_KEYS)
    insert_sql = (
        f"INSERT INTO indicators({', '.join(columns)}) "
        f"VALUES({', '.join('?' * len(columns))}) "
        f"ON CONFLICT(code, date) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in INDICATOR_KEYS)
    )

    total = 0
    for i, (code, _) in enumerate(targets, start=1):
        prices = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE code = ? ORDER BY date",
            conn,
            params=(code,),
            index_col="date",
            parse_dates=["date"],
        )
        if prices.empty:
            continue

        frame = compute_indicators(prices)
        if (since := last_indicator.get(code)) is not None:
            # 最終日そのものも計算し直す (当日中に株価が更新されている場合があるため)
            frame = frame[frame.index >= pd.Timestamp(since)]
        # 全列 NaN の行 (履歴不足の期間) は保存しない
        frame = frame.dropna(how="all")
        if frame.empty:
            continue

        rows = [
            (code, idx.strftime("%Y-%m-%d"), *[_f(v) for v in row])
            for idx, row in zip(frame.index, frame.to_numpy())
        ]
        conn.executemany(insert_sql, rows)
        total += len(rows)

        if i % 200 == 0:
            conn.commit()
            progress(f"  indicators {i}/{len(targets)} ({total} 行)")

    conn.commit()
    progress(f"indicators: {total} 行")
    return total


# --- ファンダメンタルズ ------------------------------------------------------


def ingest_fundamentals(
    conn: sqlite3.Connection,
    provider: FundamentalsProvider,
    *,
    codes: Sequence[str] | None = None,
    limit: int | None = None,
    deep: bool = False,
    as_of: str | None = None,
    sleep: float | None = None,
    progress: ProgressFn = _noop,
) -> int:
    """ファンダメンタルズを「取得日つき」で追記する。

    yfinance が返すのは現在値スナップショットで履歴が無いため、
    日次で回して as_of を積み上げることで自前の履歴にする。
    同じ as_of で 2 回走らせた場合は上書きになる。
    """
    import time

    as_of = as_of or date.today().isoformat()
    sleep = config.REQUEST_SLEEP if sleep is None else sleep
    targets = _codes_and_tickers(conn, codes, limit)

    # 配当利回りを自前で計算するために直近終値を渡す
    closes = {
        r["code"]: r["close"]
        for r in conn.execute(
            "SELECT p.code, p.close FROM prices p "
            "JOIN (SELECT code, MAX(date) AS d FROM prices GROUP BY code) m "
            "  ON m.code = p.code AND m.d = p.date"
        )
    }

    columns = ("code", "as_of", *FUNDAMENTAL_KEYS, "source")
    insert_sql = (
        f"INSERT INTO fundamentals({', '.join(columns)}) "
        f"VALUES({', '.join('?' * len(columns))}) "
        f"ON CONFLICT(code, as_of) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in (*FUNDAMENTAL_KEYS, "source"))
    )

    saved = failed = 0
    for i, (code, ticker) in enumerate(targets, start=1):
        data = provider.fetch_fundamentals(ticker, close=closes.get(code), deep=deep)
        if data is None:
            failed += 1
        else:
            conn.execute(
                insert_sql,
                (code, as_of, *[_f(data.get(k)) for k in FUNDAMENTAL_KEYS], provider.name),
            )
            saved += 1
        if sleep:
            time.sleep(sleep)
        if i % 50 == 0:
            conn.commit()
            progress(f"  fundamentals {i}/{len(targets)} (取得 {saved} / 失敗 {failed})")

    conn.commit()
    progress(f"fundamentals: {saved} 銘柄 (失敗 {failed}) as_of={as_of}")
    return saved


# --- 検索スナップショット ----------------------------------------------------


def rebuild_snapshot(
    conn: sqlite3.Connection,
    *,
    max_stale_days: int | None = _UNSET,
    progress: ProgressFn = _noop,
) -> int:
    """銘柄マスタ + 最新指標 + 最新ファンダを 1 テーブルに畳む。

    検索はこのテーブルだけを見るので、株価テーブルが何百万行あっても
    レスポンスは銘柄数 (数千行) のスキャンで済む。
    """
    if max_stale_days is _UNSET:
        max_stale_days = config.MAX_STALE_DAYS

    ind_cols = ", ".join(f"i.{c}" for c in INDICATOR_KEYS)
    fund_cols = ", ".join(f"f.{c}" for c in FUNDAMENTAL_KEYS)
    target_cols = ", ".join((*INDICATOR_KEYS, *FUNDAMENTAL_KEYS))

    stale_clause = ""
    if max_stale_days is not None and max_stale_days >= 0:
        # 全銘柄で最も新しい株価日を基準に、そこから離れすぎた銘柄を落とす
        stale_clause = (
            "AND li.date >= date((SELECT MAX(date) FROM indicators), "
            f"'-{int(max_stale_days)} day')"
        )

    conn.execute("DELETE FROM screen_snapshot")
    conn.execute(
        f"""
        INSERT INTO screen_snapshot (
            code, name, market, sector33, sector17, scale,
            price_date, fundamental_date, {target_cols}, updated_at
        )
        SELECT s.code, s.name, s.market, s.sector33, s.sector17, s.scale,
               i.date, f.as_of, {ind_cols}, {fund_cols}, datetime('now')
        FROM stocks s
        JOIN (SELECT code, MAX(date) AS date FROM indicators GROUP BY code) li
          ON li.code = s.code
        JOIN indicators i
          ON i.code = li.code AND i.date = li.date
        LEFT JOIN (SELECT code, MAX(as_of) AS as_of FROM fundamentals GROUP BY code) lf
          ON lf.code = s.code
        LEFT JOIN fundamentals f
          ON f.code = lf.code AND f.as_of = lf.as_of
        WHERE 1 = 1 {stale_clause}
        """
    )
    conn.commit()
    count = db.table_count(conn, "screen_snapshot")
    progress(f"snapshot: {count} 銘柄")
    return count


# --- まとめて実行 ------------------------------------------------------------


def run_daily(
    conn: sqlite3.Connection,
    price_provider: PriceProvider,
    fundamentals_provider: FundamentalsProvider | None = None,
    *,
    codes: Sequence[str] | None = None,
    limit: int | None = None,
    deep: bool = False,
    progress: ProgressFn = _noop,
) -> dict[str, int]:
    """日次バッチ。cron や GitHub Actions からはこれを 1 本呼べばよい。"""
    run_id = db.start_run(conn, "daily")
    try:
        result = {
            "prices": ingest_prices(
                conn, price_provider, codes=codes, limit=limit, progress=progress
            ),
            "indicators": rebuild_indicators(
                conn, codes=codes, limit=limit, progress=progress
            ),
        }
        if fundamentals_provider is not None:
            result["fundamentals"] = ingest_fundamentals(
                conn, fundamentals_provider, codes=codes, limit=limit,
                deep=deep, progress=progress,
            )
        result["snapshot"] = rebuild_snapshot(conn, progress=progress)
    except Exception as e:
        db.finish_run(conn, run_id, "failed", detail=f"{type(e).__name__}: {e}")
        raise
    db.finish_run(conn, run_id, "ok", rows=result.get("prices", 0))
    return result
