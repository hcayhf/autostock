"""コマンドラインインターフェース。

    python -m autostock_app.cli init-db
    python -m autostock_app.cli ingest universe
    python -m autostock_app.cli ingest daily
    python -m autostock_app.cli screen --preset value_dividend
    python -m autostock_app.cli screen --max per=15 --min dividend_yield=3
    python -m autostock_app.cli serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, db, ingest
from .fields import FIELDS_BY_KEY, field_or_raise
from .providers import (
    FUNDAMENTALS_PROVIDERS,
    PRICE_PROVIDERS,
    get_fundamentals_provider,
    get_price_provider,
)
from .screener import (
    PRESETS,
    PRESETS_BY_ID,
    Range,
    ScreenQuery,
    query_from_dict,
    screen,
    snapshot_status,
)
from .universe import load_universe_file, save_universe


# --- 引数のヘルパ ------------------------------------------------------------


def _parse_condition(raw: str) -> tuple[str, float]:
    """"per=15" を ("per", 15.0) にする。"""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"条件は field=値 の形式で指定してください (例: per=15): {raw!r}"
        )
    key, _, value = raw.partition("=")
    key = key.strip()
    try:
        field_or_raise(key)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"未知の指標です: {key!r}\n"
            f"利用できる指標は `python -m autostock_app.cli fields` で確認できます。"
        ) from None
    try:
        return key, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値として読めません: {value!r}") from None


def _format(key: str, value) -> str:
    if value is None:
        return "—"
    field = FIELDS_BY_KEY.get(key)
    if field is None or field.dtype == "text":
        return str(value)
    unit = field.unit
    if unit == "%":
        return f"{value:+.2f}%" if field.signed else f"{value:.2f}%"
    if unit == "倍":
        return f"{value:.2f}"
    if unit in ("円", "億円", "百万円", "株"):
        return f"{value:,.0f}"
    return f"{value:.1f}"


# --- 各コマンド --------------------------------------------------------------


def cmd_init_db(args) -> int:
    with db.session(args.db) as conn:
        print(f"初期化しました: {conn.execute('PRAGMA database_list').fetchone()['file']}")
    return 0


def cmd_status(args) -> int:
    with db.session(args.db) as conn:
        status = snapshot_status(conn)
        print(f"DB: {args.db or config.DB_PATH}")
        for table in ("stocks", "prices", "indicators", "fundamentals", "screen_snapshot"):
            print(f"  {table:16s} {db.table_count(conn, table):>10,} 行")
        print(f"  最新株価日        {status.get('price_date') or '—'}")
        print(f"  最新財務取得日    {status.get('fundamental_date') or '—'}")
        runs = conn.execute(
            "SELECT kind, started_at, finished_at, status, rows FROM ingest_runs "
            "ORDER BY id DESC LIMIT 5"
        ).fetchall()
        if runs:
            print("  直近の取り込み:")
            for r in runs:
                print(f"    {r['started_at']}  {r['kind']:<12} {r['status']:<8} {r['rows']:>8,} 行")
    return 0


def cmd_fields(args) -> int:
    from .fields import ALL_FIELDS, GROUP_LABELS

    current = None
    for f in ALL_FIELDS:
        if f.group != current:
            current = f.group
            print(f"\n[{GROUP_LABELS.get(current, current)}]")
        unit = f" ({f.unit})" if f.unit else ""
        print(f"  {f.key:20s} {f.label}{unit}")
        if f.desc:
            print(f"  {'':20s}   {f.desc}")
    return 0


def cmd_ingest(args) -> int:
    with db.session(args.db) as conn:
        if args.what in ("universe", "all"):
            frame = load_universe_file(args.universe_file)
            print(f"universe: {save_universe(conn, frame)} 銘柄")

        if args.what == "prices":
            ingest.ingest_prices(
                conn, get_price_provider(args.price_provider),
                limit=args.limit, codes=args.codes, full=args.full, progress=print,
            )
        elif args.what == "indicators":
            ingest.rebuild_indicators(
                conn, limit=args.limit, codes=args.codes, full=args.full, progress=print
            )
        elif args.what == "fundamentals":
            ingest.ingest_fundamentals(
                conn, get_fundamentals_provider(args.fundamentals_provider),
                limit=args.limit, codes=args.codes, deep=args.deep, progress=print,
            )
        elif args.what == "snapshot":
            ingest.rebuild_snapshot(conn, progress=print)
        elif args.what in ("daily", "all"):
            fundamentals = (
                None if args.skip_fundamentals
                else get_fundamentals_provider(args.fundamentals_provider)
            )
            ingest.run_daily(
                conn, get_price_provider(args.price_provider), fundamentals,
                limit=args.limit, codes=args.codes, deep=args.deep, progress=print,
            )
    return 0


def cmd_demo_seed(args) -> int:
    from .demo import seed_demo

    with db.session(args.db) as conn:
        result = seed_demo(conn, n_stocks=args.stocks, start=args.start, progress=print)
    print(f"\n完了: {result}")
    print("※ これは合成データです。実在の株価ではありません。")
    print("   `python -m autostock_app.cli serve` で UI を確認できます。")
    return 0


def cmd_screen(args) -> int:
    if args.preset:
        if args.preset not in PRESETS_BY_ID:
            print(f"未知のプリセット: {args.preset}", file=sys.stderr)
            print(f"利用可能: {', '.join(PRESETS_BY_ID)}", file=sys.stderr)
            return 2
        preset = PRESETS_BY_ID[args.preset]
        query = query_from_dict({**preset, "limit": args.limit})
        # --json のときに混ざると機械可読でなくなるので説明は stderr へ
        print(f"[{preset['name']}] {preset['description']}\n",
              file=sys.stderr if args.json else sys.stdout)
    else:
        ranges: dict[str, dict] = {}
        for key, value in args.min or []:
            ranges.setdefault(key, {"field": key})["min"] = value
        for key, value in args.max or []:
            ranges.setdefault(key, {"field": key})["max"] = value
        query = ScreenQuery(
            text=args.text,
            categories={
                k: v for k, v in (
                    ("market", args.market), ("sector33", args.sector), ("scale", args.scale)
                ) if v
            },
            ranges=[Range(**r) for r in ranges.values()],
            sort_by=args.sort,
            sort_desc=not args.asc,
            limit=args.limit,
        )

    with db.session(args.db) as conn:
        if not snapshot_status(conn).get("stocks"):
            print(
                "検索用スナップショットが空です。\n"
                "  実データ:   python -m autostock_app.cli ingest daily\n"
                "  合成データ: python -m autostock_app.cli demo-seed",
                file=sys.stderr,
            )
            return 1
        result = screen(conn, query)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    columns = result["columns"]
    widths = {
        c: max(len(FIELDS_BY_KEY[c].label), *(len(_format(c, r[c])) for r in result["rows"]))
        if result["rows"] else len(FIELDS_BY_KEY[c].label)
        for c in columns
    }
    print("  ".join(FIELDS_BY_KEY[c].label.rjust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in result["rows"]:
        print("  ".join(_format(c, row[c]).rjust(widths[c]) for c in columns))
    print(f"\n{result['total']} 件中 {result['count']} 件を表示")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    db_path = Path(args.db) if args.db else config.DB_PATH
    if not db_path.exists():
        print(f"DB がありません: {db_path}", file=sys.stderr)
        print("  python -m autostock_app.cli demo-seed  で合成データを作れます。", file=sys.stderr)
        return 1

    # uvicorn のワーカーからも同じ DB を見せる
    import os
    os.environ["AUTOSTOCK_DB"] = str(db_path)

    print(f"http://{args.host}:{args.port} で起動します (DB: {db_path})")
    uvicorn.run(
        "autostock_app.api:app",
        host=args.host, port=args.port, reload=args.reload, log_level="info",
    )
    return 0


def cmd_presets(args) -> int:
    for p in PRESETS:
        print(f"{p['id']:18s} {p['name']}")
        print(f"{'':18s} {p['description']}")
    return 0


# --- パーサ ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autostock",
        description="日本株の日次データ取得と指標スクリーニング",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", help=f"SQLite ファイル (既定: {config.DB_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="DB とテーブルを作る").set_defaults(func=cmd_init_db)
    sub.add_parser("status", help="取り込み状況を表示").set_defaults(func=cmd_status)
    sub.add_parser("fields", help="検索できる指標の一覧").set_defaults(func=cmd_fields)
    sub.add_parser("presets", help="プリセット条件の一覧").set_defaults(func=cmd_presets)

    p_ingest = sub.add_parser("ingest", help="データを取り込む")
    p_ingest.add_argument(
        "what",
        choices=["universe", "prices", "indicators", "fundamentals", "snapshot", "daily", "all"],
        help="daily = 株価→指標→財務→スナップショット を通しで実行",
    )
    p_ingest.add_argument("--universe-file", help=f"銘柄一覧 (既定: {config.UNIVERSE_FILE})")
    p_ingest.add_argument("--limit", type=int, help="先頭 N 銘柄だけ処理する (動作確認用)")
    p_ingest.add_argument("--codes", nargs="+", help="対象の証券コードを直接指定")
    p_ingest.add_argument("--full", action="store_true", help="差分ではなく全期間を取り直す")
    p_ingest.add_argument("--deep", action="store_true",
                          help="財務諸表も取得して自己資本比率を埋める (かなり遅い)")
    p_ingest.add_argument("--skip-fundamentals", action="store_true",
                          help="daily で財務の取得を省く")
    p_ingest.add_argument("--price-provider", default="yfinance", choices=sorted(PRICE_PROVIDERS))
    p_ingest.add_argument("--fundamentals-provider", default="yfinance",
                          choices=sorted(FUNDAMENTALS_PROVIDERS))
    p_ingest.set_defaults(func=cmd_ingest)

    p_demo = sub.add_parser(
        "demo-seed", help="合成データでデモ用 DB を作る (ネットワーク不要)"
    )
    p_demo.add_argument("--stocks", type=int, default=150)
    p_demo.add_argument("--start", default="2021-01-01")
    p_demo.set_defaults(func=cmd_demo_seed)

    p_screen = sub.add_parser(
        "screen", help="銘柄を検索する",
        epilog="例: screen --max per=15 --min dividend_yield=3 --sort dividend_yield",
    )
    p_screen.add_argument("--preset", help="プリセット条件を使う (presets コマンドで一覧)")
    p_screen.add_argument("--min", action="append", type=_parse_condition, metavar="FIELD=値",
                          help="下限。繰り返し指定可")
    p_screen.add_argument("--max", action="append", type=_parse_condition, metavar="FIELD=値",
                          help="上限。繰り返し指定可")
    p_screen.add_argument("--text", help="コード / 銘柄名の部分一致")
    p_screen.add_argument("--market", nargs="+", help="市場区分で絞る")
    p_screen.add_argument("--sector", nargs="+", help="33業種で絞る")
    p_screen.add_argument("--scale", nargs="+", help="規模区分で絞る")
    p_screen.add_argument("--sort", default="turnover_20d", help="並び替えに使う指標")
    p_screen.add_argument("--asc", action="store_true", help="昇順にする")
    p_screen.add_argument("--limit", type=int, default=20)
    p_screen.add_argument("--json", action="store_true", help="JSON で出力")
    p_screen.set_defaults(func=cmd_screen)

    p_serve = sub.add_parser("serve", help="Web UI と API を起動する")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
