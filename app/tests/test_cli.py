import json

import pytest

from autostock_app.cli import main


def run(capsys, *argv) -> tuple[int, str, str]:
    """(終了コード, 標準出力, 標準エラー) を返す。

    capsys.readouterr() は呼ぶたびにバッファを空にするので、
    ここで 1 度だけ読んで両方返す。
    """
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_fields_lists_every_group(capsys):
    code, out, _ = run(capsys, "fields")
    assert code == 0
    for label in ("銘柄属性", "テクニカル", "ファンダメンタルズ"):
        assert label in out


def test_presets_are_listed(capsys):
    code, out, _ = run(capsys, "presets")
    assert code == 0
    assert "value_dividend" in out


def test_status_on_a_seeded_db(capsys, seeded_db_path):
    code, out, _ = run(capsys, "--db", str(seeded_db_path), "status")
    assert code == 0
    assert "screen_snapshot" in out
    assert "最新株価日" in out


def test_screen_json_output(capsys, seeded_db_path):
    code, out, _ = run(capsys, "--db", str(seeded_db_path), "screen", "--limit", "3", "--json")
    assert code == 0
    body = json.loads(out)
    assert body["count"] <= 3
    assert body["rows"]


def test_screen_applies_min_and_max(capsys, seeded_db_path):
    code, out, _ = run(
        capsys, "--db", str(seeded_db_path), "screen",
        "--max", "rsi_14=50", "--min", "pbr=0.1", "--json",
    )
    assert code == 0
    for row in json.loads(out)["rows"]:
        assert row["rsi_14"] <= 50
        assert row["pbr"] >= 0.1


def test_screen_with_preset(capsys, seeded_db_path):
    code, out, _ = run(
        capsys, "--db", str(seeded_db_path), "screen", "--preset", "pullback", "--json"
    )
    assert code == 0
    assert "total" in json.loads(out)


def test_unknown_preset_is_rejected(capsys, seeded_db_path):
    code, _, err = run(capsys, "--db", str(seeded_db_path), "screen", "--preset", "nope")
    assert code == 2


def test_unknown_field_is_rejected_by_the_parser(seeded_db_path):
    with pytest.raises(SystemExit):
        main(["--db", str(seeded_db_path), "screen", "--max", "nonexistent=1"])


def test_screen_on_an_empty_db_explains_what_to_run(capsys, tmp_path):
    code, _, err = run(capsys, "--db", str(tmp_path / "empty.db"), "screen")
    assert code == 1
    assert "demo-seed" in err


def test_ingest_universe_reports_a_missing_file(capsys, tmp_path):
    code, _, err = run(
        capsys, "--db", str(tmp_path / "x.db"), "ingest", "universe",
        "--universe-file", str(tmp_path / "missing.xls"),
    )
    assert code == 1
    assert "data_j.xls" in err


def test_demo_seed_builds_a_searchable_db(capsys, tmp_path):
    db_path = tmp_path / "demo.db"
    code, _, err = run(
        capsys, "--db", str(db_path), "demo-seed", "--stocks", "3", "--start", "2024-01-01"
    )
    assert code == 0
    code, out, _ = run(capsys, "--db", str(db_path), "screen", "--limit", "5", "--json")
    assert code == 0
    assert json.loads(out)["total"] == 3
