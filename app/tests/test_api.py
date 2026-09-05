import pytest
from fastapi.testclient import TestClient

from autostock_app.api import create_app


@pytest.fixture
def client(seeded_db_path):
    return TestClient(create_app(seeded_db_path))


@pytest.fixture
def empty_client(tmp_path):
    return TestClient(create_app(tmp_path / "empty.db"))


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["stocks"] > 0


def test_meta_describes_the_search_form(client):
    body = client.get("/api/meta").json()
    assert body["empty"] is False
    assert len(body["fields"]) > 20
    assert body["presets"]
    assert body["categories"]["market"]
    assert all({"key", "label", "group", "unit", "signed"} <= set(f) for f in body["fields"])


def test_meta_reports_an_empty_database(empty_client):
    body = empty_client.get("/api/meta").json()
    assert body["empty"] is True
    assert body["status"]["stocks"] == 0


def test_screen_applies_filters(client):
    res = client.post("/api/screen", json={
        "ranges": [{"field": "rsi_14", "max": 60}],
        "sort_by": "rsi_14", "sort_desc": False, "limit": 5,
    })
    assert res.status_code == 200
    body = res.json()
    assert all(row["rsi_14"] <= 60 for row in body["rows"])
    assert body["count"] <= 5


def test_screen_with_no_conditions_returns_everything(client):
    body = client.post("/api/screen", json={}).json()
    assert body["total"] == client.get("/api/health").json()["stocks"]


@pytest.mark.parametrize("payload", [
    {"sort_by": "; DROP TABLE stocks"},
    {"ranges": [{"field": "evil", "max": 1}]},
    {"categories": {"per": ["x"]}},
])
def test_screen_rejects_unknown_fields(client, payload):
    assert client.post("/api/screen", json=payload).status_code == 400


def test_screen_validates_limit(client):
    assert client.post("/api/screen", json={"limit": 9999}).status_code == 422
    assert client.post("/api/screen", json={"offset": -1}).status_code == 422


def test_stock_detail(client):
    code = client.post("/api/screen", json={"limit": 1}).json()["rows"][0]["code"]
    body = client.get(f"/api/stocks/{code}", params={"days": 40}).json()
    assert body["stock"]["code"] == code
    assert len(body["prices"]) == 40
    dates = [p["date"] for p in body["prices"]]
    assert dates == sorted(dates)  # チャート用に昇順で返る
    assert body["snapshot"]["code"] == code


def test_stock_detail_404(client):
    assert client.get("/api/stocks/0000").status_code == 404


def test_ui_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/favicon.ico").status_code == 204
