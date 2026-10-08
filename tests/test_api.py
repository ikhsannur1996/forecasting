import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_request.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_atms(client):
    assert client.get("/health").json()["model_loaded"]
    atms = client.get("/atms").json()
    assert len(atms) == 20 and {"atm_id", "location_type", "cassette_capacity"} <= set(atms[0])


def test_forecast_sample(client):
    body = client.post("/forecast", json=SAMPLE).json()
    assert [a["atm_id"] for a in body["atms"]] == SAMPLE["atm_ids"]
    for atm in body["atms"]:
        assert len(atm["days"]) == SAMPLE["horizon"]
        for d in atm["days"]:
            assert d["p05"] <= d["p10"] <= d["forecast"] <= d["p90"] <= d["p95"]
            assert d["recommended_load"] <= atm["cassette_capacity"] + 1e-6
        assert atm["total_forecast"] == pytest.approx(sum(d["forecast"] for d in atm["days"]), abs=0.1)


def test_all_atms_totals_only(client):
    body = client.post("/forecast", json={"horizon": 7, "include_daily": False}).json()
    assert len(body["atms"]) == 20 and body["atms"][0]["days"] is None


def test_long_horizon_warns(client):
    body = client.post("/forecast", json={"atm_ids": ["ATM01"], "horizon": 28}).json()
    assert any("belum divalidasi" in w for w in body["warnings"])


def test_recent_actuals_moves_origin(client):
    path = ROOT / "sample_request_with_actuals.json"
    if not path.exists():
        pytest.skip("model terpilih bukan GBR")
    req = json.loads(path.read_text())
    body = client.post("/forecast", json=req).json()
    n_new = len(next(iter(req["recent_actuals"].values())))
    base = client.post("/forecast", json={"horizon": 1}).json()
    assert body["data_until"] > base["data_until"]
    from datetime import date
    assert (date.fromisoformat(body["data_until"]) - date.fromisoformat(base["data_until"])).days == n_new


@pytest.mark.parametrize("payload", [
    {"atm_ids": ["ATM99"]},
    {"horizon": 0},
    {"horizon": 60},
    {"unknown": 1},
    {"recent_actuals": {"ATM01": {"2026-01-01": 100.0}}},           # tidak semua ATM
    {"recent_actuals": {"ATM01": {"2026-01-01": -5.0}}},            # negatif
])
def test_invalid_requests(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422
