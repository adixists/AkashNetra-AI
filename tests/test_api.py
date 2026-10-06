import pytest
from fastapi.testclient import TestClient

from akashnetra.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "data_mode" in data

def test_init_dates(client):
    res = client.get("/init-dates")
    assert res.status_code == 200
    data = res.json()
    assert "dates" in data
    assert isinstance(data["dates"], list)

def test_alerts(client):
    dates = client.get("/init-dates").json()["dates"]
    if not dates:
        pytest.skip("No init dates available")

    date = dates[0]
    res = client.get(f"/alerts?init_date={date}&lead_day=1")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data

def test_alerts_summary(client):
    dates = client.get("/init-dates").json()["dates"]
    if not dates:
        pytest.skip("No init dates available")

    date = dates[0]
    res = client.get(f"/alerts/summary?init_date={date}")
    assert res.status_code == 200
    data = res.json()
    assert "tiles" in data
    assert len(data["tiles"]) > 0
    assert data["tiles"][0]["lead_day"] >= 1

def test_box_detail(client):
    dates = client.get("/init-dates").json()["dates"]
    if not dates:
        pytest.skip("No init dates available")

    date = dates[0]
    alerts = client.get(f"/alerts?init_date={date}&lead_day=1").json()
    if not alerts["features"]:
        pytest.skip("No features for date")

    box_id = alerts["features"][0]["properties"]["box_id"]
    res = client.get(f"/box/{box_id}?init_date={date}&lead_day=1")
    assert res.status_code == 200
    data = res.json()
    assert data["box_id"] == box_id
    assert "reason" in data
    assert "drivers" in data

def test_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "metrics" in res.json()
