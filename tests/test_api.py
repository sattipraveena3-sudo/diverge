from fastapi.testclient import TestClient

from api.main import app
from diverge.data import generate_synthetic_record

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze():
    frame = generate_synthetic_record(
        duration_s=30, divergence_start_s=15, adverse_start_s=20, seed=9
    ).frame
    body = {k: frame[k].ffill().bfill().tolist() for k in ["time_s", "hr", "spo2"]}
    r = client.post("/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["divergence_score"] <= 1
