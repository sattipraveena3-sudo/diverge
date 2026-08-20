from fastapi.testclient import TestClient

from api.main import app
from diverge.data import generate_synthetic_record

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["clinical_use"] is False
    assert "sensor-agnostic" in payload["scope"]


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


def test_analyze_pair_is_sensor_agnostic():
    frame = generate_synthetic_record(
        duration_s=150, divergence_start_s=80, adverse_start_s=120, seed=11
    ).frame
    body = {
        "time_s": frame.time_s.tolist(),
        "signal_a": frame.hr.ffill().bfill().tolist(),
        "signal_b": frame.spo2.ffill().bfill().tolist(),
        "signal_a_name": "wearable_channel_1",
        "signal_b_name": "wearable_channel_2",
        "divergence_threshold": 0.5,
    }
    r = client.post("/analyze-pair", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["signal_a_name"] == "wearable_channel_1"
    assert data["signal_b_name"] == "wearable_channel_2"
    assert data["samples_analyzed"] >= 100
    assert 0 <= data["divergence_score"] <= 1
    assert isinstance(data["risk_flag"], bool)


def test_analyze_pair_rejects_mismatched_lengths():
    r = client.post(
        "/analyze-pair",
        json={
            "time_s": list(range(10)),
            "signal_a": [1.0] * 10,
            "signal_b": [1.0] * 9,
        },
    )
    assert r.status_code == 422
