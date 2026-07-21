"""Single and batch RUL prediction endpoints."""

from __future__ import annotations

RUL_CLIP = 125


def assert_valid_prediction(p: dict) -> None:
    assert 0 <= p["predicted_rul"] <= RUL_CLIP
    assert p["risk_level"] in {"High", "Medium", "Low"}
    assert 0 <= p["risk_score"] <= 1
    assert p["recommended_action"]


def test_predict_single(client, sample_reading):
    r = client.post("/predict", json=sample_reading)
    assert r.status_code == 200
    assert_valid_prediction(r.json())


def test_predict_echoes_identifiers(client, sample_reading):
    payload = {**sample_reading, "unit_number": 42, "time_in_cycles": 7}
    p = client.post("/predict", json=payload).json()
    assert p["unit_number"] == 42
    assert p["time_in_cycles"] == 7


def test_risk_level_consistent_with_thresholds(client, sample_reading):
    thresholds = client.get("/model-info").json()["risk_thresholds"]
    p = client.post("/predict", json=sample_reading).json()
    rul = p["predicted_rul"]
    if rul < thresholds["high_below"]:
        expected = "High"
    elif rul < thresholds["medium_below"]:
        expected = "Medium"
    else:
        expected = "Low"
    assert p["risk_level"] == expected


def test_predict_batch(client, batch_readings):
    r = client.post("/predict/batch", json={"readings": batch_readings})
    assert r.status_code == 200
    body = r.json()

    assert len(body["predictions"]) == len(batch_readings)
    for p in body["predictions"]:
        assert_valid_prediction(p)

    summary = body["summary"]
    assert summary["total_engines"] == len(batch_readings)
    assert (
        summary["high_risk"] + summary["medium_risk"] + summary["low_risk"]
        == len(batch_readings)
    )
