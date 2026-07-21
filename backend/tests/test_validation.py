"""Input validation and clear error handling."""

from __future__ import annotations


def test_missing_field_returns_structured_422(client, sample_reading):
    bad = dict(sample_reading)
    bad.pop("sensor_21")
    r = client.post("/predict", json=bad)

    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == "Invalid input data."
    assert "body.sensor_21" in [e["field"] for e in body["errors"]]


def test_wrong_type_returns_422(client, sample_reading):
    bad = {**sample_reading, "sensor_01": "not-a-number"}
    r = client.post("/predict", json=bad)

    assert r.status_code == 422
    assert any(e["field"] == "body.sensor_01" for e in r.json()["errors"])


def test_empty_batch_returns_422(client):
    r = client.post("/predict/batch", json={"readings": []})
    assert r.status_code == 422


def test_unexpected_fields_are_ignored(client, sample_reading):
    payload = {**sample_reading, "unexpected_sensor": 999}
    assert client.post("/predict", json=payload).status_code == 200
