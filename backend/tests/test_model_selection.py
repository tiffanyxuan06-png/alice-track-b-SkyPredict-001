"""Per-request model selection via the ?model= query parameter."""

from __future__ import annotations

from backend.tests.test_prediction import assert_valid_prediction


def test_predict_with_each_model(client, sample_reading):
    for model in ("rf", "xgb"):
        r = client.post("/predict", params={"model": model}, json=sample_reading)
        assert r.status_code == 200, r.text
        assert_valid_prediction(r.json())


def test_default_matches_explicit_default(client, sample_reading):
    default = client.get("/health").json()["default_model"]
    implicit = client.post("/predict", json=sample_reading).json()
    explicit = client.post(
        "/predict", params={"model": default}, json=sample_reading
    ).json()
    assert implicit == explicit


def test_models_can_disagree(client, batch_readings):
    # Two different fitted models need not predict identically.
    rf = client.post("/predict/batch", params={"model": "rf"},
                     json={"readings": batch_readings}).json()
    xgb = client.post("/predict/batch", params={"model": "xgb"},
                      json={"readings": batch_readings}).json()
    rf_ruls = [p["predicted_rul"] for p in rf["predictions"]]
    xgb_ruls = [p["predicted_rul"] for p in xgb["predictions"]]
    assert rf_ruls != xgb_ruls


def test_unknown_model_returns_422(client, sample_reading):
    r = client.post("/predict", params={"model": "banana"}, json=sample_reading)
    assert r.status_code == 422


def test_permutation_importance_per_model(client):
    rf = client.get("/explain/permutation-importance", params={"model": "rf"}).json()
    xgb = client.get("/explain/permutation-importance", params={"model": "xgb"}).json()
    assert len(rf) == len(xgb) == 24
    # Each is sorted most-important first.
    for items in (rf, xgb):
        means = [i["importance_mean"] for i in items]
        assert means == sorted(means, reverse=True)
