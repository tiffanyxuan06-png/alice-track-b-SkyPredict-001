"""Global and per-reading explainability endpoints."""

from __future__ import annotations

from backend.tests.test_prediction import assert_valid_prediction


def test_global_importances(client):
    importances = client.get("/explain/global").json()
    values = [item["importance"] for item in importances]

    assert len(importances) == 24
    assert values == sorted(values, reverse=True)  # ranked most-important first
    assert all(0 <= v <= 1 for v in values)
    # Forest importances sum to 1; values are rounded to 5 dp, so allow slack.
    assert abs(sum(values) - 1.0) < 1e-3


def test_explain_reading(client, sample_reading):
    body = client.post("/explain", json=sample_reading).json()

    assert_valid_prediction(body["prediction"])
    assert len(body["global_importances"]) == 8  # default top_k
    assert len(body["local_contributions"]) == 8

    top = body["local_contributions"][0]
    assert {"feature", "value", "scaled_value", "saliency"} <= set(top)
    # local_contributions are ranked by descending saliency
    saliencies = [c["saliency"] for c in body["local_contributions"]]
    assert saliencies == sorted(saliencies, reverse=True)


def test_explain_prediction_matches_predict(client, sample_reading):
    predicted = client.post("/predict", json=sample_reading).json()
    explained = client.post("/explain", json=sample_reading).json()["prediction"]
    assert predicted["predicted_rul"] == explained["predicted_rul"]
    assert predicted["risk_level"] == explained["risk_level"]
