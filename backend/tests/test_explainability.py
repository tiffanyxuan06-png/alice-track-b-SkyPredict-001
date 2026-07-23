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


def test_permutation_importance(client):
    items = client.get("/explain/permutation-importance").json()
    assert len(items) == 24
    means = [i["importance_mean"] for i in items]
    assert means == sorted(means, reverse=True)  # ranked most-important first
    for item in items:
        assert {"feature", "importance_mean", "importance_std"} <= set(item)
        assert item["importance_std"] >= 0


def test_partial_dependence(client):
    curves = client.get("/explain/partial-dependence").json()
    assert len(curves) > 0  # non-constant features
    for c in curves:
        assert {"feature", "grid", "average"} <= set(c)
        assert len(c["grid"]) == len(c["average"]) > 1

    # single-feature filter returns just that curve
    name = curves[0]["feature"]
    one = client.get("/explain/partial-dependence", params={"feature": name}).json()
    assert len(one) == 1 and one[0]["feature"] == name


def test_explain_reading(client, sample_reading):
    body = client.post("/explain", json=sample_reading).json()

    assert_valid_prediction(body["prediction"])
    assert isinstance(body["base_value"], (int, float))
    assert len(body["global_importances"]) == 8  # default top_k
    assert len(body["local_contributions"]) == 8

    top = body["local_contributions"][0]
    assert {"feature", "value", "shap_value"} <= set(top)
    # local_contributions are ranked by descending |shap_value|
    magnitudes = [abs(c["shap_value"]) for c in body["local_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_prediction_matches_predict(client, sample_reading):
    predicted = client.post("/predict", json=sample_reading).json()
    explained = client.post("/explain", json=sample_reading).json()["prediction"]
    assert predicted["predicted_rul"] == explained["predicted_rul"]
    assert predicted["risk_level"] == explained["risk_level"]
