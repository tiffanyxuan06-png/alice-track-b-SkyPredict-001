"""Health and model-info endpoints."""

from __future__ import annotations


def test_root(client):
    body = client.get("/").json()
    assert "docs" in body


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["app_version"]


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["target"] == "RUL"
    assert info["n_features"] == len(info["feature_names"]) == 24
    assert set(info["risk_thresholds"]) == {"high_below", "medium_below"}
    assert {"rmse", "mae", "r2"} <= set(info["metrics"]["test"])
