"""Shared fixtures for the backend end-to-end tests.

These are true e2e tests: the ``client`` fixture starts the app through its
lifespan, which loads the real exported models (RandomForest + XGBoost), and
requests hit the real routers and services over HTTP. Sample payloads are built
from real rows of ``data/processed/test.csv`` (no hardcoded feature data).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = REPO_ROOT / "data" / "processed" / "test.csv"


@pytest.fixture(scope="session")
def client():
    # The context manager runs startup/shutdown, so the model loads once.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def feature_names(client) -> list[str]:
    return client.get("/model-info").json()["feature_names"]


@pytest.fixture(scope="session")
def _test_rows(feature_names) -> pd.DataFrame:
    if not TEST_CSV.exists():
        pytest.skip(f"processed test data not found at {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    return df[feature_names]  # drop the target column(s), keep model features


def _row_payload(rows: pd.DataFrame, i: int, feature_names: list[str]) -> dict:
    row = rows.iloc[i]
    return {name: float(row[name]) for name in feature_names}


@pytest.fixture
def sample_reading(_test_rows, feature_names) -> dict:
    """A single real engine reading (an early-life, healthy row)."""
    return _row_payload(_test_rows, 100, feature_names)


@pytest.fixture
def batch_readings(_test_rows, feature_names) -> list[dict]:
    """Several real readings spread across the file for variety."""
    n = len(_test_rows)
    indices = [i for i in (10, 300, 900, 2000) if i < n] or [0]
    return [_row_payload(_test_rows, i, feature_names) for i in indices]
