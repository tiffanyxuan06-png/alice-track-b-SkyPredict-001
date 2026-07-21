"""Train and export a baseline RUL regression model for Track B.

This reproduces the finalized pipeline defined in the notebooks and exports the
artifact contract the FastAPI backend loads at startup:

    models/model.pkl            -> a fitted sklearn Pipeline (scaler + regressor)
    models/model_metadata.json  -> feature names, target, metrics, versions, thresholds

The team replaces the estimator here with their tuned model from session 2/3.
As long as the artifact contract (same feature order + metadata keys) is kept,
the backend keeps working unchanged.

Run:
    python models/train_baseline.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

# --- Target / feature contract (must match the notebook) ---------------------
TARGET = "RUL"
RUL_CLIP = 125  # C-MAPSS standard clip applied to the target in the notebook

# Decision-support risk bands on predicted RUL (cycles remaining).
RISK_THRESHOLDS = {"high_below": 30, "medium_below": 75}


def load_split(name: str) -> pd.DataFrame:
    """Load a processed split, dropping the notebook's duplicated RUL column."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    # The notebook wrote RUL twice; pandas renames the second to "RUL.1".
    df = df.loc[:, ~df.columns.str.fullmatch(r"RUL\.\d+")]
    return df


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = np.clip(model.predict(X), 0, RUL_CLIP)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def main() -> None:
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    feature_names = [c for c in train_df.columns if c != TARGET]
    constant_features = [c for c in feature_names if train_df[c].nunique() == 1]

    X_train, y_train = train_df[feature_names], train_df[TARGET]
    X_val, y_val = val_df[feature_names], val_df[TARGET]
    X_test, y_test = test_df[feature_names], test_df[TARGET]

    # MinMaxScaler mirrors the normalization explored in the notebook; the tree
    # ensemble is scale-invariant but the scaler keeps the artifact contract
    # explicit (a fitted preprocessing object travels with the model).
    model = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
    }

    # Note: feature importances are NOT stored here. The backend reads them
    # live from the fitted model (models/model.pkl) to avoid duplicating —
    # and risking drift from — what the model already carries.

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.pkl")

    metadata = {
        "project": "ALICE Track B - Explainable Engine Health & RUL Prediction",
        "dataset": "NASA C-MAPSS FD001",
        "model_type": "sklearn.Pipeline(MinMaxScaler + RandomForestRegressor)",
        "target": TARGET,
        "rul_clip": RUL_CLIP,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "constant_features": constant_features,
        "risk_thresholds": RISK_THRESHOLDS,
        "metrics": metrics,
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("Saved:")
    print(f"  {MODELS_DIR / 'model.pkl'}")
    print(f"  {MODELS_DIR / 'model_metadata.json'}")
    print("\nMetrics (RUL cycles):")
    for split, m in metrics.items():
        print(f"  {split:5} | RMSE {m['rmse']:6.2f} | MAE {m['mae']:6.2f} | R2 {m['r2']:.3f}")


if __name__ == "__main__":
    main()
