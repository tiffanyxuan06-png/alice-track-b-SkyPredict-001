"""Train and export the RUL regression models for Track B.

Trains two models on the same StandardScaler pipeline and exports one artifact
pair per model, which the FastAPI backend loads into a selectable registry:

    models/model_rf.pkl  + models/model_rf_metadata.json    (RandomForest)
    models/model_xgb.pkl + models/model_xgb_metadata.json   (XGBoost)

Each metadata file is the self-contained contract the backend loads: feature
names, target, RUL clip, risk thresholds, metrics, permutation importance and
versions. Keep the contract (same feature order + metadata keys) and the backend
keeps working unchanged.

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
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

# The models to train and export, keyed by the name the API selects with ?model=.
ESTIMATORS = {
    "rf": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "xgb": XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
}


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


def compute_partial_dependence(model, X, feature_names, constant_features, grid_resolution=20):
    """1-way partial dependence per non-constant feature.

    The marginal effect of a feature on predicted RUL, averaging over the others
    (sklearn.inspection.partial_dependence). Grid values are in the original
    feature units. Constant features are skipped — they have no effect to plot.
    Needs a reference dataset, so this runs at export and the result is stored.
    """
    # sklearn's partial_dependence rejects integer-dtype columns; some sensors
    # are integer-valued in the CSV, so cast to float first.
    X = X.astype("float64")
    curves = []
    for feature in feature_names:
        if feature in constant_features:
            continue
        pd_result = partial_dependence(
            model, X, [feature], grid_resolution=grid_resolution, kind="average"
        )
        curves.append(
            {
                "feature": feature,
                "grid": [float(v) for v in pd_result["grid_values"][0]],
                "average": [float(v) for v in pd_result["average"][0]],
            }
        )
    return curves


def train_and_export(name, estimator, data, feature_names, constant_features):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

    # StandardScaler standardizes each feature to zero mean / unit variance. The
    # tree ensembles are scale-invariant, but the scaler keeps the artifact
    # contract explicit and its stats power the backend's explanations and docs.
    model = Pipeline(steps=[("scaler", StandardScaler()), ("regressor", estimator)])
    model.fit(X_train, y_train)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
    }

    # Permutation importance is model-agnostic and data-driven: it measures the
    # drop in score when each feature is shuffled. It cannot be recovered from
    # the pickled model, so we compute it here (on validation) and store it.
    perm = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
    )
    perm_importance = [
        {"feature": f, "importance_mean": float(m), "importance_std": float(s)}
        for f, m, s in zip(feature_names, perm.importances_mean, perm.importances_std)
    ]

    # Partial dependence on a validation subsample (bounded for speed).
    pdp_sample = X_val.sample(n=min(500, len(X_val)), random_state=RANDOM_SEED)
    partial_dep = compute_partial_dependence(
        model, pdp_sample, feature_names, constant_features
    )

    joblib.dump(model, MODELS_DIR / f"model_{name}.pkl")
    metadata = {
        "project": "ALICE Track B - Explainable Engine Health & RUL Prediction",
        "dataset": "NASA C-MAPSS FD001",
        "model_key": name,
        "model_type": f"sklearn.Pipeline(StandardScaler + {type(estimator).__name__})",
        "target": TARGET,
        "rul_clip": RUL_CLIP,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "constant_features": constant_features,
        "permutation_importance": perm_importance,
        "partial_dependence": partial_dep,
        "risk_thresholds": RISK_THRESHOLDS,
        "metrics": metrics,
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / f"model_{name}_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metrics


def main() -> None:
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    feature_names = [c for c in train_df.columns if c != TARGET]
    constant_features = [c for c in feature_names if train_df[c].nunique() == 1]
    data = [
        (df[feature_names], df[TARGET]) for df in (train_df, val_df, test_df)
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for name, estimator in ESTIMATORS.items():
        all_metrics[name] = train_and_export(
            name, estimator, data, feature_names, constant_features
        )
        print(f"Saved: models/model_{name}.pkl + models/model_{name}_metadata.json")

    print("\nTest-set comparison (RUL cycles):")
    print(f"  {'model':5} | {'RMSE':>6} | {'MAE':>6} | {'R2':>6}")
    for name, m in all_metrics.items():
        t = m["test"]
        print(f"  {name:5} | {t['rmse']:6.2f} | {t['mae']:6.2f} | {t['r2']:6.3f}")


if __name__ == "__main__":
    main()
