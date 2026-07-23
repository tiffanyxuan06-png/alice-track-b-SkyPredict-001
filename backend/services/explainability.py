"""Explainability service.

Three views:
  * global      — the model's own feature importances (live from the estimator).
  * permutation — model-agnostic importances computed on validation data at
                  export time (served from the metadata).
  * local       — exact per-reading SHAP values (TreeExplainer). Each sensor's
                  signed contribution (in RUL cycles) to this engine's
                  prediction, with base_value + sum(shap) = the raw prediction.
"""

from __future__ import annotations

import numpy as np
import shap

from backend.schemas import (
    EngineReading,
    ExplanationResponse,
    FeatureImportance,
    LocalContribution,
    PartialDependence,
    PermutationImportance,
    PredictionResponse,
)
from backend.services.artifacts import ModelArtifacts
from backend.services.preprocessing import readings_to_frame

# TreeExplainer construction is not free, so build one per model and reuse it.
# Keyed by the model artifact's key; the registry loads each model once.
_explainers: dict[str, shap.TreeExplainer] = {}


def _get_explainer(artifacts: ModelArtifacts) -> shap.TreeExplainer:
    key = artifacts.metadata.get("model_key") or str(id(artifacts.model))
    explainer = _explainers.get(key)
    if explainer is None:
        # Explain the final estimator; it sees the scaled features.
        explainer = shap.TreeExplainer(artifacts.model[-1])
        _explainers[key] = explainer
    return explainer


def global_importances(
    artifacts: ModelArtifacts, top_k: int = 10
) -> list[FeatureImportance]:
    items = sorted(
        artifacts.feature_importances.items(), key=lambda kv: kv[1], reverse=True
    )
    return [FeatureImportance(feature=f, importance=round(i, 5)) for f, i in items[:top_k]]


def permutation_importances(artifacts: ModelArtifacts) -> list[PermutationImportance]:
    items = sorted(
        artifacts.permutation_importance,
        key=lambda d: d["importance_mean"],
        reverse=True,
    )
    return [PermutationImportance(**item) for item in items]


def partial_dependence_curves(
    artifacts: ModelArtifacts, feature: str | None = None
) -> list[PartialDependence]:
    """Partial dependence curves; optionally filtered to a single feature."""
    curves = artifacts.partial_dependence
    if feature is not None:
        curves = [c for c in curves if c["feature"] == feature]
    return [PartialDependence(**c) for c in curves]


def explain_reading(
    reading: EngineReading,
    prediction: PredictionResponse,
    artifacts: ModelArtifacts,
    top_k: int = 8,
) -> ExplanationResponse:
    frame = readings_to_frame([reading], artifacts.feature_names)
    values = frame.iloc[0]
    # Apply the pipeline's preprocessing, then SHAP-explain the final estimator
    # on the scaled features it actually sees.
    scaled = artifacts.model[:-1].transform(frame)
    explainer = _get_explainer(artifacts)
    shap_values = np.asarray(explainer.shap_values(scaled))[0]
    base_value = float(np.ravel(explainer.expected_value)[0])

    contributions = [
        LocalContribution(
            feature=name,
            value=float(values[name]),
            shap_value=round(float(sv), 4),
        )
        for name, sv in zip(artifacts.feature_names, shap_values)
    ]
    # Rank by magnitude: the sensors that moved this engine's RUL the most.
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return ExplanationResponse(
        prediction=prediction,
        base_value=round(base_value, 4),
        global_importances=global_importances(artifacts, top_k=top_k),
        local_contributions=contributions[:top_k],
    )
