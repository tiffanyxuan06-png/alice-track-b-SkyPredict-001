"""Explainability service.

Two complementary views:
  * global  — the model's overall feature importances (from the exported forest).
  * local   — a lightweight per-reading saliency: global importance weighted by
              each feature's min-max scaled value. This is a transparent,
              dependency-free heuristic (not SHAP); it highlights which sensors
              sit at the extremes of their trained range for this engine.
"""

from __future__ import annotations

from backend.schemas import (
    EngineReading,
    ExplanationResponse,
    FeatureImportance,
    LocalContribution,
    PredictionResponse,
)
from backend.services.artifacts import ModelArtifacts
from backend.services.preprocessing import readings_to_frame


def global_importances(
    artifacts: ModelArtifacts, top_k: int = 10
) -> list[FeatureImportance]:
    items = sorted(
        artifacts.feature_importances.items(), key=lambda kv: kv[1], reverse=True
    )
    return [FeatureImportance(feature=f, importance=round(i, 5)) for f, i in items[:top_k]]


def explain_reading(
    reading: EngineReading,
    prediction: PredictionResponse,
    artifacts: ModelArtifacts,
    top_k: int = 8,
) -> ExplanationResponse:
    frame = readings_to_frame([reading], artifacts.feature_names)
    values = frame.iloc[0]
    # Apply the pipeline's preprocessing (all steps except the final estimator).
    scaled = artifacts.model[:-1].transform(frame)[0]

    importances = artifacts.feature_importances
    contributions: list[LocalContribution] = []
    for name, scaled_val in zip(artifacts.feature_names, scaled):
        imp = importances[name]
        contributions.append(
            LocalContribution(
                feature=name,
                value=float(values[name]),
                scaled_value=round(float(scaled_val), 4),
                saliency=round(float(imp) * float(scaled_val), 5),
            )
        )
    contributions.sort(key=lambda c: c.saliency, reverse=True)

    return ExplanationResponse(
        prediction=prediction,
        global_importances=global_importances(artifacts, top_k=top_k),
        local_contributions=contributions[:top_k],
    )
