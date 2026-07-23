"""Explainability routes: global importances + per-reading explanation."""

from __future__ import annotations

from fastapi import APIRouter

from backend.dependencies import ArtifactsDep
from backend.schemas import (
    EngineReading,
    ExplanationResponse,
    FeatureImportance,
    PartialDependence,
    PermutationImportance,
)
from backend.services.explainability import (
    explain_reading,
    global_importances,
    partial_dependence_curves,
    permutation_importances,
)
from backend.services.prediction import predict_batch

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.get("/global")
async def explain_global(artifacts: ArtifactsDep) -> list[FeatureImportance]:
    """Model-wide feature importances (top drivers of RUL), from the estimator."""
    return global_importances(artifacts, top_k=len(artifacts.feature_names))


@router.get("/permutation-importance")
async def explain_permutation(artifacts: ArtifactsDep) -> list[PermutationImportance]:
    """Model-agnostic permutation importances, computed on validation data."""
    return permutation_importances(artifacts)


@router.get("/partial-dependence")
async def explain_partial_dependence(
    artifacts: ArtifactsDep, feature: str | None = None
) -> list[PartialDependence]:
    """Partial dependence curves: the marginal effect of each feature on RUL.

    Pass ?feature=<name> to get a single curve (non-constant features only).
    """
    return partial_dependence_curves(artifacts, feature)


@router.post("")
async def explain(reading: EngineReading, artifacts: ArtifactsDep) -> ExplanationResponse:
    """Predict a single reading and explain which sensors drove the result."""
    prediction = predict_batch([reading], artifacts)[0]
    return explain_reading(reading, prediction, artifacts)
