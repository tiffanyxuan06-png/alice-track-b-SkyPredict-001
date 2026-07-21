"""Explainability routes: global importances + per-reading explanation."""

from __future__ import annotations

from fastapi import APIRouter

from backend.dependencies import ArtifactsDep
from backend.schemas import EngineReading, ExplanationResponse, FeatureImportance
from backend.services.explainability import explain_reading, global_importances
from backend.services.prediction import predict_batch

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.get("/global")
async def explain_global(artifacts: ArtifactsDep) -> list[FeatureImportance]:
    """Model-wide feature importances (top drivers of RUL)."""
    return global_importances(artifacts, top_k=len(artifacts.feature_names))


@router.post("")
async def explain(reading: EngineReading, artifacts: ArtifactsDep) -> ExplanationResponse:
    """Predict a single reading and explain which sensors drove the result."""
    prediction = predict_batch([reading], artifacts)[0]
    return explain_reading(reading, prediction, artifacts)
