"""Prediction routes: single and batch RUL prediction."""

from __future__ import annotations

from fastapi import APIRouter

from backend.dependencies import ArtifactsDep
from backend.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EngineReading,
    PredictionResponse,
)
from backend.services.prediction import predict_batch, summarize

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("")
async def predict(reading: EngineReading, artifacts: ArtifactsDep) -> PredictionResponse:
    """Predict RUL and maintenance risk for a single engine reading."""
    return predict_batch([reading], artifacts)[0]


@router.post("/batch")
async def predict_batch_route(
    request: BatchPredictionRequest, artifacts: ArtifactsDep
) -> BatchPredictionResponse:
    """Predict RUL for many engines at once and summarize fleet risk."""
    predictions = predict_batch(request.readings, artifacts)
    return BatchPredictionResponse(
        predictions=predictions, summary=summarize(predictions)
    )
