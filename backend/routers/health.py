"""Health + model-info routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.config import settings
from backend.dependencies import ArtifactsDep
from backend.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "models", {})
    return HealthResponse(
        status="ok",
        models_loaded=sorted(registry),
        default_model=settings.default_model,
        app_version=settings.app_version,
    )


@router.get("/model-info")
async def model_info(artifacts: ArtifactsDep) -> ModelInfoResponse:
    md = artifacts.metadata
    return ModelInfoResponse(
        model_key=md.get("model_key", "unknown"),
        model_type=md.get("model_type", "unknown"),
        dataset=md.get("dataset", "unknown"),
        target=md.get("target", "RUL"),
        rul_clip=artifacts.rul_clip,
        n_features=md.get("n_features", len(artifacts.feature_names)),
        feature_names=artifacts.feature_names,
        metrics=md.get("metrics", {}),
        risk_thresholds=artifacts.risk_thresholds,
        sklearn_version=md.get("sklearn_version"),
        trained_at=md.get("trained_at"),
    )
