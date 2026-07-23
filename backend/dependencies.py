"""Shared dependencies (FastAPI dependency injection).

Both models are loaded once during the app lifespan into a registry on
``app.state.models``. ``get_artifacts`` picks one per request from the optional
``?model=`` query parameter (falling back to the configured default), so every
route that depends on ``ArtifactsDep`` becomes model-selectable for free.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from backend.config import settings
from backend.schemas import ModelName
from backend.services.artifacts import ModelArtifacts


def get_artifacts(request: Request, model: ModelName | None = None) -> ModelArtifacts:
    registry: dict[str, ModelArtifacts] = getattr(request.app.state, "models", {})
    if not registry:
        raise HTTPException(
            status_code=503,
            detail="No models are loaded. Check server logs and model artifacts.",
        )
    key = model.value if model else settings.default_model
    artifacts = registry.get(key)
    if artifacts is None:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{key}' is not available. Loaded: {sorted(registry)}.",
        )
    return artifacts


ArtifactsDep = Annotated[ModelArtifacts, Depends(get_artifacts)]
