"""Shared dependencies (FastAPI dependency injection).

The model artifacts are loaded once during the app lifespan and stored on
``app.state``. Routers depend on ``ArtifactsDep`` to access them, which keeps the
model out of module globals and makes the routes easy to test with overrides.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from backend.services.artifacts import ModelArtifacts


def get_artifacts(request: Request) -> ModelArtifacts:
    artifacts: ModelArtifacts | None = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check server logs and model artifacts.",
        )
    return artifacts


ArtifactsDep = Annotated[ModelArtifacts, Depends(get_artifacts)]
