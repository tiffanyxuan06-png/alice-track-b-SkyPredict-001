"""Optional advanced track route: risk-based maintenance prioritization."""

from __future__ import annotations

from fastapi import APIRouter

from backend.dependencies import ArtifactsDep
from backend.schemas import BatchPredictionRequest, PrioritizationResponse
from backend.services.optional_module import prioritize

router = APIRouter(prefix="/prioritize", tags=["decision-support"])


@router.post("")
async def prioritize_fleet(
    request: BatchPredictionRequest, artifacts: ArtifactsDep
) -> PrioritizationResponse:
    """Rank a fleet of engines by maintenance urgency (shortest RUL first)."""
    return prioritize(request.readings, artifacts)
