"""Optional advanced track: risk-based maintenance prioritization.

Given a batch of engine readings (e.g. the latest cycle per engine in a fleet),
rank them by urgency so a maintenance team knows which engines to inspect first.
Ordering is by ascending predicted RUL (shortest life first), with risk_score as
a tie-breaker. This is a decision-support layer on top of the model predictions.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.schemas import (
    EngineReading,
    PrioritizationResponse,
    PrioritizedEngine,
)
from backend.services.artifacts import ModelArtifacts
from backend.services.prediction import predict_batch, summarize


def prioritize(
    readings: Sequence[EngineReading], artifacts: ModelArtifacts
) -> PrioritizationResponse:
    predictions = predict_batch(readings, artifacts)

    order = sorted(
        predictions, key=lambda p: (p.predicted_rul, -p.risk_score)
    )
    ranked = [
        PrioritizedEngine(priority_rank=rank, **p.model_dump())
        for rank, p in enumerate(order, start=1)
    ]
    return PrioritizationResponse(ranked_engines=ranked, summary=summarize(predictions))
