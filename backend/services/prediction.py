"""Prediction + decision-support logic.

Runs the loaded pipeline over prepared features, clips predictions to the RUL
range and maps each result to a maintenance risk band. Keeping this here (rather
than in the routers) keeps the HTTP layer thin and the logic unit-testable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from backend.schemas import (
    EngineReading,
    FleetSummary,
    PredictionResponse,
    RiskLevel,
)
from backend.services.artifacts import ModelArtifacts
from backend.services.preprocessing import readings_to_frame

_ACTIONS = {
    RiskLevel.high: "Prioritize for inspection before the next flight cycle.",
    RiskLevel.medium: "Schedule maintenance and monitor the degradation trend.",
    RiskLevel.low: "Continue normal operation with routine monitoring.",
}


def _risk_band(
    rul: float, thresholds: dict[str, float]
) -> tuple[RiskLevel, str]:
    if rul < thresholds["high_below"]:
        level = RiskLevel.high
    elif rul < thresholds["medium_below"]:
        level = RiskLevel.medium
    else:
        level = RiskLevel.low
    return level, _ACTIONS[level]


def _risk_score(rul: float, rul_clip: int) -> float:
    """Map RUL to [0, 1]: 0 = healthy (RUL at clip), 1 = failure imminent."""
    return round(float(np.clip(1.0 - rul / rul_clip, 0.0, 1.0)), 4)


def predict_batch(
    readings: Sequence[EngineReading], artifacts: ModelArtifacts
) -> list[PredictionResponse]:
    """Predict RUL and attach decision-support metadata for each reading."""
    rul_clip = artifacts.rul_clip
    thresholds = artifacts.risk_thresholds

    frame = readings_to_frame(readings, artifacts.feature_names)
    raw = artifacts.model.predict(frame)
    preds = np.clip(raw, 0, rul_clip)

    responses: list[PredictionResponse] = []
    for reading, rul in zip(readings, preds):
        rul = float(rul)
        level, action = _risk_band(rul, thresholds)
        responses.append(
            PredictionResponse(
                unit_number=reading.unit_number,
                time_in_cycles=reading.time_in_cycles,
                predicted_rul=round(rul, 2),
                risk_level=level,
                risk_score=_risk_score(rul, rul_clip),
                recommended_action=action,
            )
        )
    return responses


def summarize(predictions: Sequence[PredictionResponse]) -> FleetSummary:
    counts = Counter(p.risk_level for p in predictions)
    return FleetSummary(
        total_engines=len(predictions),
        high_risk=counts[RiskLevel.high],
        medium_risk=counts[RiskLevel.medium],
        low_risk=counts[RiskLevel.low],
    )
