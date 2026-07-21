"""Pydantic request/response models — the public API contract.

Input validation mirrors the notebook's feature contract: 24 features
(3 operational settings + 21 sensors). ``unit_number`` / ``time_in_cycles`` are
optional identifiers echoed back for the dashboard; they are not model inputs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"


class EngineReading(BaseModel):
    """A single engine snapshot (one operating cycle)."""

    unit_number: int | None = Field(
        default=None, description="Optional engine identifier (echoed back)."
    )
    time_in_cycles: int | None = Field(
        default=None, ge=1, description="Optional cycle index (echoed back)."
    )

    op_setting_1: float = Field(description="Operational setting 1.")
    op_setting_2: float = Field(description="Operational setting 2.")
    op_setting_3: float = Field(description="Operational setting 3.")

    sensor_01: float = Field(description="Sensor 1 measurement.")
    sensor_02: float = Field(description="Sensor 2 measurement.")
    sensor_03: float = Field(description="Sensor 3 measurement.")
    sensor_04: float = Field(description="Sensor 4 measurement.")
    sensor_05: float = Field(description="Sensor 5 measurement.")
    sensor_06: float = Field(description="Sensor 6 measurement.")
    sensor_07: float = Field(description="Sensor 7 measurement.")
    sensor_08: float = Field(description="Sensor 8 measurement.")
    sensor_09: float = Field(description="Sensor 9 measurement.")
    sensor_10: float = Field(description="Sensor 10 measurement.")
    sensor_11: float = Field(description="Sensor 11 measurement.")
    sensor_12: float = Field(description="Sensor 12 measurement.")
    sensor_13: float = Field(description="Sensor 13 measurement.")
    sensor_14: float = Field(description="Sensor 14 measurement.")
    sensor_15: float = Field(description="Sensor 15 measurement.")
    sensor_16: float = Field(description="Sensor 16 measurement.")
    sensor_17: float = Field(description="Sensor 17 measurement.")
    sensor_18: float = Field(description="Sensor 18 measurement.")
    sensor_19: float = Field(description="Sensor 19 measurement.")
    sensor_20: float = Field(description="Sensor 20 measurement.")
    sensor_21: float = Field(description="Sensor 21 measurement.")


class BatchPredictionRequest(BaseModel):
    """A batch of engine readings, e.g. one row per engine from a CSV upload."""

    readings: list[EngineReading] = Field(
        min_length=1, description="One or more engine readings."
    )


class PredictionResponse(BaseModel):
    unit_number: int | None = None
    time_in_cycles: int | None = None
    predicted_rul: float = Field(description="Predicted remaining useful life (cycles).")
    risk_level: RiskLevel
    risk_score: float = Field(
        ge=0, le=1, description="0 = healthy, 1 = imminent failure."
    )
    recommended_action: str


class FleetSummary(BaseModel):
    total_engines: int
    high_risk: int
    medium_risk: int
    low_risk: int


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: FleetSummary


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class LocalContribution(BaseModel):
    feature: str
    value: float
    scaled_value: float = Field(description="Feature min-max scaled to [0, 1].")
    saliency: float = Field(
        description="Heuristic attribution: global importance x scaled value."
    )


class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    global_importances: list[FeatureImportance] = Field(
        description="Top model-wide feature importances."
    )
    local_contributions: list[LocalContribution] = Field(
        description="Top drivers for this specific reading (heuristic saliency)."
    )


class PrioritizedEngine(PredictionResponse):
    priority_rank: int = Field(ge=1, description="1 = most urgent.")


class PrioritizationResponse(BaseModel):
    ranked_engines: list[PrioritizedEngine]
    summary: FleetSummary


class ModelInfoResponse(BaseModel):
    model_type: str
    dataset: str
    target: str
    rul_clip: int
    n_features: int
    feature_names: list[str]
    metrics: dict
    risk_thresholds: dict
    sklearn_version: str | None = None
    trained_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    app_version: str
