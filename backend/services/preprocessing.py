"""Deterministic preprocessing for prediction.

The heavy lifting (scaling) lives inside the exported sklearn Pipeline. This
module's job is the deterministic glue the guidelines require: turn validated
API payloads into a DataFrame whose columns are in the exact training order,
so the pipeline receives features the way it was fitted.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from backend.schemas import EngineReading


def readings_to_frame(
    readings: Sequence[EngineReading], feature_names: list[str]
) -> pd.DataFrame:
    """Build a feature matrix ordered to match the trained model.

    Only the model's ``feature_names`` are selected (identifier fields such as
    ``unit_number`` are dropped), guaranteeing column order and preventing
    extra keys from leaking into the model.
    """
    rows = [r.model_dump() for r in readings]
    frame = pd.DataFrame(rows)
    return frame[feature_names]
