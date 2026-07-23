"""Load and hold the exported model + metadata.

The backend loads these artifacts once at startup (see the app lifespan) rather
than retraining, per the project guidelines. Everything downstream reads the
feature order, RUL clip and risk thresholds from here so the API stays in lock
step with whatever model the notebook exported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any  # fitted sklearn Pipeline (scaler + regressor)
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return self.metadata["feature_names"]

    @property
    def rul_clip(self) -> int:
        return int(self.metadata["rul_clip"])

    @property
    def risk_thresholds(self) -> dict[str, float]:
        return self.metadata["risk_thresholds"]

    @property
    def feature_importances(self) -> dict[str, float]:
        """Feature importances read live from the fitted pipeline's estimator."""
        importances = self.model[-1].feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }

    @property
    def permutation_importance(self) -> list[dict[str, float]]:
        """Permutation importances computed on validation data at export time.

        Data-driven (not recoverable from the model alone), so it is stored in
        the metadata. Empty list if the artifact did not export it.
        """
        return self.metadata.get("permutation_importance", [])


# Everything the API needs to drive prediction must come from the exported
# metadata, not from constants baked into the backend.
_REQUIRED_METADATA_KEYS = ("feature_names", "rul_clip", "risk_thresholds")
_REQUIRED_THRESHOLD_KEYS = ("high_below", "medium_below")


def load_artifacts(model_path: Path, metadata_path: Path) -> ModelArtifacts:
    """Load the pickled pipeline and its JSON metadata.

    Raises FileNotFoundError if artifacts are missing (e.g. the training script
    has not been run) and ValueError if the metadata is missing required keys,
    so a malformed artifact fails loudly instead of silently using defaults.
    """
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found (looked for {model_path} and "
            f"{metadata_path}). Run `python models/train_baseline.py` first."
        )

    model = joblib.load(model_path)
    with open(metadata_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    missing = [k for k in _REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(
            f"{metadata_path.name} is missing required keys: {missing}. "
            "Re-export the model with models/train_baseline.py."
        )
    missing_thresholds = [
        k for k in _REQUIRED_THRESHOLD_KEYS if k not in metadata["risk_thresholds"]
    ]
    if missing_thresholds:
        raise ValueError(
            f"risk_thresholds in {metadata_path.name} is missing keys: "
            f"{missing_thresholds}."
        )

    return ModelArtifacts(model=model, metadata=metadata)
