"""Application settings.

Centralizes paths and runtime configuration. Values can be overridden with
environment variables (prefixed ``SKYPREDICT_``) or a local ``.env`` file, which
keeps the backend portable between a laptop, CI and a deployment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ holds this file and its .env; repo root is one level up.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

# The selectable models and their exported artifacts (produced by
# models/train_baseline.py). The key is what the API accepts as ?model=.
MODEL_FILES: dict[str, tuple[str, str]] = {
    "rf": ("model_rf.pkl", "model_rf_metadata.json"),
    "xgb": ("model_xgb.pkl", "model_xgb_metadata.json"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SKYPREDICT_",
        # Absolute path so the .env is found regardless of the launch directory.
        env_file=BACKEND_DIR / ".env",
        extra="ignore",
    )

    app_name: str = "SkyPredict - Engine Health & RUL API"
    app_version: str = "1.0.0"
    description: str = (
        "Explainable Remaining Useful Life (RUL) prediction for aircraft engines "
        "(ALICE Track B, NASA C-MAPSS)."
    )

    # Directory holding the exported model artifacts.
    models_dir: Path = REPO_ROOT / "models"
    # Model served when a request does not specify ?model=.
    default_model: str = "xgb"

    # Streamlit dashboard origin(s) allowed to call this API.
    cors_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    def artifact_paths(self, model_key: str) -> tuple[Path, Path]:
        """(model.pkl, metadata.json) paths for a model key."""
        pkl, meta = MODEL_FILES[model_key]
        return self.models_dir / pkl, self.models_dir / meta


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
