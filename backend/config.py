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

    # Exported artifacts produced by models/train_baseline.py (the notebook pipeline).
    models_dir: Path = REPO_ROOT / "models"
    model_filename: str = "model.pkl"
    metadata_filename: str = "model_metadata.json"

    # Streamlit dashboard origin(s) allowed to call this API.
    cors_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / self.metadata_filename


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
