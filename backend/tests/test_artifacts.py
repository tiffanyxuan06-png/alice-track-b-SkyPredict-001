"""Artifact loading fails loudly on missing or malformed inputs."""

from __future__ import annotations

import json

import pytest

from backend.config import settings
from backend.services.artifacts import load_artifacts


def test_missing_artifacts_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(tmp_path / "nope.pkl", tmp_path / "nope.json")


def test_metadata_missing_required_key_raises(tmp_path):
    # Real model, but strip a required key from a copy of the metadata.
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    metadata.pop("rul_clip")
    bad_meta = tmp_path / "model_metadata.json"
    bad_meta.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="rul_clip"):
        load_artifacts(settings.model_path, bad_meta)
