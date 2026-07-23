"""Frontend configuration.

The dashboard talks to the FastAPI backend over HTTP; the base URL is
configurable via the SKYPREDICT_API_URL environment variable so the app works
against a local server or a deployed one without code changes.
"""

from __future__ import annotations

import os

API_BASE_URL = os.environ.get("SKYPREDICT_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 30  # seconds

# Consistent colours for the three risk bands across every chart/badge.
RISK_COLORS = {
    "High": "#e74c3c",
    "Medium": "#f39c12",
    "Low": "#2ecc71",
}
