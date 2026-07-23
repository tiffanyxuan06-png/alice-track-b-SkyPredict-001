"""Frontend configuration.

The dashboard talks to the FastAPI backend over HTTP; the base URL is
configurable via the SKYPREDICT_API_URL environment variable so the app works
against a local server or a deployed one without code changes.
"""

from __future__ import annotations

import os

API_BASE_URL = os.environ.get("SKYPREDICT_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 30  # seconds

# Aviation CAS (crew alerting) colours for the three risk bands.
RISK_COLORS = {
    "High": "#FF3B30",    # warning (red)
    "Medium": "#FFB300",  # caution (amber)
    "Low": "#30D158",     # normal (green)
}
