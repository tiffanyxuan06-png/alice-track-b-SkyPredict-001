#!/usr/bin/env bash
# Starts the SkyPredict FastAPI backend.
# Creates the .venv and installs backend dependencies first if they are missing.
# Run from the repository root: ./start_backend.sh [PORT] [HOST]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8000}"
BIND_HOST="${2:-127.0.0.1}"
VENV_PY=".venv/bin/python"

backend_ready() {
    [ -x "$VENV_PY" ] && "$VENV_PY" -c "import fastapi, uvicorn" >/dev/null 2>&1
}

if ! backend_ready; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment in .venv ..."
        python3 -m venv .venv
    fi
    echo "Installing backend dependencies ..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r backend/requirements.txt
fi

echo "Starting backend on http://$BIND_HOST:$PORT  (docs: /docs, stop: Ctrl+C)"
exec "$VENV_PY" -m uvicorn backend.main:app --host "$BIND_HOST" --port "$PORT" --reload
