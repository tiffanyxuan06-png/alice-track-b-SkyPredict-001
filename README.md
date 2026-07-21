# ALICE Track B - Team SkyPredict

## Setup

Create the virtual environment and install all dependencies (notebooks/data science + backend API):

```bash
# Windows
.\setup_venv.ps1

```

This creates a `.venv` at the repo root, activates it, and installs `requirements.txt` and `backend/requirements.txt`. See [backend/README.md](backend/README.md) for running the API.

## Run the backend

Starts the FastAPI backend, creating the `.venv` and installing backend dependencies automatically if they are missing:

```bash
# Windows
.\start_backend.ps1

```

Then open the interactive API docs at <http://127.0.0.1:8000/docs>. Override the port with `-Port 9000` (PowerShell) or `./start_backend.sh 9000` (bash).