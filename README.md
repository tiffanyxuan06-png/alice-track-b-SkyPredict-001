# ALICE Track B - Team SkyPredict

Explainable Engine Health & Remaining Useful Life (RUL) prediction for the NASA
C-MAPSS (FD001) dataset. A FastAPI backend serves predictions and explanations
from trained models, and a Streamlit dashboard provides the user-facing UI.

## Structure

```
backend/     FastAPI service (RUL prediction, explainability, prioritization)
frontend/    Streamlit dashboard (thin client over the backend API)
notebooks/   Data exploration, model training, explainability/export
models/      Trained model artifacts + training script
data/        Raw and processed C-MAPSS data
docs/        Project guidelines
```

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md)
for details on each service.

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

## Run the dashboard

With the backend running, install the frontend dependencies and start Streamlit:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Then open <http://localhost:8501>. See [frontend/README.md](frontend/README.md) for
page details and configuration (pointing the dashboard at a different backend URL).