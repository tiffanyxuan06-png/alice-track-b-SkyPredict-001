# SkyPredict — Streamlit Dashboard

The user-facing product for **ALICE Track B**. It is a thin client over the
FastAPI backend: every prediction and explanation comes from the API — the
dashboard runs no model logic itself.

## Structure

```
frontend/
├── app.py                       # home: backend status + model overview
├── pages/                       # Streamlit multipage nav (auto-discovered)
│   ├── 1_Single_Prediction.py       # one reading -> RUL, risk, local drivers
│   ├── 2_Fleet_Prioritization.py    # CSV upload -> ranked fleet + summary
│   └── 3_Model_Insights.py          # metrics + global feature importances
├── components/                  # reusable UI pieces
│   ├── charts.py                    # Plotly gauge / bar charts
│   ├── inputs.py                    # engine form + CSV -> readings
│   └── display.py                   # risk badges, prediction summary
└── utils/
    ├── api_client.py            # the only place that calls the backend
    └── config.py                # API URL + risk colours
```

## Run

The backend must be running first (see [../backend/README.md](../backend/README.md)).
From the repo root:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The dashboard opens at <http://localhost:8501>.

## Configuration

By default the dashboard calls the backend at `http://127.0.0.1:8000`. Point it
elsewhere with an environment variable:

```bash
# Windows (PowerShell)
$env:SKYPREDICT_API_URL = "http://my-host:8000"; streamlit run frontend/app.py
# macOS/Linux
SKYPREDICT_API_URL=http://my-host:8000 streamlit run frontend/app.py
```

## Pages

- **Single Prediction** — a pre-filled engine form (defaults come from the
  backend's model-derived example); returns the RUL gauge, risk band,
  recommended action and the sensors driving the result.
- **Fleet Prioritization** — upload a CSV of engine readings (the repo's
  `data/processed/test.csv` works); returns engines ranked by urgency plus a
  fleet risk summary.
- **Model Insights** — model metrics (train/val/test) and global feature
  importances.
