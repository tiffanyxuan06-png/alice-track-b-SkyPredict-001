# SkyPredict — FastAPI Backend

Automated prediction service for **ALICE Track B — Explainable Engine Health & RUL
Prediction** (NASA C-MAPSS FD001). It loads the model artifacts exported from the
notebooks and serves Remaining Useful Life (RUL) predictions, explanations and a
risk-based maintenance prioritization endpoint.

## Structure

```
backend/
├── main.py              # app entrypoint: lifespan (loads artifacts), CORS, routers
├── config.py            # settings (paths, CORS) via pydantic-settings
├── schemas.py           # Pydantic request/response models (the API contract)
├── dependencies.py      # DI: inject loaded artifacts into routes
├── routers/             # HTTP layer — one operation per function
│   ├── health.py            # GET /health, GET /model-info
│   ├── prediction.py        # POST /predict, POST /predict/batch
│   ├── explainability.py    # GET /explain/global, /explain/permutation-importance, POST /explain
│   └── optional_module.py   # POST /prioritize  (advanced decision-support)
├── services/            # business logic
│   ├── artifacts.py         # load model.pkl + model_metadata.json
│   ├── preprocessing.py     # order features to match training
│   ├── prediction.py        # run model + risk banding
│   ├── explainability.py    # global importances + local saliency
│   └── optional_module.py   # risk-based prioritization
└── tests/               # end-to-end tests (pytest + TestClient)
```

## Model artifacts

Two models are trained on the same `StandardScaler` pipeline and loaded into a
selectable registry (the backend does **not** retrain):

- `models/model_rf.pkl`  + `model_rf_metadata.json`  — RandomForest
- `models/model_xgb.pkl` + `model_xgb_metadata.json` — XGBoost

Each metadata file carries the contract: feature names, RUL clip, risk thresholds,
metrics and permutation importance. Regenerate both from the processed data with:

```bash
python models/train_baseline.py
```

Pick the model per request with `?model=rf|xgb` (defaults to `SKYPREDICT_DEFAULT_MODEL`).
Add or swap estimators in `ESTIMATORS` in that script; as long as the artifact
contract (same feature order + metadata keys) holds, the API keeps working.

## Setup & run

From the **repository root**:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -r backend/requirements.txt

uvicorn backend.main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>

## Configuration (`backend/.env`)

Deployment settings are read from `backend/.env` (loaded regardless of the launch
directory), prefixed with `SKYPREDICT_`. Copy the template and edit as needed:

```bash
cp backend/.env.example backend/.env
```

| Variable | Default | Purpose |
|---|---|---|
| `SKYPREDICT_APP_NAME` / `SKYPREDICT_APP_VERSION` | see `config.py` | API metadata |
| `SKYPREDICT_MODELS_DIR` | `models` | Where artifacts are loaded from |
| `SKYPREDICT_DEFAULT_MODEL` | `xgb` | Model used when `?model=` is omitted (`rf`/`xgb`) |
| `SKYPREDICT_CORS_ORIGINS` | Streamlit localhost | Allowed dashboard origins (JSON array) |

`.env` holds **deployment config only**. The RUL clip and risk thresholds are
model parameters and live solely in `model_metadata.json` — edit that file (or
re-export from `train_baseline.py`) to change them. `.env` is gitignored — commit
`.env.example`, not `.env`.

## Endpoints

All model-backed routes accept an optional `?model=rf|xgb` query parameter.

| Method | Path                            | Purpose                                        |
|--------|---------------------------------|------------------------------------------------|
| GET    | `/health`                       | Liveness + which models are loaded             |
| GET    | `/model-info`                   | Selected model's type, features, metrics       |
| POST   | `/predict`                      | RUL + risk for one engine reading              |
| POST   | `/predict/batch`                | RUL for many readings + fleet risk summary     |
| GET    | `/explain/global`               | Model's own feature importances                |
| GET    | `/explain/permutation-importance` | Permutation importances (validation set)     |
| POST   | `/explain`                      | Predict + per-reading sensor saliency          |
| POST   | `/prioritize`                   | Rank a fleet by maintenance urgency (advanced) |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"op_setting_1":-0.0007,"op_setting_2":-0.0004,"op_setting_3":100.0,
       "sensor_01":518.67,"sensor_02":641.82,"sensor_03":1589.7,"sensor_04":1400.6,
       "sensor_05":14.62,"sensor_06":21.61,"sensor_07":554.36,"sensor_08":2388.06,
       "sensor_09":9046.19,"sensor_10":1.3,"sensor_11":47.47,"sensor_12":521.66,
       "sensor_13":2388.02,"sensor_14":8138.62,"sensor_15":8.4195,"sensor_16":0.03,
       "sensor_17":392,"sensor_18":2388,"sensor_19":100.0,"sensor_20":39.06,
       "sensor_21":23.419}'
```

```json
{
  "unit_number": null, "time_in_cycles": null,
  "predicted_rul": 121.41, "risk_level": "Low", "risk_score": 0.0288,
  "recommended_action": "Continue normal operation with routine monitoring."
}
```

## Testing

End-to-end tests under `backend/tests/` start the app through its lifespan (loading
the real models) and drive every endpoint over HTTP, using real rows from
`data/processed/test.csv` as payloads. From the repo root:

```bash
pip install -r backend/requirements-dev.txt
pytest
```

Coverage: health/model-info, single & batch prediction (RUL bounds, risk banding,
fleet summary), global + local explainability, prioritization ranking, input
validation (`422` shape), and artifact-loading failure modes.

## Input contract

Each reading requires the **24 model features**: `op_setting_1..3` and
`sensor_01..21` (floats). `unit_number` and `time_in_cycles` are optional
identifiers echoed back for the dashboard. Missing or non-numeric fields return a
structured `422` error listing the offending field.

## Risk bands

Derived from predicted RUL (thresholds live in `model_metadata.json`):

| RUL (cycles) | Risk   | Recommended action                                      |
|--------------|--------|---------------------------------------------------------|
| `< 30`       | High   | Prioritize for inspection before the next flight cycle. |
| `30 – 75`    | Medium | Schedule maintenance and monitor the degradation trend. |
| `>= 75`      | Low    | Continue normal operation with routine monitoring.      |

## Notes for the dashboard team

CORS allows `http://localhost:8501` (Streamlit) by default — override with the
`SKYPREDICT_CORS_ORIGINS` environment variable. The dashboard should call these
endpoints rather than loading the model itself.
