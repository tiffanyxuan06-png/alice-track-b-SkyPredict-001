# ✈️ SkyPredict — Explainable Engine Health & RUL Prediction

**ALICE Track B — Predictive Aircraft Maintenance**

An end-to-end predictive-maintenance platform that predicts the **Remaining
Useful Life (RUL)** of turbofan engines from sensor data, explains its
predictions with SHAP, and helps maintenance teams prioritize which engines to
inspect first. Built on the **NASA C-MAPSS (FD001)** dataset.

---

## What it does

Given an aircraft engine's sensor readings, the system predicts how many
operating cycles it has left before failure and translates that into a simple
maintenance risk level:

| Predicted RUL | Risk band | Recommended action |
|---|---|---|
| `< 30` cycles | 🔴 High | Prioritize for inspection before the next flight cycle |
| `30 – 74` cycles | 🟡 Medium | Schedule maintenance and monitor the degradation trend |
| `>= 75` cycles | 🟢 Low | Continue normal operation with routine monitoring |

Beyond a single-engine prediction, the dashboard supports **fleet-level
prioritization** (upload many readings, get a ranked service list) and **model
insights** (metrics, feature importances, and per-reading SHAP explanations).

---

## Architecture

Three loosely-coupled pieces, connected in one direction:

```
Notebooks (Colab)             Backend (FastAPI)            Dashboard (Streamlit)
┌────────────────────┐        ┌─────────────────────┐      ┌──────────────────────┐
│ 1. Data exploration│  ─▶    │ /predict            │ ─▶   │ Single Prediction    │
│ 2. Model training  │  save  │ /predict/batch      │ HTTP │ Fleet Prioritization │
│ 3. Explainability  │  ─▶    │ /explain (SHAP)     │ ─▶   │ Model Insights       │
│    + export        │        │ /prioritize         │      │                      │
└────────────────────┘        │ /model-info /health │      └──────────────────────┘
                              └─────────────────────┘
                                        ▲
                                     models/
                                model_rf.pkl (RandomForest)
                                model_xgb.pkl (XGBoost)
```

The dashboard **never loads the model itself** — it only calls the backend.
The backend **never retrains** — it only serves the models exported from the
notebooks.

---

## Repository structure

```
alice-track-b-SkyPredict-001/
├── notebooks/                       # data → model pipeline
│   ├── 01_data_exploration.ipynb     # load, clean, RUL target, engine-level split
│   ├── 02_model_training.ipynb       # features, models, metrics
│   └── 03_explainability_and_export.ipynb   # SHAP, export artifacts
├── backend/                         # FastAPI service
│   ├── main.py                       # app entrypoint
│   ├── config.py                     # settings (via pydantic-settings)
│   ├── schemas.py                    # request/response contracts
│   ├── routers/                      # one file per HTTP concern
│   └── services/                     # artifact loading, prediction, explainability
├── frontend/                        # Streamlit dashboard
│   ├── app.py                        # entry point
│   ├── pages/
│   │   ├── 1_Single_Prediction.py
│   │   ├── 2_Fleet_Prioritization.py
│   │   └── 3_Model_Insights.py
│   ├── components/                   # charts, inputs, display helpers
│   └── utils/                        # api_client, config
├── models/                          # exported artifacts (loaded, never trained here)
│   ├── model_rf.pkl                  # RandomForest pipeline (scaler + regressor)
│   ├── model_rf_metadata.json
│   ├── model_xgb.pkl                 # XGBoost pipeline (scaler + regressor)
│   └── model_xgb_metadata.json
├── data/
│   ├── raw/CMAPSSData/               # NASA C-MAPSS .txt files (FD001..FD004)
│   └── processed/                    # train / val / test CSVs from Session 1
├── docs/                            # supplementary documentation
├── images/                          # screenshots for docs
├── README.md
├── requirements.txt
├── setup_venv.ps1                    # Windows one-shot environment setup
└── start_backend.ps1 / .sh           # convenience launchers
```

---

## Quick start

**Requires Python 3.11+** (tested on 3.11 and 3.12).

### 1. Set up the environment

Windows:

```powershell
.\setup_venv.ps1
```

Manual (Mac / Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Start the backend

From the repo root, in a terminal with `.venv` active:

```bash
uvicorn backend.main:app --reload
```

- Health check: <http://127.0.0.1:8000/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

### 3. Start the dashboard

In a **second terminal** (`.venv` also active), leaving the backend running:

```bash
streamlit run frontend/app.py
```

The dashboard opens at <http://localhost:8501>.

---

## Screenshots

Add PNGs to `images/` and reference them here — for example:

```markdown
![Single Prediction — healthy engine](images/single_prediction_low.png)
![Single Prediction — degraded engine](images/single_prediction_high.png)
![Fleet Prioritization](images/fleet_prioritization.png)
![Model Insights](images/model_insights.png)
```

---

## API endpoints

All endpoints accept an optional `?model=rf|xgb` query parameter; defaults to
the configured primary model.

| Method | Path                | Purpose                                          |
|--------|---------------------|--------------------------------------------------|
| GET    | `/health`           | Liveness + whether the model is loaded           |
| GET    | `/model-info`       | Model type, features, metrics, risk thresholds   |
| POST   | `/predict`          | RUL + risk for one engine reading                |
| POST   | `/predict/batch`    | RUL for many readings + fleet risk summary       |
| POST   | `/explain`          | Prediction + SHAP contributions per feature      |
| GET    | `/explain/global`   | Model-wide feature importances                   |
| POST   | `/prioritize`       | Rank a fleet by maintenance urgency              |

### Example request

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

Response:

```json
{
  "predicted_rul": 96.1,
  "risk_level": "Low",
  "risk_score": 0.23,
  "recommended_action": "Continue normal operation with routine monitoring."
}
```

---

## The models

Both models are exported as scikit-learn `Pipeline` objects (`scaler + regressor`)
so preprocessing travels with the model and cannot drift between training and
serving.

| Model            | Scaler          | Regressor                   |
|------------------|-----------------|-----------------------------|
| `model_rf.pkl`   | MinMaxScaler    | RandomForestRegressor       |
| `model_xgb.pkl`  | StandardScaler  | XGBRegressor                |

**Inputs:** 24 raw features per reading — `op_setting_1..3` and `sensor_01..21`.
**Target:** RUL clipped at 125 cycles (standard C-MAPSS convention).
**Split:** engine-level (whole engines held out), preventing data leakage
between train / val / test.

The dashboard lets you switch between the two models to compare predictions —
useful for surfacing model disagreement on borderline cases.

---

## Explainability

Every prediction is accompanied by a **SHAP explanation** showing each sensor's
signed contribution (in cycles) to that specific prediction. This satisfies the
project's explainable-AI requirement and, more importantly, lets a maintenance
engineer see *why* the model flagged a particular engine — not just *that* it
did.

The Model Insights page additionally shows global feature importance and
permutation importance for the model as a whole.

---

## Integration-tested workflows

The full pipeline has been validated end-to-end:

- ✅ Single Prediction — healthy engine returns Low risk; degraded engine
  correctly flips to Medium/High, with SHAP identifying the changed sensors as
  the top drivers.
- ✅ Fleet Prioritization — batches of 2,000+ readings ranked by urgency with
  color-coded risk levels.
- ✅ Model Insights — MAE / RMSE / R² and importance charts render for both
  models.
- ✅ Error handling — dashboard shows a clean message when the backend is
  unreachable and recovers on restart.

---

## Troubleshooting

**"No module named `shap` / `xgboost`" when starting the backend.**
Some ML dependencies are used by `backend/services/explainability.py` but may
not be pinned in `requirements.txt`. Install them explicitly:
```
pip install shap xgboost
```

**`InconsistentVersionWarning` (yellow) when loading the model.**
Harmless — indicates the model was pickled with a different scikit-learn
version than the one running. Predictions remain valid.

**Dashboard says "Could not reach the backend."**
Check that the backend is running (`http://127.0.0.1:8000/health` returns
`model_loaded: true`).

**Fleet Prioritization shows `None` for engine ID / cycle.**
The processed test CSV was originally saved without `unit_number` /
`time_in_cycles`. Re-run Session 1 with the updated save cell to regenerate
`data/processed/*.csv`.

---

## Team & contributions

| Member | Focus |
|--------|------|
| _[Name]_ | Notebooks — data exploration, model training, SHAP + export |
| _[Name]_ | Backend — FastAPI service, artifact loading, endpoints |
| _[Name]_ | Backend + shared utilities |
| _[Name]_ | Frontend — Streamlit dashboard, charts, UX |
| _[Name]_ | Integration lead — end-to-end testing, README, coordination |

*(Replace names/roles before submission.)*

---

## Dataset & credits

- **Dataset:** [NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (FD001).
- **Reference:** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
  *Damage propagation modeling for aircraft engine run-to-failure simulation*.
- Built for the **ALICE Track B** workshop, 2026.
