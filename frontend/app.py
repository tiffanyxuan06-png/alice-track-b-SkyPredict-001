"""SkyPredict dashboard — home page.

ALICE Track B: Explainable Engine Health & RUL Prediction. This dashboard is a
thin client over the FastAPI backend; every prediction and explanation comes
from the API. Use the pages in the sidebar to move from model understanding
to prediction, explanation and fleet decision-support.

Run from the repo root:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import streamlit as st

from components.display import backend_status
from utils.api_client import APIError, health, model_info
from utils.config import API_BASE_URL

st.set_page_config(page_title="SkyPredict", page_icon="✈️", layout="wide")

st.title("✈️ SkyPredict — Engine Health & RUL")
st.caption(
    "Explainable Remaining Useful Life prediction for aircraft engines "
    "(ALICE Track B · NASA C-MAPSS)."
)

try:
    backend_status(health())
    info = model_info()
except APIError as exc:
    st.error(str(exc))
    st.markdown(
        f"""
        The dashboard needs the backend running at `{API_BASE_URL}`.

        Start it from the repo root:
        ```bash
        # Windows
        .\\start_backend.ps1
        # macOS/Linux
        ./start_backend.sh
        ```
        Or point the dashboard elsewhere with the `SKYPREDICT_API_URL`
        environment variable.
        """
    )
    st.stop()

st.subheader("Model at a glance")
test = info.get("metrics", {}).get("test", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", info.get("dataset", "—"))
c2.metric("Features", info.get("n_features", "—"))
c3.metric("Test RMSE", f"{test.get('rmse', float('nan')):.1f}")
c4.metric("Test R²", f"{test.get('r2', float('nan')):.2f}")

st.markdown(
    """
    ### How to use
    - **Single Prediction** — enter one engine reading and get its RUL, risk band
      and the sensors driving the result.
    - **Fleet Prioritization** — upload a CSV of engine readings and get them
      ranked by maintenance urgency.
    - **Model Insights** — the model's metrics and the features that matter most.
    """
)
