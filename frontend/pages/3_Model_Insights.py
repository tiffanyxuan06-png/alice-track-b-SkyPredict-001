"""Model insights — metrics, configuration and feature importances."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import importance_bar, permutation_importance_bar
from components.display import model_selector
from utils.api_client import (
    APIError,
    global_importances,
    health,
    model_info,
    permutation_importances,
)

st.set_page_config(page_title="Model Insights · SkyPredict", page_icon="✈️", layout="wide")
st.title("Model Insights")
st.caption("How the model performs and which sensors drive its predictions.")

try:
    model = model_selector(health())
    info = model_info(model)
    global_imp = global_importances(model)
    perm_imp = permutation_importances(model)
except APIError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Configuration")
c1, c2, c3 = st.columns(3)
c1.metric("Target", info.get("target", "—"))
c2.metric("RUL clip", info.get("rul_clip", "—"))
c3.metric("Features", info.get("n_features", "—"))
st.write(
    f"**Model:** {info.get('model_type', '—')}  \n"
    f"**Risk bands (cycles):** High < {info['risk_thresholds']['high_below']} · "
    f"Medium < {info['risk_thresholds']['medium_below']} · Low otherwise"
)

st.subheader("Performance")
metrics = info.get("metrics", {})
if metrics:
    st.dataframe(
        pd.DataFrame(metrics).T.rename(columns={"rmse": "RMSE", "mae": "MAE", "r2": "R²"}),
        width="stretch",
    )

st.subheader("Feature importances")
model_tab, perm_tab = st.tabs(["Model importances", "Permutation importance"])
with model_tab:
    st.caption("The model's own importances, read live from the trained estimator.")
    st.plotly_chart(
        importance_bar(global_imp, "Feature importance (from the trained model)"),
        width="stretch",
    )
with perm_tab:
    st.caption(
        "Model-agnostic: the drop in validation score when each feature is "
        "shuffled. Error bars show the spread across repeats."
    )
    st.plotly_chart(permutation_importance_bar(perm_imp), width="stretch")
