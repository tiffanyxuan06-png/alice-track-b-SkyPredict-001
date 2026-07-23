"""Model insights — metrics, configuration and feature importances."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import (
    importance_bar,
    partial_dependence_line,
    permutation_importance_bar,
)
from components.display import model_selector
from components.theme import apply_theme
from utils.api_client import (
    APIError,
    global_importances,
    health,
    model_info,
    partial_dependence,
    permutation_importances,
)

st.set_page_config(page_title="Model Insights · SkyPredict", page_icon="✈️", layout="wide")
bg_image_url = "https://images3.alphacoders.com/888/thumb-1920-888034.png"

st.markdown(
    f"""
    <style>
    /* 1. Main Background Image */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(0, 0, 0, 0.60), rgba(0, 0, 0, 0.99)), url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* 2. Clear Header */
    [data-testid="stHeader"] {{
        background: transparent !important;
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    [data-testid="stAppViewContainer"] h1 {{
            font-size: 5.5rem !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
apply_theme()
st.title("Model Insights")
st.caption("How the model performs and which sensors drive its predictions.")

try:
    model = model_selector(health())
    info = model_info(model)
    global_imp = global_importances(model)
    perm_imp = permutation_importances(model)
    pdp = partial_dependence(model)
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
model_tab, perm_tab, pdp_tab = st.tabs(
    ["Model importances", "Permutation importance", "Partial dependence"]
)
with model_tab:
    st.caption("The model's own importances, read live from the trained estimator.")
    st.plotly_chart(
        importance_bar(global_imp, "Feature importance (from the trained model)"),
        width="stretch",
        theme=None,
    )
with perm_tab:
    st.caption(
        "Model-agnostic: the drop in validation score when each feature is "
        "shuffled. Error bars show the spread across repeats."
    )
    st.plotly_chart(permutation_importance_bar(perm_imp), width="stretch", theme=None)
with pdp_tab:
    st.caption(
        "How predicted RUL changes as one sensor varies, averaging over the "
        "others (scikit-learn partial dependence)."
    )
    if pdp:
        feature = st.selectbox("Feature", [c["feature"] for c in pdp], key="pdp_feature")
        curve = next(c for c in pdp if c["feature"] == feature)
        st.plotly_chart(partial_dependence_line(curve), width="stretch", theme=None)
    else:
        st.info("Partial dependence is not available for this model.")
