"""Single-engine prediction + explanation."""

from __future__ import annotations

import streamlit as st

from components.charts import local_contribution_bar, rul_gauge
from components.display import model_selector, prediction_summary
from components.inputs import IDENTIFIERS, engine_reading_form
from utils.api_client import APIError, example_reading, explain, health, model_info

st.set_page_config(page_title="Single Prediction · SkyPredict", page_icon="✈️", layout="wide")
st.title("Single Prediction")
st.caption("Enter one engine reading to get its RUL, risk band and the sensors driving it.")

try:
    model = model_selector(health())
    info = model_info(model)
    defaults = example_reading()
except APIError as exc:
    st.error(str(exc))
    st.stop()

st.markdown("Fields are pre-filled with a realistic example — adjust and predict.")
reading = engine_reading_form(info["feature_names"], defaults)

if reading is not None:
    # Optional identifiers are not part of the form; send them if the example has them.
    for key in IDENTIFIERS:
        if key in defaults:
            reading[key] = defaults[key]
    try:
        result = explain(reading, model)
    except APIError as exc:
        st.error(str(exc))
        st.stop()

    prediction = result["prediction"]
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            rul_gauge(prediction["predicted_rul"], info["rul_clip"], info["risk_thresholds"]),
            width="stretch",
        )
        prediction_summary(prediction)
    with right:
        st.plotly_chart(
            local_contribution_bar(result["local_contributions"]),
            width="stretch",
        )
    st.caption(
        f"Top drivers are **SHAP values** — each sensor's signed contribution (in "
        f"cycles) to this engine's predicted RUL. Baseline (mean prediction): "
        f"{result['base_value']:.1f} cycles; base + all SHAP values = the raw prediction."
    )
