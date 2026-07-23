"""Fleet prioritization — rank engines by maintenance urgency (decision support)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import risk_distribution
from components.display import model_selector
from components.inputs import csv_to_readings
from utils.api_client import APIError, health, model_info, prioritize
from utils.config import RISK_COLORS

st.set_page_config(page_title="Fleet Prioritization · SkyPredict", page_icon="✈️", layout="wide")
st.title("Fleet Prioritization")
st.caption("Upload engine readings to rank the fleet by maintenance urgency (shortest RUL first).")

try:
    model = model_selector(health())
    info = model_info(model)
except APIError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Expected CSV format"):
    st.markdown(
        "One row per engine reading, with these feature columns "
        "(optional `unit_number` / `time_in_cycles` are echoed back; a `RUL` "
        "column, if present, is ignored):"
    )
    st.code(", ".join(info["feature_names"]))
    st.caption("Tip: `data/processed/test.csv` from the repo works directly.")

uploaded = st.file_uploader("Engine readings (CSV)", type="csv")
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)
st.write(f"Loaded **{len(df)}** rows.")

try:
    readings = csv_to_readings(df, info["feature_names"])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not st.button("Prioritize fleet", type="primary"):
    st.stop()

try:
    result = prioritize(readings, model)
except APIError as exc:
    st.error(str(exc))
    st.stop()

summary = result["summary"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Engines", summary["total_engines"])
c2.metric("High risk", summary["high_risk"])
c3.metric("Medium risk", summary["medium_risk"])
c4.metric("Low risk", summary["low_risk"])

st.plotly_chart(risk_distribution(summary), width="stretch")

st.subheader("Ranked engines")
ranked = pd.DataFrame(result["ranked_engines"])
columns = [
    c
    for c in [
        "priority_rank",
        "unit_number",
        "time_in_cycles",
        "predicted_rul",
        "risk_level",
        "risk_score",
        "recommended_action",
    ]
    if c in ranked.columns
]
styled = ranked[columns].style.map(
    lambda level: f"background-color: {RISK_COLORS.get(level, '')}; color: white;",
    subset=["risk_level"],
)
st.dataframe(styled, width="stretch", hide_index=True)
