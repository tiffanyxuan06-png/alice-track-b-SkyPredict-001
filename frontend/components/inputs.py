"""Input widgets: the single-engine form and CSV parsing."""

from __future__ import annotations

import pandas as pd
import streamlit as st

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
IDENTIFIERS = ["unit_number", "time_in_cycles"]


def engine_reading_form(feature_names: list[str], defaults: dict) -> dict | None:
    """Render a form for one engine reading; return the payload on submit.

    ``defaults`` (the backend's example) pre-fills the fields so the form is
    usable in one click. Returns None until the user submits.
    """
    sensors = [f for f in feature_names if f.startswith("sensor_")]

    with st.form("engine_reading"):
        reading: dict = {}

        st.caption("Operational settings")
        cols = st.columns(3)
        for i, name in enumerate(OP_SETTINGS):
            reading[name] = cols[i].number_input(
                name, value=float(defaults.get(name, 0.0)), format="%.4f"
            )

        st.caption("Sensor measurements")
        cols = st.columns(3)
        for i, name in enumerate(sensors):
            reading[name] = cols[i % 3].number_input(
                name, value=float(defaults.get(name, 0.0)), format="%.4f"
            )

        submitted = st.form_submit_button("Predict", type="primary")

    return reading if submitted else None


def csv_to_readings(df: pd.DataFrame, feature_names: list[str]) -> list[dict]:
    """Convert an uploaded CSV to a list of reading payloads.

    Requires the model's feature columns; passes through optional identifiers
    (unit_number, time_in_cycles) when present and ignores everything else
    (e.g. a RUL target column).
    """
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")

    keep = feature_names + [c for c in IDENTIFIERS if c in df.columns]
    records = df[keep].to_dict(orient="records")
    # JSON needs plain Python types; pandas may hand back numpy scalars.
    return [{k: (int(v) if k in IDENTIFIERS else float(v)) for k, v in r.items()} for r in records]
