"""Small display helpers: risk badges, prediction summary, backend status, model picker."""

from __future__ import annotations

import streamlit as st

from utils.config import RISK_COLORS


def risk_badge(level: str) -> str:
    """Return an inline HTML pill for a risk level (render with unsafe_allow_html)."""
    color = RISK_COLORS.get(level, "#7f8c8d")
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:12px;font-weight:600'>{level} risk</span>"
    )


def prediction_summary(prediction: dict) -> None:
    """Show RUL, risk badge and the recommended maintenance action."""
    left, right = st.columns(2)
    left.metric("Predicted RUL", f"{prediction['predicted_rul']:.1f} cycles")
    right.metric("Risk score", f"{prediction['risk_score']:.2f}")
    st.markdown(risk_badge(prediction["risk_level"]), unsafe_allow_html=True)
    st.info(f"**Recommended action:** {prediction['recommended_action']}")


def backend_status(health: dict) -> None:
    """Render a compact backend health indicator."""
    models = health.get("models_loaded", [])
    if models:
        st.success(
            f"Backend online · models: {', '.join(models)} · "
            f"default: {health.get('default_model', '?')} · "
            f"v{health.get('app_version', '?')}"
        )
    else:
        st.warning("Backend online, but no models are loaded.")


def model_selector(health: dict) -> str:
    """Sidebar picker for the model; persists across pages via session state."""
    models = health.get("models_loaded", [])
    if not models:
        return health.get("default_model", "")
    if st.session_state.get("selected_model") not in models:
        st.session_state["selected_model"] = health.get("default_model", models[0])
    return st.sidebar.selectbox("Model", models, key="selected_model")
