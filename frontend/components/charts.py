"""Plotly charts used across the dashboard (flight-deck instrument styling)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.config import RISK_COLORS

_MONO = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
_DISPLAY = "Chakra Petch, Space Grotesk, system-ui, sans-serif"


def _palette() -> dict:
    """Colours matched to the current theme mode (see components.theme)."""
    light = st.session_state.get("theme_mode") == "light"
    return {
        "text": "#0B1A24" if light else "#E4EAF0",
        "dim": "#5B6B78" if light else "#8A97A6",
        "grid": "rgba(20,45,65,0.10)" if light else "rgba(122,162,186,0.12)",
        "accent": "#0898B0" if light else "#4FD8EB",
        "steel": "#4A7C93" if light else "#7FB3C9",
    }


def _deck_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    """Transparent, monospace-typeset layout so charts read as instrument readouts."""
    p = _palette()
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family=_DISPLAY, size=15, color=p["text"])) if title else None,
        margin=dict(l=10, r=16, t=48 if title else 16, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_MONO, color=p["text"], size=12),
        barcornerradius=3,
        showlegend=False,
    )
    # automargin lets Plotly reserve space for tick labels + axis titles so they
    # are never clipped by the margins above.
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["grid"], automargin=True)
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["grid"], automargin=True)
    return fig


def rul_gauge(rul: float, rul_clip: int, thresholds: dict) -> go.Figure:
    """Primary engine instrument: RUL on a gauge with CAS status arcs."""
    p = _palette()
    high = thresholds["high_below"]
    medium = thresholds["medium_below"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul,
            number={"suffix": " cyc", "font": {"family": _MONO, "color": p["text"]}},
            title={"text": "REMAINING USEFUL LIFE", "font": {"family": _DISPLAY, "size": 13, "color": p["text"]}},
            gauge={
                "axis": {"range": [0, rul_clip], "tickcolor": p["dim"], "tickfont": {"family": _MONO, "size": 10}},
                "bar": {"color": p["accent"], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, high], "color": RISK_COLORS["High"]},
                    {"range": [high, medium], "color": RISK_COLORS["Medium"]},
                    {"range": [medium, rul_clip], "color": RISK_COLORS["Low"]},
                ],
            },
        )
    )
    return _deck_layout(fig, 280)


def importance_bar(importances: list[dict], title: str) -> go.Figure:
    """Horizontal bar chart of feature importances (expects feature/importance)."""
    items = sorted(importances, key=lambda d: d["importance"])
    fig = go.Figure(
        go.Bar(
            x=[d["importance"] for d in items],
            y=[d["feature"] for d in items],
            orientation="h",
            marker_color=_palette()["accent"],
        )
    )
    fig = _deck_layout(fig, max(300, 22 * len(items)), title)
    fig.update_xaxes(title_text="Importance")
    return fig


def local_contribution_bar(contributions: list[dict]) -> go.Figure:
    """Bar chart of the top SHAP drivers for a single reading (signed).

    Green bars raise the predicted RUL (healthier), red bars lower it (more worn).
    """
    items = sorted(contributions, key=lambda d: abs(d["shap_value"]))
    colors = [
        RISK_COLORS["Low"] if d["shap_value"] >= 0 else RISK_COLORS["High"]
        for d in items
    ]
    fig = go.Figure(
        go.Bar(
            x=[d["shap_value"] for d in items],
            y=[d["feature"] for d in items],
            orientation="h",
            marker_color=colors,
            customdata=[[d["value"]] for d in items],
            hovertemplate="%{y}<br>value=%{customdata[0]}<br>SHAP=%{x} cycles<extra></extra>",
        )
    )
    fig = _deck_layout(fig, max(280, 30 * len(items)), "Top SHAP drivers for this reading")
    fig.update_xaxes(title_text="SHAP value (cycles) — green raises RUL, red lowers it")
    return fig


def permutation_importance_bar(items: list[dict]) -> go.Figure:
    """Horizontal bar of permutation importances with std error bars.

    Expects feature/importance_mean/importance_std (model-agnostic, validation set).
    """
    data = sorted(items, key=lambda d: d["importance_mean"])
    p = _palette()
    fig = go.Figure(
        go.Bar(
            x=[d["importance_mean"] for d in data],
            y=[d["feature"] for d in data],
            orientation="h",
            marker_color=p["steel"],
            error_x=dict(
                type="data",
                array=[d["importance_std"] for d in data],
                visible=True,
                color=p["dim"],
            ),
        )
    )
    fig = _deck_layout(fig, max(300, 22 * len(data)), "Permutation importance (validation set)")
    fig.update_xaxes(title_text="Mean score drop when the feature is shuffled")
    return fig


def partial_dependence_line(curve: dict) -> go.Figure:
    """Line of one feature's partial dependence: avg predicted RUL vs feature value."""
    p = _palette()
    fig = go.Figure(
        go.Scatter(
            x=curve["grid"],
            y=curve["average"],
            mode="lines+markers",
            line=dict(color=p["accent"], width=2),
            marker=dict(size=5, color=p["accent"]),
            hovertemplate="%{x:.3g} → %{y:.1f} cycles<extra></extra>",
        )
    )
    fig = _deck_layout(fig, 360, f"Partial dependence — {curve['feature']}")
    fig.update_xaxes(title_text=f"{curve['feature']} value")
    fig.update_yaxes(title_text="Average predicted RUL (cycles)")
    return fig


def risk_distribution(summary: dict) -> go.Figure:
    """Bar chart of fleet risk counts."""
    order = ["High", "Medium", "Low"]
    counts = {
        "High": summary["high_risk"],
        "Medium": summary["medium_risk"],
        "Low": summary["low_risk"],
    }
    fig = go.Figure(
        go.Bar(
            x=order,
            y=[counts[k] for k in order],
            marker_color=[RISK_COLORS[k] for k in order],
            text=[counts[k] for k in order],
            textposition="auto",
        )
    )
    fig = _deck_layout(fig, 300, "Fleet risk distribution")
    fig.update_yaxes(title_text="Engines")
    return fig
