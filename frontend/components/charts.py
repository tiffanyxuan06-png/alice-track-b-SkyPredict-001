"""Plotly charts used across the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

from utils.config import RISK_COLORS


def rul_gauge(rul: float, rul_clip: int, thresholds: dict) -> go.Figure:
    """Gauge of predicted RUL with the three risk bands shaded."""
    high = thresholds["high_below"]
    medium = thresholds["medium_below"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul,
            number={"suffix": " cycles"},
            title={"text": "Predicted Remaining Useful Life"},
            gauge={
                "axis": {"range": [0, rul_clip]},
                "bar": {"color": "#34495e"},
                "steps": [
                    {"range": [0, high], "color": RISK_COLORS["High"]},
                    {"range": [high, medium], "color": RISK_COLORS["Medium"]},
                    {"range": [medium, rul_clip], "color": RISK_COLORS["Low"]},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def importance_bar(importances: list[dict], title: str) -> go.Figure:
    """Horizontal bar chart of feature importances (expects feature/importance)."""
    items = sorted(importances, key=lambda d: d["importance"])
    fig = go.Figure(
        go.Bar(
            x=[d["importance"] for d in items],
            y=[d["feature"] for d in items],
            orientation="h",
            marker_color="#3498db",
        )
    )
    fig.update_layout(
        title=title,
        height=max(300, 22 * len(items)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Importance",
    )
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
    fig.update_layout(
        title="Top SHAP drivers for this reading",
        height=max(280, 30 * len(items)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="SHAP value (cycles) — green raises RUL, red lowers it",
    )
    return fig


def permutation_importance_bar(items: list[dict]) -> go.Figure:
    """Horizontal bar of permutation importances with std error bars.

    Expects feature/importance_mean/importance_std (model-agnostic, from the
    validation set).
    """
    data = sorted(items, key=lambda d: d["importance_mean"])
    fig = go.Figure(
        go.Bar(
            x=[d["importance_mean"] for d in data],
            y=[d["feature"] for d in data],
            orientation="h",
            marker_color="#16a085",
            error_x=dict(
                type="data", array=[d["importance_std"] for d in data], visible=True
            ),
        )
    )
    fig.update_layout(
        title="Permutation importance (validation set)",
        height=max(300, 22 * len(data)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Mean score drop when the feature is shuffled",
    )
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
    fig.update_layout(
        title="Fleet risk distribution",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="Engines",
    )
    return fig
