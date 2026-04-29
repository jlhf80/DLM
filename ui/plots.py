"""Plotly renderers used by workflow steps.

Each function takes the current session objects (series, filter_result, etc.)
and returns a `plotly.graph_objects.Figure`. The set of function names
registered in `PLOT_FN_REGISTRY` is the source of truth for what lessons can
reference via WorkflowStep.plot_fn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.diagnostics import acf_pacf, residual_diagnostics
from engine.filter import FilterResult
from engine.forecast import Forecast
from engine.models import DLMSpec
from engine.simulate import SimulatedSeries
from engine.smoother import SmoothResult


def _blank() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="No plot for this step.",
                          showarrow=False, font=dict(color="#888"))],
    )
    return fig


def time_series(series: SimulatedSeries, **_: Any) -> go.Figure:
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", name="y_t",
                             marker=dict(size=4)))
    fig.update_layout(title="Observed series", xaxis_title="t",
                      yaxis_title="y", height=350)
    return fig


def visual_decomposition(series: SimulatedSeries, **_: Any) -> go.Figure:
    """y_t plus a simple centered moving average to suggest a trend."""
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    window = max(5, len(y) // 20)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    trend = np.convolve(y, kernel, mode="same")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="y_t",
                             line=dict(color="#888", width=1)))
    fig.add_trace(go.Scatter(x=t, y=trend, mode="lines",
                             name=f"MA({window})", line=dict(width=3)))
    fig.update_layout(title="Visual decomposition", xaxis_title="t",
                      height=350)
    return fig


def acf_pacf_fig(series: SimulatedSeries, **_: Any) -> go.Figure:
    ap = acf_pacf(series.y, nlags=min(40, len(series.y) // 4))
    n = len(series.y)
    band = 1.96 / np.sqrt(n)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    fig.add_trace(go.Bar(x=ap.lags, y=ap.acf, name="acf"), row=1, col=1)
    fig.add_trace(go.Bar(x=ap.lags, y=ap.pacf, name="pacf"), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=band, line_dash="dash", line_color="#aaa",
                      row=1, col=col)
        fig.add_hline(y=-band, line_dash="dash", line_color="#aaa",
                      row=1, col=col)
    fig.update_layout(height=350, showlegend=False)
    return fig


def acf_pacf_and_seasonal_subseries(
    series: SimulatedSeries, period: int = 4, **_: Any
) -> go.Figure:
    """ACF/PACF plus a seasonal subseries grid underneath.

    Caller passes `period` via kwargs from the sim_params dict.
    """
    ap = acf_pacf(series.y, nlags=min(40, len(series.y) // 4))
    y = series.y[:, 0]
    n = len(y)
    band = 1.96 / np.sqrt(n)
    period = int(period)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("ACF", "PACF", "Seasonal subseries", ""),
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )
    fig.add_trace(go.Bar(x=ap.lags, y=ap.acf, name="acf"), row=1, col=1)
    fig.add_trace(go.Bar(x=ap.lags, y=ap.pacf, name="pacf"), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=band, line_dash="dash", line_color="#aaa",
                      row=1, col=col)
        fig.add_hline(y=-band, line_dash="dash", line_color="#aaa",
                      row=1, col=col)
    for k in range(period):
        sub = y[k::period]
        x = np.arange(len(sub))
        fig.add_trace(
            go.Scatter(x=x, y=sub, mode="lines+markers",
                       name=f"season {k + 1}"),
            row=2, col=1,
        )
    fig.update_layout(height=600, showlegend=False)
    return fig


def spec_preview(spec: DLMSpec | None, **_: Any) -> go.Figure:
    """Render the matrices F, G, V, W as a heatmap grid."""
    if spec is None:
        return _blank()
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("F", "G", "V", "W"),
    )
    for (r, c, M, name) in [
        (1, 1, spec.F, "F"), (1, 2, spec.G, "G"),
        (2, 1, spec.V, "V"), (2, 2, spec.W, "W"),
    ]:
        fig.add_trace(
            go.Heatmap(z=M, colorscale="Greys", showscale=False,
                       hovertemplate=f"{name}"
                                     "[%{y}, %{x}] = %{z:.4g}<extra></extra>"),
            row=r, col=c,
        )
    fig.update_layout(height=500)
    return fig


def filter_state(
    series: SimulatedSeries, fr: FilterResult, **_: Any
) -> go.Figure:
    """Filtered predictive obs mean with 95% bands, alongside observed series."""
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    F = series.spec.F
    mean_y = (fr.m @ F.T)[:, 0]
    var_y = np.einsum("ij,tjk,ik->t", F, fr.C, F)
    sd_y = np.sqrt(var_y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="markers", name="y_t",
                             marker=dict(color="#888", size=4)))
    fig.add_trace(go.Scatter(x=t, y=mean_y, mode="lines",
                             name="filtered E[y|past]"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([mean_y + 1.96 * sd_y, (mean_y - 1.96 * sd_y)[::-1]]),
        fill="toself", line=dict(color="rgba(0,0,0,0)"),
        fillcolor="rgba(31, 119, 180, 0.2)",
        name="95% band", hoverinfo="skip",
    ))
    fig.update_layout(title="Filtered state", xaxis_title="t", height=400)
    return fig


def diagnostics_fig(
    fr: FilterResult, sr: SmoothResult | None = None, **_: Any
) -> go.Figure:
    diag = residual_diagnostics(fr.e, fr.Q, nlags=min(20, len(fr.e) // 4))
    t = np.arange(1, diag.standardized.shape[0] + 1)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Standardized residuals",
            f"Ljung-Box p = {diag.ljung_box_pvalue:.3f}",
            "Residual ACF",
            "Residual PACF",
        ),
    )
    fig.add_trace(go.Scatter(x=t, y=diag.standardized[:, 0],
                             mode="lines+markers",
                             marker=dict(size=3), name="std resid"),
                  row=1, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="#f66", row=1, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="#f66", row=1, col=1)
    fig.add_trace(go.Histogram(x=diag.standardized[:, 0], nbinsx=30,
                               name="hist"), row=1, col=2)
    fig.add_trace(go.Bar(x=diag.acf_pacf.lags, y=diag.acf_pacf.acf,
                         name="acf"), row=2, col=1)
    fig.add_trace(go.Bar(x=diag.acf_pacf.lags, y=diag.acf_pacf.pacf,
                         name="pacf"), row=2, col=2)
    fig.update_layout(height=600, showlegend=False)
    return fig


def forecast_fig(
    series: SimulatedSeries, fr: FilterResult, fc: Forecast, **_: Any
) -> go.Figure:
    y = series.y[:, 0]
    T = len(y)
    t_hist = np.arange(1, T + 1)
    t_fc = np.arange(T + 1, T + 1 + fc.horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_hist, y=y, mode="lines+markers",
                             name="observed", marker=dict(size=3)))
    if fc.horizon > 0:
        fig.add_trace(go.Scatter(x=t_fc, y=fc.means[:, 0], mode="lines",
                                 name="forecast"))
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_fc, t_fc[::-1]]),
            y=np.concatenate([fc.upper[:, 0], fc.lower[:, 0][::-1]]),
            fill="toself", line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(214, 39, 40, 0.25)", name="95% band",
            hoverinfo="skip",
        ))
    fig.update_layout(title=f"{fc.horizon}-step forecast",
                      xaxis_title="t", height=400)
    return fig


def reveal_overlay(
    series: SimulatedSeries,
    fr_true: FilterResult,
    fr_user: FilterResult | None = None,
    **_: Any,
) -> go.Figure:
    """Overlay the true DLM's filtered state against the user's (challenge)."""
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    F = series.spec.F
    true_mean_y = (fr_true.m @ F.T)[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="markers",
                             name="y_t", marker=dict(color="#aaa", size=3)))
    fig.add_trace(go.Scatter(x=t, y=true_mean_y, mode="lines",
                             name="ground-truth filtered"))
    if fr_user is not None:
        user_mean_y = (fr_user.m @ F.T)[:, 0]
        fig.add_trace(go.Scatter(x=t, y=user_mean_y, mode="lines",
                                 name="your filtered",
                                 line=dict(dash="dash")))
    fig.update_layout(title="Reveal: your DLM vs ground truth",
                      height=400, xaxis_title="t")
    return fig


PLOT_FN_REGISTRY: dict[str, Callable[..., go.Figure]] = {
    "time_series": time_series,
    "visual_decomposition": visual_decomposition,
    "acf_pacf": acf_pacf_fig,
    "acf_pacf_and_seasonal_subseries": acf_pacf_and_seasonal_subseries,
    "blank": lambda **_: _blank(),
    "spec_preview": spec_preview,
    "filter_state": filter_state,
    "diagnostics": diagnostics_fig,
    "forecast": forecast_fig,
    "reveal_overlay": reveal_overlay,
}
