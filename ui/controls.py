"""Shared Streamlit widget panels."""

from __future__ import annotations

import streamlit as st

from engine.models import DLMSpec
from engine.simulate import SimulatedSeries, simulate
from lessons.workflow import Lesson


def render_sidebar(lesson: Lesson) -> tuple[dict[str, float], int]:
    """Render the simulation parameter sliders and seed input.

    Returns the current (sim_params, seed) tuple.
    """
    st.sidebar.header("Simulation parameters")
    params: dict[str, float] = {}
    for p in lesson.param_schema:
        if p.step >= 1:
            params[p.name] = st.sidebar.slider(
                p.label,
                min_value=int(p.min), max_value=int(p.max),
                value=int(p.default), step=int(p.step),
                help=p.help, key=f"slider_{lesson.id}_{p.name}",
            )
        else:
            params[p.name] = st.sidebar.slider(
                p.label,
                min_value=float(p.min), max_value=float(p.max),
                value=float(p.default), step=float(p.step),
                help=p.help, key=f"slider_{lesson.id}_{p.name}",
            )
    seed = st.sidebar.number_input(
        "Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
        key=f"seed_{lesson.id}",
    )
    return params, int(seed)


@st.cache_data(show_spinner=False)
def _cached_simulate(
    spec_hash: int, spec: DLMSpec, n: int, seed: int
) -> SimulatedSeries:
    # spec_hash included so the cache key invalidates on any spec change.
    return simulate(spec, n=n, seed=seed)


def simulate_from_params(
    lesson: Lesson, params: dict[str, float], seed: int
) -> tuple[DLMSpec, SimulatedSeries]:
    """Build spec from lesson params and simulate a series, with caching."""
    n = int(params["n"])
    spec_params = {k: v for k, v in params.items() if k != "n"}
    spec = lesson.model_builder(spec_params)
    series = _cached_simulate(hash(spec), spec, n=n, seed=seed)
    return spec, series


def render_mode_selector() -> str:
    mode = st.radio(
        "Mode",
        options=["lesson", "challenge"],
        format_func=lambda x: "Lesson (transparent walkthrough)"
                              if x == "lesson"
                              else "Challenge (hidden ground truth)",
        horizontal=True,
    )
    assert mode is not None  # default index=0 always selects an option
    return mode
