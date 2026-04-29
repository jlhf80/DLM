"""Render a single WorkflowStep in Lesson or Challenge mode."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from engine.filter import FilterResult, kalman_filter
from engine.forecast import Forecast, forecast_horizon
from engine.models import DLMSpec
from engine.smoother import SmoothResult, rts_smoother
from lessons.workflow import ChallengeQuestion, Lesson, WorkflowStep
from ui.plots import PLOT_FN_REGISTRY


@st.cache_data(show_spinner=False)
def _cached_filter(
    spec_hash: int, spec: DLMSpec, y_bytes: bytes, y_shape: tuple[int, ...]
) -> FilterResult:
    y = np.frombuffer(y_bytes, dtype=np.float64).reshape(y_shape)
    return kalman_filter(spec, y)


@st.cache_data(show_spinner=False)
def _cached_smooth(
    spec_hash: int, spec: DLMSpec, fr: FilterResult
) -> SmoothResult:
    return rts_smoother(spec, fr)


@st.cache_data(show_spinner=False)
def _cached_forecast(
    spec_hash: int, spec: DLMSpec, fr: FilterResult, h: int
) -> Forecast:
    return forecast_horizon(spec, fr, h=h)


def _run_filter(spec: DLMSpec) -> FilterResult:
    series = st.session_state["series"]
    y = np.ascontiguousarray(series.y, dtype=np.float64)
    return _cached_filter(hash(spec), spec, y.tobytes(), y.shape)


def _render_plot(step: WorkflowStep, lesson: Lesson, mode: str) -> None:
    series = st.session_state["series"]
    spec_true = st.session_state["spec_true"]
    sim_params = st.session_state.get("sim_params") or {}

    fn = PLOT_FN_REGISTRY[step.plot_fn]
    kwargs: dict[str, Any] = {"series": series}

    if step.plot_fn == "acf_pacf_and_seasonal_subseries":
        kwargs["period"] = int(sim_params.get("period", 4))

    if step.id in ("fit", "diagnose", "forecast", "reveal"):
        spec_for_fit = (
            st.session_state.get("user_spec")
            if mode == "challenge" and st.session_state.get("user_spec") is not None
            else spec_true
        )
        fr = _run_filter(spec_for_fit)
        st.session_state["filter_result"] = fr
        kwargs["fr"] = fr
        if step.id == "diagnose":
            sr = _cached_smooth(hash(spec_for_fit), spec_for_fit, fr)
            st.session_state["smooth_result"] = sr
            kwargs["sr"] = sr
        if step.id == "forecast":
            h = st.slider("Forecast horizon h",
                          min_value=1, max_value=100, value=20, step=1)
            fc = _cached_forecast(hash(spec_for_fit), spec_for_fit, fr, int(h))
            kwargs["fc"] = fc
        if step.id == "reveal":
            fr_true = _run_filter(spec_true)
            kwargs["fr_true"] = fr_true
            kwargs["fr_user"] = fr if mode == "challenge" else None

    if step.plot_fn == "spec_preview":
        kwargs = {"spec": (
            st.session_state.get("user_spec") if mode == "challenge"
            else spec_true
        )}
    if step.plot_fn == "blank":
        kwargs = {}

    fig = fn(**kwargs)
    st.plotly_chart(fig, use_container_width=True)


def _render_lesson_mode(step: WorkflowStep, lesson: Lesson) -> None:
    st.markdown(step.prompt_md)
    _render_plot(step, lesson, mode="lesson")


def _render_challenge_widget(
    step: WorkflowStep, challenge: ChallengeQuestion
) -> None:
    key = f"answer_{step.id}"
    prior = st.session_state["answers"].get(step.id)
    answer: Any
    if challenge.kind == "component_toggle":
        cols = st.columns(3)
        level = cols[0].checkbox("level",
                                 value=(prior or {}).get("level", False),
                                 key=f"{key}_level")
        slope = cols[1].checkbox("slope",
                                 value=(prior or {}).get("slope", False),
                                 key=f"{key}_slope")
        seasonal = cols[2].checkbox("seasonal",
                                    value=(prior or {}).get("seasonal", False),
                                    key=f"{key}_seasonal")
        answer = {"level": level, "slope": slope, "seasonal": seasonal}
    elif challenge.kind == "numeric_range":
        answer = st.number_input(
            "Your estimate", min_value=0.0, max_value=10.0,
            value=float(prior) if prior is not None else 0.1,
            step=0.001, format="%.4f", key=key,
        )
    elif challenge.kind == "multiple_choice":
        options = ["2", "4", "7", "12"]
        answer = st.radio(
            "Pick one", options=options,
            index=options.index(prior) if prior in options else 1,
            key=key,
        )
    else:
        answer = None

    if st.button("Submit answer", key=f"{key}_submit"):
        st.session_state["answers"][step.id] = answer
        sim_params = st.session_state.get("sim_params") or {}
        if _is_correct(challenge, answer, sim_params):
            st.success(challenge.feedback_correct)
        else:
            st.warning(challenge.feedback_incorrect)


def _is_correct(
    q: ChallengeQuestion, answer: Any, params: dict[str, float]
) -> bool:
    correct = q.resolve(params)
    if q.kind == "component_toggle":
        return bool(answer == correct)
    if q.kind == "multiple_choice":
        return bool(answer == correct)
    if q.kind == "numeric_range":
        lo, hi = correct
        return bool(lo <= float(answer) <= hi)
    return False


def _render_challenge_mode(step: WorkflowStep, lesson: Lesson) -> None:
    st.markdown(step.prompt_md)
    _render_plot(step, lesson, mode="challenge")
    if step.challenge is not None:
        _render_challenge_widget(step, step.challenge)


def render_workflow_step(
    lesson: Lesson, step: WorkflowStep, mode: str
) -> None:
    st.header(step.title)
    if mode == "challenge":
        _render_challenge_mode(step, lesson)
    else:
        _render_lesson_mode(step, lesson)
    if step.hints:
        with st.expander("Hints"):
            for h in step.hints:
                st.markdown(f"- {h}")
