"""Streamlit entry point.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from lessons import ALL_LESSONS, get_lesson
from lessons.workflow import validate_lesson
from ui.controls import render_mode_selector, render_sidebar, simulate_from_params
from ui.plots import PLOT_FN_REGISTRY
from ui.state import (
    init_state,
    is_lesson_unlocked,
    mark_lesson_completed,
    reset_lesson_state,
)
from ui.workflow_render import render_workflow_step

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"

st.set_page_config(page_title="DLM Tutorial — Beginner", layout="wide")


def _validate_lessons_once() -> None:
    """Validate lessons against the plot-fn registry at startup."""
    if "lessons_validated" in st.session_state:
        return
    allowed = set(PLOT_FN_REGISTRY.keys())
    for lesson in ALL_LESSONS:
        validate_lesson(lesson, allowed_plot_fns=allowed)
    st.session_state["lessons_validated"] = True


def _render_home() -> None:
    st.title("DLM Intuition Tutorial — Beginner tier")
    st.markdown(
        "Pick a lesson. Lessons unlock in order — complete Challenge mode to "
        "unlock the next. Completed lessons remain revisitable in this session."
    )
    ordered_ids = [lesson.id for lesson in ALL_LESSONS]
    completed = st.session_state.get("completed_lessons") or set()
    for lesson in ALL_LESSONS:
        unlocked = is_lesson_unlocked(lesson.id, ordered_ids)
        is_done = lesson.id in completed
        badge = (
            "✅ completed" if is_done
            else ("🔓 available" if unlocked else "🔒 locked")
        )
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.subheader(f"{lesson.title}  — {badge}")
            st.markdown(lesson.description)
        with col_b:
            if unlocked and st.button("Open", key=f"open_{lesson.id}"):
                st.session_state["lesson_id"] = lesson.id
                st.session_state["mode"] = None
                reset_lesson_state()
                st.rerun()


def _render_lesson_page(lesson_id: str) -> None:
    lesson = get_lesson(lesson_id)
    st.title(lesson.title)
    st.caption(lesson.description)

    mode = render_mode_selector()
    if st.session_state.get("mode") != mode:
        st.session_state["mode"] = mode
        reset_lesson_state()
        st.rerun()

    params, seed = render_sidebar(lesson)
    st.session_state["sim_params"] = params
    st.session_state["seed"] = seed

    spec, series = simulate_from_params(lesson, params, seed)
    st.session_state["spec_true"] = spec
    st.session_state["series"] = series

    step_idx = st.session_state.get("step_idx") or 0
    step = lesson.workflow_steps[step_idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Progress:** step {step_idx + 1} / {len(lesson.workflow_steps)}"
    )
    if st.sidebar.button("Back to lessons"):
        st.session_state["lesson_id"] = None
        reset_lesson_state()
        st.rerun()

    render_workflow_step(lesson=lesson, step=step, mode=mode)

    col_back, col_next = st.columns([1, 1])
    with col_back:
        if step_idx > 0 and st.button("◂ Back"):
            st.session_state["step_idx"] = step_idx - 1
            st.rerun()
    with col_next:
        if step_idx < len(lesson.workflow_steps) - 1:
            if st.button("Next ▸"):
                st.session_state["step_idx"] = step_idx + 1
                st.rerun()
        elif st.button("Finish lesson"):
            if mode == "challenge":
                mark_lesson_completed(lesson.id)
            st.session_state["lesson_id"] = None
            reset_lesson_state()
            st.rerun()


def _render_reference() -> None:
    path = CONTENT_DIR / "modeling-guide.md"
    if not path.exists():
        st.warning("modeling-guide.md not found.")
        return
    st.markdown(path.read_text(encoding="utf-8"))


def main() -> None:
    init_state()
    _validate_lessons_once()

    page = st.sidebar.radio(
        "Page", options=["Home", "Reference"], horizontal=False,
        key="page_radio",
    )
    if page == "Reference":
        _render_reference()
        return

    lesson_id = st.session_state.get("lesson_id")
    if lesson_id is None:
        _render_home()
    else:
        _render_lesson_page(lesson_id)


main()
