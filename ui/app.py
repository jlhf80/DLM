"""Streamlit entry point.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts ui/ on sys.path, not the project root. Make sibling packages
# (engine, lessons, ui) importable when invoked via `streamlit run ui/app.py`
# regardless of whether the project is pip-installed in the active Python.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from lessons import ALL_LESSONS, get_lesson  # noqa: E402
from lessons.workflow import validate_lesson  # noqa: E402
from ui.content_index import extract_anchors, find_missing_anchors  # noqa: E402
from ui.controls import (  # noqa: E402
    render_mode_selector,
    render_sidebar,
    simulate_from_params,
)
from ui.plots import PLOT_FN_REGISTRY  # noqa: E402
from ui.state import (  # noqa: E402
    init_state,
    is_lesson_fully_completed,
    is_lesson_unlocked,
    lesson_modes_completed,
    mark_lesson_mode_completed,
    reset_lesson_state,
)
from ui.workflow_render import render_workflow_step  # noqa: E402

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"

st.set_page_config(page_title="DLM Tutorial — Beginner", layout="wide")


def _validate_lessons_once() -> None:
    """Validate lessons (plot-fns + reference anchors) at startup."""
    if "lessons_validated" in st.session_state:
        return
    allowed = set(PLOT_FN_REGISTRY.keys())
    for lesson in ALL_LESSONS:
        validate_lesson(lesson, allowed_plot_fns=allowed)

    guide_path = CONTENT_DIR / "modeling-guide.md"
    if guide_path.exists():
        anchors = extract_anchors(guide_path.read_text(encoding="utf-8"))
        missing = find_missing_anchors(ALL_LESSONS, available_anchors=anchors)
        if missing:
            raise RuntimeError(
                "modeling-guide.md is missing anchors referenced by "
                f"workflow steps: {missing}"
            )
    st.session_state["lessons_validated"] = True


def _render_home() -> None:
    st.title("DLM Intuition Tutorial — Beginner tier")
    st.markdown(
        "Pick a lesson. Lessons unlock in order — finish **both Lesson and "
        "Challenge mode** for a lesson to unlock the next one. Completed "
        "lessons remain revisitable in this session."
    )
    ordered_ids = [lesson.id for lesson in ALL_LESSONS]
    for lesson in ALL_LESSONS:
        unlocked = is_lesson_unlocked(lesson.id, ordered_ids)
        modes = lesson_modes_completed(lesson.id)
        fully_done = is_lesson_fully_completed(lesson.id)
        if fully_done:
            status = "✅ completed"
        elif unlocked:
            status = "🔓 available"
        else:
            status = "🔒 locked"
        lesson_tick = "✅" if "lesson" in modes else "◻"
        challenge_tick = "✅" if "challenge" in modes else "◻"
        progress = (
            f"{lesson_tick} Lesson mode &nbsp;·&nbsp; "
            f"{challenge_tick} Challenge mode"
        )
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.subheader(f"{lesson.title}  — {status}")
            st.markdown(lesson.description)
            st.markdown(progress, unsafe_allow_html=True)
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
            mark_lesson_mode_completed(lesson.id, mode)
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
