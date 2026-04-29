"""Streamlit session-state helpers.

All state keys live in `st.session_state` with no disk persistence. Helpers
below keep key names consistent and centralize initialization.

Lesson completion is tracked per mode: a lesson is "fully completed" only
once both Lesson and Challenge mode have been finished. The next lesson
unlocks only when all prior lessons are fully completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

REQUIRED_MODES: frozenset[str] = frozenset({"lesson", "challenge"})


@dataclass
class SessionState:
    tier: str = "beginner"
    lesson_id: str | None = None
    mode: str | None = None
    sim_params: dict[str, float] | None = None
    seed: int = 42
    spec_true: Any = None
    series: Any = None
    step_idx: int = 0
    answers: dict[str, Any] | None = None
    filter_result: Any = None
    smooth_result: Any = None
    user_spec: Any = None
    completed_lessons: dict[str, set[str]] | None = None  # lesson_id -> {modes}


def init_state() -> None:
    """Initialize session state on first render (idempotent)."""
    defaults = SessionState().__dict__
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.get("answers") is None:
        st.session_state["answers"] = {}
    if st.session_state.get("completed_lessons") is None:
        st.session_state["completed_lessons"] = {}


def reset_lesson_state() -> None:
    """Clear everything that depends on the chosen lesson/mode/params."""
    for k in ("sim_params", "spec_true", "series", "filter_result",
              "smooth_result", "user_spec"):
        st.session_state[k] = None
    st.session_state["step_idx"] = 0
    st.session_state["answers"] = {}


def mark_lesson_mode_completed(lesson_id: str, mode: str) -> None:
    """Record that the user finished `lesson_id` in the given `mode`."""
    completed = st.session_state.get("completed_lessons") or {}
    completed.setdefault(lesson_id, set()).add(mode)
    st.session_state["completed_lessons"] = completed


def lesson_modes_completed(lesson_id: str) -> set[str]:
    completed = st.session_state.get("completed_lessons") or {}
    return set(completed.get(lesson_id, set()))


def is_lesson_fully_completed(lesson_id: str) -> bool:
    return REQUIRED_MODES.issubset(lesson_modes_completed(lesson_id))


def is_lesson_unlocked(lesson_id: str, ordered_ids: list[str]) -> bool:
    """Linear unlock: a lesson is unlocked iff all prior lessons are fully
    completed (both Lesson and Challenge mode finished). The first lesson
    is always unlocked.
    """
    if lesson_id == ordered_ids[0]:
        return True
    idx = ordered_ids.index(lesson_id)
    return all(is_lesson_fully_completed(prev) for prev in ordered_ids[:idx])
