"""Lesson registry.

Import this module to access the list of all lessons and look them up by id.
"""

from __future__ import annotations

from lessons.local_level import LESSON as _local_level
from lessons.local_linear_trend import LESSON as _llt
from lessons.seasonal import LESSON as _seasonal
from lessons.workflow import Lesson

ALL_LESSONS: list[Lesson] = [_local_level, _llt, _seasonal]

_BY_ID: dict[str, Lesson] = {lesson.id: lesson for lesson in ALL_LESSONS}


def get_lesson(lesson_id: str) -> Lesson:
    if lesson_id not in _BY_ID:
        raise KeyError(f"unknown lesson id {lesson_id!r}; available: {list(_BY_ID)}")
    return _BY_ID[lesson_id]
