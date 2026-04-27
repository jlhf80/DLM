"""Declarative types that describe a lesson's content and structure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from engine.models import DLMSpec

ChallengeKind = Literal["multiple_choice", "numeric_range", "component_toggle"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    min: float
    max: float
    default: float
    step: float
    help: str

    def __post_init__(self) -> None:
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"default ({self.default}) must be within [min, max] = "
                f"[{self.min}, {self.max}]"
            )
        if self.step <= 0:
            raise ValueError(f"step must be > 0, got {self.step}")


@dataclass(frozen=True)
class ChallengeQuestion:
    kind: ChallengeKind
    correct: Any
    feedback_correct: str
    feedback_incorrect: str

    def __post_init__(self) -> None:
        if self.kind == "multiple_choice":
            if not isinstance(self.correct, str):
                raise ValueError(
                    f"multiple_choice 'correct' must be str, got {type(self.correct).__name__}"
                )
        elif self.kind == "numeric_range":
            ok = (
                isinstance(self.correct, tuple)
                and len(self.correct) == 2
                and all(isinstance(x, (int, float)) for x in self.correct)
                and self.correct[0] <= self.correct[1]
            )
            if not ok:
                raise ValueError(
                    f"numeric_range 'correct' must be (low, high) tuple, got {self.correct!r}"
                )
        elif self.kind == "component_toggle":
            if not (isinstance(self.correct, dict)
                    and all(isinstance(v, bool) for v in self.correct.values())):
                raise ValueError(
                    f"component_toggle 'correct' must be dict[str, bool], got {self.correct!r}"
                )
        else:
            raise ValueError(f"unknown ChallengeQuestion kind {self.kind!r}")


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    title: str
    prompt_md: str
    plot_fn: str
    hints: list[str] = field(default_factory=list)
    challenge: ChallengeQuestion | None = None


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    tier: str
    description: str
    model_builder: Callable[[dict[str, float]], DLMSpec]
    param_schema: list[ParamSpec]
    workflow_steps: list[WorkflowStep]

    def __post_init__(self) -> None:
        if not self.workflow_steps:
            raise ValueError(f"Lesson {self.id!r} has empty workflow_steps")
        if not self.param_schema:
            raise ValueError(f"Lesson {self.id!r} has empty param_schema")
        names = [p.name for p in self.param_schema]
        if len(set(names)) != len(names):
            raise ValueError(f"Lesson {self.id!r} has duplicate param names {names}")
        step_ids = [s.id for s in self.workflow_steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError(f"Lesson {self.id!r} has duplicate step ids {step_ids}")


def validate_lesson(lesson: Lesson, allowed_plot_fns: set[str]) -> None:
    """Validate a lesson against the set of plot function names available in the UI.

    Raises ValueError on any of:
    - model_builder fails with default param values
    - any WorkflowStep.plot_fn is not in allowed_plot_fns
    """
    defaults = {p.name: p.default for p in lesson.param_schema}
    try:
        spec = lesson.model_builder(defaults)
    except Exception as e:  # pragma: no cover — behavior surfaced via test_lessons
        raise ValueError(
            f"Lesson {lesson.id!r}: model_builder failed on defaults {defaults}: {e}"
        ) from e
    if not isinstance(spec, DLMSpec):
        raise ValueError(
            f"Lesson {lesson.id!r}: model_builder returned {type(spec).__name__}, not DLMSpec"
        )
    for step in lesson.workflow_steps:
        if step.plot_fn not in allowed_plot_fns:
            raise ValueError(
                f"Lesson {lesson.id!r} step {step.id!r}: unknown plot_fn {step.plot_fn!r}; "
                f"available: {sorted(allowed_plot_fns)}"
            )
