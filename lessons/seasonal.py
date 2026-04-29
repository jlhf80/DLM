"""Beginner lesson 3: simple seasonal (factor form)."""

from __future__ import annotations

from engine.models import DLMSpec, make_seasonal_factor
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    graft_challenges,
    make_default_workflow_steps,
)


def _build(params: dict[str, float]) -> DLMSpec:
    return make_seasonal_factor(
        period=int(params["period"]),
        V=params["V"],
        W_season=params["W_season"],
    )


_PARAMS = [
    ParamSpec(name="V", label="Observation variance V",
              min=1e-6, max=5.0, default=0.5, step=0.01, help=""),
    ParamSpec(name="W_season", label="Seasonal innovation variance",
              min=1e-6, max=1.0, default=0.02, step=0.001,
              help="How quickly the seasonal pattern drifts over cycles."),
    ParamSpec(name="period", label="Period (e.g. 4 quarterly, 12 monthly)",
              min=2, max=12, default=4, step=1,
              help="Seasonal period in observation units."),
    ParamSpec(name="n", label="Series length n",
              min=20, max=2000, default=200, step=10, help=""),
]


_PICK_COMPONENTS_Q = ChallengeQuestion(
    kind="component_toggle",
    correct={"level": False, "slope": False, "seasonal": True},
    feedback_correct=(
        "Right. The repeating pattern at a fixed frequency is the "
        "seasonal signature — clear spikes at multiples of the period "
        "in the ACF."
    ),
    feedback_incorrect=(
        "Look at the ACF spikes. Non-decaying peaks at multiples of "
        "a fixed lag are the seasonal signature."
    ),
    question="Which components does this series need?",
)

_SPECIFY_Q = ChallengeQuestion(
    kind="multiple_choice",
    # Resolve at grade-time so moving the period slider stays consistent
    # with the answer key.
    correct=lambda params: str(int(params["period"])),
    feedback_correct="Right — the ACF spike location gives the period.",
    feedback_incorrect=(
        "The period is the lag at which the ACF first shows a large "
        "non-decaying spike. Try matching that to one of the options."
    ),
    question=(
        "Pick the seasonal **period** — the smallest lag at which the "
        "ACF shows a strong non-decaying spike."
    ),
)

_CHALLENGES = {"pick_components": _PICK_COMPONENTS_Q, "specify": _SPECIFY_Q}


LESSON = Lesson(
    id="seasonal",
    title="Simple seasonal",
    tier="beginner",
    description=(
        "A sum-to-zero seasonal factor model. The state carries the last "
        "period-1 seasonal effects; observations read the current one."
    ),
    model_builder=_build,
    param_schema=_PARAMS,
    workflow_steps=graft_challenges(
        make_default_workflow_steps(has_seasonal=True), _CHALLENGES
    ),
)
