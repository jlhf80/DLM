"""Beginner lesson 1: local level (random walk plus observation noise)."""

from __future__ import annotations

from engine.models import DLMSpec, make_local_level
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    graft_challenges,
    make_default_workflow_steps,
)


def _build(params: dict[str, float]) -> DLMSpec:
    return make_local_level(V=params["V"], W_level=params["W_level"])


_PARAMS = [
    ParamSpec(
        name="V",
        label="Observation variance V",
        min=1e-6, max=5.0, default=0.5, step=0.01,
        help="Variance of the measurement noise v_t ~ N(0, V).",
    ),
    ParamSpec(
        name="W_level",
        label="Level innovation variance W",
        min=1e-6, max=1.0, default=0.05, step=0.001,
        help="Variance of the level innovation w_t ~ N(0, W).",
    ),
    ParamSpec(
        name="n",
        label="Series length n",
        min=20, max=2000, default=200, step=10,
        help="Number of observations to simulate.",
    ),
]


_PICK_COMPONENTS_Q = ChallengeQuestion(
    kind="component_toggle",
    correct={"level": True, "slope": False, "seasonal": False},
    feedback_correct=(
        "Right. A local level is the simplest DLM: only an "
        "unobserved level that drifts, plus observation noise."
    ),
    feedback_incorrect=(
        "Not quite. Look at the ACF — a slow monotone decay "
        "without a spike is the local-level signature. No slope, "
        "no seasonal."
    ),
)

_SPECIFY_Q = ChallengeQuestion(
    kind="numeric_range",
    # Order-of-magnitude match for W_level (accept within factor of 3x)
    correct=(0.001, 0.5),
    feedback_correct="Good — within the expected order of magnitude.",
    feedback_incorrect=(
        "Check the series variability. W controls how quickly the "
        "level drifts; larger W means the level changes more per step."
    ),
)


_CHALLENGES = {"pick_components": _PICK_COMPONENTS_Q, "specify": _SPECIFY_Q}


LESSON = Lesson(
    id="local_level",
    title="Local level",
    tier="beginner",
    description=(
        "The simplest DLM: an unobserved level that evolves by a random walk, "
        "observed with Gaussian noise."
    ),
    model_builder=_build,
    param_schema=_PARAMS,
    workflow_steps=graft_challenges(
        make_default_workflow_steps(has_seasonal=False), _CHALLENGES
    ),
)
