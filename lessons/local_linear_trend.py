"""Beginner lesson 2: local linear trend (level + slope)."""

from __future__ import annotations

from engine.models import DLMSpec, make_local_linear_trend
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    graft_challenges,
    make_default_workflow_steps,
)


def _build(params: dict[str, float]) -> DLMSpec:
    return make_local_linear_trend(
        V=params["V"],
        W_level=params["W_level"],
        W_slope=params["W_slope"],
    )


_PARAMS = [
    ParamSpec(name="V", label="Observation variance V",
              min=1e-6, max=5.0, default=0.5, step=0.01,
              help="Variance of the measurement noise."),
    ParamSpec(name="W_level", label="Level innovation variance",
              min=1e-6, max=1.0, default=0.01, step=0.001,
              help="Variance on mu_t innovation."),
    ParamSpec(name="W_slope", label="Slope innovation variance",
              min=1e-8, max=0.1, default=0.001, step=1e-4,
              help="Variance on beta_t innovation — typically much smaller than W_level."),
    ParamSpec(name="n", label="Series length n",
              min=20, max=2000, default=200, step=10, help=""),
]


_PICK_COMPONENTS_Q = ChallengeQuestion(
    kind="component_toggle",
    correct={"level": True, "slope": True, "seasonal": False},
    feedback_correct=(
        "Right. The series is drifting over time — a slope component "
        "captures the persistent direction."
    ),
    feedback_incorrect=(
        "Watch the trajectory. Does the series return to a long-run "
        "level, or does it keep moving? If it keeps moving, you need "
        "a slope component."
    ),
)

_SPECIFY_Q = ChallengeQuestion(
    kind="numeric_range",
    correct=(0.0001, 0.1),  # W_slope order-of-magnitude
    feedback_correct=(
        "Good. W_slope is almost always smaller than W_level: "
        "slopes usually change more slowly than levels."
    ),
    feedback_incorrect=(
        "W_slope should be orders of magnitude smaller than W_level. "
        "Too large and the forecast will be wildly unstable."
    ),
)


_CHALLENGES = {"pick_components": _PICK_COMPONENTS_Q, "specify": _SPECIFY_Q}


LESSON = Lesson(
    id="local_linear_trend",
    title="Local linear trend",
    tier="beginner",
    description=(
        "Level + slope. The state carries both the current level mu_t and the "
        "local slope beta_t; both evolve by random walks."
    ),
    model_builder=_build,
    param_schema=_PARAMS,
    workflow_steps=graft_challenges(
        make_default_workflow_steps(has_seasonal=False), _CHALLENGES
    ),
)
