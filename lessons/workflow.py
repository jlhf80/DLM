"""Declarative types that describe a lesson's content and structure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
    """A graded prompt attached to a workflow step.

    `correct` is either a static value matching `kind`, or a callable
    `(params: dict[str, float]) -> static_value` that resolves to the
    correct answer at grade-time. Use a callable when the right answer
    depends on the user's current parameter settings (e.g. the seasonal
    period). The static value is validated at construction; the callable's
    return value is validated lazily by `resolve()`.
    """

    kind: ChallengeKind
    correct: Any
    feedback_correct: str
    feedback_incorrect: str

    def __post_init__(self) -> None:
        if self.kind not in ("multiple_choice", "numeric_range", "component_toggle"):
            raise ValueError(f"unknown ChallengeQuestion kind {self.kind!r}")
        if not callable(self.correct):
            self._validate_value(self.correct)

    def _validate_value(self, value: Any) -> None:
        if self.kind == "multiple_choice":
            if not isinstance(value, str):
                raise ValueError(
                    f"multiple_choice 'correct' must be str, got {type(value).__name__}"
                )
        elif self.kind == "numeric_range":
            ok = (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(x, (int, float)) for x in value)
                and value[0] <= value[1]
            )
            if not ok:
                raise ValueError(
                    f"numeric_range 'correct' must be (low, high) tuple, got {value!r}"
                )
        elif self.kind == "component_toggle" and not (
            isinstance(value, dict)
            and all(isinstance(v, bool) for v in value.values())
        ):
            raise ValueError(
                f"component_toggle 'correct' must be dict[str, bool], got {value!r}"
            )

    def resolve(self, params: dict[str, float]) -> Any:
        """Return the canonical correct answer for `params`.

        If `correct` was provided as a callable, it is invoked with `params`
        and its return value is validated against `kind`. Otherwise the
        static value is returned unchanged.
        """
        if callable(self.correct):
            value = self.correct(params)
            self._validate_value(value)
            return value
        return self.correct


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    title: str
    prompt_md: str
    plot_fn: str
    hints: list[str] = field(default_factory=list)
    challenge: ChallengeQuestion | None = None

    def with_challenge(self, challenge: ChallengeQuestion) -> WorkflowStep:
        """Return a copy of this step with `challenge` attached."""
        return replace(self, challenge=challenge)


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


def graft_challenges(
    steps: list[WorkflowStep],
    mapping: dict[str, ChallengeQuestion],
) -> list[WorkflowStep]:
    """Return a copy of `steps` with challenges attached by step id.

    Steps whose `id` is not a key in `mapping` are returned unchanged.
    Raises ValueError if `mapping` references an id not present in `steps`.
    """
    step_ids = {s.id for s in steps}
    unknown = set(mapping) - step_ids
    if unknown:
        raise ValueError(f"graft_challenges: unknown step ids {sorted(unknown)}")
    return [s.with_challenge(mapping[s.id]) if s.id in mapping else s for s in steps]


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


_CANONICAL_STEP_IDS: tuple[str, ...] = (
    "inspect_data",
    "decompose",
    "quantify",
    "pick_components",
    "specify",
    "fit",
    "diagnose",
    "forecast",
    "reveal",
)


def canonical_step_ids() -> list[str]:
    return list(_CANONICAL_STEP_IDS)


def make_default_workflow_steps(has_seasonal: bool) -> list[WorkflowStep]:
    """The 9-step workflow with default parameter-agnostic prompts.

    Lessons customize steps 4 (pick_components), 5 (specify), and 9 (reveal)
    by passing in `challenge` questions at construction time. The defaults
    here are info-only (challenge=None) and serve the Lesson-mode narrative.
    """
    quantify_plot_fn = "acf_pacf_and_seasonal_subseries" if has_seasonal else "acf_pacf"
    return [
        WorkflowStep(
            id="inspect_data",
            title="Step 1 — Inspect the data",
            prompt_md=(
                "Any time series analysis starts with looking at the data. "
                "What patterns stand out? A persistent level? A drift? "
                "Anything periodic? Jot a mental answer before moving on."
            ),
            plot_fn="time_series",
        ),
        WorkflowStep(
            id="decompose",
            title="Step 2 — Decompose visually",
            prompt_md=(
                "Visual decomposition is the first hypothesis step. "
                "A shaded overlay of a moving-average trend helps surface a drift "
                "even when the noise is large. [More detail]"
                "(/reference#baseline-procedure)."
            ),
            plot_fn="visual_decomposition",
        ),
        WorkflowStep(
            id="quantify",
            title="Step 3 — Quantify autocorrelation",
            prompt_md=(
                "The sample ACF and PACF let us *measure* what the eye suggested. "
                "A slow ACF decay is a trend signature; a spike at lag s is seasonal. "
                "[Reference](/reference#baseline-procedure)."
            ),
            plot_fn=quantify_plot_fn,
        ),
        WorkflowStep(
            id="pick_components",
            title="Step 4 — Pick components",
            prompt_md=(
                "Decide which DLM components you need: a level, a slope, a "
                "seasonal. This is where your answers to steps 2 and 3 "
                "crystallize into a model — keep the visual decomposition in "
                "view while you choose."
            ),
            plot_fn="visual_decomposition",
        ),
        WorkflowStep(
            id="specify",
            title="Step 5 — Specify the DLM",
            prompt_md=(
                "Now write down the DLM matrices for the components you chose:\n\n"
                "- **F (observation matrix)** — how the state projects to the "
                "observation: $y_t = F\\, \\theta_t + v_t$. For a level "
                "component F picks out the level; for a seasonal factor F "
                "picks out the current season's effect.\n"
                "- **G (state evolution)** — how the state evolves between "
                "steps: $\\theta_t = G\\, \\theta_{t-1} + w_t$. A pure level "
                "uses $G = I$ (random walk); a slope adds a column that "
                "carries the slope into the level each step.\n"
                "- **V (observation variance)** — how noisy the measurement "
                "$y_t$ is.\n"
                "- **W (state innovation variance)** — how quickly each state "
                "component drifts. Larger $W$ means faster drift: the filter "
                "follows the data more closely but is jumpier.\n\n"
                "The heatmaps below show the matrices the simulator built "
                "from the sidebar parameters. "
                "[Reference on specification](/reference#baseline-procedure)."
            ),
            plot_fn="spec_preview",
        ),
        WorkflowStep(
            id="fit",
            title="Step 6 — Fit (Kalman filter)",
            prompt_md=(
                "Run the Kalman filter. The filtered state means "
                "$E[\\theta_t \\mid y_{1:t}]$ track the underlying components, "
                "with 95% credible bands showing the posterior uncertainty."
            ),
            plot_fn="filter_state",
        ),
        WorkflowStep(
            id="diagnose",
            title="Step 7 — Diagnose (residuals + smoothing)",
            prompt_md=(
                "One-step forecast residuals should look like white noise. "
                "We also compare the smoothed state (using all of "
                "$y_{1:T}$) against the filtered state. "
                "[Reference on diagnostics](/reference#diagnostics)."
            ),
            plot_fn="diagnostics",
        ),
        WorkflowStep(
            id="forecast",
            title="Step 8 — Forecast",
            prompt_md=(
                "h-step-ahead forecasts with 95% credible bands. Bands widen "
                "with horizon — that is the price of uncertainty."
            ),
            plot_fn="forecast",
        ),
        WorkflowStep(
            id="reveal",
            title="Step 9 — Review",
            prompt_md=(
                "Lesson mode: recap of the method. "
                "Challenge mode: your fitted DLM overlaid against the ground-truth DLM."
            ),
            plot_fn="reveal_overlay",
        ),
    ]
