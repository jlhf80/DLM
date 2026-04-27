"""Tests for lesson content types and lesson content integrity."""

import pytest

from engine.models import DLMSpec, make_local_level
from lessons.local_level import LESSON as LOCAL_LEVEL_LESSON
from lessons.local_linear_trend import LESSON as LLT_LESSON
from lessons.seasonal import LESSON as SEASONAL_LESSON
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    WorkflowStep,
    canonical_step_ids,
    graft_challenges,
    make_default_workflow_steps,
)


class TestParamSpec:
    def test_valid_paramspec(self):
        p = ParamSpec(
            name="V", label="Obs variance", min=1e-6, max=5.0,
            default=0.5, step=0.01, help="blah",
        )
        assert p.name == "V"
        assert p.min < p.default < p.max

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="min"):
            ParamSpec(name="x", label="x", min=1.0, max=0.5,
                      default=0.7, step=0.01, help="")

    def test_default_outside_range_raises(self):
        with pytest.raises(ValueError, match="default"):
            ParamSpec(name="x", label="x", min=0.0, max=1.0,
                      default=2.0, step=0.01, help="")


class TestChallengeQuestion:
    def test_multiple_choice_correct_must_be_str(self):
        with pytest.raises(ValueError, match="multiple_choice"):
            ChallengeQuestion(
                kind="multiple_choice", correct=42,
                feedback_correct="", feedback_incorrect="",
            )

    def test_numeric_range_correct_must_be_tuple(self):
        with pytest.raises(ValueError, match="numeric_range"):
            ChallengeQuestion(
                kind="numeric_range", correct="string",
                feedback_correct="", feedback_incorrect="",
            )

    def test_component_toggle_correct_must_be_dict(self):
        with pytest.raises(ValueError, match="component_toggle"):
            ChallengeQuestion(
                kind="component_toggle", correct=[True, False],
                feedback_correct="", feedback_incorrect="",
            )


class TestWorkflowStepAndLesson:
    def test_lesson_rejects_empty_workflow(self):
        with pytest.raises(ValueError, match="workflow"):
            Lesson(
                id="x", title="x", tier="beginner", description="",
                model_builder=lambda p: make_local_level(V=1.0, W_level=0.1),
                param_schema=[],
                workflow_steps=[],
            )

    def test_validate_lesson_accepts_good_lesson(self):
        params = [
            ParamSpec(name="V", label="V", min=1e-6, max=5.0,
                      default=0.5, step=0.01, help=""),
            ParamSpec(name="W_level", label="W", min=1e-6, max=1.0,
                      default=0.1, step=0.01, help=""),
        ]
        steps = [
            WorkflowStep(
                id="inspect", title="Inspect", prompt_md="Look.",
                plot_fn="time_series", hints=[], challenge=None,
            ),
        ]
        lesson = Lesson(
            id="test", title="Test", tier="beginner", description="",
            model_builder=lambda p: make_local_level(V=p["V"], W_level=p["W_level"]),
            param_schema=params,
            workflow_steps=steps,
        )
        # Should build a valid DLMSpec with default params
        spec = lesson.model_builder(
            {p.name: p.default for p in lesson.param_schema}
        )
        assert isinstance(spec, DLMSpec)


class TestWorkflowStepWithChallenge:
    def test_with_challenge_returns_new_step_carrying_question(self):
        step = WorkflowStep(
            id="x", title="X", prompt_md="", plot_fn="time_series",
        )
        q = ChallengeQuestion(
            kind="multiple_choice", correct="a",
            feedback_correct="", feedback_incorrect="",
        )
        new = step.with_challenge(q)
        assert new is not step
        assert new.challenge is q
        assert step.challenge is None
        assert new.id == step.id and new.plot_fn == step.plot_fn


class TestGraftChallenges:
    def test_only_targeted_steps_get_challenge(self):
        steps = make_default_workflow_steps(has_seasonal=False)
        q = ChallengeQuestion(
            kind="multiple_choice", correct="a",
            feedback_correct="", feedback_incorrect="",
        )
        out = graft_challenges(steps, {"specify": q})
        targeted = next(s for s in out if s.id == "specify")
        untouched = next(s for s in out if s.id == "fit")
        assert targeted.challenge is q
        assert untouched.challenge is None
        assert len(out) == len(steps)

    def test_unknown_step_id_raises(self):
        steps = make_default_workflow_steps(has_seasonal=False)
        q = ChallengeQuestion(
            kind="multiple_choice", correct="a",
            feedback_correct="", feedback_incorrect="",
        )
        with pytest.raises(ValueError, match="not_a_real_step"):
            graft_challenges(steps, {"not_a_real_step": q})


class TestCanonicalWorkflow:
    def test_nine_step_ids(self):
        assert canonical_step_ids() == [
            "inspect_data",
            "decompose",
            "quantify",
            "pick_components",
            "specify",
            "fit",
            "diagnose",
            "forecast",
            "reveal",
        ]

    def test_default_steps_have_all_nine(self):
        steps = make_default_workflow_steps(has_seasonal=False)
        assert len(steps) == 9
        assert [s.id for s in steps] == canonical_step_ids()

    def test_seasonal_flag_changes_quantify_plot(self):
        no = make_default_workflow_steps(has_seasonal=False)
        yes = make_default_workflow_steps(has_seasonal=True)
        quant_no = next(s for s in no if s.id == "quantify")
        quant_yes = next(s for s in yes if s.id == "quantify")
        assert quant_no.plot_fn == "acf_pacf"
        assert quant_yes.plot_fn == "acf_pacf_and_seasonal_subseries"


class TestLocalLevelLesson:
    def test_builds_valid_spec_on_defaults(self):
        params = {p.name: p.default for p in LOCAL_LEVEL_LESSON.param_schema}
        spec = LOCAL_LEVEL_LESSON.model_builder(params)
        assert spec.d == 1 and spec.p == 1

    def test_has_nine_steps(self):
        assert len(LOCAL_LEVEL_LESSON.workflow_steps) == 9

    def test_pick_components_question_is_toggle(self):
        step4 = next(s for s in LOCAL_LEVEL_LESSON.workflow_steps if s.id == "pick_components")
        assert step4.challenge is not None
        assert step4.challenge.kind == "component_toggle"
        assert step4.challenge.correct == {"level": True, "slope": False, "seasonal": False}

    def test_specify_variance_order_of_magnitude(self):
        step5 = next(s for s in LOCAL_LEVEL_LESSON.workflow_steps if s.id == "specify")
        assert step5.challenge is not None
        assert step5.challenge.kind == "numeric_range"


class TestLocalLinearTrendLesson:
    def test_builds_valid_spec_on_defaults(self):
        params = {p.name: p.default for p in LLT_LESSON.param_schema}
        spec = LLT_LESSON.model_builder(params)
        assert spec.d == 2 and spec.p == 1

    def test_pick_components_correct_has_slope(self):
        step4 = next(s for s in LLT_LESSON.workflow_steps if s.id == "pick_components")
        assert step4.challenge.correct == {"level": True, "slope": True, "seasonal": False}

    def test_param_schema_has_slope(self):
        names = [p.name for p in LLT_LESSON.param_schema]
        assert "W_slope" in names


class TestSeasonalLesson:
    def test_defaults_build_quarterly_spec(self):
        params = {p.name: p.default for p in SEASONAL_LESSON.param_schema}
        spec = SEASONAL_LESSON.model_builder(params)
        # With default period=4, d = period - 1 = 3
        assert spec.d == 3 and spec.p == 1

    def test_quantify_step_uses_seasonal_plot(self):
        step3 = next(s for s in SEASONAL_LESSON.workflow_steps if s.id == "quantify")
        assert step3.plot_fn == "acf_pacf_and_seasonal_subseries"

    def test_pick_components_correct_has_seasonal(self):
        step4 = next(s for s in SEASONAL_LESSON.workflow_steps if s.id == "pick_components")
        assert step4.challenge.correct == {"level": False, "slope": False, "seasonal": True}

    def test_specify_asks_period(self):
        step5 = next(s for s in SEASONAL_LESSON.workflow_steps if s.id == "specify")
        assert step5.challenge is not None
        assert step5.challenge.kind == "multiple_choice"
        # Correct must be string (period label)
        assert step5.challenge.correct in {"2", "4", "7", "12"}
