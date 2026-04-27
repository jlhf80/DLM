"""Tests for lesson content types and lesson content integrity."""

import pytest

from engine.models import DLMSpec, make_local_level
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    WorkflowStep,
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


from lessons.workflow import canonical_step_ids, make_default_workflow_steps  # noqa: E402


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
