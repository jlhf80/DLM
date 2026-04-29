"""Tests for modeling-guide.md integrity."""

from pathlib import Path

from lessons import ALL_LESSONS
from ui.content_index import (
    _slugify,
    extract_anchors,
    extract_referenced_anchors_from_prompts,
    find_missing_anchors,
)

GUIDE_PATH = Path(__file__).resolve().parent.parent / "content" / "modeling-guide.md"


def test_guide_file_exists():
    assert GUIDE_PATH.exists(), f"missing {GUIDE_PATH}"


def test_extract_anchors_finds_top_level_headings():
    anchors = extract_anchors(GUIDE_PATH.read_text(encoding="utf-8"))
    for required in (
        "the-dlm-framework",
        "baseline-procedure",
        "tuning",
        "diagnostics",
        "worked-example",
    ):
        assert required in anchors, f"missing anchor {required!r} in {sorted(anchors)}"


def test_slugify_matches_github_rules():
    assert _slugify("Baseline procedure") == "baseline-procedure"
    assert _slugify("The DLM framework") == "the-dlm-framework"
    # em-dash is stripped, surrounding spaces collapse to one hyphen
    assert _slugify("Step 6 — Fit") == "step-6-fit"
    assert _slugify("V vs W tradeoff") == "v-vs-w-tradeoff"


def test_referenced_anchors_match_lessons():
    """All anchors referenced by lesson prompts must exist in the guide."""
    anchors = extract_anchors(GUIDE_PATH.read_text(encoding="utf-8"))
    missing = find_missing_anchors(ALL_LESSONS, available_anchors=anchors)
    assert missing == [], f"Missing anchors: {missing}"


def test_referenced_anchors_nonempty():
    """Sanity: at least some lesson prompts deep-link into the guide."""
    referenced = extract_referenced_anchors_from_prompts(ALL_LESSONS)
    assert referenced, "no /reference#... anchors found in lesson prompts"
