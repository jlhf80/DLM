"""Utilities for extracting and validating markdown anchors.

GitHub-flavored markdown auto-generates anchors for headings by lowercasing,
stripping punctuation, and replacing spaces with hyphens. This module
duplicates that rule so we can validate deep links at startup.
"""

from __future__ import annotations

import re

from lessons.workflow import Lesson

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_ANCHOR_LINK_RE = re.compile(r"\]\(/reference#([a-z0-9\-]+)\)")


def _slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


def extract_anchors(markdown: str) -> set[str]:
    """Return the set of anchors produced by H1..H6 headings."""
    return {_slugify(m.group(2)) for m in _HEADING_RE.finditer(markdown)}


def extract_referenced_anchors_from_prompts(lessons: list[Lesson]) -> set[str]:
    """Return all /reference#<anchor> targets referenced in any prompt_md."""
    found: set[str] = set()
    for lesson in lessons:
        for step in lesson.workflow_steps:
            for m in _ANCHOR_LINK_RE.finditer(step.prompt_md):
                found.add(m.group(1))
    return found


def find_missing_anchors(
    lessons: list[Lesson], available_anchors: set[str]
) -> list[str]:
    """Return anchors referenced by lesson prompts but missing from the guide."""
    referenced = extract_referenced_anchors_from_prompts(lessons)
    return sorted(referenced - available_anchors)
