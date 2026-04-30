"""Unit tests for advanced engine extensions."""
from __future__ import annotations

import numpy as np
import pytest

from engine.filter import kalman_filter
from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import simulate
from engine.smoother import rts_smoother


# ---------------------------------------------------------------------------
# FFBS
# ---------------------------------------------------------------------------


def test_ffbs_shape() -> None:
    from engine.ffbs import ffbs

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=30, seed=1).y
    fr = kalman_filter(spec, y)
    rng = np.random.default_rng(99)
    theta = ffbs(spec, fr, rng)
    assert theta.shape == (30, 1)


def test_ffbs_mean_matches_smoother() -> None:
    """Monte Carlo mean of FFBS draws should match RTS smoother mean."""
    from engine.ffbs import ffbs

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=50, seed=0).y
    fr = kalman_filter(spec, y)
    sr = rts_smoother(spec, fr)
    rng = np.random.default_rng(42)
    draws = np.stack([ffbs(spec, fr, rng) for _ in range(3000)])  # (3000, T, d)
    np.testing.assert_allclose(draws.mean(axis=0)[:, 0], sr.s[:, 0], atol=0.08)


def test_ffbs_terminal_matches_filter() -> None:
    """Last draw theta[-1] is sampled from the terminal filter distribution N(m_T, C_T)."""
    from engine.ffbs import ffbs

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=20, seed=7).y
    fr = kalman_filter(spec, y)
    rng = np.random.default_rng(11)
    draws = np.stack([ffbs(spec, fr, rng) for _ in range(2000)])
    # Monte Carlo mean of last step ≈ m_T
    np.testing.assert_allclose(draws[:, -1, 0].mean(), fr.m[-1, 0], atol=0.1)


def test_ffbs_llt() -> None:
    """FFBS works for a 2-d state (local linear trend)."""
    from engine.ffbs import ffbs

    spec = make_local_linear_trend(V=1.0, W_level=0.5, W_slope=0.1)
    y = simulate(spec, n=40, seed=3).y
    fr = kalman_filter(spec, y)
    rng = np.random.default_rng(55)
    theta = ffbs(spec, fr, rng)
    assert theta.shape == (40, 2)
