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


# ---------------------------------------------------------------------------
# Missing-data filter
# ---------------------------------------------------------------------------


def test_missing_filter_skips_update() -> None:
    """When y[t] is NaN the posterior equals the prior."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=20, seed=2).y.copy()
    y[5] = np.nan
    fr = kalman_filter_missing(spec, y)
    np.testing.assert_allclose(fr.m[5], fr.a[5])
    np.testing.assert_allclose(fr.C[5], fr.R[5])
    assert np.isnan(fr.e[5]).all()


def test_missing_filter_loglik_finite() -> None:
    """Missing obs contribute 0 to the log-likelihood; result is finite."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=10, seed=3).y.copy()
    fr_full = kalman_filter(spec, y)
    y[4] = np.nan
    fr_miss = kalman_filter_missing(spec, y)
    assert np.isfinite(fr_miss.loglik)
    assert fr_miss.loglik > fr_full.loglik  # one obs dropped → fewer terms → higher (less negative) loglik


def test_missing_filter_non_missing_steps_unchanged() -> None:
    """Non-NaN steps match standard kalman_filter."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=15, seed=6).y.copy()
    y[7] = np.nan
    fr_miss = kalman_filter_missing(spec, y)
    # Re-run standard filter up to t=6 (before missing)
    fr_std = kalman_filter(spec, y[:7])
    np.testing.assert_allclose(fr_miss.m[:7], fr_std.m, atol=1e-10)
    np.testing.assert_allclose(fr_miss.C[:7], fr_std.C, atol=1e-10)


def test_missing_filter_all_missing() -> None:
    """All-NaN series: every step is posterior=prior; loglik=0."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = np.full((10, 1), np.nan)
    fr = kalman_filter_missing(spec, y)
    assert fr.loglik == 0.0
    np.testing.assert_allclose(fr.m, fr.a)
