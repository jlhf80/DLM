"""Tests for the RTS smoother."""

import numpy as np

from engine.filter import kalman_filter
from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import simulate
from engine.smoother import SmoothResult, rts_smoother


class TestSmootherBasics:
    def test_returns_shapes(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=40, seed=0)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        assert isinstance(sr, SmoothResult)
        assert sr.s.shape == (40, 1)
        assert sr.S.shape == (40, 1, 1)

    def test_terminal_equals_filtered(self):
        """At t=T, smoother output equals filter output."""
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        series = simulate(spec, n=30, seed=1)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        np.testing.assert_allclose(sr.s[-1], fr.m[-1])
        np.testing.assert_allclose(sr.S[-1], fr.C[-1])

    def test_covariance_symmetric(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        series = simulate(spec, n=30, seed=1)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        for t in range(sr.S.shape[0]):
            np.testing.assert_allclose(sr.S[t], sr.S[t].T, atol=1e-10)

    def test_smoother_variance_not_greater_than_filter(self):
        """Smoothed variance <= filtered variance at each t (smoother uses more info)."""
        spec = make_local_level(V=1.0, W_level=0.1)
        series = simulate(spec, n=100, seed=2)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        for t in range(99):
            assert sr.S[t, 0, 0] <= fr.C[t, 0, 0] + 1e-10
