"""Tests for the DLM h-step-ahead forecast."""

import numpy as np
import pytest

from engine.filter import kalman_filter
from engine.forecast import Forecast, forecast_horizon
from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import simulate


class TestForecastBasics:
    def test_shapes(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=50, seed=0)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=10)
        assert isinstance(fc, Forecast)
        assert fc.horizon == 10
        assert fc.means.shape == (10, 1)
        assert fc.lower.shape == (10, 1)
        assert fc.upper.shape == (10, 1)

    def test_horizon_zero_empty(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=20, seed=0)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=0)
        assert fc.means.shape == (0, 1)
        assert fc.lower.shape == (0, 1)
        assert fc.upper.shape == (0, 1)

    def test_negative_horizon_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=20, seed=0)
        fr = kalman_filter(spec, series.y)
        with pytest.raises(ValueError, match="horizon"):
            forecast_horizon(spec, fr, h=-1)

    def test_constant_level_forecast_is_flat(self):
        """With W=0 (tiny nugget), forecast mean is constant = last filtered mean."""
        # Use tiny W to stay within the positive-definite constraint.
        spec = make_local_level(V=0.5, W_level=1e-10)
        series = simulate(spec, n=60, seed=2)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=5)
        for t in range(5):
            np.testing.assert_allclose(fc.means[t], fr.m[-1], atol=1e-6)


class TestForecastBandsMonotone:
    def test_band_width_increases_with_horizon(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        series = simulate(spec, n=80, seed=3)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=20)
        widths = fc.upper[:, 0] - fc.lower[:, 0]
        # Strictly non-decreasing
        assert np.all(np.diff(widths) >= -1e-10)
        # And strictly increasing over the whole horizon
        assert widths[-1] > widths[0]
