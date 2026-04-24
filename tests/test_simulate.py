"""Tests for forward simulation."""

import numpy as np
import pytest

from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import SimulatedSeries, simulate


class TestSimulateBasics:
    def test_returns_simulated_series(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=50, seed=42)
        assert isinstance(series, SimulatedSeries)
        assert series.y.shape == (50, 1)
        assert series.theta_true.shape == (50, 1)
        assert series.seed == 42
        assert series.spec is spec

    def test_multivariate_shapes(self):
        # Hand-construct a p=2, d=1 spec
        from engine.models import DLMSpec
        spec = DLMSpec(
            F=np.array([[1.0], [0.5]]),
            G=np.eye(1),
            V=np.eye(2),
            W=np.eye(1),
            m0=np.zeros(1),
            C0=np.eye(1),
        )
        series = simulate(spec, n=30, seed=0)
        assert series.y.shape == (30, 2)
        assert series.theta_true.shape == (30, 1)


class TestSimulateReproducibility:
    def test_same_seed_bit_identical(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        s1 = simulate(spec, n=100, seed=7)
        s2 = simulate(spec, n=100, seed=7)
        np.testing.assert_array_equal(s1.y, s2.y)
        np.testing.assert_array_equal(s1.theta_true, s2.theta_true)

    def test_different_seed_different_output(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        s1 = simulate(spec, n=100, seed=7)
        s2 = simulate(spec, n=100, seed=8)
        assert not np.array_equal(s1.y, s2.y)


class TestSimulateDistribution:
    def test_local_level_empirical_variance(self):
        """For local level with V=1, W≈0, y_t = theta_0 for all t, so Var(y_t)=0
        across realizations given the same prior mean. With a diffuse C0, y_t
        becomes very variable, which masks the noise. Use a tight C0 here."""
        from engine.models import DLMSpec
        spec = DLMSpec(
            F=np.array([[1.0]]),
            G=np.array([[1.0]]),
            V=np.array([[1.0]]),
            W=np.array([[1e-11]]),  # effectively zero
            m0=np.array([0.0]),
            C0=np.array([[1e-11]]),  # effectively deterministic prior
        )
        rng_outputs = [simulate(spec, n=1, seed=s).y[0, 0] for s in range(5000)]
        emp_var = float(np.var(rng_outputs))
        # Expected Var(y_0) = V + C0 ≈ 1.0
        assert 0.9 < emp_var < 1.1

    def test_invalid_n_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        with pytest.raises(ValueError, match="n"):
            simulate(spec, n=0, seed=0)
