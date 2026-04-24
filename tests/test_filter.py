"""Tests for the Kalman filter."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from engine.filter import FilterResult, kalman_filter
from engine.models import DLMSpec, make_local_level
from engine.simulate import simulate


def _ll_from_joint(spec: DLMSpec, y: np.ndarray) -> float:
    """Reference log-likelihood by direct computation of the marginal joint
    N(0, Sigma_y) covariance of y_{1:T}.

    For a stationary-prior local level this is tractable for small T.
    """
    T = y.shape[0]
    p = spec.p  # noqa: F841
    d = spec.d
    # Compute the full (T*p, T*p) covariance of y_{1:T}.
    F, G, V, W, m0, C0 = spec.F, spec.G, spec.V, spec.W, spec.m0, spec.C0
    # Mean of y_t: F @ G^t @ m0
    mean = np.concatenate([(F @ np.linalg.matrix_power(G, t + 1) @ m0) for t in range(T)])
    # Cov(theta_s, theta_t) = G^s C0 (G^t).T + sum_{k=1}^{min(s,t)} G^{s-k} W (G^{t-k}).T
    Sigma_theta = np.zeros((T * d, T * d))
    for s in range(T):
        Gs = np.linalg.matrix_power(G, s + 1)
        for t in range(T):
            Gt = np.linalg.matrix_power(G, t + 1)
            cov = Gs @ C0 @ Gt.T
            for k in range(1, min(s, t) + 2):
                Gsk = np.linalg.matrix_power(G, s + 1 - k)
                Gtk = np.linalg.matrix_power(G, t + 1 - k)
                cov = cov + Gsk @ W @ Gtk.T
            Sigma_theta[s * d:(s + 1) * d, t * d:(t + 1) * d] = cov
    # Full F_block and V_block
    F_block = np.kron(np.eye(T), F)
    V_block = np.kron(np.eye(T), V)
    Sigma_y = F_block @ Sigma_theta @ F_block.T + V_block
    # Symmetrize to guard fp drift
    Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
    y_flat = y.reshape(-1)
    return float(multivariate_normal.logpdf(y_flat, mean=mean, cov=Sigma_y))


class TestKalmanFilterBasics:
    def test_returns_shapes(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=50, seed=0)
        res = kalman_filter(spec, series.y)
        assert isinstance(res, FilterResult)
        assert res.m.shape == (50, 1)
        assert res.C.shape == (50, 1, 1)
        assert res.a.shape == (50, 1)
        assert res.R.shape == (50, 1, 1)
        assert res.f.shape == (50, 1)
        assert res.Q.shape == (50, 1, 1)
        assert res.e.shape == (50, 1)
        assert np.isfinite(res.loglik)

    def test_innovations_consistency(self):
        """e_t must equal y_t - f_t."""
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=30, seed=1)
        res = kalman_filter(spec, series.y)
        np.testing.assert_allclose(res.e, series.y - res.f)


class TestKalmanFilterAgainstReference:
    def test_loglik_matches_joint_small_T(self):
        """For T=5, filter loglik must match direct joint-normal logpdf."""
        spec = make_local_level(V=1.0, W_level=0.3, m0=0.0, C0=2.0)
        series = simulate(spec, n=5, seed=42)
        res = kalman_filter(spec, series.y)
        ref = _ll_from_joint(spec, series.y)
        np.testing.assert_allclose(res.loglik, ref, atol=1e-10)


class TestKalmanFilterConvergence:
    def test_local_level_steady_state(self):
        """For constant local level, the filtered variance C_t converges to
        the positive root of the Riccati:  C_inf such that
            C_inf = (C_inf + W) * V / (C_inf + W + V)
        (see West & Harrison §2.3 / Petris §2.6)."""
        V = 1.0
        W = 0.1
        spec = make_local_level(V=V, W_level=W, m0=0.0, C0=1e6)
        # Long enough to reach steady state
        series = simulate(spec, n=500, seed=0)
        res = kalman_filter(spec, series.y)
        # Analytic fixed point: solve C = (C+W)V/(C+W+V)
        # => C(C+W+V) = V(C+W)
        # => C^2 + CW + CV = VC + VW
        # => C^2 + CW - VW = 0
        # => C = (-W + sqrt(W^2 + 4VW)) / 2
        C_inf = 0.5 * (-W + np.sqrt(W ** 2 + 4 * V * W))
        np.testing.assert_allclose(res.C[-1, 0, 0], C_inf, rtol=1e-3)


class TestKalmanFilterEdgeCases:
    def test_length_mismatch_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        y = np.zeros((10, 2))  # p=2 but spec has p=1
        with pytest.raises(ValueError, match="observation dimension"):
            kalman_filter(spec, y)

    def test_zero_length_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        with pytest.raises(ValueError, match="at least one"):
            kalman_filter(spec, np.zeros((0, 1)))
