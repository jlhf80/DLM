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


# ---------------------------------------------------------------------------
# ARIMA DLM builder
# ---------------------------------------------------------------------------


def test_arima_dlm_ar1_shape() -> None:
    from engine.models import make_arima_dlm

    spec = make_arima_dlm(ar=[0.7], ma=[], sigma2=1.0)
    assert spec.d == 1
    assert spec.F.shape == (1, 1)
    assert spec.G.shape == (1, 1)
    assert spec.V.shape == (1, 1)
    assert spec.W.shape == (1, 1)


def test_arima_dlm_arma_shape() -> None:
    """ARMA(2,1): r = max(2, 1+1) = 2."""
    from engine.models import make_arima_dlm

    spec = make_arima_dlm(ar=[0.5, -0.3], ma=[0.2], sigma2=1.0)
    assert spec.d == 2
    assert spec.F.shape == (1, 2)
    assert spec.G.shape == (2, 2)


def test_arima_dlm_ar1_companion_form() -> None:
    """AR(1): G should be [[phi]], W[0,0] should be sigma2 (plus nugget)."""
    from engine.models import make_arima_dlm

    phi = 0.7
    spec = make_arima_dlm(ar=[phi], ma=[], sigma2=1.0)
    np.testing.assert_allclose(spec.G[0, 0], phi)
    np.testing.assert_allclose(spec.W[0, 0], 1.0 + 1e-10, atol=1e-9)


def test_arima_dlm_ma1_companion_form() -> None:
    """MA(1): r=2, G[0,:] = [0,0], W reflects MA structure."""
    from engine.models import make_arima_dlm

    theta = 0.4
    spec = make_arima_dlm(ar=[], ma=[theta], sigma2=2.0)
    assert spec.d == 2  # r = max(0, 1+1) = 2
    # kappa = [1, theta, 0, ...] -> W = sigma2 * outer(kappa, kappa) + nugget*I
    np.testing.assert_allclose(spec.W[0, 0], 2.0 * 1.0 + 1e-10, atol=1e-9)
    np.testing.assert_allclose(spec.W[0, 1], 2.0 * theta, atol=1e-9)
    np.testing.assert_allclose(spec.W[1, 0], 2.0 * theta, atol=1e-9)


def test_arima_dlm_loglik_matches_statsmodels_ar1() -> None:
    """AR(1) log-lik from DLM filter should match statsmodels ARIMA."""
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-untyped]

    from engine.models import make_arima_dlm

    rng = np.random.default_rng(7)
    phi, sigma2 = 0.7, 1.5
    T = 150
    y_1d = np.zeros(T)
    eps = rng.normal(scale=np.sqrt(sigma2), size=T)
    for t in range(1, T):
        y_1d[t] = phi * y_1d[t - 1] + eps[t]
    spec = make_arima_dlm(ar=[phi], ma=[], sigma2=sigma2)
    fr = kalman_filter(spec, y_1d[:, None])
    res = ARIMA(y_1d, order=(1, 0, 0)).fit()
    # Allow generous tolerance: initialisation differs slightly
    # (diffuse C0=100*I vs statsmodels exact stationary init)
    np.testing.assert_allclose(fr.loglik, res.llf, atol=6.0)


def test_arima_dlm_filter_runs() -> None:
    """ARMA(2,1) DLM filter returns finite log-lik."""
    from engine.models import make_arima_dlm

    rng = np.random.default_rng(8)
    spec = make_arima_dlm(ar=[0.5, -0.2], ma=[0.3], sigma2=1.0)
    y = rng.standard_normal((80, 1))
    fr = kalman_filter(spec, y)
    assert np.isfinite(fr.loglik)


# ---------------------------------------------------------------------------
# Multivariate local level
# ---------------------------------------------------------------------------


def test_multivariate_local_level_shape() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=3, V=[1.0, 2.0, 3.0], W_level=0.5)
    assert spec.F.shape == (3, 1)
    assert spec.G.shape == (1, 1)
    assert spec.V.shape == (3, 3)
    assert spec.W.shape == (1, 1)
    assert spec.d == 1
    assert spec.p == 3


def test_multivariate_local_level_scalar_v() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=2, V=1.5, W_level=0.5)
    np.testing.assert_allclose(np.diag(spec.V), [1.5, 1.5])


def test_multivariate_local_level_list_v() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=2, V=[1.0, 3.0], W_level=0.5)
    np.testing.assert_allclose(np.diag(spec.V), [1.0, 3.0])
    # Off-diagonals zero
    assert spec.V[0, 1] == 0.0


def test_multivariate_local_level_filter_runs() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=2, V=1.0, W_level=0.5)
    rng = np.random.default_rng(10)
    y = rng.normal(size=(50, 2))
    fr = kalman_filter(spec, y)
    assert fr.m.shape == (50, 1)
    assert np.isfinite(fr.loglik)


def test_multivariate_local_level_f_is_ones() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=4, V=1.0, W_level=0.5)
    np.testing.assert_array_equal(spec.F, np.ones((4, 1)))


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


def test_intervention_level_shift_changes_posterior() -> None:
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=4).y
    iv = {15: Intervention(kind="level", delta=10.0, component=0)}
    fr_iv = kalman_filter_interventions(spec, y, iv)
    fr_plain = kalman_filter(spec, y)
    # Posterior mean at t=15 shifts up by ~10
    assert fr_iv.m[15, 0] > fr_plain.m[15, 0] + 5.0


def test_intervention_variance_inflation() -> None:
    """Variance-inflation intervention increases R at t+1."""
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=5).y
    iv = {10: Intervention(kind="variance", scale=100.0)}
    fr_iv = kalman_filter_interventions(spec, y, iv)
    fr_plain = kalman_filter(spec, y)
    # R at t=11 should be larger with variance inflation
    assert fr_iv.R[11, 0, 0] > fr_plain.R[11, 0, 0]


def test_intervention_outlier_downweights_obs() -> None:
    """Outlier intervention down-weights a spike observation."""
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=6).y.copy()
    y[10, 0] = 100.0  # spike
    iv = {10: Intervention(kind="outlier", scale=1e6)}
    fr_plain = kalman_filter(spec, y)
    fr_iv = kalman_filter_interventions(spec, y, iv)
    # With outlier intervention, the filter at t=11 should be less pulled toward 100
    assert abs(fr_iv.m[11, 0]) < abs(fr_plain.m[11, 0])


def test_intervention_returns_filter_result() -> None:
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=20, seed=9).y
    iv: dict[int, Intervention] = {}
    fr = kalman_filter_interventions(spec, y, iv)
    # Empty interventions == standard filter
    fr_plain = kalman_filter(spec, y)
    np.testing.assert_allclose(fr.loglik, fr_plain.loglik, atol=1e-10)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


def test_monitor_result_fields() -> None:
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=20).y
    mr = kalman_filter_monitor(spec, y, inflation=100.0, threshold=0.1)
    assert mr.H.shape == (30,)
    assert mr.L.shape == (30,)
    assert mr.alert.shape == (30,)
    assert mr.alert.dtype == bool
    # L is cumulative product of H
    np.testing.assert_allclose(mr.L, np.cumprod(mr.H), atol=1e-10)


def test_monitor_alert_on_structural_break() -> None:
    """Monitor should alert near or after a structural break in state noise."""
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.1)
    rng = np.random.default_rng(21)
    T = 300
    y_arr = np.zeros((T, 1))
    level = 0.0
    for t in range(T):
        w_var = 0.1 if t < 200 else 10.0  # structural break at t=200
        level += rng.normal(scale=np.sqrt(w_var))
        y_arr[t, 0] = level + rng.normal(scale=1.0)
    mr = kalman_filter_monitor(spec, y_arr, inflation=100.0, threshold=0.1)
    alert_times = np.where(mr.alert)[0]
    assert len(alert_times) > 0
    assert alert_times[0] >= 180  # no false alarm well before break


def test_monitor_no_alert_on_stable_series() -> None:
    """Well-specified stable series should produce few false alarms."""
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=200, seed=22).y
    mr = kalman_filter_monitor(spec, y, inflation=100.0, threshold=0.1)
    # Cumulative product resets logic means few alerts for in-model series
    assert mr.alert.sum() < 20


def test_monitor_l_is_cumprod_h() -> None:
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=50, seed=23).y
    mr = kalman_filter_monitor(spec, y)
    np.testing.assert_allclose(mr.L, np.cumprod(mr.H), atol=1e-12)
