"""Tests for ACF/PACF and residual diagnostics."""

import numpy as np
import pytest

from engine.diagnostics import (
    AcfPacfResult,
    ResidualDiagnostics,
    acf_pacf,
    residual_diagnostics,
)


class TestAcfPacf:
    def test_returns_expected_shapes(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(500, 1))
        res = acf_pacf(y, nlags=20)
        assert isinstance(res, AcfPacfResult)
        assert res.lags.shape == (21,)       # lag 0 .. 20
        assert res.acf.shape == (21,)
        assert res.pacf.shape == (21,)
        assert res.lags[0] == 0
        assert res.lags[-1] == 20

    def test_white_noise_acf_within_band(self):
        rng = np.random.default_rng(7)
        y = rng.normal(size=(2000, 1))
        res = acf_pacf(y, nlags=20)
        # 95% band ~ 1.96 / sqrt(n) ≈ 0.0438; some small exceedances are OK
        n_exceed = int(np.sum(np.abs(res.acf[1:]) > 1.96 / np.sqrt(2000)))
        assert n_exceed <= 2     # at most 2 of 20 outside band

    def test_rejects_multivariate(self):
        y = np.zeros((50, 2))
        with pytest.raises(ValueError, match="univariate"):
            acf_pacf(y, nlags=10)


class TestResidualDiagnostics:
    def test_shapes_and_fields(self):
        rng = np.random.default_rng(0)
        e = rng.normal(size=(200, 1))        # innovations
        Q = np.ones((200, 1, 1))             # unit predictive variance
        res = residual_diagnostics(e, Q, nlags=15)
        assert isinstance(res, ResidualDiagnostics)
        assert res.standardized.shape == (200, 1)
        assert res.acf_pacf.lags.shape == (16,)
        assert 0.0 <= res.ljung_box_pvalue <= 1.0

    def test_white_noise_ljung_box_not_rejected(self):
        rng = np.random.default_rng(11)
        e = rng.normal(size=(500, 1))
        Q = np.ones((500, 1, 1))
        res = residual_diagnostics(e, Q, nlags=10)
        assert res.ljung_box_pvalue > 0.05

    def test_correlated_residuals_ljung_box_rejects(self):
        """AR(1) residuals should be flagged."""
        rng = np.random.default_rng(12)
        phi = 0.7
        eps = rng.normal(size=600)
        e = np.zeros(600)
        for t in range(1, 600):
            e[t] = phi * e[t - 1] + eps[t]
        e2d = e[:, None]
        Q = np.ones((600, 1, 1))
        res = residual_diagnostics(e2d, Q, nlags=10)
        assert res.ljung_box_pvalue < 0.01
