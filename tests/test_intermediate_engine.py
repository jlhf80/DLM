"""Unit tests for intermediate-tier engine extensions."""

from __future__ import annotations

import numpy as np
import pytest

from engine.comparison import compare_models, log_bayes_factor
from engine.conjugate import kalman_filter_conjugate
from engine.filter import (
    FilterResult,
    kalman_filter,
    kalman_filter_discount,
    kalman_filter_tv,
)
from engine.models import (
    DLMSpec,
    DLMSpecTV,
    make_fourier_seasonal,
    make_local_level,
    make_local_linear_trend,
    make_seasonal_factor,
)
from engine.simulate import simulate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_level(V: float = 1.0, W: float = 0.1) -> DLMSpec:
    return make_local_level(V=V, W_level=W)


def _sim(spec: DLMSpec, T: int = 100, seed: int = 42) -> np.ndarray:
    return simulate(spec, n=T, seed=seed).y


# ---------------------------------------------------------------------------
# Task 1: Discount-factor filter
# ---------------------------------------------------------------------------


class TestKalmanFilterDiscount:
    def test_delta1_analytically(self) -> None:
        """delta=1: first-step R_1 = G C0 G' (no inflation), matching standard filter with W→0."""
        V = 1.0
        C0 = 100.0
        # Use a tiny W so DLMSpec validates; discount filter ignores it.
        spec = make_local_level(V=V, W_level=1e-9, C0=C0)
        y = np.array([[2.0]])
        fr = kalman_filter_discount(spec, y, delta=1.0)
        # For local level: G=I, so R_1 = C0/delta = C0; Q_1 = R_1 + V
        R1_expected = C0  # G=I, delta=1
        Q1_expected = R1_expected + V
        np.testing.assert_allclose(fr.Q[0, 0, 0], Q1_expected, atol=1e-6)

    def test_smaller_delta_inflates_R(self) -> None:
        """Smaller discount factor → larger R_t → wider posterior uncertainty."""
        spec = make_local_level(V=1.0, W_level=1e-9)  # W ignored; tiny for validation
        y = _sim(spec)
        fr_09 = kalman_filter_discount(spec, y, delta=0.9)
        fr_099 = kalman_filter_discount(spec, y, delta=0.99)
        # Mean posterior variance should be larger for smaller delta
        assert fr_09.C.mean() > fr_099.C.mean()

    def test_returns_filter_result(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec)
        fr = kalman_filter_discount(spec, y, delta=0.95)
        assert isinstance(fr, FilterResult)
        assert fr.m.shape == (len(y), spec.d)

    def test_invalid_delta_raises(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec)
        with pytest.raises(ValueError, match="delta"):
            kalman_filter_discount(spec, y, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            kalman_filter_discount(spec, y, delta=1.1)

    def test_one_step_posterior_variance_local_level(self) -> None:
        """After one obs from prior N(0, C0), R_1 = C0/delta, Q_1 = R_1 + V."""
        V = 1.0
        C0 = 100.0
        delta = 0.9
        spec = make_local_level(V=V, W_level=1e-9, C0=C0)  # tiny W; ignored by discount
        y = np.array([[2.0]])
        fr = kalman_filter_discount(spec, y, delta=delta)
        R1_expected = C0 / delta
        Q1_expected = R1_expected + V
        assert abs(float(fr.Q[0, 0, 0]) - Q1_expected) < 1e-9


# ---------------------------------------------------------------------------
# Task 2: Conjugate unknown-V filter
# ---------------------------------------------------------------------------


class TestKalmanFilterConjugate:
    def test_v_estimate_converges_to_true_V(self) -> None:
        """With many observations the running V estimate should track truth."""
        V_true = 2.0
        spec = make_local_level(V=V_true, W_level=0.05)
        y = _sim(spec, T=500)
        cr = kalman_filter_conjugate(spec, y, n0=2.0, d0=2.0 * V_true)
        # Last posterior estimate should be within 20% of true V
        v_final = cr.v_hat[-1]
        assert abs(v_final - V_true) / V_true < 0.20

    def test_requires_p1(self) -> None:
        spec = make_local_linear_trend(V=1.0, W_level=0.1, W_slope=0.01)
        # LLT has p=1, so this should work
        y = _sim(spec)
        cr = kalman_filter_conjugate(spec, y, n0=2.0, d0=2.0)
        assert cr.m.shape[0] == len(y)

    def test_n_increments_each_step(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec, T=50)
        cr = kalman_filter_conjugate(spec, y, n0=4.0, d0=4.0)
        np.testing.assert_allclose(cr.n, np.arange(5.0, 5.0 + 50))

    def test_invalid_n0_d0_raises(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec)
        with pytest.raises(ValueError, match="positive"):
            kalman_filter_conjugate(spec, y, n0=-1.0, d0=1.0)
        with pytest.raises(ValueError, match="positive"):
            kalman_filter_conjugate(spec, y, n0=1.0, d0=0.0)

    def test_large_n0_approximates_fixed_v_filter(self) -> None:
        """For very informative prior (large n0), estimates close to fixed-V filter."""
        V_true = 1.5
        spec = make_local_level(V=V_true, W_level=0.05)
        y = _sim(spec, T=80)
        # Tight IG prior centred on V_true
        cr = kalman_filter_conjugate(spec, y, n0=1000.0, d0=1000.0 * V_true)
        fr = kalman_filter(spec, y)
        # State means should be close (within ~1.5% of signal scale)
        np.testing.assert_allclose(cr.m, fr.m, atol=0.15)


# ---------------------------------------------------------------------------
# Task 3: Time-varying observation matrix (kalman_filter_tv)
# ---------------------------------------------------------------------------


class TestKalmanFilterTV:
    def _constant_F_seq(self, spec: DLMSpec, T: int) -> np.ndarray:
        return np.tile(spec.F, (T, 1, 1))  # (T, p, d)

    def test_constant_F_seq_matches_standard_filter(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec)
        T = len(y)
        F_seq = self._constant_F_seq(spec, T)
        fr_tv = kalman_filter_tv(
            F_seq, spec.G, spec.V, spec.W, spec.m0, spec.C0, y
        )
        fr_std = kalman_filter(spec, y)
        np.testing.assert_allclose(fr_tv.m, fr_std.m, atol=1e-10)
        np.testing.assert_allclose(fr_tv.C, fr_std.C, atol=1e-10)
        np.testing.assert_allclose(fr_tv.loglik, fr_std.loglik, atol=1e-10)

    def test_wrong_F_seq_shape_raises(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec, T=30)
        bad_F_seq = np.ones((30, 1, 3))   # wrong d
        with pytest.raises(ValueError, match="F_seq"):
            kalman_filter_tv(bad_F_seq, spec.G, spec.V, spec.W, spec.m0, spec.C0, y)

    def test_returns_filter_result(self) -> None:
        spec = make_local_linear_trend(V=1.0, W_level=0.1, W_slope=0.01)
        y = _sim(spec)
        T = len(y)
        F_seq = self._constant_F_seq(spec, T)
        fr = kalman_filter_tv(F_seq, spec.G, spec.V, spec.W, spec.m0, spec.C0, y)
        assert isinstance(fr, FilterResult)


# ---------------------------------------------------------------------------
# Task 3b: DLMSpecTV dataclass
# ---------------------------------------------------------------------------


class TestDLMSpecTV:
    def test_valid_construction(self) -> None:
        T, p, d = 50, 1, 2
        F_seq = np.ones((T, p, d))
        G = np.eye(d)
        V = np.eye(p)
        W = 0.1 * np.eye(d)
        m0 = np.zeros(d)
        C0 = np.eye(d)
        spec_tv = DLMSpecTV(F_seq=F_seq, G=G, V=V, W=W, m0=m0, C0=C0)
        assert spec_tv.T == T and spec_tv.p == p and spec_tv.d == d

    def test_wrong_G_shape_raises(self) -> None:
        T, p, d = 10, 1, 2
        F_seq = np.ones((T, p, d))
        with pytest.raises(ValueError, match="G"):
            DLMSpecTV(
                F_seq=F_seq, G=np.eye(3), V=np.eye(p),
                W=np.eye(d), m0=np.zeros(d), C0=np.eye(d),
            )


# ---------------------------------------------------------------------------
# Task 4: Fourier seasonal builder
# ---------------------------------------------------------------------------


class TestMakeFourierSeasonal:
    def test_state_dimension_two_harmonics(self) -> None:
        spec = make_fourier_seasonal(period=12, n_harmonics=2, V=1.0, W_season=0.01)
        assert spec.d == 4   # 2 harmonics * 2 state dims each

    def test_state_dimension_one_harmonic(self) -> None:
        spec = make_fourier_seasonal(period=12, n_harmonics=1, V=1.0, W_season=0.01)
        assert spec.d == 2

    def test_nyquist_harmonic_reduces_dim(self) -> None:
        # period=4, max J=2; J=2 hits Nyquist (1x1 block) → d = 2+1 = 3
        spec = make_fourier_seasonal(period=4, n_harmonics=2, V=1.0, W_season=0.01)
        assert spec.d == 3   # 2 (J=1 block) + 1 (Nyquist block)

    def test_p_equals_one(self) -> None:
        spec = make_fourier_seasonal(period=12, n_harmonics=3, V=1.0, W_season=0.01)
        assert spec.p == 1

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            make_fourier_seasonal(period=1, n_harmonics=1, V=1.0, W_season=0.01)

    def test_too_many_harmonics_raises(self) -> None:
        with pytest.raises(ValueError, match="n_harmonics"):
            make_fourier_seasonal(period=12, n_harmonics=7, V=1.0, W_season=0.01)

    def test_rotation_block_structure(self) -> None:
        """J=1 G should be a 2x2 rotation matrix."""
        spec = make_fourier_seasonal(period=12, n_harmonics=1, V=1.0, W_season=0.01)
        G = spec.G
        # Rotation: G G' = I
        np.testing.assert_allclose(G @ G.T, np.eye(2), atol=1e-10)
        # Correct angle: 2*pi/12
        omega = 2 * np.pi / 12
        G_expected = np.array([[np.cos(omega), np.sin(omega)],
                                [-np.sin(omega), np.cos(omega)]])
        np.testing.assert_allclose(G, G_expected, atol=1e-10)

    def test_filter_runs_without_error(self) -> None:
        spec = make_fourier_seasonal(period=4, n_harmonics=2, V=0.5, W_season=0.05)
        _spec_sim = make_seasonal_factor(period=4, V=0.5, W_season=0.05)
        y = _sim(_spec_sim, T=100)
        fr = kalman_filter(spec, y)
        assert np.isfinite(fr.loglik)


# ---------------------------------------------------------------------------
# Task 5: Model comparison
# ---------------------------------------------------------------------------


class TestModelComparison:
    def test_identical_models_zero_bayes_factor(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec)
        df = compare_models({"m1": spec, "m2": spec}, y)
        assert abs(df.loc[df["model"] == "m1", "log_bf_vs_best"].iloc[0]
                   - df.loc[df["model"] == "m2", "log_bf_vs_best"].iloc[0]) < 1e-10

    def test_true_model_wins(self) -> None:
        """Simulate from local-level; local-level should beat LLT on log-lik."""
        spec_true = make_local_level(V=1.0, W_level=0.2)
        spec_llt = make_local_linear_trend(V=1.0, W_level=0.2, W_slope=0.001)
        y = _sim(spec_true, T=200)
        df = compare_models({"local_level": spec_true, "llt": spec_llt}, y)
        best = df.iloc[0]["model"]
        assert best == "local_level"

    def test_delta_loglik_best_model_is_zero(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        spec2 = make_local_level(V=2.0, W_level=0.1)
        y = _sim(spec)
        df = compare_models({"m1": spec, "m2": spec2}, y)
        assert df.iloc[0]["delta_loglik"] == 0.0

    def test_log_bayes_factor(self) -> None:
        assert log_bayes_factor(10.0, 5.0) == pytest.approx(5.0)
        assert log_bayes_factor(5.0, 10.0) == pytest.approx(-5.0)

    def test_empty_models_raises(self) -> None:
        spec = make_local_level(V=1.0, W_level=0.1)
        y = _sim(spec)
        with pytest.raises(ValueError, match="models"):
            compare_models({}, y)
