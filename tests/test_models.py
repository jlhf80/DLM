"""Tests for DLMSpec and model builders."""

import numpy as np
import pytest

from engine.models import DLMSpec, make_local_level, make_local_linear_trend


def _valid_spec_kwargs(p: int = 1, d: int = 2) -> dict:
    return dict(
        F=np.ones((p, d)),
        G=np.eye(d),
        V=np.eye(p),
        W=np.eye(d),
        m0=np.zeros(d),
        C0=np.eye(d),
    )


class TestDLMSpecValidation:
    def test_valid_spec_constructs(self):
        spec = DLMSpec(**_valid_spec_kwargs())
        assert spec.F.shape == (1, 2)
        assert spec.G.shape == (2, 2)

    def test_F_wrong_shape_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["F"] = np.ones((2, 2))  # p=2 but V is (1,1)
        with pytest.raises(ValueError, match="V"):
            DLMSpec(**kwargs)

    def test_G_not_square_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["G"] = np.ones((2, 3))
        with pytest.raises(ValueError, match="G"):
            DLMSpec(**kwargs)

    def test_V_not_symmetric_raises(self):
        kwargs = _valid_spec_kwargs(p=2)
        kwargs["V"] = np.array([[1.0, 0.5], [0.1, 1.0]])  # asymmetric
        with pytest.raises(ValueError, match=r"V.*symmetric"):
            DLMSpec(**kwargs)

    def test_W_negative_diagonal_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["W"] = np.diag([-1.0, 1.0])
        with pytest.raises(ValueError, match=r"W.*positive"):
            DLMSpec(**kwargs)

    def test_V_zero_diagonal_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["V"] = np.array([[0.0]])
        with pytest.raises(ValueError, match=r"V.*positive"):
            DLMSpec(**kwargs)

    def test_m0_wrong_dim_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["m0"] = np.zeros(3)  # should be length 2
        with pytest.raises(ValueError, match="m0"):
            DLMSpec(**kwargs)


class TestDLMSpecEqualityAndHash:
    def test_equal_specs_are_equal(self):
        a = DLMSpec(**_valid_spec_kwargs())
        b = DLMSpec(**_valid_spec_kwargs())
        assert a == b
        assert hash(a) == hash(b)

    def test_different_F_not_equal(self):
        a = DLMSpec(**_valid_spec_kwargs())
        kwargs = _valid_spec_kwargs()
        kwargs["F"] = np.full((1, 2), 2.0)
        b = DLMSpec(**kwargs)
        assert a != b
        assert hash(a) != hash(b)

    def test_hash_stable_across_instances(self):
        a = DLMSpec(**_valid_spec_kwargs())
        b = DLMSpec(**_valid_spec_kwargs())
        assert hash(a) == hash(b)

    def test_spec_usable_as_dict_key(self):
        a = DLMSpec(**_valid_spec_kwargs())
        d = {a: "value"}
        b = DLMSpec(**_valid_spec_kwargs())
        assert d[b] == "value"


class TestMakeLocalLevel:
    def test_default_returns_valid_spec(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        assert spec.p == 1 and spec.d == 1
        assert spec.F.shape == (1, 1)
        assert spec.F[0, 0] == 1.0
        assert spec.G.shape == (1, 1)
        assert spec.G[0, 0] == 1.0
        assert spec.V[0, 0] == 0.5
        assert spec.W[0, 0] == 0.1

    def test_accepts_scalar_V(self):
        spec = make_local_level(V=2.0, W_level=0.1)
        assert isinstance(spec.V, np.ndarray)
        assert spec.V.shape == (1, 1)
        assert spec.V[0, 0] == 2.0

    def test_prior_defaults(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        assert spec.m0[0] == 0.0
        assert spec.C0[0, 0] == 1e3  # diffuse default

    def test_custom_prior(self):
        spec = make_local_level(V=0.5, W_level=0.1, m0=10.0, C0=4.0)
        assert spec.m0[0] == 10.0
        assert spec.C0[0, 0] == 4.0

    def test_rejects_zero_W(self):
        with pytest.raises(ValueError, match="W"):
            make_local_level(V=0.5, W_level=0.0)


class TestMakeLocalLinearTrend:
    def test_shapes_and_matrices(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        assert spec.p == 1 and spec.d == 2
        np.testing.assert_array_equal(spec.F, [[1.0, 0.0]])
        np.testing.assert_array_equal(spec.G, [[1.0, 1.0], [0.0, 1.0]])

    def test_W_is_diagonal_of_level_and_slope(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        np.testing.assert_array_equal(spec.W, [[0.05, 0.0], [0.0, 0.01]])

    def test_default_prior_is_diffuse(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        np.testing.assert_array_equal(spec.m0, [0.0, 0.0])
        np.testing.assert_array_equal(spec.C0, 1e3 * np.eye(2))

    def test_custom_prior_shapes(self):
        spec = make_local_linear_trend(
            V=0.5, W_level=0.05, W_slope=0.01,
            m0=np.array([100.0, 1.0]),
            C0=np.diag([10.0, 0.5]),
        )
        np.testing.assert_array_equal(spec.m0, [100.0, 1.0])
        np.testing.assert_array_equal(spec.C0, np.diag([10.0, 0.5]))
