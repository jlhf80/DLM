"""Core DLM data types and model builders.

DLMSpec holds the constant (time-invariant) matrices F, G, V, W of a Gaussian
DLM, plus the prior (m0, C0). All Beginner-tier lessons are time-invariant, so
we do not support time-varying components here — that belongs in a future
revision when the Intermediate tier is added.

Notation follows West & Harrison, Bayesian Forecasting and Dynamic Models (2nd
ed., 1997), chapter 4:

    Observation:  y_t = F_t theta_t + v_t,    v_t ~ N(0, V_t)
    State:        theta_t = G_t theta_{t-1} + w_t,    w_t ~ N(0, W_t)

with y_t in R^p, theta_t in R^d, F_t (p, d), G_t (d, d), V_t (p, p), W_t (d, d).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TOL = 1e-8
_DIAG_MIN = 1e-12


def _check_psd_symmetric(M: np.ndarray, name: str) -> None:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square 2D array, got shape {M.shape}")
    if not np.allclose(M, M.T, atol=_TOL):
        raise ValueError(f"{name} must be symmetric")
    # PSD check via eigenvalues; tolerate tiny negative due to fp noise
    eigs = np.linalg.eigvalsh(M)
    if np.any(eigs < -_TOL):
        raise ValueError(f"{name} must be positive semidefinite (min eigenvalue {eigs.min():.3e})")
    if np.any(np.diag(M) <= _DIAG_MIN):
        raise ValueError(f"{name} must have strictly positive diagonal (> {_DIAG_MIN})")


@dataclass(frozen=True, eq=False)
class DLMSpec:
    """A time-invariant Gaussian DLM specification.

    Attributes
    ----------
    F : (p, d) ndarray — observation matrix
    G : (d, d) ndarray — state transition matrix
    V : (p, p) ndarray — observation noise covariance
    W : (d, d) ndarray — state evolution noise covariance
    m0 : (d,) ndarray — prior state mean
    C0 : (d, d) ndarray — prior state covariance
    """

    F: np.ndarray
    G: np.ndarray
    V: np.ndarray
    W: np.ndarray
    m0: np.ndarray
    C0: np.ndarray

    def __post_init__(self) -> None:
        # Pull shapes from F (source of truth for p and d).
        if self.F.ndim != 2:
            raise ValueError(f"F must be 2D with shape (p, d), got shape {self.F.shape}")
        p, d = self.F.shape

        if self.G.shape != (d, d):
            raise ValueError(f"G must be ({d}, {d}), got {self.G.shape}")
        if self.V.shape != (p, p):
            raise ValueError(f"V must be ({p}, {p}), got {self.V.shape}")
        if self.W.shape != (d, d):
            raise ValueError(f"W must be ({d}, {d}), got {self.W.shape}")
        if self.m0.shape != (d,):
            raise ValueError(f"m0 must be shape ({d},), got {self.m0.shape}")
        if self.C0.shape != (d, d):
            raise ValueError(f"C0 must be ({d}, {d}), got {self.C0.shape}")

        _check_psd_symmetric(self.V, "V")
        _check_psd_symmetric(self.W, "W")
        _check_psd_symmetric(self.C0, "C0")

    @property
    def p(self) -> int:
        """Observation dimension."""
        return int(self.F.shape[0])

    @property
    def d(self) -> int:
        """State dimension."""
        return int(self.F.shape[1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DLMSpec):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, k), getattr(other, k))
            for k in ("F", "G", "V", "W", "m0", "C0")
        )

    def __hash__(self) -> int:
        return hash(
            tuple(np.ascontiguousarray(getattr(self, k)).tobytes()
                  for k in ("F", "G", "V", "W", "m0", "C0"))
        )


def make_local_level(
    V: float,
    W_level: float,
    m0: float = 0.0,
    C0: float = 1e3,
) -> DLMSpec:
    """Local-level (random-walk-plus-noise) DLM.

    State dim d = 1. theta_t is the unobserved level.
        y_t = theta_t + v_t,     v_t ~ N(0, V)
        theta_t = theta_{t-1} + w_t,  w_t ~ N(0, W_level)
    """
    return DLMSpec(
        F=np.array([[1.0]]),
        G=np.array([[1.0]]),
        V=np.array([[float(V)]]),
        W=np.array([[float(W_level)]]),
        m0=np.array([float(m0)]),
        C0=np.array([[float(C0)]]),
    )


def make_local_linear_trend(
    V: float,
    W_level: float,
    W_slope: float,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
) -> DLMSpec:
    """Local-linear-trend DLM (level + slope).

    State theta_t = (mu_t, beta_t); d = 2.
        y_t = mu_t + v_t
        mu_t = mu_{t-1} + beta_{t-1} + w1_t
        beta_t = beta_{t-1} + w2_t
    with w1 ~ N(0, W_level), w2 ~ N(0, W_slope), independent.
    """
    F = np.array([[1.0, 0.0]])
    G = np.array([[1.0, 1.0], [0.0, 1.0]])
    W = np.diag([float(W_level), float(W_slope)])
    if m0 is None:
        m0 = np.zeros(2)
    if C0 is None:
        C0 = 1e3 * np.eye(2)
    return DLMSpec(
        F=F, G=G,
        V=np.array([[float(V)]]),
        W=W,
        m0=np.asarray(m0, dtype=float),
        C0=np.asarray(C0, dtype=float),
    )
