"""Core DLM data types and model builders.

DLMSpec holds the constant (time-invariant) matrices F, G, V, W of a Gaussian
DLM, plus the prior (m0, C0).

For time-varying observation matrices (dynamic regression) use DLMSpecTV;
run through engine.filter.kalman_filter_tv.

Notation follows West & Harrison, Bayesian Forecasting and Dynamic Models (2nd
ed., 1997), chapter 4:

    Observation:  y_t = F_t theta_t + v_t,    v_t ~ N(0, V_t)
    State:        theta_t = G_t theta_{t-1} + w_t,    w_t ~ N(0, W_t)

with y_t in R^p, theta_t in R^d, F_t (p, d), G_t (d, d), V_t (p, p), W_t (d, d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag as _block_diag  # type: ignore[import-untyped]

_TOL = 1e-8
_DIAG_MIN = 1e-12


def _check_psd_symmetric(M: NDArray[Any], name: str) -> None:
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

    F: NDArray[Any]
    G: NDArray[Any]
    V: NDArray[Any]
    W: NDArray[Any]
    m0: NDArray[Any]
    C0: NDArray[Any]

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
    m0: NDArray[Any] | None = None,
    C0: NDArray[Any] | None = None,
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


_SEASONAL_NUGGET = 1e-8


def make_seasonal_factor(
    period: int,
    V: float,
    W_season: float,
    m0: NDArray[Any] | None = None,
    C0: NDArray[Any] | None = None,
) -> DLMSpec:
    """Seasonal DLM in factor form with sum-to-zero constraint.

    For a period-s seasonal pattern, the state has dim d = s - 1 and encodes
    the last s-1 seasonal effects; the current-time seasonal factor is their
    negated sum (enforcing sum-to-zero across a full cycle).

        F = [1, 0, ..., 0]                   # observation reads first slot
        G = companion form:
            [[-1, -1, ..., -1],              # new factor = -sum of previous
             [ 1,  0, ...,  0],
             [ 0,  1, ...,  0],
              ...
             [ 0,  0, ..., 1, 0]]

    Only the top-left entry of W carries the innovation variance W_season; a
    small nugget on the remaining diagonal keeps W strictly PD for validation.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    d = period - 1

    F = np.zeros((1, d))
    F[0, 0] = 1.0

    G = np.zeros((d, d))
    G[0, :] = -1.0
    if d > 1:
        # Identity-like shift: row i (>=1) picks up column i-1
        for i in range(1, d):
            G[i, i - 1] = 1.0

    W = np.diag([float(W_season)] + [_SEASONAL_NUGGET] * (d - 1))

    if m0 is None:
        m0 = np.zeros(d)
    if C0 is None:
        C0 = 1e3 * np.eye(d)

    return DLMSpec(
        F=F, G=G,
        V=np.array([[float(V)]]),
        W=W,
        m0=np.asarray(m0, dtype=float),
        C0=np.asarray(C0, dtype=float),
    )


@dataclass(frozen=True)
class DLMSpecTV:
    """A DLM with a time-varying observation matrix F_t (dynamic regression).

    F_seq : (T, p, d) — one observation matrix per time step.
    G, V, W, m0, C0 : constant system matrices (same semantics as DLMSpec).

    Use engine.filter.kalman_filter_tv to run the filter on this spec.
    """

    F_seq: NDArray[Any]   # (T, p, d)
    G: NDArray[Any]        # (d, d)
    V: NDArray[Any]        # (p, p)
    W: NDArray[Any]        # (d, d)
    m0: NDArray[Any]       # (d,)
    C0: NDArray[Any]       # (d, d)

    def __post_init__(self) -> None:
        if self.F_seq.ndim != 3:
            raise ValueError(
                f"F_seq must be 3D (T, p, d), got shape {self.F_seq.shape}"
            )
        _T, p, d = self.F_seq.shape
        if self.G.shape != (d, d):
            raise ValueError(f"G must be ({d}, {d}), got {self.G.shape}")
        if self.V.shape != (p, p):
            raise ValueError(f"V must be ({p}, {p}), got {self.V.shape}")
        if self.W.shape != (d, d):
            raise ValueError(f"W must be ({d}, {d}), got {self.W.shape}")
        if self.m0.shape != (d,):
            raise ValueError(f"m0 must be ({d},), got {self.m0.shape}")
        if self.C0.shape != (d, d):
            raise ValueError(f"C0 must be ({d}, {d}), got {self.C0.shape}")

    @property
    def T(self) -> int:
        return int(self.F_seq.shape[0])

    @property
    def p(self) -> int:
        return int(self.F_seq.shape[1])

    @property
    def d(self) -> int:
        return int(self.F_seq.shape[2])


def combine(*specs: DLMSpec) -> DLMSpec:
    """Superpose component DLMs into a single block-diagonal DLM.

    Components must share the same observation dimension p and observation
    covariance V. The state vector is the concatenation of each component's
    state; G and W become block-diagonal; F is horizontally stacked.
    """
    if len(specs) < 2:
        raise ValueError(f"combine requires at least two specs, got {len(specs)}")
    p0 = specs[0].p
    V0 = specs[0].V
    for s in specs[1:]:
        if s.p != p0:
            raise ValueError(f"all specs must share observation dimension; got {p0} and {s.p}")
        if not np.allclose(s.V, V0):
            raise ValueError("all specs must share observation noise covariance V")

    F = np.hstack([s.F for s in specs])
    G = _block_diag(*[s.G for s in specs])
    W = _block_diag(*[s.W for s in specs])
    m0 = np.concatenate([s.m0 for s in specs])
    C0 = _block_diag(*[s.C0 for s in specs])
    return DLMSpec(F=F, G=G, V=V0, W=W, m0=m0, C0=C0)


def make_fourier_seasonal(
    period: int,
    n_harmonics: int,
    V: float,
    W_season: float,
) -> DLMSpec:
    """Fourier-form seasonal DLM (W&H ch. 8).

    Represents a seasonal pattern with ``n_harmonics`` sinusoidal components.
    Each harmonic j = 1, ..., n_harmonics contributes a 2x2 rotation block
    (or 1x1 at the Nyquist frequency when period is even and
    j == period // 2).

    Parameters
    ----------
    period      : seasonal period s (e.g. 12 for monthly, 4 for quarterly).
    n_harmonics : number of Fourier harmonics J (1 ≤ J ≤ floor(period / 2)).
    V           : scalar observation noise variance.
    W_season    : innovation variance applied equally to each harmonic's
                  first component; the second (sine) component gets a nugget.

    Returns
    -------
    DLMSpec with d = 2*J (or 2*J-1 if the Nyquist harmonic is included).
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    max_harmonics = period // 2
    if not (1 <= n_harmonics <= max_harmonics):
        raise ValueError(
            f"n_harmonics must be in [1, {max_harmonics}] for period={period}, "
            f"got {n_harmonics}"
        )

    V_mat = np.array([[float(V)]])
    _nugget = 1e-8

    harmonic_specs: list[DLMSpec] = []
    for j in range(1, n_harmonics + 1):
        omega = 2.0 * np.pi * j / period
        nyquist = (period % 2 == 0) and (j == period // 2)

        if nyquist:
            # Scalar harmonic at frequency pi: G_j = [[-1]]
            F_j = np.array([[1.0]])
            G_j = np.array([[-1.0]])
            W_j = np.array([[float(W_season)]])
            m0_j = np.zeros(1)
            C0_j = 1e3 * np.eye(1)
        else:
            c, s = np.cos(omega), np.sin(omega)
            F_j = np.array([[1.0, 0.0]])
            G_j = np.array([[c, s], [-s, c]])
            W_j = np.diag([float(W_season), _nugget])
            m0_j = np.zeros(2)
            C0_j = 1e3 * np.eye(2)

        harmonic_specs.append(
            DLMSpec(F=F_j, G=G_j, V=V_mat, W=W_j, m0=m0_j, C0=C0_j)
        )

    if len(harmonic_specs) == 1:
        return harmonic_specs[0]
    return combine(*harmonic_specs)


def make_arima_dlm(
    ar: list[float],
    ma: list[float],
    sigma2: float,
) -> DLMSpec:
    """DLM in companion form for a causal stationary ARIMA(p, 0, q) process.

    r = max(p, q + 1) where p = len(ar), q = len(ma).
    State dim d = r.

    Matrices (W&H §9.1-9.4):
        F  = [[1, 0, ..., 0]]              (1, r)
        G  = companion matrix              (r, r)
             row 0 = [phi_1, ..., phi_p, 0, ..., 0]
             rows 1..r-1: sub-diagonal of 1s
        V  = 1e-10 * I(1)    (exact obs in theory; nugget for PSD check)
        W  = sigma2 * outer(kappa, kappa) + 1e-10 * I(r)
             kappa = [1, theta_1, ..., theta_q, 0, ..., 0]  (length r)
        m0 = zeros(r)
        C0 = 100 * I(r)

    Parameters
    ----------
    ar : list[float]
        AR coefficients phi_1, ..., phi_p (empty list = pure MA/noise).
    ma : list[float]
        MA coefficients theta_1, ..., theta_q (empty list = pure AR).
    sigma2 : float
        Innovation variance.
    """
    p_ar = len(ar)
    q_ma = len(ma)
    r = max(p_ar, q_ma + 1)

    # Observation matrix
    F = np.zeros((1, r))
    F[0, 0] = 1.0

    # Companion transition matrix
    G = np.zeros((r, r))
    for i, phi in enumerate(ar):
        G[0, i] = phi
    for i in range(1, r):
        G[i, i - 1] = 1.0

    # Moving-average polynomial coefficients (kappa)
    kappa = np.zeros(r)
    kappa[0] = 1.0
    for j, theta in enumerate(ma):
        kappa[j + 1] = theta

    # State evolution noise covariance: rank-1 + nugget
    _NUGGET = 1e-10
    W = sigma2 * np.outer(kappa, kappa) + _NUGGET * np.eye(r)

    # Observation noise: tiny nugget keeps DLMSpec PSD validation happy
    V = _NUGGET * np.eye(1)

    return DLMSpec(
        F=F,
        G=G,
        V=V,
        W=W,
        m0=np.zeros(r),
        C0=100.0 * np.eye(r),
    )
