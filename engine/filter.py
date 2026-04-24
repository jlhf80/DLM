"""Kalman filter for time-invariant Gaussian DLMs.

This module is a teaching artifact. Each update step is annotated with the
corresponding equation number in West & Harrison, *Bayesian Forecasting and
Dynamic Models* (2nd ed., 1997), chapter 4. Reading this file alongside that
chapter should make the correspondence obvious.

Notation (West-Harrison):
    m_t, C_t       filtered  posterior     theta_t | y_{1:t}   ~ N(m_t, C_t)
    a_t, R_t       predictive state prior  theta_t | y_{1:t-1} ~ N(a_t, R_t)
    f_t, Q_t       predictive obs          y_t    | y_{1:t-1} ~ N(f_t, Q_t)
    e_t            innovation (forecast error)  e_t = y_t - f_t
    A_t            Kalman gain             A_t = R_t F' Q_t^{-1}

For numerical stability we use the Joseph form for the covariance update.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve  # type: ignore[import-untyped]

from engine.models import DLMSpec


@dataclass(frozen=True)
class FilterResult:
    """Output of the Kalman filter.

    Attributes
    ----------
    m : (T, d)   filtered state means          E[theta_t | y_{1:t}]
    C : (T, d, d) filtered state covariances
    a : (T, d)   one-step predictive state means E[theta_t | y_{1:t-1}]
    R : (T, d, d) one-step predictive state covariances
    f : (T, p)   one-step predictive obs means   E[y_t | y_{1:t-1}]
    Q : (T, p, p) one-step predictive obs covariances
    e : (T, p)   innovations y_t - f_t
    loglik : total log-marginal-likelihood sum_t log p(y_t | y_{1:t-1})
    """

    m: np.ndarray
    C: np.ndarray
    a: np.ndarray
    R: np.ndarray
    f: np.ndarray
    Q: np.ndarray
    e: np.ndarray
    loglik: float


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (M + M.T))


def _solve_psd(Q: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve Q X = B for X using Cholesky; fall back to pinv if singular."""
    try:
        c, low = cho_factor(Q, lower=True)
        return np.asarray(cho_solve((c, low), B))
    except LinAlgError:
        warnings.warn("Q_t singular in Kalman filter; falling back to pinv", RuntimeWarning, stacklevel=2)
        return np.asarray(np.linalg.pinv(Q) @ B)


def _logdet_psd(Q: np.ndarray) -> float:
    """log|Q| for symmetric PSD Q via Cholesky (or pinv fallback)."""
    try:
        c, _ = cho_factor(Q, lower=True)
        return 2.0 * float(np.sum(np.log(np.diag(c))))
    except LinAlgError:
        _sign, logabsdet = np.linalg.slogdet(Q)
        return float(logabsdet)


def kalman_filter(spec: DLMSpec, y: np.ndarray) -> FilterResult:
    """Run the Kalman filter on observations `y` using `spec`.

    `y` has shape (T, p). Returns a FilterResult.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (T, p), got shape {y.shape}")
    T, p = y.shape
    if T < 1:
        raise ValueError("y must have at least one observation")
    if p != spec.p:
        raise ValueError(f"y observation dimension {p} does not match spec.p {spec.p}")

    d = spec.d
    F, G, V, W = spec.F, spec.G, spec.V, spec.W
    m_prev = spec.m0
    C_prev = spec.C0

    m = np.empty((T, d))
    C = np.empty((T, d, d))
    a = np.empty((T, d))
    R = np.empty((T, d, d))
    f = np.empty((T, p))
    Q = np.empty((T, p, p))
    e = np.empty((T, p))
    loglik = 0.0

    # Constant used by the log-likelihood: -p/2 * log(2 pi)
    log2pi = float(np.log(2.0 * np.pi))

    for t in range(T):
        # --- Prior for theta_t given y_{1:t-1}   (West-Harrison eq 4.3) ----
        a_t = G @ m_prev                                       # E[theta_t | y_{1:t-1}]
        R_t = _symmetrize(G @ C_prev @ G.T + W)                # Cov[theta_t | y_{1:t-1}]

        # --- One-step-ahead obs forecast         (West-Harrison eq 4.4) ----
        f_t = F @ a_t                                          # E[y_t | y_{1:t-1}]
        Q_t = _symmetrize(F @ R_t @ F.T + V)                   # Cov[y_t | y_{1:t-1}]

        # --- Innovation ----
        e_t = y[t] - f_t                                        # forecast error

        # --- Kalman gain and posterior update    (West-Harrison eq 4.5-6) ---
        #   A_t = R_t F' Q_t^{-1}
        # Use _solve_psd to compute Q_t^{-1} (F R_t)'  stably.
        FRt = F @ R_t                                          # (p, d)
        A_t = _solve_psd(Q_t, FRt).T                           # (d, p)

        m_t = a_t + A_t @ e_t                                  # posterior mean
        # Joseph form for PSD stability:  C_t = (Id - A F) R (Id - A F)' + A V A'
        Id = np.eye(d)
        IAF = Id - A_t @ F
        C_t = _symmetrize(IAF @ R_t @ IAF.T + A_t @ V @ A_t.T)

        # --- Accumulate log-likelihood  log N(y_t; f_t, Q_t) ---
        logdet_Q = _logdet_psd(Q_t)
        quad = float(e_t @ _solve_psd(Q_t, e_t))
        loglik += -0.5 * (p * log2pi + logdet_Q + quad)

        m[t], C[t] = m_t, C_t
        a[t], R[t] = a_t, R_t
        f[t], Q[t] = f_t, Q_t
        e[t] = e_t

        m_prev, C_prev = m_t, C_t

    return FilterResult(m=m, C=C, a=a, R=R, f=f, Q=Q, e=e, loglik=loglik)
