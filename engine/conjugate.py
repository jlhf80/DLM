"""Kalman filter with conjugate inverse-gamma prior on the observation variance.

For a scalar observation (p = 1) DLM, V is treated as unknown with prior
V ~ IG(n0/2, d0/2).  The conjugate Bayesian update at each step yields a
Normal-Inverse-Gamma posterior:

    (theta_t, V) | y_{1:t} ~ NIG(m_t, C_t * V, n_t/2, d_t/2)

where the marginal for theta_t is Student-t with n_t degrees of freedom.

Equations follow West & Harrison, *Bayesian Forecasting and Dynamic Models*
(2nd ed., 1997), chapter 10, §10.2 (scalar observation, known G, W).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln  # type: ignore[import-untyped]

from engine.filter import _symmetrize
from engine.models import DLMSpec


@dataclass(frozen=True)
class ConjugateFilterResult:
    """Output of the conjugate Kalman filter (unknown V, p = 1).

    Attributes
    ----------
    m : (T, d)    filtered state means  E[theta_t | y_{1:t}]
    C : (T, d, d) filtered state covariance *scale matrices* (not V·C; these
                  absorb the current V̂_t = d_t / n_t estimate)
    n : (T,)      IG shape parameter sequence  (n_0, n_1, ..., n_{T-1})
    d : (T,)      IG scale parameter sequence
    f : (T, 1)    one-step predictive obs means
    Q : (T, 1, 1) one-step predictive obs scale matrices (not covariances)
    e : (T, 1)    innovations
    v_hat : (T,)  running V estimate  d_t / n_t
    loglik : float  summed log predictive density (Student-t marginals)
    """

    m: NDArray[Any]
    C: NDArray[Any]
    n: NDArray[Any]
    d: NDArray[Any]
    f: NDArray[Any]
    Q: NDArray[Any]
    e: NDArray[Any]
    v_hat: NDArray[Any]
    loglik: float


def kalman_filter_conjugate(
    spec: DLMSpec,
    y: NDArray[Any],
    n0: float,
    d0: float,
) -> ConjugateFilterResult:
    """Conjugate Kalman filter with unknown observation variance V.

    Restricted to scalar observations (p = 1). `spec.V` is used only for its
    structural shape; the actual V is estimated online.

    Parameters
    ----------
    spec : DLMSpec with p == 1.
    y    : (T, 1) or (T,) observations.
    n0   : prior IG shape parameter (degrees of freedom ≥ 2 for finite prior
           variance; use small values like 2 for a vague prior).
    d0   : prior IG scale parameter (d0/n0 is the prior point estimate of V).
    """
    if spec.p != 1:
        raise ValueError(
            f"kalman_filter_conjugate requires p=1, got p={spec.p}"
        )
    if n0 <= 0 or d0 <= 0:
        raise ValueError(f"n0 and d0 must be positive, got n0={n0}, d0={d0}")

    y2d = np.atleast_2d(y)
    if y2d.shape[0] == 1 and y2d.shape[1] != 1:
        y2d = y2d.T
    if y2d.ndim != 2 or y2d.shape[1] != 1:
        raise ValueError(f"y must be (T,) or (T, 1), got shape {y.shape}")
    T = y2d.shape[0]

    d_state = spec.d
    F, G, W = spec.F, spec.G, spec.W
    m_prev = spec.m0.copy()
    C_prev = spec.C0.copy()
    n_prev = float(n0)
    d_prev = float(d0)

    m_arr = np.empty((T, d_state))
    C_arr = np.empty((T, d_state, d_state))
    n_arr = np.empty(T)
    d_arr = np.empty(T)
    f_arr = np.empty((T, 1))
    Q_arr = np.empty((T, 1, 1))
    e_arr = np.empty((T, 1))
    v_hat_arr = np.empty(T)
    loglik = 0.0

    for t in range(T):
        # --- Predictive state prior ----------------------------------------
        a_t = G @ m_prev
        R_t = _symmetrize(G @ C_prev @ G.T + W)

        # --- Predictive obs (scale, not covariance) ------------------------
        # Q_t* = F R_t F' + 1   (scaled; true Q_t = V * Q_t*)
        q_scalar = float((F @ R_t @ F.T)[0, 0]) + 1.0   # scalar (p=1)
        f_t = F @ a_t                             # (1,)

        # --- Innovation ---------------------------------------------------
        e_t = y2d[t] - f_t   # (1,)

        # --- Conjugate update ---------------------------------------------
        n_t = n_prev + 1.0
        # d_t = d_{t-1} + e_t^2 / q_scalar
        d_t = d_prev + float(e_t[0] ** 2) / q_scalar

        # Kalman gain using current V estimate  V̂ = d_prev / n_prev
        # (use *prior* V estimate for the gain, as in W&H eq 10.13)
        v_est_prior = d_prev / n_prev
        Q_t_val = v_est_prior * q_scalar          # scalar predictive variance
        A_t = (R_t @ F.T) / q_scalar              # (d, 1)

        m_t = a_t + A_t.squeeze() * float(e_t[0])

        # Posterior covariance scale: C_t = (d_t / n_t) * (R_t - A_t q A_t')
        # Factor out V̂_t = d_t/n_t; store the shape matrix only.
        v_est_post = d_t / n_t
        C_t = _symmetrize(R_t - A_t * q_scalar @ A_t.T)  # shape matrix (no V)

        # --- Log predictive density: Student-t(n_prev, f_t, V_hat * q_scalar)
        nu = n_prev
        sigma2 = v_est_prior * q_scalar
        loglik += float(
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * sigma2)
            - ((nu + 1) / 2) * np.log(1 + float(e_t[0] ** 2) / (nu * sigma2))
        )

        m_arr[t] = m_t
        C_arr[t] = C_t
        n_arr[t] = n_t
        d_arr[t] = d_t
        f_arr[t] = f_t
        Q_arr[t, 0, 0] = Q_t_val
        e_arr[t] = e_t
        v_hat_arr[t] = v_est_post

        m_prev, C_prev = m_t, C_t
        n_prev, d_prev = n_t, d_t

    return ConjugateFilterResult(
        m=m_arr, C=C_arr, n=n_arr, d=d_arr,
        f=f_arr, Q=Q_arr, e=e_arr,
        v_hat=v_hat_arr, loglik=loglik,
    )
