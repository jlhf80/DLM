"""Forward Filtering Backward Sampling (Carter & Kohn 1994).

Given a completed Kalman filter pass (FilterResult), draws one exact sample
from the joint smoothing distribution p(theta_{1:T} | y_{1:T}).

Backward sampling equations (W&H §15.2):
    theta_T | y_{1:T}  ~  N(m_T, C_T)
    for t = T-1 downto 0:
        B_t  = C_t G' R_{t+1}^{-1}
        h_t  = m_t + B_t (theta_{t+1} - a_{t+1})
        H_t  = C_t - B_t R_{t+1} B_t'
        theta_t | theta_{t+1}, y_{1:t}  ~  N(h_t, H_t)

B_t is the same backward gain used in the RTS smoother.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from engine.filter import FilterResult, _solve_psd, _symmetrize
from engine.models import DLMSpec


def ffbs(spec: DLMSpec, fr: FilterResult, rng: np.random.Generator) -> NDArray[Any]:
    """Draw one sample from the joint smoothing distribution.

    Parameters
    ----------
    spec : DLMSpec
        Time-invariant DLM specification (provides G).
    fr : FilterResult
        Output of kalman_filter(spec, y) — the forward pass.
    rng : numpy.random.Generator
        Random number generator (e.g. np.random.default_rng(42)).

    Returns
    -------
    theta : (T, d) ndarray
        One draw from p(theta_{1:T} | y_{1:T}).
    """
    T, d = fr.m.shape
    G = spec.G
    theta = np.empty((T, d))

    # Terminal draw from the marginal filter distribution
    theta[-1] = rng.multivariate_normal(fr.m[-1], fr.C[-1])

    # Backward pass
    for t in range(T - 2, -1, -1):
        CG_T = fr.C[t] @ G.T                              # (d, d)
        # B_t = C_t G' R_{t+1}^{-1}  <==>  R_{t+1} B_t' = (C_t G')'
        B_t = _solve_psd(fr.R[t + 1], CG_T.T).T           # (d, d)
        h_t = fr.m[t] + B_t @ (theta[t + 1] - fr.a[t + 1])
        H_t = _symmetrize(fr.C[t] - B_t @ fr.R[t + 1] @ B_t.T)
        theta[t] = rng.multivariate_normal(h_t, H_t)

    return theta
