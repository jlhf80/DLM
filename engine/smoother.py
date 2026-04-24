"""Rauch-Tung-Striebel (RTS) smoother for Gaussian DLMs.

Standard backward pass:
    s_T = m_T,  S_T = C_T
    for t = T-1 down to 1:
        B_t = C_t G' R_{t+1}^{-1}
        s_t = m_t + B_t (s_{t+1} - a_{t+1})
        S_t = C_t + B_t (S_{t+1} - R_{t+1}) B_t'

See West & Harrison chapter 4.8 and Petris chapter 2.4.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.filter import FilterResult, _solve_psd, _symmetrize
from engine.models import DLMSpec


@dataclass(frozen=True)
class SmoothResult:
    """Output of the RTS smoother.

    Attributes
    ----------
    s : (T, d) smoothed state means  E[theta_t | y_{1:T}]
    S : (T, d, d) smoothed state covariances
    """

    s: np.ndarray
    S: np.ndarray


def rts_smoother(spec: DLMSpec, fr: FilterResult) -> SmoothResult:
    T, d = fr.m.shape
    G = spec.G

    s = np.empty((T, d))
    S = np.empty((T, d, d))
    s[-1] = fr.m[-1]
    S[-1] = fr.C[-1]

    for t in range(T - 2, -1, -1):
        # B_t = C_t G' R_{t+1}^{-1}
        C_t = fr.C[t]
        R_next = fr.R[t + 1]
        # Solve R_next X = (C_t G')'  =>  X = R_next^{-1} G C_t
        # then B_t = C_t G' R_next^{-1} = X.T
        CG_T = C_t @ G.T                     # (d, d)
        B_t = _solve_psd(R_next, CG_T.T).T   # (d, d)
        s[t] = fr.m[t] + B_t @ (s[t + 1] - fr.a[t + 1])
        S[t] = _symmetrize(C_t + B_t @ (S[t + 1] - R_next) @ B_t.T)

    return SmoothResult(s=s, S=S)
