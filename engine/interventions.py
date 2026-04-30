"""Interventions and monitoring for Gaussian DLMs.

Interventions (W&H §11.1-11.3):
    level     — shift the state prior mean by a known amount delta at time t
    variance  — inflate the state evolution noise W_t by scale at time t
    outlier   — inflate the observation noise V_t by scale at time t

Monitoring (W&H §11.4-11.5):
    Sequential Bayes factor H_t = p(y_t | M1) / p(y_t | M0)
    where M0 has observation variance V * inflation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from engine.filter import FilterResult, _logdet_psd, _solve_psd, _symmetrize
from engine.models import DLMSpec


@dataclass(frozen=True)
class Intervention:
    """Specification of a single-step model intervention.

    Attributes
    ----------
    kind : str
        One of "level", "variance", "outlier".
    delta : float
        Amount to add to the state prior mean (kind="level").
        Applied to the state component indexed by `component`.
    component : int
        State index to shift (kind="level").  Default 0.
    scale : float
        Multiplicative factor applied to W_t (kind="variance") or
        V_t (kind="outlier").  Default 10.0.
    """

    kind: str
    delta: float = 0.0
    component: int = 0
    scale: float = 10.0


def kalman_filter_interventions(
    spec: DLMSpec,
    y: NDArray[Any],
    interventions: dict[int, Intervention],
) -> FilterResult:
    """Kalman filter with user-specified interventions (W&H §11.1-11.3).

    Parameters
    ----------
    spec : DLMSpec
        Time-invariant baseline DLM.
    y : (T, p) ndarray
        Observations.
    interventions : dict mapping time index -> Intervention
        At each keyed time step the specified modification is applied
        before the Kalman update.

    Returns
    -------
    FilterResult
        Standard filter output.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (T, p), got {y.shape}")
    T, p = y.shape
    if T < 1:
        raise ValueError("y must have at least one observation")
    if p != spec.p:
        raise ValueError(f"y observation dim {p} != spec.p {spec.p}")

    d = spec.d
    F, G, V_base, W_base = spec.F, spec.G, spec.V, spec.W
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
    log2pi = float(np.log(2.0 * np.pi))

    for t in range(T):
        W_t = W_base
        V_t = V_base

        iv = interventions.get(t)
        if iv is not None:
            if iv.kind == "level":
                shift = np.zeros(d)
                shift[iv.component] = iv.delta
                m_prev = m_prev + shift
            elif iv.kind == "variance":
                W_t = W_t * iv.scale
            elif iv.kind == "outlier":
                V_t = V_t * iv.scale
            else:
                raise ValueError(f"Unknown intervention kind: {iv.kind!r}")

        a_t = G @ m_prev
        R_t = _symmetrize(G @ C_prev @ G.T + W_t)
        f_t = F @ a_t
        Q_t = _symmetrize(F @ R_t @ F.T + V_t)
        e_t = y[t] - f_t

        FRt = F @ R_t
        A_t = _solve_psd(Q_t, FRt).T
        m_t = a_t + A_t @ e_t
        Id = np.eye(d)
        IAF = Id - A_t @ F
        C_t = _symmetrize(IAF @ R_t @ IAF.T + A_t @ V_t @ A_t.T)

        logdet_Q = _logdet_psd(Q_t)
        quad = float(e_t @ _solve_psd(Q_t, e_t))
        loglik += -0.5 * (p * log2pi + logdet_Q + quad)

        m[t], C[t] = m_t, C_t
        a[t], R[t] = a_t, R_t
        f[t], Q[t] = f_t, Q_t
        e[t] = e_t
        m_prev, C_prev = m_t, C_t

    return FilterResult(m=m, C=C, a=a, R=R, f=f, Q=Q, e=e, loglik=loglik)


@dataclass(frozen=True)
class MonitorResult:
    """Output of the sequential Bayes factor monitor (W&H §11.4-11.5).

    Attributes
    ----------
    filter_result : FilterResult
        Standard Kalman filter output under the current model M1.
    H : (T,) ndarray
        Per-step Bayes factor H_t = p(y_t | M1) / p(y_t | M0).
    L : (T,) ndarray
        Running product L_t = prod_{s=1}^{t} H_s.
    alert : (T,) bool ndarray
        True where L_t < threshold.
    """

    filter_result: FilterResult
    H: NDArray[Any]
    L: NDArray[Any]
    alert: NDArray[Any]


def _gaussian_log_predictive(
    y_t: NDArray[Any],
    f_t: NDArray[Any],
    Q_t: NDArray[Any],
) -> float:
    """log p(y_t | y_{1:t-1}) = log N(y_t; f_t, Q_t)."""
    p = len(y_t)
    log2pi = float(np.log(2.0 * np.pi))
    e_t = y_t - f_t
    logdet_Q = _logdet_psd(Q_t)
    quad = float(e_t @ _solve_psd(Q_t, e_t))
    return -0.5 * (p * log2pi + logdet_Q + quad)


def kalman_filter_monitor(
    spec: DLMSpec,
    y: NDArray[Any],
    inflation: float = 100.0,
    threshold: float = 0.1,
) -> MonitorResult:
    """Sequential Bayes factor monitoring (W&H §11.4-11.5).

    At each step t, compare the one-step predictive density under the
    current model M1 to a reference model M0 with observation variance
    V * inflation:

        H_t = p(y_t | y_{1:t-1}, M1) / p(y_t | y_{1:t-1}, M0)

    The cumulative product L_t = prod_{s=1}^{t} H_s falls when the data
    repeatedly favours M0, signalling possible model failure.

    Parameters
    ----------
    spec : DLMSpec
        Current model M1.
    y : (T, p) ndarray
        Observations.
    inflation : float
        Factor to inflate V for the reference model M0.
    threshold : float
        Alert when L_t < threshold.

    Returns
    -------
    MonitorResult
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (T, p), got {y.shape}")
    T, p = y.shape
    if T < 1:
        raise ValueError("y must have at least one observation")
    if p != spec.p:
        raise ValueError(f"y observation dim {p} != spec.p {spec.p}")

    V0 = spec.V * inflation

    d = spec.d
    F, G = spec.F, spec.G
    m_prev = spec.m0
    C_prev = spec.C0

    ms = np.empty((T, d))
    Cs = np.empty((T, d, d))
    as_ = np.empty((T, d))
    Rs = np.empty((T, d, d))
    fs = np.empty((T, p))
    Qs = np.empty((T, p, p))
    es = np.empty((T, p))
    loglik = 0.0
    log2pi = float(np.log(2.0 * np.pi))

    H_arr = np.empty(T)
    L_arr = np.empty(T)
    L_t = 1.0

    for t in range(T):
        a_t = G @ m_prev
        R_t = _symmetrize(G @ C_prev @ G.T + spec.W)
        f_t = F @ a_t
        Q_t = _symmetrize(F @ R_t @ F.T + spec.V)
        Q0_t = _symmetrize(F @ R_t @ F.T + V0)

        e_t = y[t] - f_t
        log_p_M1 = _gaussian_log_predictive(y[t], f_t, Q_t)
        log_p_M0 = _gaussian_log_predictive(y[t], f_t, Q0_t)
        H_t = float(np.exp(log_p_M1 - log_p_M0))
        L_t *= H_t
        H_arr[t] = H_t
        L_arr[t] = L_t

        FRt = F @ R_t
        A_t = _solve_psd(Q_t, FRt).T
        m_t = a_t + A_t @ e_t
        Id = np.eye(d)
        IAF = Id - A_t @ F
        C_t = _symmetrize(IAF @ R_t @ IAF.T + A_t @ spec.V @ A_t.T)
        logdet_Q = _logdet_psd(Q_t)
        quad = float(e_t @ _solve_psd(Q_t, e_t))
        loglik += -0.5 * (p * log2pi + logdet_Q + quad)

        ms[t], Cs[t] = m_t, C_t
        as_[t], Rs[t] = a_t, R_t
        fs[t], Qs[t] = f_t, Q_t
        es[t] = e_t
        m_prev, C_prev = m_t, C_t

    fr = FilterResult(m=ms, C=Cs, a=as_, R=Rs, f=fs, Q=Qs, e=es, loglik=loglik)
    alert = L_arr < threshold
    return MonitorResult(filter_result=fr, H=H_arr, L=L_arr, alert=alert)
