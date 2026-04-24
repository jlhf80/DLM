"""Forward-propagate filtered posterior to produce h-step-ahead forecasts.

For k = 1..h, the predictive distribution is
    theta_{T+k} | y_{1:T} ~ N(a^{(k)}, R^{(k)})
    y_{T+k}     | y_{1:T} ~ N(F a^{(k)}, F R^{(k)} F' + V)
with recursion
    a^{(1)} = G m_T,   R^{(1)} = G C_T G' + W
    a^{(k)} = G a^{(k-1)},  R^{(k)} = G R^{(k-1)} G' + W

Marginal credible bands at level 1 - alpha (default 95%) are
    [f_k - z * sqrt(diag(Q_k)),  f_k + z * sqrt(diag(Q_k))]
where z = Phi^{-1}(1 - alpha/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm  # type: ignore[import-untyped]

from engine.filter import FilterResult, _symmetrize
from engine.models import DLMSpec


@dataclass(frozen=True)
class Forecast:
    """h-step-ahead forecast, marginal per observation component.

    Attributes
    ----------
    horizon : int
    means   : (h, p) predictive means
    lower   : (h, p) predictive 1-alpha/2 lower bound
    upper   : (h, p) predictive 1-alpha/2 upper bound
    """

    horizon: int
    means: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def forecast_horizon(
    spec: DLMSpec,
    fr: FilterResult,
    h: int,
    alpha: float = 0.05,
) -> Forecast:
    """Compute h-step-ahead forecast with credible bands.

    Parameters
    ----------
    spec : DLMSpec
        Model specification.
    fr : FilterResult
        Output from Kalman filter.
    h : int
        Forecast horizon (number of steps ahead).
    alpha : float, default 0.05
        Credible band significance level; bands are at level 1 - alpha.

    Returns
    -------
    Forecast
        Forecast with means and credible bands.
    """
    if h < 0:
        raise ValueError(f"horizon h must be >= 0, got {h}")
    p = spec.p

    if h == 0:
        zeros = np.empty((0, p))
        return Forecast(horizon=0, means=zeros, lower=zeros, upper=zeros)

    F, G, V, W = spec.F, spec.G, spec.V, spec.W
    a = fr.m[-1].copy()
    R = fr.C[-1].copy()

    means = np.empty((h, p))
    lower = np.empty((h, p))
    upper = np.empty((h, p))
    z = float(norm.ppf(1 - alpha / 2))

    for k in range(h):
        a = G @ a
        R = _symmetrize(G @ R @ G.T + W)
        f_k = F @ a                             # predictive mean of y_{T+k+1}
        Q_k = _symmetrize(F @ R @ F.T + V)      # predictive covariance
        sd = np.sqrt(np.diag(Q_k))
        means[k] = f_k
        lower[k] = f_k - z * sd
        upper[k] = f_k + z * sd

    return Forecast(horizon=h, means=means, lower=lower, upper=upper)
