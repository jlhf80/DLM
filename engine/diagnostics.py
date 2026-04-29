"""ACF/PACF plotting data and residual diagnostics.

Uses statsmodels for ACF, PACF, and Ljung-Box as these are well-tested
reference implementations that would be wasteful to reinvent for a tutorial.
Everything else (standardization, assembly into result dataclasses) is
pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as _acf
from statsmodels.tsa.stattools import pacf as _pacf


@dataclass(frozen=True)
class AcfPacfResult:
    """ACF and PACF values at lags 0..nlags."""

    lags: NDArray[Any]   # (nlags+1,)
    acf: NDArray[Any]    # (nlags+1,)
    pacf: NDArray[Any]   # (nlags+1,)


@dataclass(frozen=True)
class ResidualDiagnostics:
    """One-step-forecast residual diagnostics.

    Attributes
    ----------
    standardized : (T, p) standardized innovations e_t / sqrt(diag(Q_t))
    acf_pacf     : ACF/PACF of univariate standardized residuals
                   (only populated for p=1; for p>1 acf_pacf is of the first
                   component for visualization convenience)
    ljung_box_pvalue : float — portmanteau p-value (null: white noise)
    """

    standardized: NDArray[Any]
    acf_pacf: AcfPacfResult
    ljung_box_pvalue: float


def acf_pacf(y: NDArray[Any], nlags: int = 20) -> AcfPacfResult:
    """Compute ACF and PACF for a univariate series of shape (T, 1) or (T,)."""
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(
                f"acf_pacf requires univariate series; got shape {arr.shape}"
            )
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"acf_pacf requires a 1D or (T,1) array; got {arr.shape}")
    a = _acf(arr, nlags=nlags, fft=False)
    p = _pacf(arr, nlags=nlags, method="yw")
    return AcfPacfResult(
        lags=np.arange(nlags + 1),
        acf=np.asarray(a),
        pacf=np.asarray(p),
    )


def residual_diagnostics(
    e: NDArray[Any],
    Q: NDArray[Any],
    nlags: int = 20,
) -> ResidualDiagnostics:
    """Diagnostics for innovations `e` with predictive covariances `Q`.

    Parameters
    ----------
    e : (T, p) innovations from the Kalman filter
    Q : (T, p, p) predictive covariances
    nlags : number of ACF/PACF lags to compute for the first component
    """
    if e.ndim != 2:
        raise ValueError(f"e must be 2D (T, p), got {e.shape}")
    n, p = e.shape
    if Q.shape != (n, p, p):
        raise ValueError(f"Q shape {Q.shape} must match (T, p, p) = ({n}, {p}, {p})")

    # Standardize each component by its marginal predictive sd.
    sd = np.sqrt(np.diagonal(Q, axis1=1, axis2=2))   # (T, p)
    standardized = e / sd

    # Ljung-Box on the first component (for p=1, this is the only component;
    # for p>1, per-component test is sufficient for the Beginner tier).
    first = standardized[:, 0]
    lb = acorr_ljungbox(first, lags=[nlags], return_df=True)
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    ap = acf_pacf(first[:, None], nlags=nlags)

    return ResidualDiagnostics(
        standardized=standardized,
        acf_pacf=ap,
        ljung_box_pvalue=lb_pvalue,
    )
