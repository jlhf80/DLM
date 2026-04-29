"""Bayesian model comparison utilities for DLMs.

The Kalman filter computes the exact log marginal likelihood
    log p(y_{1:T} | M) = sum_t log p(y_t | y_{1:t-1}, M)
via the Gaussian predictive densities.  Comparing these likelihoods gives
log Bayes factors without approximation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from numpy.typing import NDArray

from engine.filter import kalman_filter
from engine.models import DLMSpec


def compare_models(
    models: dict[str, DLMSpec],
    y: NDArray[Any],
) -> pd.DataFrame:
    """Run the Kalman filter for each model and compare log marginal likelihoods.

    Parameters
    ----------
    models : mapping from model name to DLMSpec.
    y      : (T, p) observation array.

    Returns
    -------
    DataFrame with columns:
        model          — model name (str)
        loglik         — log p(y | model)
        delta_loglik   — loglik - max(loglik)   (0 for the best model)
        log_bf_vs_best — log Bayes factor vs. the best model (0 for the best)
    Rows are sorted by loglik descending (best first).
    """
    if not models:
        raise ValueError("models dict must contain at least one entry")

    rows = []
    for name, spec in models.items():
        fr = kalman_filter(spec, y)
        rows.append({"model": name, "loglik": fr.loglik})

    df = pd.DataFrame(rows).sort_values("loglik", ascending=False).reset_index(drop=True)
    best = float(df["loglik"].iloc[0])
    df["delta_loglik"] = df["loglik"] - best
    df["log_bf_vs_best"] = df["delta_loglik"]   # log BF = log p(y|M) - log p(y|M*)
    return df


def log_bayes_factor(loglik1: float, loglik2: float) -> float:
    """Log Bayes factor: log p(y | M1) - log p(y | M2).

    Positive values favour M1; negative values favour M2.
    """
    return loglik1 - loglik2
