"""Forward-simulate observations and latent states from a DLMSpec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.models import DLMSpec


@dataclass(frozen=True)
class SimulatedSeries:
    """One realization of a DLM.

    Attributes
    ----------
    y : (T, p) ndarray — observations
    theta_true : (T, d) ndarray — true latent states
    spec : DLMSpec — the generating specification
    seed : int — RNG seed used
    """

    y: np.ndarray
    theta_true: np.ndarray
    spec: DLMSpec
    seed: int


def simulate(spec: DLMSpec, n: int, seed: int) -> SimulatedSeries:
    """Forward-simulate n steps from a DLMSpec.

    Draws theta_0 ~ N(m0, C0), then for t = 1..n:
        theta_t = G theta_{t-1} + w_t,   w_t ~ N(0, W)
        y_t = F theta_t + v_t,           v_t ~ N(0, V)

    Returns a SimulatedSeries with y of shape (n, p) and theta_true (n, d).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    rng = np.random.default_rng(seed)
    p, d = spec.p, spec.d

    theta = np.empty((n, d))
    y = np.empty((n, p))

    # Draw theta_0 from the prior and advance one step to get theta_1; by
    # convention, t=1 is the first observed time.
    theta_prev = rng.multivariate_normal(spec.m0, spec.C0)
    for t in range(n):
        w_t = rng.multivariate_normal(np.zeros(d), spec.W)
        theta_t = spec.G @ theta_prev + w_t
        v_t = rng.multivariate_normal(np.zeros(p), spec.V)
        y_t = spec.F @ theta_t + v_t
        theta[t] = theta_t
        y[t] = y_t
        theta_prev = theta_t

    return SimulatedSeries(y=y, theta_true=theta, spec=spec, seed=seed)
