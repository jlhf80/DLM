# Advanced DLM Notebook Series — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four new engine functions, unit tests, and six advanced Jupyter notebooks covering ARIMA↔DLM equivalence, FFBS/MCMC, interventions, monitoring, and multivariate/missing-data DLMs.

**Architecture:** Pure-NumPy engine extensions (no PyMC dependency) with TDD; notebooks import from `engine/` and use PyMC only in notebook 07. All work on branch `feat/advanced-notebooks`; PR to main at the end.

**Tech Stack:** NumPy/SciPy, nbformat 4.5, pytest/nbmake, PyMC ≥ 5.0 + arviz (notebooks only), statsmodels (ARIMA log-lik comparison test).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `engine/ffbs.py` | `ffbs(spec, fr, rng)` — joint smoothing draw |
| Modify | `engine/filter.py` | Add `kalman_filter_missing(spec, y)` |
| Modify | `engine/models.py` | Add `make_arima_dlm()`, `make_multivariate_local_level()` |
| Create | `engine/interventions.py` | `Intervention`, `kalman_filter_interventions()`, `MonitorResult`, `kalman_filter_monitor()` |
| Create | `tests/test_advanced_engine.py` | Unit tests for all new engine functions |
| Modify | `pyproject.toml` | Add `pymc>=5.0`, `arviz>=0.17` to `[dev]` |
| Create | `notebooks/advanced/00_advanced_setup.ipynb` | Environment check + engine smoke test |
| Create | `notebooks/advanced/06_arima_dlm_equivalence.ipynb` | ARIMA companion-form DLM |
| Create | `notebooks/advanced/07_ffbs_and_mcmc.ipynb` | FFBS + Gibbs + PyMC |
| Create | `notebooks/advanced/08_interventions_outliers.ipynb` | Level shift, variance inflation, outlier model |
| Create | `notebooks/advanced/09_monitoring_structural_breaks.ipynb` | Sequential Bayes factor monitoring |
| Create | `notebooks/advanced/10_multivariate_missing_data.ipynb` | Missing-data filter + multivariate local level |
| Modify | `.github/workflows/ci.yml` | Add `notebooks/advanced/` to nbmake step |

---

## Task 1: FFBS backward sampler (`engine/ffbs.py`)

**Files:**
- Create: `tests/test_advanced_engine.py`
- Create: `engine/ffbs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_advanced_engine.py`:

```python
"""Unit tests for advanced engine extensions."""
from __future__ import annotations

import numpy as np
import pytest

from engine.filter import kalman_filter
from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import simulate
from engine.smoother import rts_smoother


# ---------------------------------------------------------------------------
# FFBS
# ---------------------------------------------------------------------------


def test_ffbs_shape() -> None:
    from engine.ffbs import ffbs

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=30, seed=1).y
    fr = kalman_filter(spec, y)
    rng = np.random.default_rng(99)
    theta = ffbs(spec, fr, rng)
    assert theta.shape == (30, 1)


def test_ffbs_mean_matches_smoother() -> None:
    """Monte Carlo mean of FFBS draws should match RTS smoother mean."""
    from engine.ffbs import ffbs

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=50, seed=0).y
    fr = kalman_filter(spec, y)
    sr = rts_smoother(spec, fr)
    rng = np.random.default_rng(42)
    draws = np.stack([ffbs(spec, fr, rng) for _ in range(3000)])  # (3000, T, d)
    np.testing.assert_allclose(draws.mean(axis=0)[:, 0], sr.s[:, 0], atol=0.08)


def test_ffbs_terminal_matches_filter() -> None:
    """Last draw theta[-1] is sampled from the terminal filter distribution N(m_T, C_T)."""
    from engine.ffbs import ffbs

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=20, seed=7).y
    fr = kalman_filter(spec, y)
    rng = np.random.default_rng(11)
    draws = np.stack([ffbs(spec, fr, rng) for _ in range(2000)])
    # Monte Carlo mean of last step ≈ m_T
    np.testing.assert_allclose(draws[:, -1, 0].mean(), fr.m[-1, 0], atol=0.1)


def test_ffbs_llt() -> None:
    """FFBS works for a 2-d state (local linear trend)."""
    from engine.ffbs import ffbs

    spec = make_local_linear_trend(V=1.0, W_level=0.5, W_slope=0.1)
    y = simulate(spec, n=40, seed=3).y
    fr = kalman_filter(spec, y)
    rng = np.random.default_rng(55)
    theta = ffbs(spec, fr, rng)
    assert theta.shape == (40, 2)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_advanced_engine.py::test_ffbs_shape tests/test_advanced_engine.py::test_ffbs_mean_matches_smoother tests/test_advanced_engine.py::test_ffbs_terminal_matches_filter tests/test_advanced_engine.py::test_ffbs_llt -v
```

Expected: `ModuleNotFoundError: No module named 'engine.ffbs'`

- [ ] **Step 3: Implement `engine/ffbs.py`**

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_advanced_engine.py::test_ffbs_shape tests/test_advanced_engine.py::test_ffbs_mean_matches_smoother tests/test_advanced_engine.py::test_ffbs_terminal_matches_filter tests/test_advanced_engine.py::test_ffbs_llt -v
```

Expected: 4 PASSED

- [ ] **Step 5: Lint check**

```bash
ruff check engine/ffbs.py && mypy engine/ffbs.py
```

Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add engine/ffbs.py tests/test_advanced_engine.py
git commit -m "feat(engine): FFBS backward sampler (Carter & Kohn 1994)"
```

---

## Task 2: Missing-data Kalman filter (`engine/filter.py`)

**Files:**
- Modify: `tests/test_advanced_engine.py` (add tests)
- Modify: `engine/filter.py` (add function)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_advanced_engine.py`:

```python
# ---------------------------------------------------------------------------
# Missing-data filter
# ---------------------------------------------------------------------------


def test_missing_filter_skips_update() -> None:
    """When y[t] is NaN the posterior equals the prior."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=20, seed=2).y.copy()
    y[5] = np.nan
    fr = kalman_filter_missing(spec, y)
    np.testing.assert_allclose(fr.m[5], fr.a[5])
    np.testing.assert_allclose(fr.C[5], fr.R[5])
    assert np.isnan(fr.e[5]).all()


def test_missing_filter_loglik_finite() -> None:
    """Missing obs contribute 0 to the log-likelihood; result is finite."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=10, seed=3).y.copy()
    fr_full = kalman_filter(spec, y)
    y[4] = np.nan
    fr_miss = kalman_filter_missing(spec, y)
    assert np.isfinite(fr_miss.loglik)
    assert fr_miss.loglik < fr_full.loglik  # one obs dropped


def test_missing_filter_non_missing_steps_unchanged() -> None:
    """Non-NaN steps match standard kalman_filter."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=15, seed=6).y.copy()
    y[7] = np.nan
    fr_miss = kalman_filter_missing(spec, y)
    # Re-run standard filter up to t=6 (before missing)
    fr_std = kalman_filter(spec, y[:7])
    np.testing.assert_allclose(fr_miss.m[:7], fr_std.m, atol=1e-10)
    np.testing.assert_allclose(fr_miss.C[:7], fr_std.C, atol=1e-10)


def test_missing_filter_all_missing() -> None:
    """All-NaN series: every step is posterior=prior; loglik=0."""
    from engine.filter import kalman_filter_missing

    spec = make_local_level(V=1.0, W_level=0.5)
    y = np.full((10, 1), np.nan)
    fr = kalman_filter_missing(spec, y)
    assert fr.loglik == 0.0
    np.testing.assert_allclose(fr.m, fr.a)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_advanced_engine.py -k "missing" -v
```

Expected: `ImportError: cannot import name 'kalman_filter_missing'`

- [ ] **Step 3: Implement `kalman_filter_missing` in `engine/filter.py`**

Add at the end of `engine/filter.py` (after `kalman_filter_tv`):

```python
def kalman_filter_missing(spec: DLMSpec, y: NDArray[Any]) -> FilterResult:
    """Kalman filter handling NaN observations (W&H §16.1; Petris §2.7).

    For each time step t where `np.isnan(y[t]).any()` is True, the update
    step is skipped entirely: the posterior equals the prior
    (m_t = a_t, C_t = R_t), and no log-likelihood contribution is added.
    Missing time points have e[t] = NaN.
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

    log2pi = float(np.log(2.0 * np.pi))

    for t in range(T):
        # Prior for theta_t
        a_t = G @ m_prev
        R_t = _symmetrize(G @ C_prev @ G.T + W)

        # One-step-ahead forecast
        f_t = F @ a_t
        Q_t = _symmetrize(F @ R_t @ F.T + V)

        a[t], R[t] = a_t, R_t
        f[t], Q[t] = f_t, Q_t

        if np.isnan(y[t]).any():
            # Skip update: posterior = prior
            m_t, C_t = a_t, R_t
            e[t] = np.full(p, np.nan)
        else:
            e_t = y[t] - f_t
            FRt = F @ R_t
            A_t = _solve_psd(Q_t, FRt).T
            m_t = a_t + A_t @ e_t
            Id = np.eye(d)
            IAF = Id - A_t @ F
            C_t = _symmetrize(IAF @ R_t @ IAF.T + A_t @ V @ A_t.T)
            logdet_Q = _logdet_psd(Q_t)
            quad = float(e_t @ _solve_psd(Q_t, e_t))
            loglik += -0.5 * (p * log2pi + logdet_Q + quad)
            e[t] = e_t

        m[t], C[t] = m_t, C_t
        m_prev, C_prev = m_t, C_t

    return FilterResult(m=m, C=C, a=a, R=R, f=f, Q=Q, e=e, loglik=loglik)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_advanced_engine.py -k "missing" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Lint check**

```bash
ruff check engine/filter.py && mypy engine/filter.py
```

Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add engine/filter.py tests/test_advanced_engine.py
git commit -m "feat(engine): kalman_filter_missing — skip update for NaN observations"
```

---

## Task 3: ARIMA DLM builder (`engine/models.py`)

**Files:**
- Modify: `tests/test_advanced_engine.py` (add tests)
- Modify: `engine/models.py` (add function)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_advanced_engine.py`:

```python
# ---------------------------------------------------------------------------
# ARIMA DLM builder
# ---------------------------------------------------------------------------


def test_arima_dlm_ar1_shape() -> None:
    from engine.models import make_arima_dlm

    spec = make_arima_dlm(ar=[0.7], ma=[], sigma2=1.0)
    assert spec.d == 1
    assert spec.F.shape == (1, 1)
    assert spec.G.shape == (1, 1)
    assert spec.V.shape == (1, 1)
    assert spec.W.shape == (1, 1)


def test_arima_dlm_arma_shape() -> None:
    """ARMA(2,1): r = max(2, 1+1) = 2."""
    from engine.models import make_arima_dlm

    spec = make_arima_dlm(ar=[0.5, -0.3], ma=[0.2], sigma2=1.0)
    assert spec.d == 2
    assert spec.F.shape == (1, 2)
    assert spec.G.shape == (2, 2)


def test_arima_dlm_ar1_companion_form() -> None:
    """AR(1): G should be [[phi]], W[0,0] should be sigma2 (plus nugget)."""
    from engine.models import make_arima_dlm

    phi = 0.7
    spec = make_arima_dlm(ar=[phi], ma=[], sigma2=1.0)
    np.testing.assert_allclose(spec.G[0, 0], phi)
    np.testing.assert_allclose(spec.W[0, 0], 1.0 + 1e-10, atol=1e-9)


def test_arima_dlm_ma1_companion_form() -> None:
    """MA(1): r=2, G[0,:] = [0,0], W reflects MA structure."""
    from engine.models import make_arima_dlm

    theta = 0.4
    spec = make_arima_dlm(ar=[], ma=[theta], sigma2=2.0)
    assert spec.d == 2  # r = max(0, 1+1) = 2
    # kappa = [1, theta, 0, ...] -> W = sigma2 * outer(kappa, kappa) + nugget*I
    np.testing.assert_allclose(spec.W[0, 0], 2.0 * 1.0 + 1e-10, atol=1e-9)
    np.testing.assert_allclose(spec.W[0, 1], 2.0 * theta, atol=1e-9)
    np.testing.assert_allclose(spec.W[1, 0], 2.0 * theta, atol=1e-9)


def test_arima_dlm_loglik_matches_statsmodels_ar1() -> None:
    """AR(1) log-lik from DLM filter should match statsmodels ARIMA."""
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-untyped]

    from engine.models import make_arima_dlm

    rng = np.random.default_rng(7)
    phi, sigma2 = 0.7, 1.5
    T = 150
    y_1d = np.zeros(T)
    eps = rng.normal(scale=np.sqrt(sigma2), size=T)
    for t in range(1, T):
        y_1d[t] = phi * y_1d[t - 1] + eps[t]
    spec = make_arima_dlm(ar=[phi], ma=[], sigma2=sigma2)
    fr = kalman_filter(spec, y_1d[:, None])
    res = ARIMA(y_1d, order=(1, 0, 0)).fit(disp=False)
    # Allow generous tolerance: initialisation differs slightly
    np.testing.assert_allclose(fr.loglik, res.llf, atol=5.0)


def test_arima_dlm_filter_runs() -> None:
    """ARMA(2,1) DLM filter returns finite log-lik."""
    from engine.models import make_arima_dlm

    rng = np.random.default_rng(8)
    spec = make_arima_dlm(ar=[0.5, -0.2], ma=[0.3], sigma2=1.0)
    y = rng.standard_normal((80, 1))
    fr = kalman_filter(spec, y)
    assert np.isfinite(fr.loglik)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_advanced_engine.py -k "arima" -v
```

Expected: `ImportError: cannot import name 'make_arima_dlm'`

- [ ] **Step 3: Implement `make_arima_dlm` in `engine/models.py`**

Add after `make_fourier_seasonal` in `engine/models.py`:

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_advanced_engine.py -k "arima" -v
```

Expected: 6 PASSED

- [ ] **Step 5: Lint check**

```bash
ruff check engine/models.py && mypy engine/models.py
```

Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add engine/models.py tests/test_advanced_engine.py
git commit -m "feat(engine): make_arima_dlm — ARIMA companion-form DLM builder"
```

---

## Task 4: Multivariate local level builder (`engine/models.py`)

**Files:**
- Modify: `tests/test_advanced_engine.py` (add tests)
- Modify: `engine/models.py` (add function)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_advanced_engine.py`:

```python
# ---------------------------------------------------------------------------
# Multivariate local level
# ---------------------------------------------------------------------------


def test_multivariate_local_level_shape() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=3, V=[1.0, 2.0, 3.0], W_level=0.5)
    assert spec.F.shape == (3, 1)
    assert spec.G.shape == (1, 1)
    assert spec.V.shape == (3, 3)
    assert spec.W.shape == (1, 1)
    assert spec.d == 1
    assert spec.p == 3


def test_multivariate_local_level_scalar_v() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=2, V=1.5, W_level=0.5)
    np.testing.assert_allclose(np.diag(spec.V), [1.5, 1.5])


def test_multivariate_local_level_list_v() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=2, V=[1.0, 3.0], W_level=0.5)
    np.testing.assert_allclose(np.diag(spec.V), [1.0, 3.0])
    # Off-diagonals zero
    assert spec.V[0, 1] == 0.0


def test_multivariate_local_level_filter_runs() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=2, V=1.0, W_level=0.5)
    rng = np.random.default_rng(10)
    y = rng.normal(size=(50, 2))
    fr = kalman_filter(spec, y)
    assert fr.m.shape == (50, 1)
    assert np.isfinite(fr.loglik)


def test_multivariate_local_level_f_is_ones() -> None:
    from engine.models import make_multivariate_local_level

    spec = make_multivariate_local_level(p=4, V=1.0, W_level=0.5)
    np.testing.assert_array_equal(spec.F, np.ones((4, 1)))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_advanced_engine.py -k "multivariate" -v
```

Expected: `ImportError: cannot import name 'make_multivariate_local_level'`

- [ ] **Step 3: Implement `make_multivariate_local_level` in `engine/models.py`**

Add after `make_arima_dlm` in `engine/models.py`:

```python
def make_multivariate_local_level(
    p: int,
    V: NDArray[Any] | list[float] | float,
    W_level: float,
) -> DLMSpec:
    """Multivariate local level: p series sharing a common scalar level.

    Observation equation (p-dimensional):
        y_t = F theta_t + v_t,   F = ones((p, 1)),   v_t ~ N(0, V_obs)
    State equation (scalar random walk):
        theta_t = theta_{t-1} + w_t,   w_t ~ N(0, W_level)

    Parameters
    ----------
    p : int
        Number of observed series.
    V : float or list[float] or (p, p) ndarray
        Observation noise.  A scalar broadcasts to V * eye(p).
        A length-p list becomes diag(V).
    W_level : float
        Variance of the scalar state random walk.
    """
    if isinstance(V, (int, float)):
        V_mat = float(V) * np.eye(p)
    elif isinstance(V, list):
        V_mat = np.diag(np.array(V, dtype=float))
    else:
        V_mat = np.asarray(V, dtype=float)
        if V_mat.shape != (p, p):
            raise ValueError(f"V array must be ({p}, {p}), got {V_mat.shape}")

    return DLMSpec(
        F=np.ones((p, 1)),
        G=np.array([[1.0]]),
        V=V_mat,
        W=np.array([[float(W_level)]]),
        m0=np.zeros(1),
        C0=100.0 * np.eye(1),
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_advanced_engine.py -k "multivariate" -v
```

Expected: 5 PASSED

- [ ] **Step 5: Lint check**

```bash
ruff check engine/models.py && mypy engine/models.py
```

Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add engine/models.py tests/test_advanced_engine.py
git commit -m "feat(engine): make_multivariate_local_level — shared scalar level model"
```

---

## Task 5: Interventions filter (`engine/interventions.py`)

**Files:**
- Modify: `tests/test_advanced_engine.py` (add tests)
- Create: `engine/interventions.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_advanced_engine.py`:

```python
# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


def test_intervention_level_shift_changes_posterior() -> None:
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=4).y
    iv = {15: Intervention(kind="level", delta=10.0, component=0)}
    fr_iv = kalman_filter_interventions(spec, y, iv)
    fr_plain = kalman_filter(spec, y)
    # Posterior mean at t=15 shifts up by ~10
    assert fr_iv.m[15, 0] > fr_plain.m[15, 0] + 5.0


def test_intervention_variance_inflation() -> None:
    """Variance-inflation intervention increases R at t+1."""
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=5).y
    iv = {10: Intervention(kind="variance", scale=100.0)}
    fr_iv = kalman_filter_interventions(spec, y, iv)
    fr_plain = kalman_filter(spec, y)
    # R at t=11 should be larger with variance inflation
    assert fr_iv.R[11, 0, 0] > fr_plain.R[11, 0, 0]


def test_intervention_outlier_downweights_obs() -> None:
    """Outlier intervention down-weights a spike observation."""
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=6).y.copy()
    y[10, 0] = 100.0  # spike
    iv = {10: Intervention(kind="outlier", scale=1e6)}
    fr_plain = kalman_filter(spec, y)
    fr_iv = kalman_filter_interventions(spec, y, iv)
    # With outlier intervention, the filter at t=11 should be less pulled toward 100
    assert abs(fr_iv.m[11, 0]) < abs(fr_plain.m[11, 0])


def test_intervention_returns_filter_result() -> None:
    from engine.interventions import Intervention, kalman_filter_interventions

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=20, seed=9).y
    iv: dict[int, "Intervention"] = {}
    fr = kalman_filter_interventions(spec, y, iv)
    # Empty interventions == standard filter
    fr_plain = kalman_filter(spec, y)
    np.testing.assert_allclose(fr.loglik, fr_plain.loglik, atol=1e-10)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_advanced_engine.py -k "intervention" -v
```

Expected: `ModuleNotFoundError: No module named 'engine.interventions'`

- [ ] **Step 3: Implement `engine/interventions.py` (interventions section)**

```python
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

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve  # type: ignore[import-untyped]

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

        # Apply intervention
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_advanced_engine.py -k "intervention" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add engine/interventions.py tests/test_advanced_engine.py
git commit -m "feat(engine): kalman_filter_interventions — level/variance/outlier interventions"
```

---

## Task 6: Monitoring filter (`engine/interventions.py`)

**Files:**
- Modify: `tests/test_advanced_engine.py` (add tests)
- Modify: `engine/interventions.py` (add MonitorResult + function)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_advanced_engine.py`:

```python
# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


def test_monitor_result_fields() -> None:
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=30, seed=20).y
    mr = kalman_filter_monitor(spec, y, inflation=100.0, threshold=0.1)
    assert mr.H.shape == (30,)
    assert mr.L.shape == (30,)
    assert mr.alert.shape == (30,)
    assert mr.alert.dtype == bool
    # L is cumulative product of H
    np.testing.assert_allclose(mr.L, np.cumprod(mr.H), atol=1e-10)


def test_monitor_alert_on_structural_break() -> None:
    """Monitor should alert near or after a structural break in state noise."""
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.1)
    rng = np.random.default_rng(21)
    T = 300
    y_arr = np.zeros((T, 1))
    level = 0.0
    for t in range(T):
        w_var = 0.1 if t < 200 else 10.0  # structural break at t=200
        level += rng.normal(scale=np.sqrt(w_var))
        y_arr[t, 0] = level + rng.normal(scale=1.0)
    mr = kalman_filter_monitor(spec, y_arr, inflation=100.0, threshold=0.1)
    alert_times = np.where(mr.alert)[0]
    assert len(alert_times) > 0
    assert alert_times[0] >= 180  # no false alarm well before break


def test_monitor_no_alert_on_stable_series() -> None:
    """Well-specified stable series should produce few false alarms."""
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.1)
    y = simulate(spec, n=200, seed=22).y
    mr = kalman_filter_monitor(spec, y, inflation=100.0, threshold=0.1)
    # Cumulative product resets logic means few alerts for in-model series
    assert mr.alert.sum() < 20


def test_monitor_l_is_cumprod_h() -> None:
    from engine.interventions import kalman_filter_monitor

    spec = make_local_level(V=1.0, W_level=0.5)
    y = simulate(spec, n=50, seed=23).y
    mr = kalman_filter_monitor(spec, y)
    np.testing.assert_allclose(mr.L, np.cumprod(mr.H), atol=1e-12)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_advanced_engine.py -k "monitor" -v
```

Expected: `ImportError: cannot import name 'kalman_filter_monitor'`

- [ ] **Step 3: Add `MonitorResult` and `kalman_filter_monitor` to `engine/interventions.py`**

Add after the `kalman_filter_interventions` function:

```python
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
    if p != spec.p:
        raise ValueError(f"y observation dim {p} != spec.p {spec.p}")

    # Build M0 spec: same as M1 but V inflated
    V0 = spec.V * inflation
    spec0 = DLMSpec(
        F=spec.F, G=spec.G, V=V0, W=spec.W, m0=spec.m0, C0=spec.C0
    )

    d = spec.d
    F, G = spec.F, spec.G
    m_prev = spec.m0
    C_prev = spec.C0
    m0_prev = spec.m0
    C0_prev = spec.C0

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
        # M1 prior
        a_t = G @ m_prev
        R_t = _symmetrize(G @ C_prev @ G.T + spec.W)
        f_t = F @ a_t
        Q_t = _symmetrize(F @ R_t @ F.T + spec.V)
        # M0 prior (same state, inflated V)
        Q0_t = _symmetrize(F @ R_t @ F.T + spec0.V)

        e_t = y[t] - f_t
        log_p_M1 = _gaussian_log_predictive(y[t], f_t, Q_t)
        log_p_M0 = _gaussian_log_predictive(y[t], f_t, Q0_t)
        H_t = float(np.exp(log_p_M1 - log_p_M0))
        L_t *= H_t
        H_arr[t] = H_t
        L_arr[t] = L_t

        # M1 update
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
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_advanced_engine.py -k "monitor" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_advanced_engine.py -v
```

Expected: all tests PASSED

- [ ] **Step 6: Lint check**

```bash
ruff check engine/interventions.py && mypy engine/interventions.py
```

Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add engine/interventions.py tests/test_advanced_engine.py
git commit -m "feat(engine): MonitorResult + kalman_filter_monitor — sequential Bayes factor monitoring"
```

---

## Task 7: Dev dependencies + lint/type check sweep

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add PyMC and arviz to dev deps**

In `pyproject.toml`, update the `[project.optional-dependencies]` `dev` list to add:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
    "pandas-stubs>=2.0",
    "nbmake>=1.4",
    "jupyterlab>=4.0",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "pymc>=5.0",
    "arviz>=0.17",
]
```

- [ ] **Step 2: Install new deps**

```bash
pip install pymc>=5.0 arviz>=0.17
```

Expected: installs without error

- [ ] **Step 3: Full lint + type check**

```bash
ruff check engine/ && mypy engine/
```

Expected: no errors

- [ ] **Step 4: Full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all existing tests pass (new `test_advanced_engine.py` tests all pass)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add pymc and arviz to dev dependencies"
```

---

## Task 8: Notebook 00 — Advanced Setup

**Files:**
- Create: `notebooks/advanced/00_advanced_setup.ipynb`

- [ ] **Step 1: Create the notebook directory and notebook**

```bash
mkdir -p notebooks/advanced
```

Write `notebooks/advanced/00_advanced_setup.ipynb` as nbformat 4.5 JSON with the following cells:

**Cell 0** — markdown:
```markdown
# Advanced DLM Notebooks — Setup

**Prerequisites:** Intermediate series (notebooks 01–05), comfort with conjugate
Bayesian analysis and reading algorithm pseudocode.

**New engine modules introduced in this series:**
- `engine.ffbs` — Forward Filtering Backward Sampling
- `engine.interventions` — interventions, monitoring
- `engine.filter.kalman_filter_missing` — missing-data extension
- `engine.models.make_arima_dlm`, `make_multivariate_local_level`
```

**Cell 1** — code (imports/version check):
```python
import importlib, sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
print(f"Python {sys.version}")
print(f"NumPy  {np.__version__}")
print(f"SciPy  {scipy.__version__}")
print(f"Matplotlib {matplotlib.__version__}")
```

**Cell 2** — code (engine imports):
```python
from engine.filter import kalman_filter, kalman_filter_missing
from engine.ffbs import ffbs
from engine.interventions import (
    Intervention,
    kalman_filter_interventions,
    kalman_filter_monitor,
    MonitorResult,
)
from engine.models import (
    make_local_level,
    make_arima_dlm,
    make_multivariate_local_level,
)
from engine.simulate import simulate
from engine.smoother import rts_smoother
print("All advanced engine imports OK")
```

**Cell 3** — code (PyMC import check):
```python
try:
    import pymc as pm
    import arviz as az
    print(f"PyMC   {pm.__version__}")
    print(f"ArviZ  {az.__version__}")
    HAS_PYMC = True
except ImportError:
    print("PyMC not installed — notebook 07 MCMC cells will be skipped")
    HAS_PYMC = False
```

**Cell 4** — code (quick smoke test):
```python
spec = make_local_level(V=1.0, W_level=0.5)
sim = simulate(spec, n=50, seed=0)
fr = kalman_filter(spec, sim.y)
rng = np.random.default_rng(0)
theta = ffbs(spec, fr, rng)
assert theta.shape == (50, 1)
print("FFBS smoke test passed")
```

**Cell 5** — markdown:
```markdown
If all cells above ran without error, your environment is ready for the
advanced notebooks.
```

Each cell needs an `id` field (8-char hex string, unique within the notebook).

- [ ] **Step 2: Run nbmake smoke test**

```bash
pytest --nbmake notebooks/advanced/00_advanced_setup.ipynb -v
```

Expected: PASSED

- [ ] **Step 3: Commit**

```bash
git add notebooks/advanced/00_advanced_setup.ipynb
git commit -m "feat(notebooks): 00_advanced_setup — environment check for advanced series"
```

---

## Task 9: Notebook 06 — ARIMA ↔ DLM Equivalence

**Files:**
- Create: `notebooks/advanced/06_arima_dlm_equivalence.ipynb`

- [ ] **Step 1: Write notebook**

Write `notebooks/advanced/06_arima_dlm_equivalence.ipynb` with the following cell sequence (nbformat 4.5, each cell has unique 8-char hex `id`):

**Cell 0** — markdown: Title + references
```markdown
# Notebook 06 — ARIMA ↔ DLM Equivalence

**References:** West & Harrison §9.1–9.4; Petris §3.3

**New engine function:** `engine.models.make_arima_dlm(ar, ma, sigma2)`

Every causal stationary ARMA process has an exact state-space representation.
This notebook derives that representation step-by-step and verifies that
`FilterResult.loglik` recovers the exact ARIMA log-likelihood.
```

**Cell 1** — code: imports
```python
import numpy as np
import matplotlib.pyplot as plt
from engine.filter import kalman_filter
from engine.models import make_arima_dlm
```

**Cell 2** — markdown: AR(1) derivation
```markdown
## 1. AR(1) as a DLM

An AR(1) process $y_t = \phi y_{t-1} + \epsilon_t$, $\epsilon_t \sim N(0,\sigma^2)$,
is already a state-space model with state $\theta_t = y_t$:

$$y_t = \underbrace{[1]}_F \theta_t + v_t, \quad v_t \sim N(0,\nu)$$
$$\theta_t = \underbrace{[\phi]}_G \theta_{t-1} + w_t, \quad w_t \sim N(0,\sigma^2)$$

In the exact representation, observation noise $\nu = 0$; we use $\nu = 10^{-10}$
to satisfy the DLMSpec positive-diagonal requirement without affecting results.

**Reference:** W&H §9.1, eq. (9.2)
```

**Cell 3** — code: AR(1) spec and filter
```python
phi, sigma2 = 0.8, 1.0
spec_ar1 = make_arima_dlm(ar=[phi], ma=[], sigma2=sigma2)
print(f"State dim d={spec_ar1.d}")
print(f"F = {spec_ar1.F}")
print(f"G = {spec_ar1.G}")
print(f"W[0,0] = {spec_ar1.W[0,0]:.6f}  (≈ sigma2 = {sigma2})")

rng = np.random.default_rng(0)
T = 200
y_1d = np.zeros(T)
eps = rng.normal(scale=np.sqrt(sigma2), size=T)
for t in range(1, T):
    y_1d[t] = phi * y_1d[t - 1] + eps[t]

fr = kalman_filter(spec_ar1, y_1d[:, None])
print(f"\nDLM log-likelihood: {fr.loglik:.4f}")
```

**Cell 4** — markdown: ARMA(p,q) companion form derivation
```markdown
## 2. Companion Form for ARMA(p, q)

For ARMA(p,q) with $r = \max(p, q+1)$, the state vector
$\theta_t \in \mathbb{R}^r$ collects $r$ lags of the process in a companion form.

**Companion transition matrix** (W&H §9.2):
$$G = \begin{pmatrix} \phi_1 & \phi_2 & \cdots & \phi_p & 0 & \cdots & 0 \\
1 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 1 & \cdots & & & \\
\vdots & & \ddots & & & \end{pmatrix}$$

**MA polynomial vector** $\kappa = (1, \theta_1, \dots, \theta_q, 0, \dots, 0)^\top \in \mathbb{R}^r$.

**State noise covariance:**
$W = \sigma^2 \kappa \kappa^\top + \epsilon I$ where $\epsilon = 10^{-10}$ is a nugget.

**Key insight:** The Kalman filter innovations $e_t = y_t - f_t$ reproduce the
exact ARMA innovations, so `FilterResult.loglik` is the exact Gaussian
log-likelihood (W&H §9.4).
```

**Cell 5** — code: ARMA(2,1) example
```python
spec_arma = make_arima_dlm(ar=[0.6, -0.2], ma=[0.4], sigma2=1.0)
print(f"d = {spec_arma.d}  (r = max(2, 1+1) = 2)")
print(f"G =\n{spec_arma.G}")
print(f"W =\n{spec_arma.W}")
print(f"kappa = [1, 0.4] -> W[0,0] = 1*(1)^2 + nugget = {spec_arma.W[0,0]:.4f}")
```

**Cell 6** — code: log-lik comparison vs statsmodels
```python
from statsmodels.tsa.arima.model import ARIMA

y_ar1 = y_1d  # AR(1) series from above
res = ARIMA(y_ar1, order=(1, 0, 0)).fit(disp=False)
print(f"statsmodels log-lik : {res.llf:.4f}")
print(f"DLM Kalman log-lik  : {fr.loglik:.4f}")
print(f"Difference          : {abs(res.llf - fr.loglik):.4f}  (expect < 5 for T=200)")
```

**Cell 7** — code: plot innovations
```python
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(y_1d, lw=0.8, label="AR(1) data")
axes[0].plot(fr.f[:, 0], lw=1.0, label="one-step forecast", ls="--")
axes[0].legend(); axes[0].set_title("AR(1) data and one-step forecast")
axes[1].plot(fr.e[:, 0], lw=0.8, color="tab:orange", label="innovations e_t")
axes[1].axhline(0, lw=0.5, color="k")
axes[1].legend(); axes[1].set_title("Innovations (should be white noise for well-specified model)")
plt.tight_layout()
plt.show()
```

**Cell 8** — markdown: Exercises
```markdown
## Exercises

**Exercise 1** — Derive the state-space form for ARMA(2,1) from scratch:
write out G, kappa, W symbolically with φ₁, φ₂, θ₁, σ². Verify your result
matches `make_arima_dlm([phi1, phi2], [theta1], sigma2)`.

**Exercise 2** — Verify log-lik match for ARMA(1,1):
simulate 300 obs from an ARMA(1,1) with ar=[0.7], ma=[0.3], sigma2=2.0.
Compare `FilterResult.loglik` to `statsmodels.tsa.arima.model.ARIMA(...).fit().llf`.
The difference should be less than 5.

**Exercise 3 (challenge)** — The ARIMA(p,d,q) case: for d=1, the integration
layer means the first difference $\Delta y_t = y_t - y_{t-1}$ follows an
ARMA(p, q+1) process. Describe how you would extend `make_arima_dlm` to
handle d > 0.
```

**Cell 9** — code: Exercise 2 scaffold
```python
# Exercise 2 scaffold
# YOUR CODE HERE
# rng = np.random.default_rng(1)
# ... simulate ARMA(1,1) ...
# spec_arma11 = make_arima_dlm(ar=[0.7], ma=[0.3], sigma2=2.0)
# fr_arma11 = kalman_filter(spec_arma11, y[:, None])
# res_arma11 = ARIMA(y, order=(1, 0, 1)).fit(disp=False)
# assert abs(fr_arma11.loglik - res_arma11.llf) < 5.0, "log-lik mismatch"
```

- [ ] **Step 2: Run nbmake**

```bash
pytest --nbmake notebooks/advanced/06_arima_dlm_equivalence.ipynb -v
```

Expected: PASSED

- [ ] **Step 3: Commit**

```bash
git add notebooks/advanced/06_arima_dlm_equivalence.ipynb
git commit -m "feat(notebooks): 06_arima_dlm_equivalence — companion-form ARIMA DLM"
```

---

## Task 10: Notebook 07 — FFBS and MCMC

**Files:**
- Create: `notebooks/advanced/07_ffbs_and_mcmc.ipynb`

- [ ] **Step 1: Write notebook**

Write `notebooks/advanced/07_ffbs_and_mcmc.ipynb` (nbformat 4.5):

**Cell 0** — markdown: Title + references
```markdown
# Notebook 07 — FFBS and MCMC

**References:** W&H §15.2; Petris §4.4–4.5; Carter & Kohn (1994); Frühwirth-Schnatter (1994)

**New engine function:** `engine.ffbs.ffbs(spec, fr, rng)`

Two parts:
- **Part A:** Forward Filtering Backward Sampling — exact draws from the joint
  smoothing distribution p(θ_{1:T} | y_{1:T})
- **Part B:** Gibbs sampler for unknown V and W, then PyMC for NUTS sampling
```

**Cell 1** — code: imports
```python
import numpy as np
import matplotlib.pyplot as plt
from engine.filter import kalman_filter
from engine.ffbs import ffbs
from engine.models import make_local_level
from engine.simulate import simulate
from engine.smoother import rts_smoother
```

**Cell 2** — markdown: Part A derivation
```markdown
## Part A: Forward Filtering Backward Sampling

### The joint smoothing distribution

The marginal smoothing distributions $p(\theta_t \mid y_{1:T})$ are given by
the RTS smoother. But the joint distribution
$p(\theta_{1:T} \mid y_{1:T})$ factors as:

$$p(\theta_{1:T} \mid y_{1:T}) = p(\theta_T \mid y_{1:T}) \prod_{t=1}^{T-1} p(\theta_t \mid \theta_{t+1}, y_{1:t})$$

The conditional distributions in the product are Gaussian (W&H §15.2):

$$\theta_T \mid y_{1:T} \sim N(m_T, C_T)$$

For $t = T-1, \ldots, 1$:
$$h_t = m_t + B_t(\theta_{t+1} - a_{t+1}), \quad H_t = C_t - B_t R_{t+1} B_t^\top$$
$$\theta_t \mid \theta_{t+1}, y_{1:t} \sim N(h_t, H_t)$$

where $B_t = C_t G^\top R_{t+1}^{-1}$ is the backward gain (same as RTS smoother).
```

**Cell 3** — code: FFBS demo
```python
spec = make_local_level(V=2.0, W_level=0.5)
sim = simulate(spec, n=80, seed=0)
fr = kalman_filter(spec, sim.y)
sr = rts_smoother(spec, fr)
rng = np.random.default_rng(42)
n_draws = 200
draws = np.stack([ffbs(spec, fr, rng) for _ in range(n_draws)])  # (200, T, 1)

t_arr = np.arange(80)
fig, ax = plt.subplots(figsize=(11, 4))
for i in range(50):
    ax.plot(t_arr, draws[i, :, 0], color="steelblue", alpha=0.15, lw=0.7)
ax.plot(t_arr, sim.y[:, 0], "k.", ms=3, label="obs")
ax.plot(t_arr, sr.s[:, 0], "r-", lw=1.5, label="RTS smoother mean")
ax.plot(t_arr, draws.mean(axis=0)[:, 0], "b--", lw=1.5, label="FFBS mean (n=200)")
ax.legend(); ax.set_title("FFBS joint draws vs RTS smoother (local level)")
plt.tight_layout(); plt.show()
```

**Cell 4** — markdown: Part B — Gibbs sampler
```markdown
## Part B: Gibbs Sampler for Unknown V and W

When V and W are unknown, we place conjugate priors and alternate:

1. **Draw** $\theta_{1:T} \mid y, V, W$ — via FFBS (analytic Gaussian draw)
2. **Draw** $V \mid \theta, y$ — inverse-gamma conjugate:
   $V \mid \theta, y \sim \text{IG}\!\left(\frac{n_0 + T}{2},\, \frac{d_0 + \sum_t (y_t - \theta_t)^2}{2}\right)$
3. **Draw** $W \mid \theta$ — inverse-gamma (scalar W):
   $W \mid \theta \sim \text{IG}\!\left(\frac{a_0 + T - 1}{2},\, \frac{b_0 + \sum_t (\theta_t - \theta_{t-1})^2}{2}\right)$

**Reference:** Petris §4.4, Frühwirth-Schnatter (1994)
```

**Cell 5** — code: Gibbs sampler implementation
```python
from scipy.stats import invgamma  # type: ignore[import-untyped]

def gibbs_local_level(y: np.ndarray, n_iter: int = 1000, seed: int = 0) -> dict:
    """Gibbs sampler for local-level V and W."""
    T = len(y)
    rng = np.random.default_rng(seed)
    # Initialise
    V_curr, W_curr = 2.0, 0.5
    # Prior hyperparameters (weakly informative)
    n0, d0 = 1.0, 1.0  # V ~ IG(n0/2, d0/2)
    a0, b0 = 1.0, 1.0  # W ~ IG(a0/2, b0/2)
    V_samples, W_samples = [], []
    for _ in range(n_iter):
        # Step 1: draw theta | V, W via FFBS
        spec = make_local_level(V=V_curr, W_level=W_curr)
        fr = kalman_filter(spec, y[:, None])
        theta = ffbs(spec, fr, rng)[:, 0]
        # Step 2: draw V | theta, y
        ss_obs = float(np.sum((y - theta) ** 2))
        V_curr = float(invgamma.rvs(a=(n0 + T) / 2, scale=(d0 + ss_obs) / 2, random_state=rng))
        # Step 3: draw W | theta
        diffs = np.diff(theta)
        ss_state = float(np.sum(diffs ** 2))
        W_curr = float(invgamma.rvs(a=(a0 + T - 1) / 2, scale=(b0 + ss_state) / 2, random_state=rng))
        V_samples.append(V_curr)
        W_samples.append(W_curr)
    return {"V": np.array(V_samples), "W": np.array(W_samples)}

y_obs = sim.y[:, 0]
trace_gibbs = gibbs_local_level(y_obs, n_iter=2000, seed=1)
burnin = 500
print(f"V posterior mean: {trace_gibbs['V'][burnin:].mean():.3f}  (true 2.0)")
print(f"W posterior mean: {trace_gibbs['W'][burnin:].mean():.3f}  (true 0.5)")
```

**Cell 6** — code: Gibbs trace plots
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(trace_gibbs["V"], lw=0.5)
axes[0].axhline(2.0, color="r", ls="--", label="true V=2.0")
axes[0].set_title("Gibbs trace — V"); axes[0].legend()
axes[1].plot(trace_gibbs["W"], lw=0.5)
axes[1].axhline(0.5, color="r", ls="--", label="true W=0.5")
axes[1].set_title("Gibbs trace — W"); axes[1].legend()
plt.tight_layout(); plt.show()
```

**Cell 7** — markdown: PyMC blackbox likelihood
```markdown
## PyMC — Blackbox Likelihood

Since our Kalman filter is pure NumPy (not differentiable by PyTensor),
we use the **blackbox likelihood** approach:

```python
import pytensor.compile.ops as ops

@ops.as_op(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
def kalman_loglik_op(V_val, W_val):
    spec = make_local_level(V=float(V_val), W_level=float(W_val))
    fr = kalman_filter(spec, y_obs[:, None])
    return np.array(fr.loglik)
```

This registers our function as a PyTensor Op. Because it has no gradient,
PyMC will use the **Slice sampler** (gradient-free) automatically.

**Note:** The Slice sampler is less efficient than NUTS for smooth posteriors —
it may need more tuning steps. For production use, consider implementing the
Kalman filter in PyTensor ops (see `pymc-extras`).
```

**Cell 8** — code: PyMC sampling
```python
HAS_PYMC = False
try:
    import pymc as pm
    import pytensor.tensor as pt
    import pytensor.compile.ops as ops
    import arviz as az
    HAS_PYMC = True
except ImportError:
    print("PyMC not installed — skipping MCMC cells")

if HAS_PYMC:
    @ops.as_op(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
    def kalman_loglik_op(V_val: np.ndarray, W_val: np.ndarray) -> np.ndarray:
        spec = make_local_level(V=float(V_val), W_level=float(W_val))
        fr = kalman_filter(spec, y_obs[:, None])
        return np.array(fr.loglik)

    with pm.Model() as dlm_model:
        V_rv = pm.HalfNormal("V", sigma=3.0)
        W_rv = pm.HalfNormal("W", sigma=1.0)
        _ = pm.Potential("loglik", kalman_loglik_op(V_rv, W_rv))
        idata = pm.sample(500, tune=500, target_accept=0.9,
                          progressbar=True, random_seed=42)

    az.plot_trace(idata, var_names=["V", "W"])
    plt.tight_layout(); plt.show()
    print(az.summary(idata, var_names=["V", "W"]))
```

**Cell 9** — markdown: Exercises
```markdown
## Exercises

**Exercise 1** — Implement one full Gibbs sweep from scratch:
given current V, W, write the three steps (FFBS draw, V draw, W draw)
without calling `gibbs_local_level`. Verify your single sweep produces a
valid `(V, W)` pair.

**Exercise 2** — Extend to LLT:
adapt `gibbs_local_level` for a local linear trend model with unknown
V, W_level, W_slope. Use inverse-gamma priors on each variance separately.
Check that the Gibbs chain recovers the true variances (within Monte Carlo error).

**Exercise 3** — Compare Gibbs vs PyMC:
if PyMC is installed, compare the posterior means and standard deviations
of V and W from both samplers on the same dataset. Do they agree?
```

**Cell 10** — code: Exercise 1 scaffold
```python
# Exercise 1 — single Gibbs sweep
# YOUR CODE HERE
# Hint: use rts_smoother or ffbs for the state draw,
#       scipy.stats.invgamma.rvs for the variance draws.
```

- [ ] **Step 2: Run nbmake**

```bash
pytest --nbmake notebooks/advanced/07_ffbs_and_mcmc.ipynb -v
```

Expected: PASSED (PyMC cells guarded by `if HAS_PYMC`)

- [ ] **Step 3: Commit**

```bash
git add notebooks/advanced/07_ffbs_and_mcmc.ipynb
git commit -m "feat(notebooks): 07_ffbs_and_mcmc — FFBS, Gibbs sampler, PyMC blackbox likelihood"
```

---

## Task 11: Notebook 08 — Interventions and Outliers

**Files:**
- Create: `notebooks/advanced/08_interventions_outliers.ipynb`

- [ ] **Step 1: Write notebook**

Write `notebooks/advanced/08_interventions_outliers.ipynb` (nbformat 4.5):

**Cell 0** — markdown: Title + refs
```markdown
# Notebook 08 — Interventions and Outliers

**References:** W&H §11.1–11.3; Petris §3.6

**New engine function:** `engine.interventions.kalman_filter_interventions`

Three canonical intervention types:
- **Level shift** — add a known delta to the state prior mean at time t
- **Variance inflation** — multiply W_t by a scale factor at time t
- **Outlier** — multiply V_t by a large scale at time t (downweights obs)
```

**Cell 1** — code: imports
```python
import numpy as np
import matplotlib.pyplot as plt
from engine.filter import kalman_filter
from engine.interventions import Intervention, kalman_filter_interventions
from engine.models import make_local_level
from engine.simulate import simulate
```

**Cell 2** — markdown: Level shift derivation
```markdown
## 1. Level Shift

At time $t$ we know the series shifts by $\delta$. We encode this by
modifying the prior state mean before the update step:

$$a_t^* = a_t + \delta \, e_j$$

where $e_j$ is the unit vector selecting the level component.

Without this adjustment, the filter must learn the shift from the data —
producing a smeared step in the filtered mean over several time steps.
With the intervention, the filter instantly accounts for the shift.

**Reference:** W&H §11.1
```

**Cell 3** — code: level shift demo
```python
spec = make_local_level(V=1.0, W_level=0.1)
sim = simulate(spec, n=80, seed=0)
y = sim.y.copy()
# Inject a level shift at t=40
y[40:] += 5.0

fr_plain = kalman_filter(spec, y)
fr_iv = kalman_filter_interventions(
    spec, y, {40: Intervention(kind="level", delta=5.0, component=0)}
)

t = np.arange(80)
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(t, y[:, 0], "k.", ms=3, label="obs (with shift at t=40)")
axes[0].plot(t, fr_plain.m[:, 0], label="filter (no intervention)")
axes[0].plot(t, fr_iv.m[:, 0], label="filter (level intervention)", ls="--")
axes[0].axvline(40, color="gray", ls=":", lw=0.8)
axes[0].legend(); axes[0].set_title("Level shift: filtered means")

axes[1].plot(t, fr_plain.e[:, 0], label="innovations (no intervention)", alpha=0.7)
axes[1].plot(t, fr_iv.e[:, 0], label="innovations (intervention)", ls="--", alpha=0.7)
axes[1].axhline(0, lw=0.5, color="k")
axes[1].axvline(40, color="gray", ls=":", lw=0.8)
axes[1].legend(); axes[1].set_title("Innovations ACF: smearing vs clean step")
plt.tight_layout(); plt.show()
```

**Cell 4** — markdown: Outlier model derivation
```markdown
## 2. Outlier / Variance-Inflation Model (λ-ω model)

At time $t$, the observation is either in-model (probability $\lambda$)
or an outlier (probability $1 - \lambda$) drawn from a distribution
with inflated variance $\omega V$:

$$y_t \sim \lambda \, N(f_t, Q_t) + (1 - \lambda) \, N(f_t, \omega Q_t)$$

**Approximate filter:** run two parallel filters (M1 and M0) and compute
the posterior mixture weight:

$$\pi_t = \frac{\lambda \, p(y_t \mid M_1)}{\lambda \, p(y_t \mid M_1) + (1-\lambda) \, p(y_t \mid M_0)}$$

$\pi_t \approx 0$ signals a likely outlier. We implement the simpler
single-model version here: an outlier intervention at time $t$ sets
$V_t \leftarrow \omega V$, strongly downweighting the observation.

**Reference:** W&H §11.3
```

**Cell 5** — code: outlier intervention demo
```python
spec = make_local_level(V=1.0, W_level=0.1)
sim2 = simulate(spec, n=60, seed=1)
y2 = sim2.y.copy()
outlier_times = [15, 35, 50]
for t_out in outlier_times:
    y2[t_out, 0] += 15.0  # spike

fr2_plain = kalman_filter(spec, y2)
iv_dict = {t_out: Intervention(kind="outlier", scale=1000.0) for t_out in outlier_times}
fr2_iv = kalman_filter_interventions(spec, y2, iv_dict)

t2 = np.arange(60)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t2, y2[:, 0], "k.", ms=4, label="obs (with spikes)")
ax.plot(t2, fr2_plain.m[:, 0], label="filter (no intervention)")
ax.plot(t2, fr2_iv.m[:, 0], label="filter (outlier intervention)", ls="--")
for t_out in outlier_times:
    ax.axvline(t_out, color="red", ls=":", lw=0.8, alpha=0.6)
ax.legend(); ax.set_title("Outlier intervention: downweighting spikes")
plt.tight_layout(); plt.show()
```

**Cell 6** — markdown: Exercises
```markdown
## Exercises

**Exercise 1** — Simulate a series with three injected outliers.
Apply the outlier filter and verify that the innovations at the outlier
time steps are dramatically smaller with the intervention.

```python
# YOUR CODE HERE
# spec = make_local_level(V=1.0, W_level=0.1)
# sim = simulate(spec, n=80, seed=5)
# y = sim.y.copy()
# Inject spikes at t=20, 45, 60 with amplitude 20
# ...
# Show that |fr_iv.e[20]| < |fr_plain.e[20]|
```

**Exercise 2** — Variance inflation vs outlier:
For a single large observation, compare the filtered mean at t+1 when
using `kind="variance"` (inflates W → allows big state jump) vs
`kind="outlier"` (inflates V → ignores obs). How do the posteriors differ?

**Exercise 3 (challenge)** — Implement the two-component approximate mixture
filter: for each time step, compute $\pi_t$ using log-sum-exp and compare
$\pi_t$ across the three injected outlier times.
```

- [ ] **Step 2: Run nbmake**

```bash
pytest --nbmake notebooks/advanced/08_interventions_outliers.ipynb -v
```

Expected: PASSED

- [ ] **Step 3: Commit**

```bash
git add notebooks/advanced/08_interventions_outliers.ipynb
git commit -m "feat(notebooks): 08_interventions_outliers — level shift, variance inflation, outlier model"
```

---

## Task 12: Notebook 09 — Monitoring and Structural Breaks

**Files:**
- Create: `notebooks/advanced/09_monitoring_structural_breaks.ipynb`

- [ ] **Step 1: Write notebook**

Write `notebooks/advanced/09_monitoring_structural_breaks.ipynb` (nbformat 4.5):

**Cell 0** — markdown: Title + refs
```markdown
# Notebook 09 — Monitoring and Structural Breaks

**References:** W&H §11.4–11.5

**New engine function:** `engine.interventions.kalman_filter_monitor`

The sequential Bayes factor monitoring statistic H_t detects when the data
is better explained by a reference model with inflated observation variance.
```

**Cell 1** — code: imports
```python
import numpy as np
import matplotlib.pyplot as plt
from engine.interventions import kalman_filter_monitor
from engine.models import make_local_level
from engine.simulate import simulate
```

**Cell 2** — markdown: Theory
```markdown
## 1. Sequential Bayes Factor

At each time step $t$, compare the one-step predictive density:

$$H_t = \frac{p(y_t \mid y_{1:t-1}, M_1)}{p(y_t \mid y_{1:t-1}, M_0)}$$

- $M_1$: current model with observation variance $V$
- $M_0$: reference "robustness" model with inflated variance $\omega V$, $\omega \gg 1$

When $H_t < 1$, the data is better explained by $M_0$ at step $t$.

The running product:
$$L_t = \prod_{s=1}^{t} H_s$$

falls monotonically when the data repeatedly favours $M_0$.  An alert
is triggered when $L_t < \text{threshold}$.

**Connection to CUSUM:** $-\log L_t$ is a Bayesian analogue of the CUSUM
statistic. The threshold $L < 0.1$ corresponds to a Bayes factor $> 10$ in
favour of model inadequacy. (W&H §11.5)

**Reference:** W&H §11.4, eq. (11.22)
```

**Cell 3** — code: stable series (no alert)
```python
spec = make_local_level(V=1.0, W_level=0.1)
sim_stable = simulate(spec, n=200, seed=0)
mr_stable = kalman_filter_monitor(spec, sim_stable.y, inflation=100.0, threshold=0.1)
print(f"Alerts on stable series: {mr_stable.alert.sum()} / 200")
```

**Cell 4** — code: structural break demo
```python
rng = np.random.default_rng(42)
T = 300
y_break = np.zeros((T, 1))
level = 0.0
for t in range(T):
    w_var = 0.1 if t < 200 else 10.0  # W doubles after t=200
    level += rng.normal(scale=np.sqrt(w_var))
    y_break[t, 0] = level + rng.normal(scale=1.0)

mr_break = kalman_filter_monitor(spec, y_break, inflation=100.0, threshold=0.1)
alert_times = np.where(mr_break.alert)[0]
print(f"First alert at t={alert_times[0] if len(alert_times) > 0 else 'never'}")
```

**Cell 5** — code: plot monitor output
```python
t_arr = np.arange(T)
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axes[0].plot(t_arr, y_break[:, 0], lw=0.5, label="obs")
axes[0].axvline(200, color="red", ls="--", lw=0.8, label="structural break (t=200)")
axes[0].legend(); axes[0].set_title("Observations")

axes[1].semilogy(t_arr, mr_break.H, lw=0.8, label="H_t (per-step Bayes factor)")
axes[1].axhline(1.0, lw=0.5, color="k", ls="--")
axes[1].axvline(200, color="red", ls="--", lw=0.8)
axes[1].legend(); axes[1].set_title("Per-step Bayes factor H_t")

axes[2].semilogy(t_arr, mr_break.L, lw=1.0, label="L_t (cumulative product)")
axes[2].axhline(0.1, color="orange", ls="--", lw=0.8, label="threshold=0.1")
axes[2].axvline(200, color="red", ls="--", lw=0.8)
for ta in alert_times[:5]:
    axes[2].axvline(ta, color="orange", lw=0.5, alpha=0.5)
axes[2].legend(); axes[2].set_title("Cumulative product L_t and alert times")
plt.tight_layout(); plt.show()
```

**Cell 6** — markdown: Exercises
```markdown
## Exercises

**Exercise 1** — Threshold sensitivity:
Run `kalman_filter_monitor` on the structural-break series with
`threshold` values 0.01, 0.1, 0.5. For each, record the first alert time.
Plot first-alert time vs threshold.

```python
# YOUR CODE HERE
thresholds = [0.01, 0.1, 0.5]
first_alerts = []
for thr in thresholds:
    mr = kalman_filter_monitor(spec, y_break, inflation=100.0, threshold=thr)
    alerts = np.where(mr.alert)[0]
    first_alerts.append(alerts[0] if len(alerts) > 0 else T)
# Plot first_alerts vs thresholds
```

**Exercise 2** — Inflation sensitivity:
Fix `threshold=0.1`. Run the monitor with `inflation` values 10, 100, 1000.
How does the first-alert time change? Explain intuitively why a smaller
inflation makes the monitor less sensitive.

**Exercise 3 (challenge)** — After an alert, practitioners often "reset" L_t = 1
and restart monitoring. Implement a resetting version of `kalman_filter_monitor`
and show that it detects a second structural break in a series with two breaks.
```

- [ ] **Step 2: Run nbmake**

```bash
pytest --nbmake notebooks/advanced/09_monitoring_structural_breaks.ipynb -v
```

Expected: PASSED

- [ ] **Step 3: Commit**

```bash
git add notebooks/advanced/09_monitoring_structural_breaks.ipynb
git commit -m "feat(notebooks): 09_monitoring_structural_breaks — sequential Bayes factor monitor"
```

---

## Task 13: Notebook 10 — Multivariate DLMs and Missing Data

**Files:**
- Create: `notebooks/advanced/10_multivariate_missing_data.ipynb`

- [ ] **Step 1: Write notebook**

Write `notebooks/advanced/10_multivariate_missing_data.ipynb` (nbformat 4.5):

**Cell 0** — markdown: Title + refs
```markdown
# Notebook 10 — Multivariate DLMs and Missing Data

**References:** W&H §16.1–16.3; Petris §2.7, §4.3

**New engine functions:**
- `engine.filter.kalman_filter_missing(spec, y)` — handles NaN observations
- `engine.models.make_multivariate_local_level(p, V, W_level)` — p series, one shared level

**Two parts:**
- **Part A:** Missing-data filter — imputation and uncertainty quantification
- **Part B:** Multivariate local level — shared scalar level across multiple series
```

**Cell 1** — code: imports
```python
import numpy as np
import matplotlib.pyplot as plt
from engine.filter import kalman_filter, kalman_filter_missing
from engine.models import make_local_level, make_multivariate_local_level
from engine.simulate import simulate
from engine.comparison import compare_models
```

**Cell 2** — markdown: Part A — missing data
```markdown
## Part A: Missing-Data Filter

### Theory

When $y_t$ contains NaN entries, the update step is skipped:

$$m_t = a_t, \quad C_t = R_t \quad \text{(posterior = prior)}$$

No log-likelihood contribution is added. The filter continues forward —
uncertainty grows at rate $W$ per missing step.

This gives exact Bayesian imputation: the predictive distribution over
a missing gap follows the state evolution equations only.

**Reference:** W&H §16.1; Petris §2.7
```

**Cell 3** — code: missing data imputation
```python
spec = make_local_level(V=1.0, W_level=0.2)
sim = simulate(spec, n=100, seed=0)
y_miss = sim.y.copy()
gap_start, gap_end = 40, 60
y_miss[gap_start:gap_end] = np.nan  # 20-step gap

fr_full = kalman_filter(spec, sim.y)
fr_miss = kalman_filter_missing(spec, y_miss)

t = np.arange(100)
# 95% credible intervals from filtered distribution
ci_half = 1.96 * np.sqrt(fr_miss.f[:, 0] * 0 + fr_miss.Q[:, 0, 0])  # use Q for prediction
```

**Cell 4** — code: plot imputation
```python
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(t,
    fr_miss.f[:, 0] - 1.96 * np.sqrt(fr_miss.Q[:, 0, 0]),
    fr_miss.f[:, 0] + 1.96 * np.sqrt(fr_miss.Q[:, 0, 0]),
    alpha=0.25, color="steelblue", label="95% predictive CI")
ax.plot(t, sim.y[:, 0], "k.", ms=3, label="true obs")
ax.plot(t, y_miss[:, 0], "r.", ms=4, label="observed (NaN in gap)")
ax.plot(t, fr_miss.f[:, 0], "b-", lw=1.0, label="one-step forecast (imputation)")
ax.axvspan(gap_start, gap_end - 1, alpha=0.08, color="red", label="missing gap")
ax.legend(fontsize=8); ax.set_title("Missing-data filter: imputation over 20-step gap")
plt.tight_layout(); plt.show()
print(f"CI width at gap midpoint (t=50): {2 * 1.96 * np.sqrt(fr_miss.Q[50, 0, 0]):.3f}")
```

**Cell 5** — markdown: Part B — multivariate local level
```markdown
## Part B: Multivariate Local Level

### Model

$p$ series share a common scalar level $\theta_t$:

$$y_{i,t} = \theta_t + v_{i,t}, \quad v_{i,t} \sim N(0, \sigma^2_i), \quad i=1,\ldots,p$$
$$\theta_t = \theta_{t-1} + w_t, \quad w_t \sim N(0, W_\text{level})$$

In matrix form:
$$y_t = F\theta_t + v_t, \quad F = \mathbf{1}_p, \quad V = \text{diag}(\sigma_1^2, \ldots, \sigma_p^2)$$
$$\theta_t = G\theta_{t-1} + w_t, \quad G = [1], \quad W = [W_\text{level}]$$

State dim $d = 1$ regardless of $p$.

**When is this appropriate?** When the series share a true common component
(e.g. multiple measurements of the same underlying quantity). Compare to
$p$ independent local-level models using `compare_models`.

**Reference:** W&H §16.2; Petris §4.3
```

**Cell 6** — code: multivariate local level demo
```python
rng = np.random.default_rng(42)
T = 100
# Shared level
level = np.cumsum(rng.normal(scale=np.sqrt(0.3), size=T))
# Three series with different obs noise
y3 = np.column_stack([
    level + rng.normal(scale=1.0, size=T),
    level + rng.normal(scale=2.0, size=T),
    level + rng.normal(scale=0.5, size=T),
])

spec_shared = make_multivariate_local_level(p=3, V=[1.0, 4.0, 0.25], W_level=0.3)
fr_shared = kalman_filter(spec_shared, y3)
print(f"Shared-level log-lik: {fr_shared.loglik:.2f}")
print(f"Filtered state shape: {fr_shared.m.shape}  (T=100, d=1)")
```

**Cell 7** — code: model comparison vs independent
```python
spec_ind1 = make_local_level(V=1.0, W_level=0.3)
spec_ind2 = make_local_level(V=4.0, W_level=0.3)
spec_ind3 = make_local_level(V=0.25, W_level=0.3)
fr1 = kalman_filter(spec_ind1, y3[:, [0]])
fr2 = kalman_filter(spec_ind2, y3[:, [1]])
fr3 = kalman_filter(spec_ind3, y3[:, [2]])
loglik_ind = fr1.loglik + fr2.loglik + fr3.loglik
print(f"Independent models log-lik sum: {loglik_ind:.2f}")
print(f"Shared-level log-lik          : {fr_shared.loglik:.2f}")
print(f"Bayes factor (shared vs ind)  : exp({fr_shared.loglik - loglik_ind:.2f})")
```

**Cell 8** — code: plot shared-level filter
```python
t_arr = np.arange(T)
fig, ax = plt.subplots(figsize=(11, 4))
for i, col in enumerate(["tab:blue", "tab:orange", "tab:green"]):
    ax.plot(t_arr, y3[:, i], ".", ms=2, color=col, alpha=0.5, label=f"y_{i+1}")
ax.plot(t_arr, level, "k-", lw=1.5, label="true shared level")
ax.plot(t_arr, fr_shared.m[:, 0], "r--", lw=1.5, label="filtered shared level")
ax.legend(fontsize=8); ax.set_title("Multivariate local level: shared scalar level")
plt.tight_layout(); plt.show()
```

**Cell 9** — markdown: Exercises
```markdown
## Exercises

**Exercise 1** — Bivariate imputation:
Simulate a bivariate series (p=2) with `make_multivariate_local_level(p=2, V=[1.0, 2.0], W_level=0.3)`.
Set 20% of observations in series 1 to NaN. Run `kalman_filter_missing` and
plot the imputed values with 95% credible intervals.

```python
# YOUR CODE HERE
# spec2 = make_multivariate_local_level(p=2, V=[1.0, 2.0], W_level=0.3)
# sim2 = ...
# Set 20% of y[:, 0] to NaN
# fr_miss2 = kalman_filter_missing(spec2, y_miss2)
```

**Exercise 2** — Two independent levels:
Compare the log marginal likelihood of `make_multivariate_local_level(p=2, ...)`
to two independent `make_local_level` models run separately. When does the
shared-level model win? Generate data where they are the same (zero cross-correlation)
and where they share a true common level — show the Bayes factor flips.

**Exercise 3 (challenge)** — Dynamic factor model concept:
For $p$ series and $k < p$ latent factors, the observation matrix $F$ has
shape $(p, k)$. Each column of $F$ is a factor loading. For $p=4$, $k=2$:
write down what the shapes of $F$, $G$, $V$, $W$ should be if:
(a) each factor is an independent random walk, and
(b) factors share a single common trend.
```

- [ ] **Step 2: Run nbmake**

```bash
pytest --nbmake notebooks/advanced/10_multivariate_missing_data.ipynb -v
```

Expected: PASSED

- [ ] **Step 3: Commit**

```bash
git add notebooks/advanced/10_multivariate_missing_data.ipynb
git commit -m "feat(notebooks): 10_multivariate_missing_data — missing-data filter + multivariate local level"
```

---

## Task 14: CI update and Pull Request

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Locate nbmake step in CI**

```bash
grep -n "nbmake" .github/workflows/ci.yml
```

Expected: a line like `pytest --nbmake notebooks/intermediate/`

- [ ] **Step 2: Extend nbmake step to include advanced notebooks**

Change the nbmake command from:
```yaml
pytest --nbmake notebooks/intermediate/
```
to:
```yaml
pytest --nbmake notebooks/intermediate/ notebooks/advanced/
```

- [ ] **Step 3: Verify full test suite locally**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASSED

- [ ] **Step 4: Run smoke test on all notebooks**

```bash
pytest --nbmake notebooks/intermediate/ notebooks/advanced/ -v --tb=short
```

Expected: all 11 notebooks PASSED

- [ ] **Step 5: Commit CI update**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add notebooks/advanced/ to nbmake smoke test"
```

- [ ] **Step 6: Push branch and open PR**

```bash
git push -u origin feat/advanced-notebooks
gh pr create \
  --title "feat: Advanced DLM notebook series (FFBS, interventions, monitoring, multivariate)" \
  --body "$(cat <<'EOF'
## Summary

- **Engine extensions** (pure NumPy, ruff+mypy clean):
  - `engine/ffbs.py` — Forward Filtering Backward Sampling (Carter & Kohn 1994)
  - `engine/filter.py` — `kalman_filter_missing` for NaN observations
  - `engine/models.py` — `make_arima_dlm`, `make_multivariate_local_level`
  - `engine/interventions.py` — `Intervention`, `kalman_filter_interventions`, `MonitorResult`, `kalman_filter_monitor`
- **Unit tests** — `tests/test_advanced_engine.py` (analytic checks + Monte Carlo validation)
- **Six notebooks** — `notebooks/advanced/00`–`10` covering ARIMA↔DLM, FFBS/MCMC, interventions, monitoring, multivariate/missing data
- **PyMC + arviz** added to dev deps (used only in notebook 07)
- **CI** extended to smoke-test all 11 notebooks

## Test plan

- [ ] All `pytest tests/` pass
- [ ] All `pytest --nbmake notebooks/intermediate/ notebooks/advanced/` pass
- [ ] `ruff check engine/` and `mypy engine/` clean
- [ ] FFBS Monte Carlo mean matches RTS smoother mean (atol=0.08)
- [ ] ARIMA log-lik matches statsmodels within 5 nats
- [ ] Monitor triggers after structural break (first alert ≥ t=180)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

### Spec coverage

| Spec section | Task covering it |
|---|---|
| §3.1 `engine/ffbs.py` — `ffbs(spec, fr, rng)` | Task 1 |
| §3.2 `engine/filter.py` — `kalman_filter_missing` | Task 2 |
| §3.3 `make_arima_dlm` | Task 3 |
| §3.3 `make_multivariate_local_level` | Task 4 |
| §3.4 `Intervention`, `kalman_filter_interventions` | Task 5 |
| §3.4 `MonitorResult`, `kalman_filter_monitor` | Task 6 |
| §6 PyMC dev deps | Task 7 |
| §2 Notebook 00 setup | Task 8 |
| §5 Notebook 06 ARIMA | Task 9 |
| §5 Notebook 07 FFBS + MCMC | Task 10 |
| §5 Notebook 08 Interventions | Task 11 |
| §5 Notebook 09 Monitoring | Task 12 |
| §5 Notebook 10 Multivariate + missing | Task 13 |
| §7 CI: nbmake extended | Task 14 |
| §8 pymc≥5.0, arviz≥0.17 to dev deps | Task 7 |
| §4 Four-section notebook template | Tasks 9–13 (all) |

All spec requirements are covered.

### Type / interface consistency

- `ffbs` imports `_solve_psd`, `_symmetrize` from `engine.filter` (both present at line 57–68 of filter.py)
- `kalman_filter_missing` uses same `_solve_psd`, `_logdet_psd`, `_symmetrize` helpers — already in same file
- `kalman_filter_interventions` and `kalman_filter_monitor` both import from `engine.filter` and `engine.models`
- `MonitorResult.filter_result: FilterResult` (not flat fields) — consistent with spec §3.4
- `Intervention` dataclass fields: `kind`, `delta`, `component`, `scale` — used consistently in Task 5 tests and implementation
- `make_arima_dlm` returns `DLMSpec` — validated by `DLMSpec.__post_init__` which checks PSD diagonal ≥ 1e-12; the 1e-10 nugget satisfies this
- `make_multivariate_local_level` accepts `float | list[float] | NDArray` for V — type annotation `NDArray[Any] | list[float] | float` is consistent with the implementation

### No placeholders

All code steps contain complete, runnable code. No "TBD", "YOUR CODE" in implementation cells (only in exercise scaffolds, which are intentionally incomplete).
