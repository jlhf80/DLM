# DLM Advanced Tutorial — Notebook Design Spec

**Date:** 2026-04-29
**Status:** Approved — ready for implementation planning
**Scope:** Full Advanced tier, all six topics merged into five topic notebooks
**Delivery:** Jupyter notebook series in `notebooks/advanced/`
**Prerequisites:** Completion of `notebooks/intermediate/` (discount factors, conjugate V, dynamic regression, Fourier seasonal, model comparison)

---

## 1. Overview

A rigorous notebook series aimed at PhD-level readers comfortable with conjugate
Bayesian analysis, measure-theoretic probability, and reading algorithm pseudocode.
Each notebook derives the key equations, implements the algorithm in Python, and
connects the result to practical data-analysis decisions.

Compared to the Intermediate series, Advanced notebooks carry:
- More prose derivation — full equation chains, not just the final result
- Explicit W&H / Petris section citations at every major result, so readers can
  follow up in the primary sources
- Exercises that require the reader to extend or re-derive, not just run code

**Target reader:** a doctoral student or practitioner who has seen Bayesian
statistics formally and wants to understand DLMs at the level needed to apply
them to real research problems.

---

## 2. Notebook list

```
notebooks/advanced/
    00_advanced_setup.ipynb
    06_arima_dlm_equivalence.ipynb
    07_ffbs_and_mcmc.ipynb
    08_interventions_outliers.ipynb
    09_monitoring_structural_breaks.ipynb
    10_multivariate_missing_data.ipynb
```

The numbering continues from the Intermediate series (01–05) so both series
can coexist in `notebooks/`.

---

## 3. Engine extensions

The existing engine (`engine/`) remains dependency-free (no PyMC). All new
additions follow the same conventions: pure NumPy/SciPy, `NDArray[Any]` typing,
ruff + mypy clean.

### 3.1 `engine/ffbs.py` — Forward Filtering Backward Sampling

**Algorithm (Carter & Kohn 1994; Frühwirth-Schnatter 1994; W&H §15.2)**

Given a `FilterResult` (forward pass already run), the backward sampler draws
an exact sample from the joint smoothing distribution:

```
θ_{1:T} | y_{1:T}  ~  N(s, S)   (marginals, from RTS smoother)
```

but jointly, not marginally. The backward sampling pass is:

```
θ_T | y_{1:T}  ~  N(m_T, C_T)
for t = T-1 down to 1:
    h_t = m_t + B_t (θ_{t+1} - a_{t+1})
    H_t = C_t - B_t R_{t+1} B_t'
    θ_t | θ_{t+1}, y_{1:t}  ~  N(h_t, H_t)
```

where `B_t = C_t G' R_{t+1}^{-1}` (same gain used in the RTS smoother).

**New function:** `ffbs(spec, fr, rng)` — takes a `DLMSpec`, a `FilterResult`,
and a `numpy.random.Generator`; returns one draw `theta: (T, d)`.

Running `n_samples` calls gives a Monte Carlo approximation of the joint
smoothing distribution.

### 3.2 `engine/filter.py` — Missing-data filter

**Theory (W&H §16.1; Petris §2.7)**

When `y_t` contains `NaN` entries, the update step is skipped entirely for
that time point: the posterior equals the prior (`m_t = a_t`, `C_t = R_t`).
The one-step forecast `f_t` and `Q_t` are still computed, but no log-likelihood
contribution is added for missing observations.

**New function:** `kalman_filter_missing(spec, y)` — identical to
`kalman_filter` except it checks `np.isnan(y[t]).any()` before the update
step. Returns a standard `FilterResult`; missing time points have `e[t] = NaN`.

### 3.3 `engine/models.py` — ARIMA DLM builder and multivariate local level

**`make_arima_dlm(ar, ma, sigma2)`**

Converts a causal stationary ARIMA specification to a DLM in companion form.

For ARIMA(p, 0, q) with `r = max(p, q+1)`:
- State dim `d = r`
- `F = [1, 0, ..., 0]`
- `G` = companion matrix (first row = AR coefficients zero-padded to `r`,
  sub-diagonal = 1)
- `V = nugget * I` where `nugget = 1e-10` (observation is exact in theory;
  tiny nugget keeps `DLMSpec` PSD validation happy without affecting results)
- `W` = sparse: only `W[0,0] = sigma2` (innovation enters through first
  component)

For ARIMA(p, d, q) with d > 0: pre-difference the state representation by
stacking d integration layers on top. This is the standard state-space
representation of integrated processes (W&H §9.1–9.4).

Parameters:
- `ar: list[float]` — AR coefficients φ₁, ..., φ_p (empty list = no AR)
- `ma: list[float]` — MA coefficients θ₁, ..., θ_q (empty list = no MA)
- `sigma2: float` — innovation variance

Returns a `DLMSpec`.

**`make_multivariate_local_level(p, V, W_level)`**

Multivariate local level: p series sharing a common scalar level. Each
observation dimension has its own observation noise variance (diagonal V),
and the shared level evolves as a scalar random walk.

- State dim `d = 1` (shared scalar level)
- `F = ones((p, 1))`
- `G = [[1]]`
- `V = diag(V)` if V is a list, else `V * eye(p)`
- `W = [[W_level]]`

### 3.4 `engine/interventions.py` — Interventions and monitoring

**Interventions (W&H §11.1–11.3)**

An intervention at time `t` modifies the model specification for that step.
Three canonical types:

1. **Level shift** (`kind="level"`): add a large known amount `delta` to the
   state mean: `a_t ← a_t + delta * e_direction`, where `e_direction` is a
   unit vector selecting the level component.
2. **Variance inflation** (`kind="variance"`): multiply `W_t` by a scalar
   `scale > 1` at time `t` to allow a larger-than-usual state jump.
3. **Outlier / point** (`kind="outlier"`): inflate `V_t` by a large factor at
   time `t`, down-weighting that observation.

**New function:** `kalman_filter_interventions(spec, y, interventions)` where
`interventions: dict[int, dict]` maps time index to intervention spec.

**Monitoring (W&H §11.4–11.5)**

The Bayes factor monitoring statistic detects model inadequacy sequentially.
At each time `t`, compare the predictive density under the current model M₁
to that under a reference "robustness model" M₀ (inflated observation variance):

```
H_t = p(y_t | y_{1:t-1}, M₁) / p(y_t | y_{1:t-1}, M₀)
```

A running product `L_t = ∏_{s=s0}^{t} H_s < threshold` triggers an alert.

**New function:** `kalman_filter_monitor(spec, y, inflation=100.0, threshold=0.1)`
Returns a `MonitorResult` dataclass containing all `FilterResult` fields plus:
- `H: (T,)` — per-step Bayes factors `H_t`
- `L: (T,)` — cumulative product `L_t`
- `alert: (T,)` bool — True where `L_t < threshold`

---

## 4. Notebook structure

Each notebook follows a four-section template, expanded from the Intermediate
template to include a **Derivation** section:

1. **Motivation** — 3–5 cells of prose explaining the problem and why the
   standard approach is insufficient. Include a concrete failure example.
2. **Derivation** — 4–8 cells of LaTeX developing the key equations with
   explicit W&H / Petris section references at each major result.
3. **Implementation** — inspect or write the new engine function; run it on
   simulated and/or real-flavoured data.
4. **Exercises** — 3–4 cells with `# YOUR CODE HERE` and assert-based
   auto-grading where feasible. At least one exercise requires re-deriving or
   extending the algorithm, not just tuning parameters.

---

## 5. Notebook content

### 06 — ARIMA ↔ DLM equivalence

**W&H §9.1–9.4; Petris §3.3**

Shows that every causal ARIMA model has an exact DLM representation. The
companion-form state-space representation is derived step by step for
AR(1), AR(p), MA(q), and ARMA(p,q). The integration (differencing) extension
to ARIMA(p,d,q) follows naturally.

Key insight: the Kalman filter on the DLM form gives the exact innovations
representation of an ARIMA process, so `FilterResult.loglik` is the exact
ARIMA log-likelihood — no numerical differencing needed.

Exercises: derive the state space form for ARMA(2,1); verify that `FilterResult.loglik`
matches `statsmodels.tsa.arima.model.ARIMA.loglik` on the same series.

### 07 — FFBS and MCMC

**W&H §15.2; Petris §4.4–4.5; Carter & Kohn (1994); Frühwirth-Schnatter (1994)**

Two-part notebook:

*Part A: FFBS*
Derives the joint smoothing distribution `p(θ_{1:T} | y_{1:T})` as a sequence
of conditionals. Shows that the marginal means equal the RTS smoother means
but the joint draws are richer (they preserve temporal correlations). Implements
`ffbs` and visualises a fan of joint draws vs. the RTS smoother.

*Part B: Gibbs sampler and PyMC*
Block Gibbs sampler for unknown V and W:
1. Draw `θ_{1:T} | y, V, W` — via FFBS (analytic Gaussian draw)
2. Draw `V | θ, y` — inverse-gamma conjugate draw
3. Draw `W | θ` — inverse-Wishart conjugate draw (scalar W: inverse-gamma)

Implements the Gibbs loop in pure NumPy for the conjugate case, then
re-implements using **PyMC**: V and W are `pm.HalfNormal` priors, the Kalman
filter log-likelihood is a `pm.Potential`, and NUTS is used for sampling.
Compares Gibbs and NUTS trace plots and effective sample sizes.

Exercises: implement one Gibbs sweep from scratch; extend to a two-component
model (level + slope) and check that the MCMC recovers the true V, W_level,
W_slope.

### 08 — Interventions and outliers

**W&H §11.1–11.3; Petris §3.6**

Derives the three canonical intervention types. Shows that an unmodelled
level shift causes the standard filter to "smear" the shock across many time
steps (visible in the innovations ACF). Applying a variance-inflation
intervention at the known break point corrects this.

Also derives the λ-ω outlier model: at time `t`, `y_t` is either the standard
model (probability λ) or an outlier with inflated V (probability 1-λ). The
exact posterior requires a mixture, but the two-component filter gives an
approximation. Shows the posterior probability of outlier at each time step.

Exercises: simulate a series with three injected outliers; apply the outlier
filter and show that posterior outlier probabilities peak at the correct times.

### 09 — Monitoring and structural breaks

**W&H §11.4–11.5**

Derives the sequential Bayes factor monitoring statistic H_t. Shows that
`H_t < 1` at time `t` means the data is better explained by the inflated-
variance reference model, signalling possible model failure.

The cumulative product `L_t` plays the role of a Bayesian CUSUM. Derives
the connection to classical CUSUM control charts and explains what the
`threshold` parameter means in terms of Bayesian evidence.

Exercises: simulate a series where W doubles after t=150 (structural break in
the state evolution variance); show that the monitor triggers near t=150.
Compare threshold sensitivity.

### 10 — Multivariate DLMs and missing data

**W&H §16.1–16.3; Petris §2.7, §4.3**

*Part A: Missing data*
Derives the missing-data filter (skip update for NaN observations). Shows
that the filtered distribution over a gap follows the prediction equations
only — uncertainty grows at rate W per step. Uses the filter to impute
missing values and build credible intervals over the gap.

*Part B: Multivariate observations*
Derives the multivariate local-level model (`p > 1` observations sharing a
common scalar level). Discusses when a shared-level model is appropriate vs.
independent univariate models. Uses `compare_models` to compare the two on
simulated data.

Introduces the concept of a **dynamic factor model** (many series, few latent
factors) as the natural extension — derives the observation matrix structure
but defers full implementation to the reader as an exercise.

Exercises: simulate a bivariate series with 20% missing data; impute and plot
uncertainty intervals; extend the shared-level model to two independent levels
and compare log marginal likelihoods.

---

## 6. PyMC usage (notebook 07 only)

PyMC is used **only in the notebook layer**. The approach:

```python
import pymc as pm
from engine.filter import kalman_filter

with pm.Model() as dlm_model:
    V = pm.HalfNormal("V", sigma=2.0)
    W = pm.HalfNormal("W", sigma=0.5)
    spec = make_local_level(V=V, W_level=W)   # symbolic — won't work directly
    # Instead: use pm.Potential with a Python function
    ll = pm.Potential("loglik", kalman_filter_loglik_op(V, W, y))
    trace = pm.sample(1000, tune=500, target_accept=0.9)
```

Because the Kalman filter is pure NumPy, it is not differentiable by PyTensor
(PyMC's backend). Two options — both demonstrated in the notebook:
1. **Blackbox likelihood** via `pm.Potential` with `pytensor.compile.ops.as_op`
   (gradient-free; uses Slice sampler)
2. **PyMC's native DLM support** via `pymc_extras` if installed, or a
   re-implementation using PyTensor operations

The notebook implements option 1 (simpler, more pedagogically transparent)
and mentions option 2 with a pointer to `pymc-extras`.

---

## 7. Testing

- `tests/test_advanced_engine.py` — analytic unit tests for each new engine
  function (FFBS draws have correct mean and variance; missing filter skips
  update correctly; ARIMA log-lik matches statsmodels; monitor triggers on
  known break)
- `tests/test_notebooks.py` extended (or a new `tests/test_advanced_notebooks.py`)
  — nbmake smoke test for all 6 advanced notebooks
- CI: change the existing nbmake step from `notebooks/intermediate/` to
  `notebooks/intermediate/ notebooks/advanced/` to pick up both series

---

## 8. Dependencies

New packages added to `[project.optional-dependencies] dev`:
- `pymc>=5.0` — MCMC in notebook 07
- `arviz>=0.17` — trace plots and diagnostics in notebook 07

No new production dependencies (engine remains dependency-free of PyMC/arviz).

---

## 9. Non-goals

- No GPU acceleration or JAX backend.
- No full dynamic factor model implementation (introduced as concept only).
- No non-linear / non-Gaussian particle filter (out of scope for this series).
- No real dataset bundling (all examples use simulated data for reproducibility).
- No Quarto/R conversion.
