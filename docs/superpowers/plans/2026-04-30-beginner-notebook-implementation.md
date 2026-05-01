# Beginner Notebook Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a self-contained beginner Jupyter notebook series (`notebooks/beginner/`) that takes a Python-fluent reader with no DLM experience through the Kalman filter, local level, LLT, seasonal, and parameter estimation — leaving them ready for the intermediate series.

**Architecture:** Eight notebooks in `notebooks/beginner/` using the existing `engine/` package (no new engine functions). Two optional notebooks (B0 Bayesian primer, B6 GLM connection). PyMC used in B0 and B5 under a `HAS_PYMC` guard. CI nbmake step extended to include the new directory.

**Tech Stack:** NumPy, SciPy, Matplotlib, PyMC ≥ 5.0, ArviZ, engine/ (filter, smoother, models, simulate, forecast)

---

## Context for implementers

**Repo layout:**
```
engine/
    filter.py      — kalman_filter(spec, y), kalman_filter_tv(F_seq, G, V, W, m0, C0, y)
    models.py      — DLMSpec, DLMSpecTV, make_local_level, make_local_linear_trend,
                     make_seasonal_factor, combine
    smoother.py    — rts_smoother(spec, fr) → SmoothResult(.s, .S)
    simulate.py    — simulate(spec, n, seed) → SimResult(.y, .theta_true)
    forecast.py    — forecast_horizon(spec, fr, h) → Forecast(.means, .lower, .upper)
notebooks/
    intermediate/  — 00_setup + 01-05 topic notebooks (reference for style)
    advanced/      — 00_setup + 06-10 topic notebooks
    beginner/      ← CREATE THIS DIRECTORY (first notebook creates it)
.github/workflows/ci.yml   — nbmake step to extend
```

**Key signatures (verify before using):**
- `make_local_level(V, W_level)` → DLMSpec (d=1)
- `make_local_linear_trend(V, W_level, W_slope)` → DLMSpec (d=2, state=(level,slope))
- `make_seasonal_factor(period, V, W_season)` → DLMSpec (d=period-1)
- `combine(*specs)` → DLMSpec (block-diagonal merge, V taken from first spec)
- `kalman_filter(spec, y)` — y shape (T,1); returns FilterResult(.m, .C, .a, .R, .f, .Q, .e, .loglik)
- `rts_smoother(spec, fr)` — returns SmoothResult(.s shape (T,d), .S shape (T,d,d))
- `forecast_horizon(spec, fr, h)` — returns Forecast(.means shape (h,1), .lower, .upper)
- `simulate(spec, n, seed)` — returns SimResult(.y shape (T,1), .theta_true shape (T,d))
- Path setup in notebooks: `Path().resolve().parents[1]` (two levels up from `notebooks/beginner/`)

**Style reference:** Read `notebooks/intermediate/00_setup.ipynb` and `notebooks/intermediate/01_discount_factors.ipynb` before writing any notebook — match cell structure, header format, and plot conventions exactly.

**Smoke-testing notebooks:** Run after creating each notebook:
```bash
cd /path/to/repo
pytest --nbmake notebooks/beginner/<notebook>.ipynb -q
```
Expected: `1 passed` (or `1 passed, N warnings`). If it fails, the notebook has a runtime error — fix before committing.

**PyMC guard pattern (use in B0 and B5):**
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
```
Wrap all PyMC code in `if HAS_PYMC:` blocks.

---

## Task 1: CI update

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Open `.github/workflows/ci.yml` and locate the notebook smoke-test step**

The current last step reads:
```yaml
      - name: Notebook smoke tests
        run: pytest --nbmake notebooks/intermediate/ notebooks/advanced/ -q
```

- [ ] **Step 2: Add `notebooks/beginner/` to the nbmake command**

Replace the run line with:
```yaml
      - name: Notebook smoke tests
        run: pytest --nbmake notebooks/beginner/ notebooks/intermediate/ notebooks/advanced/ -q
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add notebooks/beginner/ to nbmake smoke tests"
```

---

## Task 2: 00_setup.ipynb

**Files:**
- Create: `notebooks/beginner/00_setup.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/00_setup.ipynb` with these cells in order:

**Cell 0 — markdown:**
```markdown
# DLM Beginner Tutorial — Setup & Orientation

This notebook verifies your environment and introduces the notation used throughout
the beginner series.

**Prerequisites:** Python fluency, basic probability (mean, variance, normal distribution).
No prior Bayesian or state-space modeling experience required.

**Series overview:**
- `B0_bayesian_primer.ipynb` *(optional)* — Bayesian modeling with PyMC from scratch
- `B1_dlm_intro.ipynb` — State space model equations and the Kalman filter
- `B2_local_level.ipynb` — Local-level model: filter, smoother, forecast
- `B3_local_linear_trend.ipynb` — Adding a slope state
- `B4_seasonal_models.ipynb` — Periodic components
- `B5_parameter_estimation.ipynb` — MLE and Bayesian estimation of V and W
- `B6_dlm_glm_connection.ipynb` *(optional)* — DLM as a generalization of regression
```

**Cell 1 — markdown:**
```markdown
## 1. Environment check
```

**Cell 2 — code:**
```python
import sys
from pathlib import Path

# Add project root to path when running from notebooks/beginner/
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

print(f"Python  : {sys.version.split()[0]}")
print(f"NumPy   : {np.__version__}")
print(f"SciPy   : {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

# Engine imports — all should succeed
from engine.models import (
    DLMSpec, DLMSpecTV,
    make_local_level, make_local_linear_trend,
    make_seasonal_factor, combine,
)
from engine.filter import kalman_filter, kalman_filter_tv
from engine.smoother import rts_smoother
from engine.simulate import simulate
from engine.forecast import forecast_horizon

print("\nEngine imports: OK")

# Optional PyMC
try:
    import pymc as pm
    import arviz as az
    print(f"PyMC    : {pm.__version__}  (optional — used in B0 and B5)")
except ImportError:
    print("PyMC    : not installed (optional — skip B0, skip PyMC cells in B5)")
```

**Cell 3 — markdown:**
```markdown
## 2. Notation

The Gaussian DLM uses West & Harrison (W&H) notation throughout this series.

**Model equations (W&H §2.2):**

$$
\begin{aligned}
y_t &= F_t \theta_t + v_t, \quad v_t \sim N(0, V_t) \quad \text{(observation equation)} \\
\theta_t &= G_t \theta_{t-1} + w_t, \quad w_t \sim N(0, W_t) \quad \text{(state equation)}
\end{aligned}
$$

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $y_t$ | $(p,)$ | observed data at time $t$ |
| $\theta_t$ | $(d,)$ | latent state at time $t$ |
| $F_t$ | $(p, d)$ | observation matrix |
| $G_t$ | $(d, d)$ | state transition matrix |
| $V_t$ | $(p, p)$ | observation noise covariance |
| $W_t$ | $(d, d)$ | state evolution noise covariance |
| $m_t$ | $(d,)$ | filtered state mean $E[\theta_t \mid y_{1:t}]$ |
| $C_t$ | $(d, d)$ | filtered state covariance |
| $a_t$ | $(d,)$ | prior state mean $E[\theta_t \mid y_{1:t-1}]$ |
| $R_t$ | $(d, d)$ | prior state covariance |
| $e_t$ | $(p,)$ | innovation $y_t - F_t a_t$ |
| $Q_t$ | $(p, p)$ | innovation covariance $F_t R_t F_t' + V_t$ |
| $A_t$ | $(d, p)$ | Kalman gain $R_t F_t' Q_t^{-1}$ |
```

**Cell 4 — markdown:**
```markdown
## 3. Reading guide

| Your background | Suggested path |
|-----------------|----------------|
| Never done Bayesian modeling | B0 → B1 → B2 → B3 → B4 → B5 |
| Know PyMC already | B1 → B2 → B3 → B4 → B5 |
| Know regression, want the GLM link | Add B6 after B5 |

After completing this series, open `notebooks/intermediate/00_setup.ipynb`.
```

- [ ] **Step 2: Smoke-test the notebook**

```bash
pytest --nbmake notebooks/beginner/00_setup.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/00_setup.ipynb
git commit -m "feat(beginner): add 00_setup notebook"
```

---

## Task 3: B0_bayesian_primer.ipynb

**Files:**
- Create: `notebooks/beginner/B0_bayesian_primer.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B0_bayesian_primer.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B0 — Bayesian Modeling Primer *(optional)*

**Skip this notebook** if you already use PyMC or have a solid Bayesian modeling background.

**You need this notebook if:** you haven't seen Bayesian inference in practice, or
you've heard of Bayes' theorem but never written a probabilistic model in code.

**What you'll learn:**
- Prior, likelihood, posterior — what each term means and how to compute them
- MCMC intuition — what the sampler does and how to read diagnostics
- The PyMC workflow you'll use in B5 (parameter estimation)
```

**Cell 1 — markdown:**
```markdown
## 1. Bayes' theorem

$$P(\theta \mid y) = \frac{P(y \mid \theta)\, P(\theta)}{P(y)} \propto P(y \mid \theta)\, P(\theta)$$

- **Prior** $P(\theta)$: what we believe about $\theta$ before seeing data
- **Likelihood** $P(y \mid \theta)$: how probable the data is given $\theta$
- **Posterior** $P(\theta \mid y)$: updated belief after seeing data

### Example: coin-flip

We observe $k$ heads in $n$ flips. We want to infer $\theta$ = P(heads).

Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$ (encodes prior belief)
Likelihood: $k \mid \theta \sim \text{Binomial}(n, \theta)$
Posterior: $\theta \mid k \sim \text{Beta}(\alpha_0 + k,\, \beta_0 + n - k)$ (conjugate)
```

**Cell 2 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# Data
n_flips, k_heads = 20, 14

# Prior: Beta(2, 2) — slight preference for fair coin
alpha0, beta0 = 2.0, 2.0

# Grid approximation of posterior
theta_grid = np.linspace(0.001, 0.999, 500)
prior = beta_dist.pdf(theta_grid, alpha0, beta0)
likelihood = theta_grid**k_heads * (1 - theta_grid)**(n_flips - k_heads)
posterior_unnorm = prior * likelihood
posterior = posterior_unnorm / np.trapz(posterior_unnorm, theta_grid)

# Exact conjugate posterior
alpha_post = alpha0 + k_heads
beta_post  = beta0 + (n_flips - k_heads)
posterior_exact = beta_dist.pdf(theta_grid, alpha_post, beta_post)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(theta_grid, prior / np.trapz(prior, theta_grid), "C0--", lw=1.5, label=f"Prior Beta({alpha0},{beta0})")
ax.plot(theta_grid, posterior, "C1-", lw=2, label="Grid posterior")
ax.plot(theta_grid, posterior_exact, "k:", lw=1.5, label=f"Conjugate Beta({alpha_post},{beta_post})")
ax.axvline(k_heads/n_flips, color="grey", ls="--", lw=1, label=f"MLE = {k_heads/n_flips:.2f}")
ax.set(xlabel="θ = P(heads)", ylabel="density", title=f"Coin flip: {k_heads}/{n_flips} heads")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Posterior mean (conjugate): {alpha_post/(alpha_post+beta_post):.3f}")
```

**Cell 3 — markdown:**
```markdown
## 2. PyMC — same model in code

PyMC lets us write the generative model symbolically. MCMC then samples from the
posterior numerically — useful when there's no conjugate formula.
```

**Cell 4 — code (PyMC guard):**
```python
HAS_PYMC = False
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    print("PyMC not installed — install with: pip install pymc arviz")

if HAS_PYMC:
    with pm.Model() as coin_model:
        theta = pm.Beta("theta", alpha=alpha0, beta=beta0)
        obs = pm.Binomial("obs", n=n_flips, p=theta, observed=k_heads)
        idata = pm.sample(2000, tune=1000, progressbar=False, random_seed=42)

    fig, ax = plt.subplots(figsize=(9, 4))
    samples = idata.posterior["theta"].values.ravel()
    ax.hist(samples, bins=60, density=True, alpha=0.5, color="C1", label="MCMC posterior")
    ax.plot(theta_grid, posterior_exact, "k-", lw=2, label="Conjugate posterior")
    ax.set(xlabel="θ", ylabel="density", title="PyMC vs conjugate posterior")
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(az.summary(idata, var_names=["theta"]))
```

**Cell 5 — markdown:**
```markdown
## 3. Gaussian mean estimation

A more DLM-relevant example: estimate the mean $\mu$ of a Gaussian with known variance.

**Model:**
$$\mu \sim N(\mu_0, \sigma_0^2), \quad y_i \mid \mu \sim N(\mu, \sigma^2)$$

**Conjugate posterior:**
$$\mu \mid y_{1:N} \sim N(\mu_N, \sigma_N^2)$$
where
$$\sigma_N^2 = \left(\frac{1}{\sigma_0^2} + \frac{N}{\sigma^2}\right)^{-1}, \quad
\mu_N = \sigma_N^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_i y_i}{\sigma^2}\right)$$
```

**Cell 6 — code:**
```python
rng = np.random.default_rng(0)
mu_true = 3.5
sigma = 1.0       # known observation std
N = 30
y_data = rng.normal(mu_true, sigma, N)

# Prior
mu0, sigma0 = 0.0, 5.0

# Conjugate posterior
sigma_N2 = 1.0 / (1/sigma0**2 + N/sigma**2)
mu_N = sigma_N2 * (mu0/sigma0**2 + y_data.sum()/sigma**2)

mu_grid = np.linspace(-2, 8, 400)
from scipy.stats import norm
prior_pdf = norm.pdf(mu_grid, mu0, sigma0)
post_pdf  = norm.pdf(mu_grid, mu_N, np.sqrt(sigma_N2))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(mu_grid, prior_pdf, "C0--", lw=1.5, label=f"Prior N({mu0},{sigma0}²)")
ax.plot(mu_grid, post_pdf,  "C1-",  lw=2,   label=f"Posterior N({mu_N:.2f},{sigma_N2:.3f})")
ax.axvline(mu_true, color="k", ls=":", lw=1.5, label=f"True μ = {mu_true}")
ax.axvline(y_data.mean(), color="grey", ls="--", lw=1, label=f"Sample mean = {y_data.mean():.2f}")
ax.set(xlabel="μ", ylabel="density", title="Gaussian mean estimation")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Posterior mean: {mu_N:.3f}  (true: {mu_true})")
print(f"Posterior std : {np.sqrt(sigma_N2):.3f}")
```

**Cell 7 — markdown:**
```markdown
## 4. Reading MCMC diagnostics

When using MCMC (as in B5), check two diagnostics before trusting results:

| Diagnostic | Good value | Meaning |
|------------|-----------|---------|
| `r_hat` | < 1.01 | chains converged to same distribution |
| `ess_bulk` | > 400 | effective sample size — enough independent draws |

`az.plot_trace(idata)` shows: left column = density of samples, right column = trace over iterations.
A healthy trace looks like white noise (no trends, no stuck periods).
```

**Cell 8 — code (PyMC guard):**
```python
if HAS_PYMC:
    with pm.Model() as gauss_model:
        mu = pm.Normal("mu", mu=mu0, sigma=sigma0)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)
        idata2 = pm.sample(2000, tune=1000, progressbar=False, random_seed=1)

    az.plot_trace(idata2, var_names=["mu"])
    plt.tight_layout()
    plt.show()
    print(az.summary(idata2, var_names=["mu"]))
    print(f"\nConjugate posterior mean: {mu_N:.3f}")
```

**Cell 9 — markdown:**
```markdown
## Exercises

**Exercise 1** — Prior sensitivity: change the prior to `Normal(mu=10, sigma=0.5)`
(strong prior far from true value) and re-run with N=5, then N=50. At which N does
the posterior "overcome" the prior?

**Exercise 2** — Posterior width formula: verify numerically that
$\sigma_N^2 = (1/\sigma_0^2 + N/\sigma^2)^{-1}$ matches `az.summary`'s `sd` column
(within floating-point tolerance) for N=30, σ=1, σ₀=5.
```

- [ ] **Step 2: Smoke-test the notebook**

```bash
pytest --nbmake notebooks/beginner/B0_bayesian_primer.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B0_bayesian_primer.ipynb
git commit -m "feat(beginner): add B0 Bayesian primer notebook"
```

---

## Task 4: B1_dlm_intro.ipynb

**Files:**
- Create: `notebooks/beginner/B1_dlm_intro.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B1_dlm_intro.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B1 — Introduction to Dynamic Linear Models

**Reference:** West & Harrison §1–2; Petris §1

**Concepts introduced:**
- The state space model: observation and state equations
- Why we need a filter: the latent state is unobserved
- Predict–update cycle: the Kalman filter algorithm
```

**Cell 1 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from engine.filter import kalman_filter
from engine.models import make_local_level
from engine.simulate import simulate
```

**Cell 2 — markdown:**
```markdown
## 1. The two-equation model

A **Dynamic Linear Model** describes how an unobserved state $\theta_t$ evolves
and how noisy observations $y_t$ are generated from it (W&H §2.2):

$$\boxed{y_t = F_t \theta_t + v_t, \quad v_t \sim N(0, V_t)}  \quad \text{(observation equation)}$$

$$\boxed{\theta_t = G_t \theta_{t-1} + w_t, \quad w_t \sim N(0, W_t)}  \quad \text{(state equation)}$$

**What each piece means:**
- $\theta_t$ — the true underlying quantity we care about (latent, unobserved)
- $y_t$ — what we actually measure (noisy version of $F_t \theta_t$)
- $F_t$ — how the state maps to the observation (often just 1)
- $G_t$ — how the state evolves from one step to the next
- $V_t$ — observation noise: how noisy is each measurement?
- $W_t$ — state noise: how much does the true state wander?

**The filtering problem:** given $y_1, \ldots, y_t$, what is the distribution
$p(\theta_t \mid y_{1:t})$? The Kalman filter answers this exactly when the model
is Gaussian.
```

**Cell 3 — markdown:**
```markdown
## 2. The simplest DLM: local level

The **local level model** (W&H §4.3) sets $F=1$, $G=1$:

$$y_t = \theta_t + v_t, \quad v_t \sim N(0, V)$$
$$\theta_t = \theta_{t-1} + w_t, \quad w_t \sim N(0, W)$$

$\theta_t$ is a random-walk level. $y_t$ is that level observed with noise $V$.
The ratio $\kappa = W/V$ (signal-to-noise) controls how quickly the filter tracks changes.
```

**Cell 4 — code:**
```python
# Simulate a local-level series by hand — no engine yet
rng = np.random.default_rng(0)
T = 80
V_true, W_true = 2.0, 0.5

theta = np.zeros(T)
y     = np.zeros(T)

theta[0] = rng.normal(0.0, 1.0)
y[0]     = theta[0] + rng.normal(0.0, np.sqrt(V_true))

for t in range(1, T):
    theta[t] = theta[t-1] + rng.normal(0.0, np.sqrt(W_true))   # state evolves
    y[t]     = theta[t]   + rng.normal(0.0, np.sqrt(V_true))   # noisy observation

t_arr = np.arange(T)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_arr, theta, "k-", lw=1.5, label="true state θ_t")
ax.plot(t_arr, y, "C0.", ms=4, alpha=0.6, label="observations y_t")
ax.set(xlabel="t", ylabel="value", title=f"Local level simulation  (V={V_true}, W={W_true})")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Obs std (expected √V={np.sqrt(V_true):.2f}): {y.std():.2f}")
```

**Cell 5 — markdown:**
```markdown
## 3. The Kalman filter: predict–update

The filter maintains a Gaussian belief $p(\theta_t \mid y_{1:t}) = N(m_t, C_t)$.
Each time step has two phases (W&H §2.3):

### Predict (prior for time $t$)

$$a_t = G\, m_{t-1} \qquad (\text{prior mean})$$
$$R_t = G\, C_{t-1}\, G' + W \qquad (\text{prior covariance})$$

### Update (incorporate $y_t$)

$$e_t = y_t - F\, a_t \qquad (\text{innovation: surprise in observation})$$
$$Q_t = F\, R_t\, F' + V \qquad (\text{innovation covariance})$$
$$A_t = R_t\, F'\, Q_t^{-1} \qquad (\text{Kalman gain: how much to trust } y_t)$$
$$m_t = a_t + A_t\, e_t \qquad (\text{posterior mean})$$
$$C_t = R_t - A_t\, Q_t\, A_t' \qquad (\text{posterior covariance})$$

**Intuition:** if $Q_t$ is large (uncertain prediction), $A_t$ is large — we update
strongly toward $y_t$. If $Q_t$ is small (confident prediction), $A_t$ is small — we
barely move.
```

**Cell 6 — code:**
```python
# Manual filter step at t=1
# Prior at t=0: m0=0, C0=1000 (diffuse)
m_prev, C_prev = np.array([[0.0]]), np.array([[[1000.0]]])
F, G = np.array([[1.0]]), np.array([[1.0]])
V_mat = np.array([[V_true]])
W_mat = np.array([[W_true]])

# --- Predict ---
a1 = G @ m_prev[-1]                        # (1,)
R1 = G @ C_prev[-1] @ G.T + W_mat         # (1,1)

# --- Update ---
e1   = y[0] - F @ a1                       # innovation
Q1   = F @ R1 @ F.T + V_mat               # (1,1)
A1   = R1 @ F.T @ np.linalg.inv(Q1)       # Kalman gain
m1   = a1 + A1 @ e1                        # posterior mean
C1   = R1 - A1 @ Q1 @ A1.T               # posterior covariance

print(f"y[0]      = {y[0]:.4f}")
print(f"a1        = {a1[0]:.4f}  (prior mean: no data yet → 0)")
print(f"e1        = {e1[0]:.4f}  (innovation)")
print(f"A1        = {A1[0,0]:.6f}  (Kalman gain)")
print(f"m1        = {m1[0]:.4f}  (posterior mean)")
print(f"√C1       = {np.sqrt(C1[0,0]):.4f}  (posterior std)")
```

**Cell 7 — code:**
```python
# Full filter via engine
spec = make_local_level(V=V_true, W_level=W_true)
fr   = kalman_filter(spec, y[:, None])   # y must be (T,1)

fig, ax = plt.subplots(figsize=(11, 4))
std = np.sqrt(fr.C[:, 0, 0])
ax.plot(t_arr, theta, "k-", lw=1, label="true θ_t")
ax.plot(t_arr, y, ".", ms=4, alpha=0.5, color="C0", label="obs y_t")
ax.plot(t_arr, fr.m[:, 0], "C1-", lw=2, label="filtered mean m_t")
ax.fill_between(t_arr,
                fr.m[:, 0] - 1.96*std,
                fr.m[:, 0] + 1.96*std,
                alpha=0.25, color="C1", label="95% filter interval")
ax.set(xlabel="t", ylabel="value", title="Kalman filter on local level")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Log marginal likelihood: {fr.loglik:.2f}")
```

**Cell 8 — markdown:**
```markdown
## Exercises

**Exercise 1** — Change W from 0.5 to 5.0 and re-run the engine filter. What happens
to the 95% interval? Does the filtered mean track $y_t$ more or less closely? Why?

**Exercise 2** — Trace the predict step manually for $t=2$. Use `m1` and `C1` computed
above as the prior, then compute `a2`, `R2` with the formulas:
```python
a2 = G @ m1
R2 = G @ C1 @ G.T + W_mat
```
Compare `a2` to `fr.a[1, 0]` — they should match (remember `fr` is 0-indexed, so t=2 maps to index 1).
```

- [ ] **Step 2: Smoke-test**

```bash
pytest --nbmake notebooks/beginner/B1_dlm_intro.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B1_dlm_intro.ipynb
git commit -m "feat(beginner): add B1 DLM intro notebook"
```

---

## Task 5: B2_local_level.ipynb

**Files:**
- Create: `notebooks/beginner/B2_local_level.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B2_local_level.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B2 — Local Level Model

**Reference:** West & Harrison §2.3, §4.3; Petris §2.1

**Concepts introduced:**
- Signal-to-noise ratio and its effect on filtering
- RTS smoother: using future observations to refine past estimates
- Multi-step forecasting with widening uncertainty bands
- Log marginal likelihood and its role in model comparison
```

**Cell 1 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from engine.models import make_local_level
from engine.filter import kalman_filter
from engine.smoother import rts_smoother
from engine.simulate import simulate
from engine.forecast import forecast_horizon
```

**Cell 2 — markdown:**
```markdown
## 1. Local level model and signal-to-noise ratio

The local level model (W&H §4.3):

$$y_t = \theta_t + v_t, \quad v_t \sim N(0, V)$$
$$\theta_t = \theta_{t-1} + w_t, \quad w_t \sim N(0, W)$$

The **signal-to-noise ratio** $\kappa = W/V$ determines how responsive the filter is:
- $\kappa \to 0$: filter barely updates — trusts the model, ignores data
- $\kappa \to \infty$: filter follows $y_t$ closely — trusts data over model
```

**Cell 3 — code:**
```python
spec = make_local_level(V=2.0, W_level=0.5)
sim  = simulate(spec, n=100, seed=0)
fr   = kalman_filter(spec, sim.y)

t = np.arange(100)
std = np.sqrt(fr.C[:, 0, 0])

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# Panel 1: observations + true state
axes[0].plot(t, sim.y[:, 0], ".", ms=4, alpha=0.5, label="obs y_t")
axes[0].plot(t, sim.theta_true[:, 0], "k-", lw=1.5, label="true θ_t")
axes[0].set_ylabel("value"); axes[0].legend()
axes[0].set_title("Local level (V=2.0, W=0.5, κ=0.25)")

# Panel 2: prior means (one-step predictions)
axes[1].plot(t, fr.a[:, 0], "C2-", lw=1.5, label="prior mean a_t")
axes[1].plot(t, sim.y[:, 0], ".", ms=3, alpha=0.3, color="grey")
axes[1].set_ylabel("value"); axes[1].legend()
axes[1].set_title("One-step-ahead predictions a_t")

# Panel 3: filtered means with uncertainty
axes[2].plot(t, sim.theta_true[:, 0], "k-", lw=1, label="true θ_t")
axes[2].plot(t, fr.m[:, 0], "C1-", lw=2, label="filtered mean m_t")
axes[2].fill_between(t, fr.m[:, 0]-1.96*std, fr.m[:, 0]+1.96*std,
                     alpha=0.25, color="C1", label="95% interval")
axes[2].set_xlabel("t"); axes[2].set_ylabel("value"); axes[2].legend()
axes[2].set_title("Filtered state")

plt.tight_layout()
plt.show()
```

**Cell 4 — markdown:**
```markdown
## 2. RTS smoother

The **Rauch-Tung-Striebel (RTS) smoother** (W&H §4.5) runs a backward pass after
the filter to compute $p(\theta_t \mid y_{1:T})$ — using all observations, not just
those up to time $t$. The smoother is never worse than the filter.

**Backward recursion** (starting from $s_T = m_T$, $S_T = C_T$):

$$B_t = C_t G' R_{t+1}^{-1}$$
$$s_t = m_t + B_t(s_{t+1} - a_{t+1}), \quad S_t = C_t + B_t(S_{t+1} - R_{t+1})B_t'$$

The smoother mean $s_t$ uses future data to correct the filter. The smoother
covariance $S_t \leq C_t$ — always tighter than the filter.
```

**Cell 5 — code:**
```python
sr = rts_smoother(spec, fr)

std_s = np.sqrt(sr.S[:, 0, 0])
std_f = np.sqrt(fr.C[:, 0, 0])

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, sim.theta_true[:, 0], "k-", lw=1, label="true θ_t")
ax.plot(t, fr.m[:, 0], "C1-", lw=1.5, alpha=0.7, label="filtered mean")
ax.fill_between(t, fr.m[:, 0]-1.96*std_f, fr.m[:, 0]+1.96*std_f,
                alpha=0.15, color="C1")
ax.plot(t, sr.s[:, 0], "C2-", lw=2, label="smoothed mean")
ax.fill_between(t, sr.s[:, 0]-1.96*std_s, sr.s[:, 0]+1.96*std_s,
                alpha=0.25, color="C2", label="95% smoother interval")
ax.set(xlabel="t", ylabel="value", title="Filter vs smoother")
ax.legend()
plt.tight_layout()
plt.show()

# Verify: smoother at T equals filter at T
print(f"s[T-1] = {sr.s[-1, 0]:.6f}  (smoother)")
print(f"m[T-1] = {fr.m[-1, 0]:.6f}  (filter)")
print("Smoother mean std (avg):", np.sqrt(sr.S[:, 0, 0]).mean().round(4))
print("Filter   mean std (avg):", np.sqrt(fr.C[:, 0, 0]).mean().round(4))
```

**Cell 6 — markdown:**
```markdown
## 3. Multi-step forecast

The one-step-ahead predictive distribution at time $t$ is (W&H §4.4):

$$p(y_{t+1} \mid y_{1:t}) = N(F\, a_{t+1},\, Q_{t+1})$$

For $h$-step forecasting we apply the state equation $h$ times, propagating
uncertainty: each step adds $W$ to the state covariance and projects through $F$.
The uncertainty widens monotonically for a local level (random walk).
```

**Cell 7 — code:**
```python
fc = forecast_horizon(spec, fr, h=20)

t_fc = np.arange(100, 120)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, sim.y[:, 0], ".", ms=3, alpha=0.5, color="grey", label="obs (in-sample)")
ax.plot(t, fr.m[:, 0], "C1-", lw=1.5, label="filtered mean")
ax.plot(t_fc, fc.means[:, 0], "C3-", lw=2, label="forecast mean")
ax.fill_between(t_fc, fc.lower[:, 0], fc.upper[:, 0],
                alpha=0.3, color="C3", label="95% forecast interval")
ax.axvline(99.5, color="k", ls="--", lw=1)
ax.set(xlabel="t", ylabel="value", title="20-step forecast — local level")
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 8 — markdown:**
```markdown
## 4. Log marginal likelihood

The Kalman filter computes the **log marginal likelihood** as a by-product:

$$\log p(y_{1:T}) = \sum_{t=1}^T \log N(e_t;\, 0, Q_t)
= -\frac{T}{2}\log(2\pi) - \frac{1}{2}\sum_t \left[\log|Q_t| + e_t^\top Q_t^{-1} e_t\right]$$

This is the probability of the data under the model. It is the gold standard for
comparing models with different structures (no penalty term needed — complexity
is already penalised by integrating out $\theta_{1:T}$).
```

**Cell 9 — code:**
```python
# Compare log-likelihoods for different V values
V_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
logliks = []
for V_try in V_grid:
    sp = make_local_level(V=V_try, W_level=0.5)
    logliks.append(kalman_filter(sp, sim.y).loglik)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(V_grid, logliks, "o-", lw=2)
ax.axvline(2.0, color="r", ls="--", label="true V=2.0")
ax.set(xlabel="V (observation noise)", ylabel="log p(y | V, W=0.5)",
       title="Likelihood profile over V (true W=0.5 fixed)")
ax.legend()
plt.tight_layout()
plt.show()
print("Best V from grid:", V_grid[int(np.argmax(logliks))])
```

**Cell 10 — markdown:**
```markdown
## Exercises

**Exercise 1** — Fix W=0.5. Run the filter for V ∈ {0.5, 2.0, 8.0} and plot the
three filtered means on one axis alongside the true state. What does large V do
to the filter?

**Exercise 2** — Verify that `sr.s[-1, 0] == fr.m[-1, 0]` (within floating-point
tolerance). Explain in one sentence why this must be true from the RTS equations.

**Exercise 3** — Compute the log-likelihood for W ∈ {0.1, 0.5, 1.0, 2.0} with
V=2.0 fixed. Plot it. At which W is it maximized?
```

- [ ] **Step 2: Smoke-test**

```bash
pytest --nbmake notebooks/beginner/B2_local_level.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B2_local_level.ipynb
git commit -m "feat(beginner): add B2 local level notebook"
```

---

## Task 6: B3_local_linear_trend.ipynb

**Files:**
- Create: `notebooks/beginner/B3_local_linear_trend.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B3_local_linear_trend.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B3 — Local Linear Trend Model

**Reference:** West & Harrison §3.1; Petris §2.2

**Concepts introduced:**
- Two-dimensional state: (level, slope)
- Evolution matrix G encoding linear trend
- Comparing local level vs LLT on trending data
- LLT forecast fan: linear projection vs flat local level
```

**Cell 1 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from engine.models import make_local_level, make_local_linear_trend
from engine.filter import kalman_filter
from engine.smoother import rts_smoother
from engine.simulate import simulate
from engine.forecast import forecast_horizon
```

**Cell 2 — markdown:**
```markdown
## 1. State equations

The **local linear trend** (LLT) model has a 2D state $\theta_t = (\mu_t, \beta_t)'$
representing level and slope (W&H §3.1):

$$\mu_t = \mu_{t-1} + \beta_{t-1} + w_{1t}, \quad w_{1t} \sim N(0, W_\text{level})$$
$$\beta_t = \beta_{t-1} + w_{2t}, \quad w_{2t} \sim N(0, W_\text{slope})$$

Written as a DLM with:

$$G = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad
F = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
W = \begin{pmatrix} W_\text{level} & 0 \\ 0 & W_\text{slope} \end{pmatrix}$$

$F$ picks out the level component as the observation.
When $W_\text{slope} \to 0$, the slope becomes a fixed constant (no slope evolution).
```

**Cell 3 — code:**
```python
# Print the matrices to verify
spec_llt = make_local_linear_trend(V=1.0, W_level=0.1, W_slope=0.01)
print("F =\n", spec_llt.F)
print("G =\n", spec_llt.G)
print("W =\n", spec_llt.W)
print("State dim d =", spec_llt.G.shape[0])
```

**Cell 4 — code:**
```python
sim_llt = simulate(spec_llt, n=100, seed=3)
fr_llt  = kalman_filter(spec_llt, sim_llt.y)
sr_llt  = rts_smoother(spec_llt, fr_llt)

t = np.arange(100)
std_mu = np.sqrt(fr_llt.C[:, 0, 0])
std_bt = np.sqrt(fr_llt.C[:, 1, 1])

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# Level
axes[0].plot(t, sim_llt.theta_true[:, 0], "k-", lw=1, label="true μ_t")
axes[0].plot(t, sim_llt.y[:, 0], ".", ms=3, alpha=0.4, color="grey", label="obs")
axes[0].plot(t, fr_llt.m[:, 0], "C1-", lw=2, label="filtered level")
axes[0].fill_between(t, fr_llt.m[:, 0]-1.96*std_mu,
                        fr_llt.m[:, 0]+1.96*std_mu, alpha=0.2, color="C1")
axes[0].set_ylabel("level μ_t"); axes[0].legend()

# Slope
axes[1].plot(t, sim_llt.theta_true[:, 1], "k-", lw=1, label="true β_t")
axes[1].plot(t, fr_llt.m[:, 1], "C2-", lw=2, label="filtered slope")
axes[1].fill_between(t, fr_llt.m[:, 1]-1.96*std_bt,
                        fr_llt.m[:, 1]+1.96*std_bt, alpha=0.2, color="C2")
axes[1].set_xlabel("t"); axes[1].set_ylabel("slope β_t"); axes[1].legend()

plt.suptitle("Local linear trend — level and slope (filtered)", y=1.01)
plt.tight_layout()
plt.show()
```

**Cell 5 — markdown:**
```markdown
## 2. Local level vs LLT on trending data

The local level model assumes no systematic trend — it treats the state as a
random walk. On data with a persistent slope, local level's filtered mean lags
behind the true state because it can't anticipate the trend direction.
```

**Cell 6 — code:**
```python
# True slope is ~0.3 per step
spec_compare = make_local_linear_trend(V=1.0, W_level=0.05, W_slope=0.001)
sim_trend = simulate(spec_compare, n=100, seed=5)

fr_ll  = kalman_filter(make_local_level(V=1.0, W_level=0.1), sim_trend.y)
fr_llt2 = kalman_filter(spec_compare, sim_trend.y)

print(f"Local level  log-lik: {fr_ll.loglik:.2f}")
print(f"LLT          log-lik: {fr_llt2.loglik:.2f}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, sim_trend.theta_true[:, 0], "k-", lw=1.5, label="true level μ_t")
ax.plot(t, sim_trend.y[:, 0], ".", ms=3, alpha=0.3, color="grey")
ax.plot(t, fr_ll.m[:, 0], "C0-", lw=1.5, label="local level filter")
ax.plot(t, fr_llt2.m[:, 0], "C1-", lw=1.5, label="LLT filter")
ax.set(xlabel="t", ylabel="value",
       title="Local level vs LLT on trending data")
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 7 — markdown:**
```markdown
## 3. Forecast comparison

The LLT forecast is linear: the level grows by the estimated slope $\hat\beta_T$ at
each step. The local level forecast is flat (random-walk has zero expected increment).
```

**Cell 8 — code:**
```python
fc_ll   = forecast_horizon(make_local_level(V=1.0, W_level=0.1), fr_ll, h=20)
fc_llt2 = forecast_horizon(spec_compare, fr_llt2, h=20)

t_fc = np.arange(100, 120)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, sim_trend.y[:, 0], ".", ms=3, alpha=0.3, color="grey", label="in-sample obs")
ax.plot(t_fc, fc_ll.means[:, 0],  "C0-", lw=2, label="local level forecast")
ax.fill_between(t_fc, fc_ll.lower[:, 0], fc_ll.upper[:, 0], alpha=0.2, color="C0")
ax.plot(t_fc, fc_llt2.means[:, 0], "C1-", lw=2, label="LLT forecast")
ax.fill_between(t_fc, fc_llt2.lower[:, 0], fc_llt2.upper[:, 0], alpha=0.2, color="C1")
ax.axvline(99.5, color="k", ls="--", lw=1)
ax.set(xlabel="t", ylabel="value", title="20-step forecasts: local level vs LLT")
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 9 — markdown:**
```markdown
## Exercises

**Exercise 1** — Set `W_slope=0` in `make_local_linear_trend`. Filter the same
trending series. Does the slope converge? Print `fr.m[:, 1]` over time to see it
stabilise around the true slope.

**Exercise 2** — Increase `W_slope` from 0.01 to 1.0. How does the slope trace
`fr.m[:, 1]` change? Does it track the true slope more or less closely?

**Exercise 3** — Compute log-likelihoods for both models on `sim_trend`. Which wins?
Does that match the visual evidence?
```

- [ ] **Step 2: Smoke-test**

```bash
pytest --nbmake notebooks/beginner/B3_local_linear_trend.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B3_local_linear_trend.ipynb
git commit -m "feat(beginner): add B3 local linear trend notebook"
```

---

## Task 7: B4_seasonal_models.ipynb

**Files:**
- Create: `notebooks/beginner/B4_seasonal_models.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B4_seasonal_models.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B4 — Seasonal Models

**Reference:** West & Harrison §3.2; Petris §2.3

**Concepts introduced:**
- Seasonal component in dummy form (sum-to-zero constraint)
- Combining specs with `combine()` (block-diagonal system matrices)
- Decomposing an observed series into trend + seasonal components
- Effect of `W_season` on seasonal rigidity
```

**Cell 1 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from engine.models import make_local_linear_trend, make_seasonal_factor, combine
from engine.filter import kalman_filter
from engine.smoother import rts_smoother
from engine.simulate import simulate
```

**Cell 2 — markdown:**
```markdown
## 1. The problem: ignored seasonality

When a series has a periodic pattern and we fit a trend-only model, the residuals
show strong autocorrelation at the seasonal lag.
```

**Cell 3 — code:**
```python
# Simulate a series with monthly seasonality but fit trend-only
rng = np.random.default_rng(7)
T, s = 120, 12

# Build true seasonal pattern (sum-to-zero)
seasonal_pattern = np.array([3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2], dtype=float)
seasonal_pattern -= seasonal_pattern.mean()   # enforce mean zero

t_arr = np.arange(T)
true_level = 10.0 + 0.1 * t_arr
true_seasonal = np.tile(seasonal_pattern, T // s)
y_seasonal = true_level + true_seasonal + rng.normal(0, 0.5, T)

# Fit a trend-only LLT — ignores seasonality
spec_ll = make_local_linear_trend(V=1.0, W_level=0.1, W_slope=0.001)
fr_ll   = kalman_filter(spec_ll, y_seasonal[:, None])
residuals = y_seasonal - fr_ll.f[:, 0]

fig, axes = plt.subplots(2, 1, figsize=(11, 6))
axes[0].plot(t_arr, y_seasonal, lw=1, label="y_t (seasonal)")
axes[0].plot(t_arr, fr_ll.m[:, 0], "C1-", lw=2, label="LLT filter (no seasonal)")
axes[0].set_ylabel("value"); axes[0].legend()

# ACF of residuals
max_lag = 30
acf = [np.corrcoef(residuals[lag:], residuals[:-lag] if lag else residuals)[0, 1]
       for lag in range(max_lag + 1)]
axes[1].bar(range(max_lag + 1), acf, color="C0")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].axhline( 1.96/np.sqrt(T), color="r", ls="--", lw=1, label="±1.96/√T")
axes[1].axhline(-1.96/np.sqrt(T), color="r", ls="--", lw=1)
axes[1].set(xlabel="lag", ylabel="ACF", title="Residual ACF — spike at lag 12")
axes[1].legend()
plt.tight_layout()
plt.show()
```

**Cell 4 — markdown:**
```markdown
## 2. Dummy seasonal representation

For a period-$s$ seasonal pattern, we use the **dummy seasonal** form (W&H §3.2).
The state has dimension $s-1$ and encodes the last $s-1$ seasonal effects.
The current seasonal factor is their negated sum — enforcing the sum-to-zero constraint.

**Evolution matrix** (companion/rotation form):

$$J_s = \begin{pmatrix}
-1 & -1 & \cdots & -1 & -1 \\
 1 &  0 & \cdots &  0 &  0 \\
 0 &  1 & \cdots &  0 &  0 \\
\vdots & & \ddots & & \vdots \\
 0 &  0 & \cdots &  1 &  0
\end{pmatrix}$$

Only $W_\text{season}$ on the top-left of $W$ drives seasonal innovation;
the remaining entries are near-zero (numerical nugget).

**`make_seasonal_factor(period, V, W_season)`** builds this spec.
Use **`combine(spec1, spec2)`** to merge it with an LLT spec (block-diagonal $G$ and $W$,
combined $F$, $V$ from the first spec).
```

**Cell 5 — code:**
```python
# Build LLT + seasonal combined model
llt_spec  = make_local_linear_trend(V=0.25, W_level=0.05, W_slope=0.001)
seas_spec = make_seasonal_factor(period=12, V=0.25, W_season=0.01)
spec      = combine(llt_spec, seas_spec)

print(f"State dim d: {spec.G.shape[0]}  (2 LLT + 11 seasonal = 13)")
print(f"F shape: {spec.F.shape}")
print(f"G shape: {spec.G.shape}")
```

**Cell 6 — code:**
```python
sim  = simulate(spec, n=T, seed=8)
fr   = kalman_filter(spec, sim.y)
sr   = rts_smoother(spec, fr)

# Extract components from smoothed state
# State layout from combine(): [level, slope, seasonal_0, ..., seasonal_10]
smooth_level    = sr.s[:, 0]           # level μ_t
smooth_slope    = sr.s[:, 1]           # slope β_t
smooth_seasonal = sr.s[:, 2]           # leading seasonal factor γ_t

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axes[0].plot(t_arr, sim.y[:, 0], ".", ms=3, alpha=0.5, label="obs y_t")
axes[0].plot(t_arr, smooth_level + smooth_seasonal, "C1-", lw=2, label="level + seasonal (smoothed)")
axes[0].set_ylabel("y_t"); axes[0].legend()

axes[1].plot(t_arr, sim.theta_true[:, 0], "k--", lw=1, label="true level")
axes[1].plot(t_arr, smooth_level, "C1-", lw=2, label="smoothed level μ_t")
axes[1].set_ylabel("level"); axes[1].legend()

axes[2].plot(t_arr, sim.theta_true[:, 2], "k--", lw=1, label="true seasonal")
axes[2].plot(t_arr, smooth_seasonal, "C2-", lw=2, label="smoothed seasonal γ_t")
axes[2].set_xlabel("t"); axes[2].set_ylabel("seasonal"); axes[2].legend()

plt.suptitle("LLT + Seasonal decomposition (smoothed)", y=1.01)
plt.tight_layout()
plt.show()
print(f"Log marginal likelihood: {fr.loglik:.2f}")
```

**Cell 7 — markdown:**
```markdown
## 3. Effect of W_season

Large `W_season`: seasonal effects are allowed to drift substantially over time —
the pattern can change shape from year to year.

Small `W_season` (near zero): seasonal effects are nearly fixed — the pattern
repeats almost identically every cycle.
```

**Cell 8 — code:**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, W_s, title in zip(
    axes,
    [0.001, 0.5],
    ["W_season=0.001 (near-fixed)", "W_season=0.5 (drifting)"],
):
    llt_s = make_local_linear_trend(V=0.25, W_level=0.05, W_slope=0.001)
    sea_s = make_seasonal_factor(period=12, V=0.25, W_season=W_s)
    sp    = combine(llt_s, sea_s)
    fr_s  = kalman_filter(sp, sim.y)
    sr_s  = rts_smoother(sp, fr_s)
    ax.plot(t_arr, sr_s.s[:, 2], lw=1.5)
    ax.set_title(title); ax.set_xlabel("t"); ax.set_ylabel("seasonal γ_t")
plt.tight_layout()
plt.show()
```

**Cell 9 — markdown:**
```markdown
## Exercises

**Exercise 1** — Set `W_season=0` (use a very small value like 1e-9 to stay
numerically stable). Verify the seasonal component repeats exactly every 12 steps
by checking `np.allclose(sr.s[0, 2], sr.s[12, 2], atol=1e-3)`.

**Exercise 2** — Fit a monthly (`period=12`) model to a series that was generated
with quarterly seasonality (`period=4`). Compare log-likelihoods to show the
mis-specified model does worse:
```python
# Generate quarterly seasonal series
llt_q  = make_local_linear_trend(V=0.25, W_level=0.05, W_slope=0.001)
seas_q = make_seasonal_factor(period=4, V=0.25, W_season=0.01)
spec_q = combine(llt_q, seas_q)
sim_q  = simulate(spec_q, n=48, seed=99)
# Then fit both period=4 and period=12 and compare fr.loglik
```
```

- [ ] **Step 2: Smoke-test**

```bash
pytest --nbmake notebooks/beginner/B4_seasonal_models.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B4_seasonal_models.ipynb
git commit -m "feat(beginner): add B4 seasonal models notebook"
```

---

## Task 8: B5_parameter_estimation.ipynb

**Files:**
- Create: `notebooks/beginner/B5_parameter_estimation.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B5_parameter_estimation.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B5 — Parameter Estimation

**Reference:** West & Harrison §4.4, §10; Petris §3

**Concepts introduced:**
- V and W are rarely known in practice — we must estimate them from data
- MLE via `scipy.optimize.minimize` on the Kalman filter log-likelihood
- 2D likelihood surface — the ridge shaped by identifiability
- Bayesian estimation with PyMC (blackbox likelihood via `as_op`)
- Comparing MLE point estimate to posterior uncertainty
```

**Cell 1 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from engine.models import make_local_level
from engine.filter import kalman_filter
from engine.simulate import simulate
```

**Cell 2 — markdown:**
```markdown
## 1. The estimation problem

In the local level model, $V$ and $W$ are unknown. We need to estimate them.

**Two approaches:**

| Approach | Tool | Returns |
|----------|------|---------|
| **MLE** | `scipy.optimize.minimize` | point estimate $(\hat V, \hat W)$ |
| **Bayesian** | PyMC + Kalman blackbox likelihood | full posterior $p(V, W \mid y)$ |

Both use `fr.loglik` from the Kalman filter as the objective / likelihood.
```

**Cell 3 — code:**
```python
# Simulate data
V_true, W_true = 2.0, 0.5
spec_true = make_local_level(V=V_true, W_level=W_true)
sim = simulate(spec_true, n=100, seed=0)
y = sim.y    # (100, 1)
```

**Cell 4 — markdown:**
```markdown
## 2. MLE via scipy.optimize

We maximise $\log p(y \mid V, W)$ by minimising its negation.

**Log-space parameterisation:** we optimise $\log V$ and $\log W$ instead of
$V$ and $W$ directly. This enforces positivity and improves numerical conditioning —
$\log V$ ranges over all of $\mathbb R$ while $V > 0$.
```

**Cell 5 — code:**
```python
def neg_loglik(params: np.ndarray) -> float:
    V_opt, W_opt = np.exp(params)
    spec = make_local_level(V=V_opt, W_level=W_opt)
    return -kalman_filter(spec, y).loglik

result = minimize(neg_loglik, x0=np.array([0.0, 0.0]), method="Nelder-Mead",
                  options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
V_mle, W_mle = np.exp(result.x)

print(f"Converged: {result.success}")
print(f"V_mle = {V_mle:.4f}  (true: {V_true})")
print(f"W_mle = {W_mle:.4f}  (true: {W_true})")
print(f"κ_mle = W/V = {W_mle/V_mle:.4f}  (true: {W_true/V_true:.4f})")
```

**Cell 6 — markdown:**
```markdown
## 3. Likelihood surface

The 2D likelihood surface $\log p(y \mid V, W)$ has a characteristic **ridge**
shape: many (V, W) pairs with the same ratio $W/V$ give nearly identical likelihoods.
This reflects partial non-identifiability — only the ratio is well-determined from
a single series of modest length.
```

**Cell 7 — code:**
```python
V_grid = np.exp(np.linspace(-1, 2.5, 50))
W_grid = np.exp(np.linspace(-2.5, 1, 50))
LL = np.zeros((len(V_grid), len(W_grid)))

for i, V_try in enumerate(V_grid):
    for j, W_try in enumerate(W_grid):
        sp = make_local_level(V=V_try, W_level=W_try)
        LL[i, j] = kalman_filter(sp, y).loglik

fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(np.log(W_grid), np.log(V_grid), LL, levels=40, cmap="viridis")
fig.colorbar(cf, ax=ax, label="log p(y | V, W)")
ax.plot(np.log(W_mle), np.log(V_mle), "r*", ms=12, label="MLE")
ax.plot(np.log(W_true), np.log(V_true), "w+", ms=12, mew=2, label="true")
ax.set(xlabel="log W", ylabel="log V", title="Log-likelihood surface")
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 8 — markdown:**
```markdown
## 4. Bayesian estimation with PyMC

The Kalman filter is pure NumPy — PyTensor (PyMC's backend) cannot differentiate
through it. We register the log-likelihood function as a **blackbox op** via
`pytensor.compile.ops.as_op`. This tells PyMC to use the **Slice sampler**
(gradient-free) instead of NUTS (requires gradients).

**Note:** The Slice sampler is less efficient than NUTS for smooth posteriors but
works correctly here. For production use, consider implementing the filter in
PyTensor for NUTS compatibility.
```

**Cell 9 — code (PyMC guard):**
```python
HAS_PYMC = False
try:
    import pymc as pm
    import pytensor.tensor as pt
    import pytensor.compile.ops as ops
    import arviz as az
    HAS_PYMC = True
except ImportError:
    print("PyMC not installed — skipping Bayesian estimation cells")

if HAS_PYMC:
    @ops.as_op(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
    def kalman_loglik_op(V_val: np.ndarray, W_val: np.ndarray) -> np.ndarray:
        spec = make_local_level(V=float(V_val), W_level=float(W_val))
        return np.array(kalman_filter(spec, y).loglik)

    with pm.Model() as dlm_model:
        V_rv = pm.HalfNormal("V", sigma=3.0)
        W_rv = pm.HalfNormal("W", sigma=1.0)
        _    = pm.Potential("loglik", kalman_loglik_op(V_rv, W_rv))
        idata = pm.sample(500, tune=500, progressbar=False, random_seed=42)

    print(az.summary(idata, var_names=["V", "W"]))
```

**Cell 10 — code (PyMC guard):**
```python
if HAS_PYMC:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, var, true_val, mle_val in zip(
        axes,
        ["V", "W"],
        [V_true, W_true],
        [V_mle, W_mle],
    ):
        samples = idata.posterior[var].values.ravel()
        ax.hist(samples, bins=50, density=True, alpha=0.6, color="C1", label="Posterior")
        ax.axvline(true_val, color="k", ls="-",  lw=2, label=f"True {var}={true_val}")
        ax.axvline(mle_val,  color="r", ls="--", lw=2, label=f"MLE {var}={mle_val:.2f}")
        ax.set(xlabel=var, ylabel="density", title=f"Posterior vs MLE: {var}")
        ax.legend()
    plt.tight_layout()
    plt.show()
```

**Cell 11 — markdown:**
```markdown
## Exercises

**Exercise 1** — Simulate a short series (T=20) and a long series (T=200) with
the same V=2.0, W=0.5. Run MLE on both. How does estimation error change with T?

**Exercise 2** — Replace `HalfNormal` priors with `pm.Exponential("V", lam=0.5)`
and `pm.Exponential("W", lam=1.0)`. Re-run. How does the posterior change, especially
for the short T=20 series?

**Exercise 3** — Plot the likelihood profile over V with W fixed at W_mle
(a 1D slice of the surface). Mark the MLE and the posterior mean. Are they close?
```

- [ ] **Step 2: Smoke-test**

```bash
pytest --nbmake notebooks/beginner/B5_parameter_estimation.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B5_parameter_estimation.ipynb
git commit -m "feat(beginner): add B5 parameter estimation notebook"
```

---

## Task 9: B6_dlm_glm_connection.ipynb

**Files:**
- Create: `notebooks/beginner/B6_dlm_glm_connection.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/beginner/B6_dlm_glm_connection.ipynb` with these cells:

**Cell 0 — markdown:**
```markdown
# B6 — DLM as a Generalization of Regression *(optional)*

**Reference:** West & Harrison §2.1; Petris §1.3

**Concepts introduced:**
- Static regression as a degenerate DLM (G=I, W=0)
- Kalman filter as recursive least squares (RLS)
- Dynamic regression: time-varying coefficients via W > 0
- When to use DLM vs ordinary regression

**Prerequisite:** you should have completed B5 before this notebook.
This notebook previews `engine.filter.kalman_filter_tv` and `engine.models.DLMSpecTV`
— both used in depth in intermediate notebook 03.
```

**Cell 1 — code:**
```python
import sys
from pathlib import Path
project_root = Path().resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from engine.filter import kalman_filter_tv
```

**Cell 2 — markdown:**
```markdown
## 1. Static regression as a DLM

Standard OLS regression:

$$y_t = x_t' \beta + \varepsilon_t, \quad \varepsilon_t \sim N(0, V)$$

This is a DLM with:
- State $\theta_t = \beta$ (constant coefficients)
- $F_t = x_t'$ (time-varying observation matrix — one row of predictors)
- $G = I$ (coefficients don't evolve: $\beta_t = \beta_{t-1}$)
- $W = 0$ (no state noise — coefficients are truly fixed)

With $W=0$ and a diffuse prior ($C_0 = \sigma^2 (X'X)^{-1}$ as $\sigma^2 \to \infty$),
the Kalman filter **is** recursive least squares (RLS): each new observation $y_t$
updates $\hat\beta$ exactly as adding one row to the OLS normal equations.
```

**Cell 3 — code:**
```python
rng = np.random.default_rng(42)
T   = 80
x1  = rng.normal(0, 1, T)
x2  = rng.normal(0, 1, T)
beta_true = np.array([1.5, -0.8])
V_true = 0.5
y = x1*beta_true[0] + x2*beta_true[1] + rng.normal(0, np.sqrt(V_true), T)

# OLS
X = np.column_stack([x1, x2])
beta_ols, _, _, _ = lstsq(X, y, rcond=None)
print(f"OLS: β₁={beta_ols[0]:.4f}, β₂={beta_ols[1]:.4f}  (true: {beta_true})")

# DLM with W=0 (static, fixed coefficients)
F_seq = np.stack([x1, x2], axis=1)[:, None, :]   # shape (T, 1, 2)
G = np.eye(2)
V_mat = np.array([[V_true]])
W_mat = 1e-10 * np.eye(2)    # near-zero: coefficients nearly fixed
m0 = np.zeros(2)
C0 = 1e6 * np.eye(2)         # diffuse prior

fr_static = kalman_filter_tv(F_seq, G, V_mat, W_mat, m0, C0, y[:, None])
beta_dlm_final = fr_static.m[-1]   # final filtered coefficients = RLS estimate
print(f"DLM (W≈0): β₁={beta_dlm_final[0]:.4f}, β₂={beta_dlm_final[1]:.4f}")
print(f"Max diff from OLS: {np.abs(beta_dlm_final - beta_ols).max():.6f}")
```

**Cell 4 — markdown:**
```markdown
## 2. Dynamic regression: time-varying coefficients

Now allow $W > 0$ so coefficients can drift. The model becomes:

$$y_t = x_t' \beta_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, V)$$
$$\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim N(0, W)$$

Each coefficient evolves as a random walk. The Kalman filter tracks the most
likely coefficient value at each time step — a useful tool for detecting structural
breaks or gradual coefficient drift.
```

**Cell 5 — code:**
```python
# Simulate data with a structural break at t=40: β₁ flips sign
beta_tv = np.zeros((T, 2))
beta_tv[:40]  = [1.5, -0.8]
beta_tv[40:]  = [-1.0, -0.8]   # β₁ reverses at t=40
y_tv = (X * beta_tv).sum(axis=1) + rng.normal(0, np.sqrt(V_true), T)

# Static OLS (cannot detect break)
beta_ols_tv, _, _, _ = lstsq(X, y_tv, rcond=None)

# Dynamic regression (W > 0)
W_dyn = 0.2 * np.eye(2)
fr_dyn = kalman_filter_tv(F_seq, G, V_mat, W_dyn, m0, C0, y_tv[:, None])

t_arr = np.arange(T)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

for i, (ax, name) in enumerate(zip(axes, ["β₁ (reverses at t=40)", "β₂ (fixed at -0.8)"])):
    ax.plot(t_arr, beta_tv[:, i], "k-", lw=1.5, label=f"true {name}")
    ax.axhline(beta_ols_tv[i], color="C0", ls="--", lw=1.5, label="OLS (static)")
    ax.plot(t_arr, fr_dyn.m[:, i], "C1-", lw=2, label="dynamic filter")
    std_i = np.sqrt(fr_dyn.C[:, i, i])
    ax.fill_between(t_arr, fr_dyn.m[:, i]-1.96*std_i,
                           fr_dyn.m[:, i]+1.96*std_i, alpha=0.2, color="C1")
    if i == 0:
        ax.axvline(40, color="grey", ls=":", lw=1.5, label="break at t=40")
    ax.set_ylabel(f"β{i+1}"); ax.legend(fontsize=8)

axes[-1].set_xlabel("t")
plt.suptitle("Static OLS vs dynamic regression with structural break")
plt.tight_layout()
plt.show()
```

**Cell 6 — markdown:**
```markdown
## 3. When to use DLM vs regression

| Situation | Preferred approach |
|-----------|-------------------|
| Coefficients truly fixed, errors IID | OLS / GLM — simpler, no tuning |
| Coefficients change slowly over time | Dynamic regression (DLM, W small) |
| Structural break at unknown time | Dynamic regression detects it automatically |
| Latent state driving observations | DLM (local level, LLT, seasonal) |
| Need exact posterior over state history | DLM + RTS smoother |

**Connection to intermediate notebook 03:** `03_dynamic_regression.ipynb` covers
`DLMSpecTV` and `kalman_filter_tv` in depth, including model comparison and
diagnostics for the time-varying coefficient model.
```

**Cell 7 — markdown:**
```markdown
## Exercises

**Exercise 1** — Static as DLM: fit the static regression via DLM (W=1e-10) on
the original `y` (no break). Verify that `fr_static.m[-1]` matches `beta_ols`
to at least 3 decimal places.

**Exercise 2** — Structural break detection: on the break dataset, try `W_dyn` values
{0.01, 0.2, 2.0} and observe how quickly the filter detects the break at t=40.
Which W detects it fastest? What is the trade-off?
```

- [ ] **Step 2: Smoke-test**

```bash
pytest --nbmake notebooks/beginner/B6_dlm_glm_connection.ipynb -q
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add notebooks/beginner/B6_dlm_glm_connection.ipynb
git commit -m "feat(beginner): add B6 DLM-GLM connection notebook (optional)"
```

---

## Task 10: Full series smoke test + CI verification

**Files:**
- Read: `.github/workflows/ci.yml` (verify Task 1 change is present)

- [ ] **Step 1: Run the full beginner series through nbmake**

```bash
pytest --nbmake notebooks/beginner/ -q
```

Expected output (order may vary):
```
8 passed in Xs
```

If any notebook fails, read the traceback carefully. Common issues:
- Wrong path setup: check `Path().resolve().parents[1]` gives the repo root
- Wrong argument name: verify `make_local_linear_trend(V=..., W_level=..., W_slope=...)`
- Wrong function name: `forecast_horizon` not `forecast`

- [ ] **Step 2: Run the combined suite to check for regressions**

```bash
pytest --nbmake notebooks/beginner/ notebooks/intermediate/ notebooks/advanced/ -q
```

Expected: all notebooks pass (previously 11 notebooks + 8 new = 19 total)

- [ ] **Step 3: Run unit tests to confirm no engine regressions**

```bash
pytest -q
```

Expected: all existing tests pass (no new unit tests added — no new engine functions)

- [ ] **Step 4: Run ruff and mypy**

```bash
ruff check .
mypy engine lessons
```

Expected: no errors (notebooks are not type-checked)

- [ ] **Step 5: Commit if any fixes were needed**

```bash
git add -p   # stage only intentional fixes
git commit -m "fix(beginner): fix smoke-test failures in beginner series"
```

If Step 1–4 all pass cleanly, no commit needed here.

---

## Self-review notes

**Spec coverage check:**

| Spec requirement | Covered by |
|------------------|-----------|
| 00_setup: env check, notation, reading guide | Task 2 |
| B0: Beta-Binomial, Gaussian mean, MCMC diagnostics, PyMC guard | Task 3 |
| B1: DLM equations, manual sim, predict-update, engine intro | Task 4 |
| B2: filter, smoother, forecast, log-likelihood | Task 5 |
| B3: LLT state, compare to local level, forecast fan | Task 6 |
| B4: dummy seasonal, combine(), decomposition, W_season effect | Task 7 |
| B5: MLE, likelihood surface, PyMC blackbox | Task 8 |
| B6: static regression = DLM, dynamic regression, DLMSpecTV | Task 9 |
| CI: notebooks/beginner/ in nbmake step | Task 1 |
| No new engine functions | All tasks — confirmed |
| PyMC guard in B0 and B5 only | Tasks 3, 8 |

**Type consistency:** all tasks use `make_local_linear_trend`, `make_seasonal_factor`,
`combine`, `forecast_horizon`, `rts_smoother` — consistent with engine signatures verified above.
