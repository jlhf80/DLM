# DLM Beginner Tutorial — Notebook Design Spec

**Date:** 2026-04-30
**Status:** Approved — ready for implementation planning
**Scope:** Full beginner notebook series, 8 notebooks (6 core + 2 optional)
**Delivery:** Jupyter notebook series in `notebooks/beginner/`
**Target reader:** Python-fluent reader with basic probability/stats background; may or may not have done Bayesian modeling; no prior state space model experience; skips the Streamlit app entirely
**Goal:** Reader completes the series ready to open `notebooks/intermediate/` without any prerequisite gaps

---

## 1. Overview

A self-contained beginner notebook series that teaches Dynamic Linear Models from
first principles using the same `engine/` package as the intermediate and advanced
series. No new engine functions are introduced — the series teaches what is already
there. PyMC is used in the Bayesian primer and parameter estimation notebooks.

The series is designed so a reader who knows Python, numpy, and basic probability
can start at `B0` (or skip it if they already know PyMC) and arrive at `B5` ready
for the intermediate series without gaps. `B6` is an optional conceptual bridge for
readers with a regression background.

**Primary references:**
- West, M. & Harrison, J. — *Bayesian Forecasting and Dynamic Models* (2nd ed.) — cited as **W&H**
- Petris, G., Petrone, S. & Campagnoli, P. — *Dynamic Linear Models with R* — cited as **Petris**

---

## 2. Notebook list

```
notebooks/beginner/
    00_setup.ipynb
    B0_bayesian_primer.ipynb        ← optional: skip if you know PyMC
    B1_dlm_intro.ipynb
    B2_local_level.ipynb
    B3_local_linear_trend.ipynb
    B4_seasonal_models.ipynb
    B5_parameter_estimation.ipynb
    B6_dlm_glm_connection.ipynb     ← optional: DLM as a generalization of GLM
```

The `B`-prefix distinguishes beginner notebooks from the intermediate (`01`–`05`)
and advanced (`06`–`10`) numbering so all three series can coexist in `notebooks/`.

---

## 3. Notebook descriptions

### 00_setup.ipynb

Environment verification and orientation. Matches the style of
`notebooks/intermediate/00_setup.ipynb`.

**Contents:**
- Python/package version check (`numpy`, `scipy`, `matplotlib`, `pymc`, `arviz`)
- Project path setup (add repo root to `sys.path`)
- Import smoke test for all engine modules
- Full notation table: F, G, V, W, θ, y, m, C, R, a, Q, e, A (W&H notation)
- Reading guide: which notebook to start from based on prior background

---

### B0_bayesian_primer.ipynb *(optional)*

Standalone introduction to Bayesian modeling with PyMC. Skippable for readers
who already use PyMC.

**Contents:**
1. Bayes' theorem: prior × likelihood ∝ posterior — stated and visualised with a
   Beta-Binomial coin-flip example. Grid approximation first, then PyMC.
2. Gaussian mean estimation: known variance, unknown mean. Prior: `Normal(0, 5)`,
   likelihood: `Normal(μ, 1)`. Derive the conjugate posterior analytically, then
   verify with PyMC `pm.sample`.
3. MCMC intuition: what the sampler is doing, how to read a trace plot, what
   `r_hat` and `ess_bulk` mean. Use `az.plot_trace` and `az.summary`.
4. PyMC workflow template: model block → sample → `az.summary` → interpret.

**PyMC pattern introduced here:**
```python
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=5)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)
    idata = pm.sample(1000, tune=1000, progressbar=False, random_seed=42)
az.summary(idata)
```

**Exercises:**
1. Change the prior to `Normal(10, 1)` and observe how a strong prior pulls the
   posterior away from the MLE with small N.
2. Double the sample size and compare posterior width to the conjugate formula
   `σ²_post = (1/σ²_prior + N/σ²_lik)⁻¹`.

---

### B1_dlm_intro.ipynb

**Reference:** W&H §1–2; Petris §1

Introduces the state space model and motivates the Kalman filter before any
code. Simulates a local level by hand (pure numpy, no engine) to make the
data-generating process concrete.

**Contents:**
1. The two-equation model (W&H §2.2):

   $$y_t = F_t \theta_t + v_t, \quad v_t \sim N(0, V_t)$$
   $$\theta_t = G_t \theta_{t-1} + w_t, \quad w_t \sim N(0, W_t)$$

   Meaning of each symbol. Why we can't observe θ directly.

2. Manual simulation of a local level (F=1, G=1, V=2, W=0.5):
   ```python
   np.random.seed(0)
   T = 60
   theta = np.zeros(T); y = np.zeros(T)
   theta[0] = rng.normal(0, 1)
   for t in range(1, T):
       theta[t] = theta[t-1] + rng.normal(0, np.sqrt(0.5))
       y[t] = theta[t] + rng.normal(0, np.sqrt(2.0))
   ```
   Plot θ and y together. Discuss why y is noisier.

3. The filtering problem stated: given y_{1:t}, what is p(θ_t | y_{1:t})?

4. Predict–update cycle (W&H §2.3) — written out in words, then as equations:

   **Predict:**
   $$a_t = G m_{t-1}, \quad R_t = G C_{t-1} G' + W$$

   **Update:**
   $$e_t = y_t - F a_t, \quad Q_t = F R_t F' + V$$
   $$A_t = R_t F' Q_t^{-1}$$
   $$m_t = a_t + A_t e_t, \quad C_t = R_t - A_t Q_t A_t'$$

5. One manual filter step traced through numerically (t=1 only), then hand the
   rest to the engine:
   ```python
   from engine.filter import kalman_filter
   from engine.models import make_local_level
   from engine.simulate import simulate
   ```

**Exercises:**
1. Increase W from 0.5 to 5.0 and re-run. What happens to filtered uncertainty?
2. Trace the predict step manually for t=2 using the m₁ and C₁ you computed in
   the notebook.

---

### B2_local_level.ipynb

**Reference:** W&H §2.3, §4.3; Petris §2.1

Full Kalman filter and smoother walkthrough on the local level model.

**Contents:**
1. Local level model as a special case: F=[[1]], G=[[1]], V scalar, W scalar.
   W&H §4.3. Signal-to-noise ratio κ = W/V and its effect on filtering.

2. Kalman filter pass — plot all three:
   - Prior means a_t (one-step-ahead predictions)
   - Filtered means m_t with 95% bands (±1.96 √C_t)
   - True state θ_t

3. The smoother — RTS equations (W&H §4.5):

   $$s_t = m_t + B_t(s_{t+1} - a_{t+1}), \quad S_t = C_t - B_t(R_{t+1} - S_{t+1})B_t'$$

   where $B_t = C_t G' R_{t+1}^{-1}$.

   Plot smoother means alongside filtered means. Explain why the smoother is
   narrower — it uses future information.

4. One-step-ahead forecast: $p(y_{t+1} | y_{1:t}) = N(f a_{t+1},\, Q_{t+1})$.
   Show a 12-step forecast with widening bands.
   ```python
   from engine.forecast import forecast_horizon
   fc = forecast_horizon(spec, fr, h=12)
   ```

5. Log marginal likelihood: $\log p(y_{1:T}) = \sum_t \log N(e_t; 0, Q_t)$.
   Compute it and explain its role in model comparison.

**Exercises:**
1. Fix W=0.5. Vary V ∈ {0.5, 2.0, 8.0} and plot the three filtered means on
   one axis. What does a large V do to the filter?
2. Show that the smoother mean at t=T equals the filtered mean at t=T.

---

### B3_local_linear_trend.ipynb

**Reference:** W&H §3.1; Petris §2.2

Extends the state to two dimensions: (level, slope).

**Contents:**
1. State vector: $\theta_t = (\mu_t, \beta_t)'$. Evolution matrices:

   $$G = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad
   F = \begin{pmatrix} 1 & 0 \end{pmatrix}$$

   W&H §3.1. Interpret: μ_t = μ_{t-1} + β_{t-1} + w_{1t}, β_t = β_{t-1} + w_{2t}.

2. Block-diagonal W for LLT:
   $$W = \begin{pmatrix} \sigma^2_\mu & 0 \\ 0 & \sigma^2_\beta \end{pmatrix}$$

   What happens when σ²_β → 0? (Slope becomes a fixed parameter.)

3. Simulate LLT, filter, smooth, plot level and slope separately:
   ```python
   from engine.models import make_local_linear_trend
   spec = make_local_linear_trend(V=1.0, W_level=0.1, W_slope=0.01)
   sim = simulate(spec, n=100, seed=1)
   fr = kalman_filter(spec, sim.y)
   sr = rts_smoother(spec, fr)
   ```

4. Compare local level vs LLT on the same trending dataset. Show that local
   level under-estimates the trend (biased filtered mean), while LLT tracks it.

5. Forecast: show that the LLT forecast fan is linear (growing by β_T per step)
   vs. flat for local level.

**Exercises:**
1. Set W_slope=0 and show the slope converges to a constant. What is it?
2. Increase W_slope from 0.01 to 1.0. How does the slope trace change?

---

### B4_seasonal_models.ipynb

**Reference:** W&H §3.2; Petris §2.3

Adds a periodic component to the state vector using the dummy seasonal
representation.

**Contents:**
1. Problem motivation: what residual autocorrelation looks like when seasonality
   is ignored. ACF plot showing period-s spike.

2. Dummy seasonal state (W&H §3.2): s seasons, state dimension s-1.
   Constraint: seasonal effects sum to zero over one period.
   Evolution matrix J_s (companion/rotation form).

3. LLT + seasonal combined model state:
   $$\theta_t = (\mu_t, \beta_t, \gamma_t, \gamma_{t-1}, \ldots, \gamma_{t-s+2})'$$

   F selects the level and first seasonal component:
   $$F = (1, 0, 1, 0, \ldots, 0)$$

4. Simulate, filter, and decompose using `combine()` to merge LLT and seasonal specs:
   ```python
   from engine.models import make_local_linear_trend, make_seasonal_factor, combine
   llt  = make_local_linear_trend(V=1.0, W_level=0.1, W_slope=0.01)
   seas = make_seasonal_factor(period=12, V=1.0, W_season=0.05)
   spec = combine(llt, seas)
   sim = simulate(spec, n=120, seed=2)
   fr  = kalman_filter(spec, sim.y)
   sr  = rts_smoother(spec, fr)
   ```
   Plot: observed y, trend component (μ_t), seasonal component (γ_t) — three
   panels stacked.

5. Effect of W_seasonal: large → seasons allowed to drift; small → near-fixed
   seasonal pattern.

**Exercises:**
1. Set W_seasonal=0 and verify the seasonal pattern is exactly periodic.
2. Fit a monthly (s=12) model to a synthetic quarterly (s=4) series. Show the
   mis-specified model has a higher AIC than the correct s=4 specification
   (use log-likelihood from `fr.loglik`).

---

### B5_parameter_estimation.ipynb

**Reference:** W&H §4.4, §10; Petris §3

Estimates V and W from data using two approaches: MLE via `scipy.optimize` and
Bayesian posterior via PyMC.

**Contents:**
1. Motivation: V and W are rarely known in practice. How do we choose them?

2. MLE — maximize log p(y | V, W) over V, W > 0:
   ```python
   from scipy.optimize import minimize

   def neg_loglik(params):
       V, W = np.exp(params)   # log-space to enforce positivity
       spec = make_local_level(V=V, W_level=W)
       return -kalman_filter(spec, y).loglik

   result = minimize(neg_loglik, x0=[0.0, 0.0], method="Nelder-Mead")
   V_mle, W_mle = np.exp(result.x)
   ```
   Discuss: log-space parameterisation, why Nelder-Mead, convergence check.

3. Likelihood surface — 2D contour plot of log p(y | V, W) over a grid.
   Mark the MLE. Observe the ridge (only ratio V/W is well-identified from
   short series).

4. Bayesian estimation with PyMC (blackbox likelihood pattern from notebook 07):
   ```python
   import pymc as pm
   import pytensor.compile.ops as ops
   import pytensor.tensor as pt

   @ops.as_op(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
   def kalman_loglik_op(V_val, W_val):
       spec = make_local_level(V=float(V_val), W_level=float(W_val))
       return np.array(kalman_filter(spec, y).loglik)

   with pm.Model() as m:
       V_rv = pm.HalfNormal("V", sigma=3.0)
       W_rv = pm.HalfNormal("W", sigma=1.0)
       pm.Potential("ll", kalman_loglik_op(V_rv, W_rv))
       idata = pm.sample(500, tune=500, progressbar=False, random_seed=42)
   ```

5. Compare MLE point estimate vs posterior mean and 94% HDI. Show the posterior
   captures the ridge-shaped uncertainty that a point estimate misses.

**Exercises:**
1. Fit V and W via MLE on a simulated series of length T=20. Repeat for T=200.
   How does estimation uncertainty change?
2. Replace `HalfNormal` with `Exponential(lam=1)` priors and re-sample. Compare
   posteriors.

---

### B6_dlm_glm_connection.ipynb *(optional)*

**Reference:** W&H §2.1; Petris §1.3

Shows how GLM/regression is a special case of a DLM, and how relaxing the
static-coefficient assumption leads naturally to dynamic regression.

**Contents:**
1. Static regression as a DLM: $y_t = x_t' \beta + \epsilon_t$.
   Set G=I (no evolution), W=0 (coefficients fixed), F_t=x_t'.
   Show that the Kalman filter reduces to recursive least squares (RLS).

2. Dynamic regression: allow W > 0 so β_t can drift. F_t=x_t' still, but now
   the coefficient is a random walk. The engine handles time-varying F via
   `DLMSpecTV` and `kalman_filter_tv`:
   ```python
   from engine.models import DLMSpecTV
   from engine.filter import kalman_filter_tv
   F_seq = x.reshape(T, 1, 1)   # shape (T, 1, d)
   spec_tv = DLMSpecTV(G=np.eye(1), V=np.array([[V]]), W=W_mat * np.eye(1),
                       m0=np.zeros(1), C0=10.0 * np.eye(1))
   fr = kalman_filter_tv(F_seq, spec_tv.G, spec_tv.V, spec_tv.W,
                         spec_tv.m0, spec_tv.C0, y.reshape(T, 1))
   ```
   (Covered in full in intermediate notebook 03 — this notebook previews it.)

3. Connection to WLS / GLS: the Kalman gain at each step is equivalent to a
   weighted regression weight, with the filter covariance playing the role of
   the precision matrix.

4. When to use DLM vs GLM: static coefficients with IID errors → GLM suffices.
   Time-varying coefficients, correlated errors, latent state → DLM.

**Exercises:**
1. Fit a static regression via both OLS and the DLM (W=0). Verify the
   coefficient estimates agree.
2. Introduce a structural break (β changes at t=50). Show OLS misses it;
   dynamic regression tracks it.

---

## 4. Depth and style

All notebooks follow the intermediate series conventions:

- **Opening header:** title, W&H/Petris reference, concepts introduced
- **Equations:** key results stated and briefly derived — full equation chains for
  the Kalman filter steps, shorter justifications elsewhere. No derivations skipped,
  none exhaustive.
- **Citations:** W&H section number at every major result.
- **Plots:** at every meaningful step — simulated data, filtered means with 95%
  uncertainty bands, decomposed components.
- **Exercises:** 2–3 per notebook. At least one traces an equation manually; at
  least one runs a code variant.
- **PyMC:** used only in B0 and B5 (B6 is engine-only). Guard pattern:
  ```python
  HAS_PYMC = False
  try:
      import pymc as pm
      ...
      HAS_PYMC = True
  except ImportError:
      print("PyMC not installed — skipping")
  ```

---

## 5. Engine usage

No new engine functions are introduced. The beginner series uses:

| Module | Used in |
|--------|---------|
| `engine.models` | B1–B5 |
| `engine.simulate` | B1–B5 |
| `engine.filter` | B1–B5 |
| `engine.smoother` | B2–B4 |
| `engine.forecast` | B2–B3 |

`B6` also uses `engine.models.DLMSpecTV` and `engine.filter.kalman_filter_tv`
as a preview of the dynamic regression pattern from intermediate notebook 03.

---

## 6. Testing & CI

No new unit tests (no new engine functions). The nbmake CI step is extended to
include `notebooks/beginner/`:

```yaml
pytest --nbmake notebooks/beginner/ notebooks/intermediate/ notebooks/advanced/ -q
```

PyMC is already in `[project.optional-dependencies] dev`, so B0 and B5
run fully in CI under the `HAS_PYMC` guard. B6 uses only the engine and
runs unconditionally.
