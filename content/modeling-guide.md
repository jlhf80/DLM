# DLM Modeling Guide

This reference accompanies the interactive tutorial. It covers the DLM framework, the baseline modeling procedure, variance tuning, and diagnostics. The five sections below map directly to the anchors that workflow steps deep-link to.

- [The DLM framework](#the-dlm-framework)
- [Baseline procedure](#baseline-procedure)
- [Tuning](#tuning)
- [Diagnostics](#diagnostics)
- [Worked example](#worked-example)

---

## The DLM framework

A **Dynamic Linear Model** is a pair of linear Gaussian equations that together define a state-space model for a time series.

Observation equation:

$$y_t = F_t\, \theta_t + v_t, \qquad v_t \sim N(0, V_t)$$

State equation:

$$\theta_t = G_t\, \theta_{t-1} + w_t, \qquad w_t \sim N(0, W_t)$$

- $y_t \in \mathbb{R}^p$ is the observation at time $t$ ($p = 1$ for all Beginner lessons)
- $\theta_t \in \mathbb{R}^d$ is the unobserved state
- $F_t$ ($p \times d$) maps the state into the observation space
- $G_t$ ($d \times d$) evolves the state one step forward
- $V_t$ ($p \times p$), $W_t$ ($d \times d$) are noise covariances
- Prior: $\theta_0 \sim N(m_0, C_0)$

For the Beginner tier we restrict to **time-invariant** DLMs: $F_t, G_t, V_t, W_t$ do not depend on $t$. Notation follows West & Harrison (1997) and cross-references Petris, Petrone & Campagnoli (2009) §2.3.

Three canonical components you will meet in the lessons:

| Component | F | G | State dim |
|---|---|---|---|
| Local level | `[1]` | `[1]` | 1 |
| Local linear trend | `[1, 0]` | `[[1, 1], [0, 1]]` | 2 |
| Seasonal factor, period $s$ | `[1, 0, …, 0]` (length $s-1$) | companion form with first row `[-1, …, -1]` | $s-1$ |

Components combine by **superposition**: stack $F$ horizontally, block-diagonalize $G$ and $W$. That is exactly what `combine()` does in `engine/models.py`.

---

## Baseline procedure

The nine-step modeling workflow is the spine of every lesson. Here is the version written out for reference — the app paces you through each step, but sometimes it helps to see the whole thing.

### Step 1 — Inspect the data

Plot $y_t$ on the vertical axis and $t$ on the horizontal. Ask:

- Is there a persistent **level**? (horizontal band, not drifting)
- Is there **drift** (trend)? (long-run direction)
- Any **periodicity**? (waves repeating at fixed lag)
- Any **heteroscedasticity**? (variance changes over time — not handled by constant-V DLMs)

Why it matters: the mental image you form here drives every subsequent step.

Pitfall: noise can disguise a slow trend. Don't commit to a hypothesis in step 1 — we quantify in step 3.

### Step 2 — Decompose visually

Overlay a centered moving average on the raw series. The MA smooths out the noise and surfaces a trend (or its absence). A seasonal signal often appears as a residual wobble after subtracting the MA.

Why it matters: a second, less noisy view of the same data lets you falsify or confirm the step-1 hypotheses.

Pitfall: if the window is too short the MA tracks the noise; if too long, it misses the trend. We pick $n/20$ (rounded to an odd number) as a starting point.

### Step 3 — Quantify autocorrelation

Compute and plot the ACF and PACF out to at least $n/4$ lags. Look for:

- **Slow monotone ACF decay** → trend or local level (persistent shocks)
- **Spike at lag $s$** not decaying → seasonal period $s$
- **PACF spike at lag 1** only → AR(1)-like behavior
- **All lags inside the** $\pm 1.96/\sqrt{n}$ **band** → white noise (no model to build)

Why it matters: the eye is unreliable; the ACF is reliable.

Pitfall: a spurious spike inside the confidence band is still possible — don't over-interpret a single lag.

### Step 4 — Pick components

Translate steps 1–3 into a set of components. Rules of thumb:

- Slow decay → local **level**
- Slow decay with *directional* drift → local level **and** slope
- Spike at lag $s$ → **seasonal** with period $s$
- Multiple patterns → combine with superposition

### Step 5 — Specify the model

Write down the matrices. This is where mis-specification most often happens:

- Mismatched dimensions → fitter will error
- Missing seasonality → residuals will spike at lag $s$
- Missing slope → forecasts will revert to the mean instead of extrapolating

Variance choice is discussed in [Tuning](#tuning).

### Step 6 — Fit

Run the Kalman filter (see `engine/filter.py`). Output:

- **Filtered means** $m_t$ — the current-time posterior mean of $\theta_t$
- **Filtered variances** $C_t$ — the associated uncertainty
- **One-step forecasts** $f_t$ — what the model *would have* predicted before seeing $y_t$
- **Innovations** $e_t = y_t - f_t$

### Step 7 — Diagnose

If your specification is right, the innovations $e_t$ look like white noise. Check:

- Residual ACF — spikes inside the band
- Residual normality — histogram roughly bell-shaped
- Portmanteau (Ljung-Box) p-value > 0.05

And compare **filtered vs smoothed** state: the smoothed state uses all of $y_{1:T}$ rather than just $y_{1:t}$, so it is smoother and less variable. Divergence of the two in low-data regions is expected; divergence in high-data regions suggests misspecification.

### Step 8 — Forecast

$h$-step-ahead forecasts propagate the filtered posterior forward through $G$, adding $W$ each step. Credible bands widen with horizon — that is intentional and informative.

### Step 9 — Reveal / review

Lesson mode: recap what you did and why. Challenge mode: compare your fitted DLM against the ground-truth DLM. If they agree, you recovered the true structure. If they disagree, inspect the residuals: that is where the misspecification shows.

---

## Tuning

Variance specification is where DLM practice becomes a craft. For the Beginner tier we treat $V$ as known; $W$ is what we tune.

**Strategy 1 — Fixed prior.** Pick $W$ based on domain knowledge. Good for simulations (where you *know* the generating $W$) and for preliminary analyses.

**Strategy 2 — Discount factors.** Write $W_t = C_t\, (1/\delta - 1)$ for some $\delta \in (0.9, 1)$. A single number controls the effective memory of the filter. $\delta = 1$ means no information discount ($W = 0$, the state is static); $\delta < 1$ lets the state evolve. Typical: $\delta = 0.95\text{–}0.99$ for a level component, $\delta \approx 0.98$ for a slope. (Intermediate tier only; Beginner works with fixed $W$ directly.)

**Strategy 3 — Empirical prior.** Use maximum-likelihood or REML to estimate $W$ from the data. (Intermediate / Advanced tier.)

### Signal-to-noise intuition

Let $r = W / V$. For the local-level model:

- $r \approx 0$ → $y_t \approx$ constant + noise (deep averaging)
- $r \approx 1$ → $y_t$ tracks the level closely (low averaging)
- $r \gg 1$ → the filter chases each observation (little smoothing)

The steady-state filtered variance is

$$C_\infty = \frac{-W + \sqrt{W^2 + 4 V W}}{2}.$$

See `engine/filter.py` — the test suite verifies this analytic fixed point.

### V vs W tradeoff

Increasing $V$ *or* decreasing $W$ both push the filter toward heavier smoothing. When both are unknown, the ratio matters more than either absolute value.

### Prior covariance $C_0$

For a diffuse prior, use $C_0 = \kappa\, I$ with $\kappa$ large ($10^3$ is typical). This lets the first few observations "inform themselves" without prior bias. Very small $C_0$ is informative; use it only when you have strong prior beliefs.

---

## Diagnostics

### Standardized residuals

Define $\varepsilon_t = e_t / \sqrt{Q_t}$. Under correct specification, $\varepsilon_t$ is approximately iid $N(0, 1)$. Plot $\varepsilon_t$ against $t$:

- No runs of same sign
- Roughly 95% within $\pm 2$
- No trend or pattern

### Residual ACF

All lags inside $\pm 1.96/\sqrt{n}$. Any persistent lag signature indicates missing structure:

- Slow decay → missed trend or level
- Spike at lag $s$ → missed seasonal

### Ljung-Box / portmanteau

A single p-value summarizing the first $k$ lags' autocorrelations. Null: white noise. Reject at p < 0.05.

### What "good" looks like

- Ljung-Box p > 0.1
- No residual ACF spikes beyond the band except possibly 1/20 by chance
- MAPE (mean absolute percent error) small relative to problem scale

---

## Worked example

We walk through the 9-step procedure on a synthetic quarterly series generated by a local-linear-trend **plus** seasonal model, noting *why* each diagnostic choice happens and *what* each reveals.

### Setup

We simulate $T = 120$ quarters (30 years) from:

- Level innovation variance $W_\mu = 0.02$
- Slope innovation variance $W_\beta = 0.001$
- Seasonal innovation variance $W_s = 0.05$
- Observation variance $V = 0.3$
- Initial level $\mu_0 = 10$, slope $\beta_0 = 0.05$, seasonal pattern $\{1.0, -0.5, 0.2, -0.7\}$

### Step 1 — Inspect

The series wanders upward over the 30-year span, with a regular four-peaked wobble each cycle. Both a trend and seasonal are plausible.

### Step 2 — Decompose

A centered 5-quarter moving average recovers a steadily rising curve. The residual ($y_t - \text{MA}$) shows a clear four-cycle pattern — confirmation of a seasonal.

### Step 3 — Quantify

The ACF decays slowly (trend signature) and shows non-decaying peaks at lags 4, 8, 12 (seasonal with period 4). The PACF has a dominant spike at lag 1 and near-zero beyond.

### Step 4 — Pick components

Level + slope + seasonal(4). Three components, combined by superposition.

### Step 5 — Specify

- Trend: $F_1 = [1, 0]$, $G_1 = \begin{pmatrix}1 & 1 \\ 0 & 1\end{pmatrix}$, $W_1 = \mathrm{diag}(0.02,\, 0.001)$
- Seasonal: $F_2 = [1, 0, 0]$, $G_2$ = companion form with first row $[-1, -1, -1]$, $W_2 = \mathrm{diag}(0.05,\, \epsilon,\, \epsilon)$
- Combined: $F = [1, 0, 1, 0, 0]$, $G$ = block-diag, $W$ = block-diag, $d = 5$
- $V = 0.3$ (known for this exercise)
- $m_0 = [10,\, 0.05,\, 1.0,\, -0.5,\, 0.2]$, $C_0 = 10^3 \cdot I_5$

### Step 6 — Fit

The Kalman filter returns filtered state means $m_t$ that separate into trend and seasonal components. Plotted alongside $y_t$, the filtered $F m_t$ tracks the observations closely without overfitting.

### Step 7 — Diagnose

- Standardized residuals: 95% within $\pm 2$, no runs — ✓
- Residual ACF: all lags inside the band — ✓
- Ljung-Box p ≈ 0.6 — ✓
- Filtered vs smoothed: visible divergence only in the first 4 quarters (diffuse prior regime)

### Step 8 — Forecast

$h = 20$ quarters ahead. The forecast extrapolates both the linear trend and the four-cycle pattern. Credible bands widen linearly because the dominant source of forecast uncertainty is the slope innovation propagating through $G$.

### Step 9 — Reveal

Because this is a worked example with known ground truth, we compare the fitted filtered state against the true state. They agree within the 95% credible band at every $t$.

### What to take away

- The method is mechanical once you can read ACF/PACF.
- The specification step is where understanding pays off — it's the only step that is not algorithmic.
- Diagnostics are how you know you got it right. Residuals that look like noise are your proof.
- The forecast is the product; everything before it is quality assurance.
