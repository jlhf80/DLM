# DLM Intermediate Tutorial — Notebook Design Spec

**Date:** 2026-04-29
**Status:** Approved — ready for implementation planning
**Scope:** Full Intermediate, Option A (all 5 topics)
**Delivery:** Jupyter notebook series in `notebooks/intermediate/`

---

## 1. Overview

A hands-on notebook series that teaches intermediate DLM methodology: discount
factors, conjugate variance estimation, dynamic regression, Fourier seasonality,
and Bayesian model comparison. Each notebook is self-contained but builds
sequentially on earlier material. Students are expected to understand the Kalman
filter at the level covered in the Beginner Streamlit tutorial.

Target reader: a master's-level student comfortable with multivariate normals,
Bayes' theorem, and reading NumPy/Python code.

---

## 2. Delivery format

**Series of 5 notebooks** in `notebooks/intermediate/`:

```
01_discount_factors.ipynb
02_conjugate_unknown_variance.ipynb
03_dynamic_regression.ipynb
04_fourier_seasonal.ipynb
05_model_comparison.ipynb
```

Rationale: each notebook introduces exactly one engine extension and can be
executed independently. A single 200-cell notebook is hard to navigate and
harder to grade. A series lets a reader stop after topic 2 and still have
complete, runnable code.

An `00_setup.ipynb` covers environment setup and a 2-page recap of the Beginner
material (filter notation, `DLMSpec`, `FilterResult`).

---

## 3. Engine extensions required

The existing `engine/` ships a time-invariant Kalman filter + RTS smoother.
Each notebook topic requires a new engine capability:

| Topic | Engine addition | Module |
|-------|----------------|--------|
| Discount factors | `kalman_filter_discount(spec, y, delta)` — replaces W_t computation | `engine/filter.py` (new overload) or `engine/discount.py` |
| Conjugate unknown V | `kalman_filter_conjugate(spec, y, n0, d0)` — inverse-gamma state | `engine/conjugate.py` |
| Dynamic regression | `DLMSpecTV` (time-varying F) + `kalman_filter_tv(spec_tv, y)` | `engine/models.py`, `engine/filter.py` |
| Fourier seasonal | `make_fourier_seasonal(period, n_harmonics, V, W_season)` | `engine/models.py` |
| Model comparison | `log_bayes_factor(ll1, ll2)`, `compare_models(specs, y)` | `engine/comparison.py` |

### 3.1 Discount factors (W&H ch. 6)

Discount factor δ ∈ (0, 1] replaces the state evolution variance W with a
state-dependent quantity:

    R_t = C_{t-1} / δ    (instead of G C_{t-1} G' + W)

When G = I this is exact. For non-identity G the discount is applied after the
G rotation:  R_t = G C_{t-1} G' / δ.

Implementation: `kalman_filter_discount(spec, y, delta)` ignores `spec.W` and
computes R_t via discounting. Returns the same `FilterResult` as the standard
filter.

### 3.2 Conjugate unknown V (W&H ch. 10)

Prior: V ~ IG(n_0/2, d_0/2). The conjugate update at each step:
    n_t = n_{t-1} + 1
    Q_t = F R_t F' + (d_{t-1}/n_{t-1})   [scalar p=1 case]
    A_t = R_t F' / Q_t
    e_t = y_t - F a_t
    d_t = d_{t-1} + e_t² / Q_t
    m_t = a_t + A_t e_t
    C_t = (d_t / n_t) * (R_t - A_t Q_t A_t')

Returns `ConjugateFilterResult` carrying (m, C, n, d, f, Q, e, loglik).

### 3.3 Time-varying observation matrix (dynamic regression)

`DLMSpecTV`: replaces constant F (p×d) with F_seq (T×p×d). G, V, W remain
constant (handles the dynamic regression case where regressors vary but state
dynamics are fixed). The filter loop picks `F_t = spec_tv.F_seq[t]`.

### 3.4 Fourier seasonal representation (W&H ch. 8)

For a period-s seasonal with J harmonics (J ≤ s/2), each harmonic j gives a
2×2 rotation block (or 1×1 for the Nyquist harmonic when s is even):

    F_j = [1, 0],   G_j = [[cos(2πj/s), sin(2πj/s)],
                             [-sin(2πj/s), cos(2πj/s)]]

`make_fourier_seasonal(period, n_harmonics, V, W_season)` assembles these
blocks via `combine()` and returns a `DLMSpec`.

### 3.5 Model comparison

Log Bayes factor: BF_{12} = log p(y | M1) - log p(y | M2) = loglik1 - loglik2
(marginal likelihoods already computed by the Kalman filter).

`compare_models(models: dict[str, DLMSpec], y) -> pd.DataFrame` runs the filter
for each model and returns a DataFrame with columns [model, loglik, delta_loglik,
bayes_factor_vs_best].

---

## 4. Notebook structure (per notebook)

Each notebook follows a consistent 4-section template:

1. **Motivation** — 1–2 cells of prose explaining the problem the technique
   solves, with a reference to W&H.
2. **Math** — 2–4 cells with LaTeX derivation of the key equations, kept
   concise (no full proofs).
3. **Implementation** — inspect or write the new engine function; run it on
   simulated data.
4. **Exercises** — 2–3 cells with `# YOUR CODE HERE` blanks and assert-based
   auto-grading where possible.

---

## 5. Testing strategy

- `tests/test_intermediate_engine.py` — analytic checks for each new engine
  function (discount filter matches manual calculation, conjugate posterior
  moments match closed-form for p=1 normal-IG case).
- `tests/test_notebooks.py` — `nbmake` smoke test: each notebook executes
  without error top-to-bottom with its default parameters.

---

## 6. Non-goals

- No new Streamlit UI for Intermediate content.
- No multi-output (p > 1) support beyond what already exists.
- No EM / MLE fitting; variance estimation is Bayesian-conjugate only.
- No MCMC.
