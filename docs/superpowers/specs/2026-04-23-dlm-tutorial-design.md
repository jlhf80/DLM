# DLM Intuition Tutorial — Design Spec

**Date:** 2026-04-23
**Status:** Approved — ready for implementation planning
**Source references in repo:**
- Petris, Petrone, Campagnoli — *Dynamic Linear Models with R* (2009)
- West & Harrison — *Bayesian Forecasting and Dynamic Models* (2nd ed., 1997)

---

## 1. Overview

An interactive tutorial that builds intuition for Dynamic Linear Models (DLMs). A user picks key parameters, the tool simulates a time series, and then a guided workflow walks the user through the modeling process that recovers the correct DLM specification. Two modes: a **Lesson** (transparent ground truth, demonstrates the method) and a **Challenge** (hidden ground truth, the student tries it themselves). The same data pipeline backs both modes; the only difference is what is revealed and when.

## 2. Goals and non-goals

### Goals
- Teach the *method* of DLM specification, not just the definition. The 9-step modeling workflow is the primary teaching surface.
- Make the Kalman filter and related math inspectable — a student who opens `engine/filter.py` sees the West-Harrison equations written out in ~40 lines of annotated NumPy.
- Ship a working Beginner tier end-to-end (simulate → specify → fit → diagnose → forecast → reveal) before expanding.
- Separate the DLM engine cleanly from the Streamlit UI so a Jupyter/Quarto notebook variant (C) can reuse the engine later.

### Non-goals (explicit, v1)
- No Intermediate or Advanced tier content. Engine supports their computational requirements, but no lessons ship.
- No user accounts, no cross-session persistence, no multi-user state.
- No deployment story. The app runs locally via `streamlit run ui/app.py`.
- No pedagogical A/B testing infrastructure.
- No mobile layout.

## 3. Target users

Three difficulty tiers, all using the same tool with a tier toggle:
- **Beginner (undergraduate):** basic linear algebra, calculus, intro stats; little to no Bayesian background. Focus on visual intuition.
- **Intermediate (masters):** comfortable with multivariate normal distributions and Bayes' theorem; can read Kalman filter equations.
- **Advanced (PhD):** comfortable with conjugate analysis, MCMC, measure-theoretic notation.

**v1 ships only the Beginner tier**, with architecture ready to extend.

## 4. Platform decision

**Python + Streamlit** web app, run locally.

- Rejected: R Shiny (narrower audience despite matching Petris code).
- Rejected: standalone Jupyter/Quarto notebook for v1 — planned as variant C, to be built later reusing the same engine.
- Rejected: CLI (less engaging for a visual topic).

**Dependencies:** `numpy`, `scipy`, `statsmodels` (ACF/PACF only), `streamlit`, `plotly`, `pytest`, `ruff`, `mypy`.

## 5. Scope — Beginner tier

Three lessons, taught in linear order. Reaching step 9 in Challenge mode marks a lesson complete (within the current Streamlit session) and unlocks the next. Completed lessons remain revisitable. Completion state is not persisted across sessions — on a fresh app launch, only lesson 1 is unlocked.

1. **Local level** — random walk plus observation noise. State dim `d = 1`.
2. **Local linear trend** — level plus slope. `d = 2`.
3. **Simple seasonal** — factor form. `d = period − 1` (e.g., `d = 3` for quarterly).

Observation variance `V` is treated as known in all Beginner lessons (sidesteps conjugate-inference content, which belongs to the Intermediate tier).

**Future tiers (not in v1, but the engine supports their math):**
- Intermediate: component combination (trend + seasonal), Fourier seasonal, dynamic regression, unknown `V` (conjugate inverse-gamma), discount factors, model comparison.
- Advanced: ARIMA ↔ DLM equivalence, Forward Filtering Backward Sampling, MCMC for unknown `V`/`W`, interventions/outliers (λ-ω model), monitoring and structural breaks, optional multivariate and missing-data extensions.

## 6. Pedagogical flow

**Hybrid model per lesson:** one Lesson-mode walkthrough (method demonstration) followed by a Challenge-mode session on a freshly simulated series (student applies the method). Both halves reuse the same components; the only difference is what the UI reveals and when.

- **Lesson mode:** ground-truth parameters visible in sidebar. Tool walks through the 9 steps showing the plots, the ACF, the specification, the fit, and the diagnostics, with narrative text explaining each step. Prompts are parameter-agnostic (they describe *what the method is*, not specific slider values) so a student can freely adjust sliders and watch the plots and derived quantities update without the accompanying text going stale.
- **Challenge mode:** student picks simulation parameters (those become the hidden truth). At each step, the UI asks a question; the student answers; the tool gives feedback. At the final step, ground truth is revealed and overlaid with the student's specification.

## 7. Lesson structure — the 9-step workflow

Every lesson in every tier uses this same workflow skeleton, parameterized by lesson content:

1. **Inspect data** — plot `y_t` over time.
2. **Decompose visually** — is there a persistent level? slope? periodic pattern?
3. **Quantify** — ACF/PACF; seasonal subseries plot if `period > 1`.
4. **Pick components** — student identifies which components are present.
5. **Specify the model** — student writes down `F`, `G`, `V`, `W` (tool assists with fill-in-the-blank; Lesson mode fills it all).
6. **Fit** — run Kalman filter; show filtered state means and 95% credible bands over time.
7. **Diagnose** — run RTS smoother; compare filtered vs smoothed state; inspect one-step forecast residuals (innovations). Residuals must be approximately white noise — ACF of residuals and standardized residual plot shown.
8. **Forecast** — h-step-ahead forecast with credible bands. Horizon `h` is a slider on this step (range 1–100, default 20).
9. **Reveal vs truth** — Challenge mode: overlay the student's fitted DLM against the ground-truth DLM on forecasts and filtered states; Lesson mode: recap the steps taken.

## 8. Architecture

Hard split between a UI-free engine and the Streamlit app, so the engine runs unchanged in the later notebook variant.

```
dlm_tutorial/
├── engine/                     # pure NumPy; no Streamlit imports
│   ├── models.py               # DLMSpec; make_local_level / _linear_trend / _seasonal_factor; combine
│   ├── simulate.py             # forward-simulate y and true states from a DLMSpec
│   ├── filter.py               # Kalman filter — annotated, West-Harrison notation
│   ├── smoother.py             # RTS smoother
│   ├── forecast.py             # h-step forecast + credible bands
│   └── diagnostics.py          # residuals, ACF/PACF (statsmodels helpers)
├── lessons/                    # lesson content — declarative
│   ├── local_level.py
│   ├── local_linear_trend.py
│   ├── seasonal.py
│   └── workflow.py             # 9-step workflow skeleton, parameterized per lesson
├── ui/                         # only layer that imports Streamlit
│   ├── app.py                  # top-level routing (tier / lesson / step)
│   ├── plots.py                # Plotly renderers
│   ├── controls.py             # shared sliders / component toggles
│   └── state.py                # session-state helpers
├── content/                    # long-form markdown, rendered in-app and standalone-readable
│   └── modeling-guide.md       # baseline application, fitting, and tuning procedure (§13)
├── tests/
│   ├── test_models.py
│   ├── test_simulate.py
│   ├── test_filter.py
│   ├── test_smoother.py
│   ├── test_forecast.py
│   ├── test_diagnostics.py
│   └── test_lessons.py
├── pyproject.toml
└── README.md
```

Three properties this buys:
- **Engine is notebook-reusable** — `from dlm_tutorial.engine import filter, simulate` has no Streamlit cost.
- **Lessons are data, not code paths** — adding a lesson is adding a `Lesson` object; no UI code changes required.
- **Engine is the teaching artifact** — `filter.py` is readable line-by-line alongside West-Harrison Ch. 4.

## 9. Components and data model

All engine types are frozen dataclasses with NumPy arrays. The engine supports multivariate observations (`p ≥ 1`); Beginner lessons all use `p = 1`.

### Core engine types

```python
# engine/models.py
@dataclass(frozen=True, eq=False)   # eq=False: custom __eq__/__hash__ below
class DLMSpec:
    F: np.ndarray        # (p, d)       observation matrix
    G: np.ndarray        # (d, d)       state transition
    V: np.ndarray        # (p, p)       observation covariance
    W: np.ndarray        # (d, d)       state innovation covariance
    m0: np.ndarray       # (d,)         prior state mean
    C0: np.ndarray       # (d, d)       prior state covariance

    def __eq__(self, other):
        return isinstance(other, DLMSpec) and all(
            np.array_equal(getattr(self, k), getattr(other, k))
            for k in ("F", "G", "V", "W", "m0", "C0")
        )

    def __hash__(self):
        return hash(tuple(getattr(self, k).tobytes() for k in ("F", "G", "V", "W", "m0", "C0")))

# engine/simulate.py
@dataclass(frozen=True)
class SimulatedSeries:
    y: np.ndarray              # (T, p)
    theta_true: np.ndarray     # (T, d)
    spec: DLMSpec
    seed: int

# engine/filter.py
@dataclass(frozen=True)
class FilterResult:
    m: np.ndarray         # (T, d)       filtered state means
    C: np.ndarray         # (T, d, d)    filtered state covariances
    a: np.ndarray         # (T, d)       one-step predictive state means
    R: np.ndarray         # (T, d, d)    one-step predictive state covariances
    f: np.ndarray         # (T, p)       one-step obs forecast means
    Q: np.ndarray         # (T, p, p)    one-step obs forecast covariances
    e: np.ndarray         # (T, p)       innovations  y_t − f_t
    loglik: float

# engine/smoother.py
@dataclass(frozen=True)
class SmoothResult:
    s: np.ndarray         # (T, d)       smoothed state means
    S: np.ndarray         # (T, d, d)    smoothed state covariances

# engine/forecast.py
@dataclass(frozen=True)
class Forecast:
    horizon: int
    means: np.ndarray     # (h, p)
    lower: np.ndarray     # (h, p)       marginal 2.5% per obs component
    upper: np.ndarray     # (h, p)       marginal 97.5% per obs component
```

### Model builders (engine/models.py)

```python
make_local_level(V, W_level, m0=0.0, C0=1e3)                       -> DLMSpec   # d=1
make_local_linear_trend(V, W_level, W_slope, ...)                  -> DLMSpec   # d=2
make_seasonal_factor(period, V, W_season, ...)                     -> DLMSpec   # d=period-1
combine(*specs)                                                    -> DLMSpec   # block-diagonal superposition
```

Builders accept scalar `V` and auto-promote to `(p, p) = (1, 1)`. `combine` exists to support future Intermediate-tier lessons; it is not UI-exposed in v1.

### Lesson content types (lessons/workflow.py)

```python
@dataclass(frozen=True)
class ParamSpec:
    name: str; label: str; min: float; max: float; default: float; step: float; help: str

@dataclass(frozen=True)
class ChallengeQuestion:
    kind: Literal["multiple_choice", "numeric_range", "component_toggle"]
    correct: Any
    feedback_correct: str
    feedback_incorrect: str

@dataclass(frozen=True)
class WorkflowStep:
    id: str                              # "inspect_data", "acf", "specify", etc.
    title: str
    prompt_md: str                       # markdown shown above the plot
    plot_fn: str                         # name of renderer in ui/plots.py
    hints: list[str]
    challenge: ChallengeQuestion | None  # None => info-only

@dataclass(frozen=True)
class Lesson:
    id: str; title: str; tier: str; description: str
    model_builder: Callable[[dict[str, float]], DLMSpec]
    param_schema: list[ParamSpec]
    workflow_steps: list[WorkflowStep]
```

### Streamlit session state (ui/state.py)

```
tier             : "beginner"
lesson_id        : "local_level" | "local_linear_trend" | "seasonal"
mode             : "lesson" | "challenge"
sim_params       : dict[str, float]     # current slider values
spec_true        : DLMSpec              # built from sim_params
series           : SimulatedSeries
step_idx         : int
answers          : dict[step_id, Any]
filter_result    : FilterResult | None
smooth_result    : SmoothResult | None
user_spec        : DLMSpec | None       # student spec in challenge mode
```

All state lives in `st.session_state`. No disk persistence.

## 10. Data flow and user journey

### Navigation

```
Home → [Tier: Beginner]
         → [Lesson: 1/2/3]
             → [Mode: Lesson ▸ or Challenge ▸]
                 → Simulation panel + 9-step workflow
                     → Complete → return to lesson menu
```

### Layout

Left-sidebar + main-panel Streamlit page. Sidebar stays visible through all 9 steps and contains:
- Lesson and mode selectors
- Simulation parameter sliders (keyed by `Lesson.param_schema`)
- Seed input (default 42)
- `n` (series length) slider — default 200, range 20–2000
- Step progress indicator

The forecast horizon `h` is *not* in the sidebar; it appears as a slider on step 8 only (default 20, range 1–100).

Main panel shows the current `WorkflowStep`: `prompt_md`, a Plotly chart produced by `plot_fn`, a challenge widget (if any), and Back/Next navigation.

### Data pipeline

```
sim_params  → model_builder  → spec_true  → simulate  → series
                                                           │
                    (student spec, Challenge only) ──────┐ ▼
                 or spec_true (Lesson) ─────── spec_to_fit ─► kalman_filter → filter_result
                                                                              │
                                             ┌────────────────────────────────┼──────────────┐
                                             ▼                                ▼              ▼
                                         smoother                       diagnostics       forecast
```

### Step → computation map

| Step | Computation | Cached |
|---|---|---|
| 1. Inspect data | plot only | – |
| 2. Decompose | plot only | – |
| 3. Quantify | `diagnostics.acf_pacf(series.y)` | ACF arrays |
| 4. Pick components | user input | answers |
| 5. Specify | user input → candidate spec | `user_spec` |
| 6. Fit | `kalman_filter(spec, series.y)` | `filter_result` |
| 7. Diagnose | `smoother(filter_result, spec)` + residual stats | `smooth_result` |
| 8. Forecast | `forecast_horizon(filter_result, spec, h)` where `h` is a step-local slider | `forecast` (keyed by `h`) |
| 9. Reveal | overlay `spec_true` vs `user_spec` | – |

### Re-run behavior

`@st.cache_data` keys are `(hash(DLMSpec), y.tobytes(), seed, ...)`. `DLMSpec` defines a custom `__hash__` (see §9) because NumPy arrays are not hashable by default. Changing a slider invalidates only the downstream computations; moving step forward/back is pure navigation and triggers no recompute.

| Action | Invalidates |
|---|---|
| Slider change | `spec_true`, `series`, all downstream |
| Seed change | `series`, all downstream |
| Mode switch | `answers`, `user_spec`, `filter_result` |
| Step Next/Back | nothing |
| Lesson switch | everything |

## 11. Error handling

Principle: turn every error into a teaching moment, not a stack trace.

| Failure | Response |
|---|---|
| `V = 0` or `W = 0` slider | Slider min is `1e-6`. Help text explains variance must be strictly positive. |
| Invalid user-constructed spec (Challenge) | `DLMSpec.__post_init__` raises `ValueError` with precise shape message; UI catches and renders via `st.error` on the specify step. |
| Non-PSD drift in `C_t` | Joseph form covariance update; symmetrize each step. |
| Singular `Q_t` | Cholesky first; fall back to `np.linalg.pinv` and attach warning to `FilterResult`; UI surfaces in a collapsed "Numerical notes" expander. |
| Series too long | `n` slider capped at 2000 (default 200). |
| Student spec mis-specified but fittable | Not an error — diagnostics will surface correlated residuals. That is the lesson. |
| Missing anchor in `modeling-guide.md` | Validated at app startup against parsed header list — raises at import-time, not runtime (caught in CI). |
| Stale cache | `DLMSpec` is a frozen dataclass; any parameter change produces a new cache key. |
| Plotly render error | Streamlit default handling; no custom path. |

`DLMSpec.__post_init__` is the one guaranteed choke point: if spec construction succeeds, nothing downstream can fail on shape or PSD issues.

### What we do not do
- No try/except swallow in the engine. Validation is the UI's job; the engine raises loudly on invalid inputs.
- No logging framework. `warnings` + `st.warning`/`st.error` is sufficient.
- No UI error-recovery flow. A bad user spec shows a clear message and keeps the UI on the specify step.

### Invariants enforced in `DLMSpec.__post_init__`
```
F.shape == (p, d)
G.shape == (d, d)
V.shape == (p, p) and V symmetric PSD
W.shape == (d, d) and W symmetric PSD
m0.shape == (d,)
C0.shape == (d, d) and C0 symmetric PSD
diag(V) > 1e-12  and  diag(W) > 1e-12
```

## 12. Testing

Engine-first. UI manually verified in the browser. No e2e harness in v1.

```
tests/
├── test_models.py        DLMSpec validation + model builders
├── test_simulate.py      reproducibility + distributional checks
├── test_filter.py        Kalman filter correctness
├── test_smoother.py      RTS smoother correctness
├── test_forecast.py      forecast horizon behavior
├── test_diagnostics.py   ACF/PACF + residual stats
└── test_lessons.py       lesson content integrity
```

### Key correctness tests
- **Filter log-likelihood** matches `scipy.stats.multivariate_normal.logpdf` on the full joint `y_{1:T}` for small `T = 5` within `1e-10`.
- **Steady-state convergence** of the filtered variance on constant local level matches the analytic Riccati fixed point.
- **Zero-noise limit** (tiny `W`, tiny `V`): filtered mean tracks true state within tolerance.
- **Joseph form vs standard form** produce equal results (1e-10) on a small case.
- **Smoother at `t = T`** equals filter output (`s_T == m_T`, `S_T == C_T`).
- **Forecast band width** strictly increases with horizon (monotonicity).
- **Model builder shapes**: `make_local_linear_trend` has `G = [[1, 1], [0, 1]]`; `make_seasonal_factor(period=4)` has state dim 3.
- **Simulate reproducibility**: same seed → bit-identical output.

### Content integrity tests (`test_lessons.py`)
- All `Lesson` objects import and construct.
- Each `model_builder` returns a valid `DLMSpec` on default params.
- Every `WorkflowStep.plot_fn` names an existing function in `ui/plots.py`.
- Every `ChallengeQuestion.correct` type matches its `kind`.
- Every deep-link anchor in `WorkflowStep.prompt_md` exists in `content/modeling-guide.md`.

### Not tested
- UI rendering (manual).
- Pedagogical quality (user testing, out of scope).
- Performance (for `n ≤ 2000`, `d ≤ 14`, a naive filter runs in < 50 ms).

### CI (GitHub Actions on push)
- `ruff check .`
- `pytest -q` (target < 30 s wall time)
- `mypy engine/ lessons/` (strict on library surface; UI layer lenient)

## 13. Content: `modeling-guide.md`

A long-form Markdown reference rendered in-app (Reference tab) and readable standalone. Workflow steps deep-link to anchors within it.

**Sections:**
1. **The DLM framework** — observation and state equations with West-Harrison notation cross-referenced to Petris.
2. **The baseline modeling procedure** — the 9-step workflow written out in prose, each step with its *why* and common pitfall.
3. **Tuning** — variance specification strategies (fixed prior, discount factors, empirical prior); signal-to-noise intuition; `V` vs `W` tradeoff; effect of `C_0` choice.
4. **Model selection & diagnostics** — one-step forecast residuals, standardized innovations, residual ACF, MAPE; what "good" looks like.
5. **Worked example** — one fully solved walkthrough (local linear trend + quarterly seasonal on a synthetic series) from raw data to forecasts.

## 14. Future work (not in v1)

- **Notebook variant (C)** — Quarto or Jupyter with ipywidgets, reusing `engine/` unchanged.
- **Intermediate tier** — component combinations, Fourier seasonal, dynamic regression, unknown `V`, discount factors, model comparison.
- **Advanced tier** — ARIMA ↔ DLM, FFBS, MCMC, interventions/outliers, monitoring, multivariate DLMs.
- **Persistence** — optional local progress tracking (JSON file) so students can resume.
- **Shareable lesson links** — URL query params encode `(lesson, mode, seed, params)` for instructors.

---

## Appendix A — Dependency pinning guidance

For the implementation plan: use loose minimum pins in `pyproject.toml` (`numpy >= 1.26`, `streamlit >= 1.30`, etc.) but commit a `uv.lock` or `requirements.lock` for reproducible installs.

## Appendix B — File naming convention

Engine files are noun-named (`filter.py`, `smoother.py`). Lesson files are lesson-id-named (`local_level.py`). UI files are component-named (`plots.py`, `controls.py`). Test files mirror the engine module names with `test_` prefix.
