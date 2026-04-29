# DLM Intermediate Notebook — Implementation Plan

**Date:** 2026-04-29
**Spec:** `docs/superpowers/specs/2026-04-29-intermediate-notebook-design.md`
**Estimated effort:** 17–22 hours across 2–3 sessions

---

## Phase 1 — Engine extensions (8–11 hrs)

### Task 1: Discount-factor filter (1–2 hrs)
- [ ] Add `kalman_filter_discount(spec, y, delta)` to `engine/filter.py`
      Returns standard `FilterResult`; ignores `spec.W`; computes
      `R_t = G C_{t-1} G' / delta` instead.
- [ ] Unit test: local-level with δ=1 must equal standard filter (W = 0);
      analytic check for 1-step posterior variance.

### Task 2: Conjugate unknown-V filter (2–3 hrs)
- [ ] Create `engine/conjugate.py` with:
  - `ConjugateFilterResult` dataclass (m, C, n_vec, d_vec, f, Q, e, loglik)
  - `kalman_filter_conjugate(spec, y, n0, d0)` — scalar p=1 implementation
- [ ] Unit test: for large n0 (V effectively known), posterior mean should
      converge to the standard Kalman filter result; d_t/n_t should track V.

### Task 3: Time-varying observation matrix (2–3 hrs)
- [ ] Add `DLMSpecTV` dataclass to `engine/models.py`:
      `F_seq: NDArray[Any]`  shape (T, p, d); constant G, V, W, m0, C0.
- [ ] Add `kalman_filter_tv(spec_tv, y)` to `engine/filter.py`.
      Same as `kalman_filter` but picks `F = spec_tv.F_seq[t]` each step.
      Returns standard `FilterResult`.
- [ ] Unit test: constant F_seq matches standard filter exactly.

### Task 4: Fourier seasonal builder (0.5–1 hr)
- [ ] Add `make_fourier_seasonal(period, n_harmonics, V, W_season)` to
      `engine/models.py`. Assemble harmonic blocks using `combine()`.
- [ ] Unit test: `d == 2 * n_harmonics` (or `2*n_harmonics - 1` at Nyquist);
      verify G_j blocks have the correct rotation structure.

### Task 5: Model comparison helper (0.5 hr)
- [ ] Create `engine/comparison.py` with:
  - `compare_models(models: dict[str, DLMSpec], y) -> pd.DataFrame`
  - Columns: model, loglik, delta_loglik, bayes_factor_vs_best
- [ ] Unit test: identical models have BF = 1.

### Task 6: mypy + ruff pass on new engine files (0.5 hr)
- [ ] `mypy engine` passes with new files.
- [ ] `ruff check .` passes.

---

## Phase 2 — Notebooks (6–8 hrs)

### Task 7: Setup notebook (0.5 hr)
- [ ] `notebooks/intermediate/00_setup.ipynb`
      Environment check, imports, 2-page filter-notation recap, simulate a
      local-level series to confirm the engine works.

### Task 8: Notebook 01 — Discount factors (1–1.5 hrs)
- [ ] Motivation: what problem does W → discount solve?
- [ ] Math: derive R_t = G C_{t-1} G' / δ from variance-scaling argument.
- [ ] Implementation: run `kalman_filter_discount` on a local-level series with
      δ ∈ {0.8, 0.9, 0.95, 1.0}; plot filtered state ± 2σ for each.
- [ ] Exercise: choose δ to minimize one-step-ahead RMSE on a held-out window.

### Task 9: Notebook 02 — Conjugate unknown V (1.5–2 hrs)
- [ ] Motivation: practitioners rarely know V; running estimate is more honest.
- [ ] Math: IG conjugate update equations; posterior predictive is Student-t.
- [ ] Implementation: run `kalman_filter_conjugate` on simulated data; plot
      posterior mean V̂_t alongside ground-truth V.
- [ ] Exercise: compare marginal likelihoods of conjugate vs. fixed-V filter.

### Task 10: Notebook 03 — Dynamic regression (1.5–2 hrs)
- [ ] Motivation: Bayesian time-varying coefficient model.
- [ ] Math: y_t = x_t' β_t + ε_t, β_t = β_{t-1} + η_t (state = coefficients).
- [ ] Implementation: generate a series with a regime-changing slope; fit with
      `DLMSpecTV` / `kalman_filter_tv`; compare to fixed-coefficient OLS.
- [ ] Exercise: extend to two regressors (2D state).

### Task 11: Notebook 04 — Fourier seasonal (1–1.5 hrs)
- [ ] Motivation: Fourier vs. factor seasonal — same spectral content, fewer
      parameters when J < s/2.
- [ ] Math: harmonic rotation block derivation; relation to DFT.
- [ ] Implementation: fit monthly series (s=12) with J=1, 2, 3, 6 harmonics;
      plot residual variance and forecast RMSE vs. J.
- [ ] Exercise: confirm J = s/2 reproduces the factor-form seasonal exactly
      (up to rotation).

### Task 12: Notebook 05 — Model comparison (1–1.5 hrs)
- [ ] Motivation: choosing among local-level, LLT, seasonal models.
- [ ] Math: log Bayes factor = difference in log marginal likelihoods.
- [ ] Implementation: simulate a seasonal series; run `compare_models` on
      local-level, LLT, and seasonal specs; show that the true model wins.
- [ ] Exercise: demonstrate that overly complex models are penalised even with
      in-sample fit improvement.

---

## Phase 3 — Tests and polish (2–3 hrs)

### Task 13: Engine unit tests (1.5 hrs)
- [ ] `tests/test_intermediate_engine.py` — cover Tasks 1–5 analytic checks.
- [ ] `pytest -q` all green including existing Beginner tests.

### Task 14: Notebook smoke test (0.5 hr)
- [ ] Add `nbmake` to `[project.optional-dependencies]` dev extras in
      `pyproject.toml`.
- [ ] `tests/test_notebooks.py` — parametrised `nbmake` run for each notebook
      in `notebooks/intermediate/`.
- [ ] CI: add `pytest --nbmake notebooks/intermediate/` step to
      `.github/workflows/ci.yml`.

### Task 15: README update (0.5 hr)
- [ ] Add `notebooks/` section to `README.md` describing the intermediate
      series and how to launch JupyterLab.
- [ ] Update `pyproject.toml` to include `jupyterlab` (or `notebook`) in dev
      extras.

---

## Dependency order

```
Task 1 (discount)
Task 2 (conjugate)        ─┐
Task 3 (DLMSpecTV + TV)   ─┤─→ Task 6 (mypy/ruff) ──→ Tasks 7-12 (notebooks)
Task 4 (Fourier)          ─┤                           ──→ Task 13 (unit tests)
Task 5 (comparison)       ─┘                           ──→ Task 14 (nbmake)
                                                        ──→ Task 15 (README)
```
