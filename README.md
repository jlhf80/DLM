# DLM Intuition Tutorial

An interactive Streamlit tutorial for building intuition about **Dynamic Linear Models (DLMs)**. Pick simulation parameters, watch the series appear, then walk through a 9-step modeling workflow — first as a demonstration, then as a challenge where you try to recover the model yourself.

Beginner-tier content covers three canonical models:

1. **Local level** — random walk plus observation noise
2. **Local linear trend** — level + slope
3. **Simple seasonal** — sum-to-zero factor model

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/jlhf80/DLM.git
cd DLM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run ui/app.py
```

The app opens in your browser at http://localhost:8501.

## Project layout

- `engine/` — pure NumPy Kalman filter, smoother, forecast, diagnostics, and intermediate-tier extensions (discount factors, conjugate unknown V, time-varying F, Fourier seasonal, model comparison). No Streamlit imports; fully reusable from notebooks.
- `lessons/` — declarative `Lesson` objects wiring a model to a 9-step workflow.
- `ui/` — Streamlit app, Plotly renderers, session-state helpers.
- `notebooks/intermediate/` — Jupyter notebook series covering intermediate DLM topics (see below).
- `content/modeling-guide.md` — long-form reference on DLM specification and fitting.
- `tests/` — pytest suite (unit + notebook smoke tests).
- `docs/superpowers/specs/` — design specifications.
- `docs/superpowers/plans/` — implementation plans.

## Intermediate notebook series

Five self-contained notebooks in `notebooks/intermediate/`, designed for a master's-level reader:

| Notebook | Topic |
|----------|-------|
| `00_setup.ipynb` | Environment check + notation recap |
| `01_discount_factors.ipynb` | Replace W with a single discount factor δ |
| `02_conjugate_unknown_variance.ipynb` | Online V estimation via inverse-gamma conjugate prior |
| `03_dynamic_regression.ipynb` | Time-varying coefficients via time-varying F_t |
| `04_fourier_seasonal.ipynb` | Fourier-form seasonal representation |
| `05_model_comparison.ipynb` | Bayes factors from log marginal likelihoods |

Launch with:

```bash
jupyter lab notebooks/intermediate/
```

## Development

```bash
pytest -q                   # run tests
ruff check .                # lint
mypy engine lessons         # type check engine + lessons
```

## References

- Petris, Petrone & Campagnoli. *Dynamic Linear Models with R.* Springer, 2009.
- West & Harrison. *Bayesian Forecasting and Dynamic Models*, 2nd ed. Springer, 1997.

## License

TBD — add a LICENSE file before distributing.
