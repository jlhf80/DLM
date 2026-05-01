# DLM Intuition Tutorial — Streamlit App

An interactive browser-based tutorial for building intuition about **Dynamic Linear Models (DLMs)**. Pick simulation parameters, watch the time series appear, then walk through a structured 9-step modeling workflow — first as a guided demonstration, then as a challenge where you recover the model yourself.

Designed for practitioners who learn by doing: no equations required to start, but the modeling-guide explains the math when you want it.

## Models covered

1. **Local level** — random walk plus observation noise
2. **Local linear trend** — level + slope state
3. **Simple seasonal** — sum-to-zero dummy factor model

## Quickstart

Requires Python 3.11+.

```bash
git clone -b streamlit_DLM https://github.com/jlhf80/DLM.git
cd DLM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run ui/app.py
```

The app opens in your browser at http://localhost:8501.

## Project layout

- `engine/` — pure NumPy Kalman filter, smoother, forecast, and diagnostics. No Streamlit imports; reusable independently.
- `lessons/` — declarative `Lesson` objects wiring each model to a 9-step workflow.
- `ui/` — Streamlit app, Plotly renderers, session-state helpers.
- `content/modeling-guide.md` — long-form reference on DLM specification and fitting.
- `tests/` — pytest suite (unit + integration).

## Development

```bash
pytest -q                   # run tests
ruff check .                # lint
mypy engine lessons         # type check
```

## References

- West, M. & Harrison, J. *Bayesian Forecasting and Dynamic Models*, 2nd ed. Springer, 1997.
- Petris, G., Petrone, S. & Campagnoli, P. *Dynamic Linear Models with R.* Springer, 2009.

## Note

The notebook-based learning path (Beginner → Intermediate → Advanced) lives on the `main` branch.

## License

TBD — add a LICENSE file before distributing.
