# DLM Intuition Tutorial

A self-contained Jupyter notebook series for practitioners who want to build genuine intuition about **Dynamic Linear Models (DLMs)** — also called state-space models or Kalman filter models.

The goal is not just to run code, but to understand *why* the Kalman filter works, *what* each parameter controls, and *when* to reach for a DLM over simpler alternatives. Notebooks progress from first principles through advanced inference, with derivations, exercises, and a custom NumPy engine that keeps the math visible.

## Three learning tiers

### Beginner — `notebooks/beginner/`

No prior state-space experience required. Assumes Python and basic probability.

| Notebook | Topic |
|----------|-------|
| `00_setup.ipynb` | Series overview, environment check, notation |
| `B0_bayesian_primer.ipynb` | *(optional)* Bayesian updating, PyMC intro |
| `B1_dlm_intro.ipynb` | The DLM equations; manual predict-update step |
| `B2_local_level.ipynb` | Kalman filter, RTS smoother, forecasting, log-likelihood |
| `B3_local_linear_trend.ipynb` | Adding a slope state; model comparison |
| `B4_seasonal_models.ipynb` | Dummy seasonal component; combining model specs |
| `B5_parameter_estimation.ipynb` | MLE via scipy; Bayesian estimation with PyMC |
| `B6_dlm_glm_connection.ipynb` | *(optional)* DLM as a generalization of regression |

### Intermediate — `notebooks/intermediate/`

Assumes familiarity with the Kalman filter (Beginner tier or equivalent).

| Notebook | Topic |
|----------|-------|
| `00_setup.ipynb` | Environment check, notation recap |
| `01_discount_factors.ipynb` | Replace W with a discount factor δ |
| `02_conjugate_unknown_variance.ipynb` | Online V estimation via inverse-gamma conjugate prior |
| `03_dynamic_regression.ipynb` | Time-varying coefficients via time-varying F_t |
| `04_fourier_seasonal.ipynb` | Fourier-form seasonal representation |
| `05_model_comparison.ipynb` | Bayes factors from log marginal likelihoods |

### Advanced — `notebooks/advanced/`

Assumes comfortable familiarity with the full filter-smoother-forecast cycle.

| Notebook | Topic |
|----------|-------|
| `00_advanced_setup.ipynb` | Environment check, advanced notation |
| `06_arima_dlm_equivalence.ipynb` | ARIMA models as DLMs; equivalence proofs |
| `07_ffbs_and_mcmc.ipynb` | Forward filtering backward sampling; Gibbs sampler; PyMC blackbox likelihood |
| `08_interventions_outliers.ipynb` | Level shifts, outlier detection, intervention analysis |
| `09_monitoring_structural_breaks.ipynb` | Sequential Bayes factors; cusum monitoring |
| `10_multivariate_missing_data.ipynb` | Multivariate DLMs; Kalman filter with missing observations |

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/jlhf80/DLM.git
cd DLM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
jupyter lab notebooks/beginner/
```

## Engine

All notebooks use a lightweight pure-NumPy engine in `engine/`. No framework imports — designed to stay readable alongside derivations in the notebooks.

Key modules: `filter`, `smoother`, `forecast`, `simulate`, `models`, `diagnostics`, `ffbs`, `interventions`.

## References

- West, M. & Harrison, J. *Bayesian Forecasting and Dynamic Models*, 2nd ed. Springer, 1997.
- Petris, G., Petrone, S. & Campagnoli, P. *Dynamic Linear Models with R.* Springer, 2009.
- Carter, C.K. & Kohn, R. On Gibbs sampling for state space models. *Biometrika*, 81(3), 1994.
- Frühwirth-Schnatter, S. Data augmentation and dynamic linear models. *Journal of Time Series Analysis*, 15(2), 1994.

## Development

```bash
pytest -q                          # unit tests
ruff check .                       # lint
mypy engine                        # type check
pytest --nbmake notebooks/ -q      # notebook smoke tests
```

## License

TBD — add a LICENSE file before distributing.
