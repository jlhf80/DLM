# DLM Intuition Tutorial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that teaches DLM modeling through three Beginner-tier lessons (local level, local linear trend, simple seasonal), each with a Lesson mode walkthrough and a Challenge mode exercise, backed by a UI-free NumPy engine designed for reuse in a future notebook variant.

**Architecture:** Hard split between a `dlm_tutorial.engine` package (pure NumPy Kalman filter/smoother/forecast/diagnostics, no Streamlit imports) and a `dlm_tutorial.ui` layer (Streamlit app + Plotly). Lessons are declarative `Lesson` objects that wire a model-builder to a 9-step workflow skeleton. Long-form reference content lives in `content/modeling-guide.md`, rendered in-app and cross-linked from workflow steps.

**Tech Stack:** Python 3.11+, NumPy, SciPy, statsmodels (ACF/PACF helpers only), Streamlit, Plotly, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-04-23-dlm-tutorial-design.md`

**Layout** (flat, package-per-responsibility; imports like `from engine.filter import kalman_filter`):

```
DLM/                                    ← project root (current working dir)
├── engine/                             ← pure NumPy; no Streamlit imports
├── lessons/                            ← declarative lesson content
├── ui/                                 ← Streamlit + Plotly
├── content/                            ← long-form markdown
├── tests/
├── docs/superpowers/{specs,plans}/
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Phase 0 — Project scaffolding

### Task 1: Initialize git repo, project structure, and `.gitignore`

**Files:**
- Create: `/Users/jameshenson/Documents/tutorials/DLM/.gitignore`
- Create: directory tree (engine/, lessons/, ui/, content/, tests/, docs/superpowers/{specs,plans}/, .github/workflows/)

- [ ] **Step 1: Initialize git repo**

Run:
```bash
cd /Users/jameshenson/Documents/tutorials/DLM
git init
```
Expected: `Initialized empty Git repository in .../DLM/.git/`.

- [ ] **Step 2: Create directory tree and empty `__init__.py` files**

Run:
```bash
cd /Users/jameshenson/Documents/tutorials/DLM
mkdir -p engine lessons ui tests .github/workflows
touch engine/__init__.py lessons/__init__.py ui/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `.gitignore`**

Create `/Users/jameshenson/Documents/tutorials/DLM/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
venv/
dist/
build/

# Streamlit
.streamlit/secrets.toml

# Superpowers workspace
.superpowers/

# macOS / editors
.DS_Store
.vscode/
.idea/

# Jupyter (future notebook variant)
.ipynb_checkpoints/
```

- [ ] **Step 4: Commit**

```bash
cd /Users/jameshenson/Documents/tutorials/DLM
git add .gitignore engine/__init__.py lessons/__init__.py ui/__init__.py tests/__init__.py
git commit -m "chore: initialize project structure"
```

Expected: a single commit adding the `.gitignore` and empty package `__init__.py` files. The two PDFs and existing `docs/superpowers/specs/...` are untracked for now — they'll be committed later.

---

### Task 2: `pyproject.toml` with dependencies + tool configuration

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

Create `/Users/jameshenson/Documents/tutorials/DLM/pyproject.toml`:

```toml
[project]
name = "dlm-tutorial"
version = "0.1.0"
description = "Interactive tutorial for building intuition about Dynamic Linear Models."
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "statsmodels>=0.14",
    "streamlit>=1.30",
    "plotly>=5.18",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["engine*", "lessons*", "ui*"]
exclude = ["tests*"]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = "-q --strict-markers"
filterwarnings = [
    "error",                                               # promote warnings to errors in tests
    "ignore::DeprecationWarning:statsmodels.*",            # statsmodels is chatty
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]  # line length enforced by formatter

[tool.mypy]
python_version = "3.11"
strict = true
files = ["engine", "lessons"]
# UI layer not strictly typed — Streamlit's stubs are incomplete
[[tool.mypy.overrides]]
module = "statsmodels.*"
ignore_missing_imports = true
[[tool.mypy.overrides]]
module = "plotly.*"
ignore_missing_imports = true
```

- [ ] **Step 2: Create and activate a virtual environment, install dev deps**

Run:
```bash
cd /Users/jameshenson/Documents/tutorials/DLM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Expected: installs numpy, scipy, statsmodels, streamlit, plotly, pytest, ruff, mypy.

- [ ] **Step 3: Verify tool configuration with empty-project smoke tests**

Run:
```bash
ruff check .
pytest -q
mypy engine lessons
```

Expected:
- `ruff check .` → `All checks passed!` (no files to lint yet)
- `pytest -q` → `no tests ran`
- `mypy engine lessons` → `Success: no issues found in 0 source files`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject with dev tooling"
```

---

## Phase 1 — Engine

### Task 3: `DLMSpec` dataclass with validation, custom `__eq__`/`__hash__`

**Files:**
- Create: `engine/models.py` (partial — only `DLMSpec` in this task)
- Create: `tests/test_models.py` (partial — only DLMSpec tests in this task)

- [ ] **Step 1: Write failing tests for DLMSpec validation and hashing**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_models.py`:

```python
"""Tests for DLMSpec and model builders."""

import numpy as np
import pytest

from engine.models import DLMSpec


def _valid_spec_kwargs(p: int = 1, d: int = 2) -> dict:
    return dict(
        F=np.ones((p, d)),
        G=np.eye(d),
        V=np.eye(p),
        W=np.eye(d),
        m0=np.zeros(d),
        C0=np.eye(d),
    )


class TestDLMSpecValidation:
    def test_valid_spec_constructs(self):
        spec = DLMSpec(**_valid_spec_kwargs())
        assert spec.F.shape == (1, 2)
        assert spec.G.shape == (2, 2)

    def test_F_wrong_shape_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["F"] = np.ones((2, 2))  # p=2 but V is (1,1)
        with pytest.raises(ValueError, match="F"):
            DLMSpec(**kwargs)

    def test_G_not_square_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["G"] = np.ones((2, 3))
        with pytest.raises(ValueError, match="G"):
            DLMSpec(**kwargs)

    def test_V_not_symmetric_raises(self):
        kwargs = _valid_spec_kwargs(p=2)
        kwargs["V"] = np.array([[1.0, 0.5], [0.1, 1.0]])  # asymmetric
        with pytest.raises(ValueError, match="V.*symmetric"):
            DLMSpec(**kwargs)

    def test_W_negative_diagonal_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["W"] = np.diag([-1.0, 1.0])
        with pytest.raises(ValueError, match="W.*positive"):
            DLMSpec(**kwargs)

    def test_V_zero_diagonal_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["V"] = np.array([[0.0]])
        with pytest.raises(ValueError, match="V.*positive"):
            DLMSpec(**kwargs)

    def test_m0_wrong_dim_raises(self):
        kwargs = _valid_spec_kwargs()
        kwargs["m0"] = np.zeros(3)  # should be length 2
        with pytest.raises(ValueError, match="m0"):
            DLMSpec(**kwargs)


class TestDLMSpecEqualityAndHash:
    def test_equal_specs_are_equal(self):
        a = DLMSpec(**_valid_spec_kwargs())
        b = DLMSpec(**_valid_spec_kwargs())
        assert a == b
        assert hash(a) == hash(b)

    def test_different_F_not_equal(self):
        a = DLMSpec(**_valid_spec_kwargs())
        kwargs = _valid_spec_kwargs()
        kwargs["F"] = np.full((1, 2), 2.0)
        b = DLMSpec(**kwargs)
        assert a != b
        assert hash(a) != hash(b)

    def test_hash_stable_across_instances(self):
        a = DLMSpec(**_valid_spec_kwargs())
        b = DLMSpec(**_valid_spec_kwargs())
        assert hash(a) == hash(b)

    def test_spec_usable_as_dict_key(self):
        a = DLMSpec(**_valid_spec_kwargs())
        d = {a: "value"}
        b = DLMSpec(**_valid_spec_kwargs())
        assert d[b] == "value"
```

- [ ] **Step 2: Run tests; expect collection error (engine.models not found)**

Run: `pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'engine.models'` — collection fails.

- [ ] **Step 3: Implement `DLMSpec` in `engine/models.py`**

Create `/Users/jameshenson/Documents/tutorials/DLM/engine/models.py`:

```python
"""Core DLM data types and model builders.

DLMSpec holds the constant (time-invariant) matrices F, G, V, W of a Gaussian
DLM, plus the prior (m0, C0). All Beginner-tier lessons are time-invariant, so
we do not support time-varying components here — that belongs in a future
revision when the Intermediate tier is added.

Notation follows West & Harrison, Bayesian Forecasting and Dynamic Models (2nd
ed., 1997), chapter 4:

    Observation:  y_t = F_t theta_t + v_t,    v_t ~ N(0, V_t)
    State:        theta_t = G_t theta_{t-1} + w_t,    w_t ~ N(0, W_t)

with y_t in R^p, theta_t in R^d, F_t (p, d), G_t (d, d), V_t (p, p), W_t (d, d).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TOL = 1e-8
_DIAG_MIN = 1e-12


def _check_psd_symmetric(M: np.ndarray, name: str) -> None:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square 2D array, got shape {M.shape}")
    if not np.allclose(M, M.T, atol=_TOL):
        raise ValueError(f"{name} must be symmetric")
    # PSD check via eigenvalues; tolerate tiny negative due to fp noise
    eigs = np.linalg.eigvalsh(M)
    if np.any(eigs < -_TOL):
        raise ValueError(f"{name} must be positive semidefinite (min eigenvalue {eigs.min():.3e})")
    if np.any(np.diag(M) <= _DIAG_MIN):
        raise ValueError(f"{name} must have strictly positive diagonal (> {_DIAG_MIN})")


@dataclass(frozen=True, eq=False)
class DLMSpec:
    """A time-invariant Gaussian DLM specification.

    Attributes
    ----------
    F : (p, d) ndarray — observation matrix
    G : (d, d) ndarray — state transition matrix
    V : (p, p) ndarray — observation noise covariance
    W : (d, d) ndarray — state evolution noise covariance
    m0 : (d,) ndarray — prior state mean
    C0 : (d, d) ndarray — prior state covariance
    """

    F: np.ndarray
    G: np.ndarray
    V: np.ndarray
    W: np.ndarray
    m0: np.ndarray
    C0: np.ndarray

    def __post_init__(self) -> None:
        # Pull shapes from F (source of truth for p and d).
        if self.F.ndim != 2:
            raise ValueError(f"F must be 2D with shape (p, d), got shape {self.F.shape}")
        p, d = self.F.shape

        if self.G.shape != (d, d):
            raise ValueError(f"G must be ({d}, {d}), got {self.G.shape}")
        if self.V.shape != (p, p):
            raise ValueError(f"V must be ({p}, {p}), got {self.V.shape}")
        if self.W.shape != (d, d):
            raise ValueError(f"W must be ({d}, {d}), got {self.W.shape}")
        if self.m0.shape != (d,):
            raise ValueError(f"m0 must be shape ({d},), got {self.m0.shape}")
        if self.C0.shape != (d, d):
            raise ValueError(f"C0 must be ({d}, {d}), got {self.C0.shape}")

        _check_psd_symmetric(self.V, "V")
        _check_psd_symmetric(self.W, "W")
        _check_psd_symmetric(self.C0, "C0")

    @property
    def p(self) -> int:
        """Observation dimension."""
        return self.F.shape[0]

    @property
    def d(self) -> int:
        """State dimension."""
        return self.F.shape[1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DLMSpec):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, k), getattr(other, k))
            for k in ("F", "G", "V", "W", "m0", "C0")
        )

    def __hash__(self) -> int:
        return hash(
            tuple(np.ascontiguousarray(getattr(self, k)).tobytes()
                  for k in ("F", "G", "V", "W", "m0", "C0"))
        )
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_models.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/__init__.py engine/models.py tests/test_models.py tests/__init__.py
git commit -m "feat(engine): DLMSpec with validation and hashing"
```

---

### Task 4: Model builder — `make_local_level`

**Files:**
- Modify: `engine/models.py` (append)
- Modify: `tests/test_models.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
from engine.models import make_local_level


class TestMakeLocalLevel:
    def test_default_returns_valid_spec(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        assert spec.p == 1 and spec.d == 1
        assert spec.F.shape == (1, 1)
        assert spec.F[0, 0] == 1.0
        assert spec.G.shape == (1, 1)
        assert spec.G[0, 0] == 1.0
        assert spec.V[0, 0] == 0.5
        assert spec.W[0, 0] == 0.1

    def test_accepts_scalar_V(self):
        spec = make_local_level(V=2.0, W_level=0.1)
        assert isinstance(spec.V, np.ndarray)
        assert spec.V.shape == (1, 1)
        assert spec.V[0, 0] == 2.0

    def test_prior_defaults(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        assert spec.m0[0] == 0.0
        assert spec.C0[0, 0] == 1e3  # diffuse default

    def test_custom_prior(self):
        spec = make_local_level(V=0.5, W_level=0.1, m0=10.0, C0=4.0)
        assert spec.m0[0] == 10.0
        assert spec.C0[0, 0] == 4.0

    def test_rejects_zero_W(self):
        with pytest.raises(ValueError, match="W"):
            make_local_level(V=0.5, W_level=0.0)
```

- [ ] **Step 2: Run tests; expect ImportError on `make_local_level`**

Run: `pytest tests/test_models.py::TestMakeLocalLevel -v`
Expected: `ImportError: cannot import name 'make_local_level'`.

- [ ] **Step 3: Implement `make_local_level`**

Append to `engine/models.py`:

```python
def make_local_level(
    V: float,
    W_level: float,
    m0: float = 0.0,
    C0: float = 1e3,
) -> DLMSpec:
    """Local-level (random-walk-plus-noise) DLM.

    State dim d = 1. theta_t is the unobserved level.
        y_t = theta_t + v_t,     v_t ~ N(0, V)
        theta_t = theta_{t-1} + w_t,  w_t ~ N(0, W_level)
    """
    return DLMSpec(
        F=np.array([[1.0]]),
        G=np.array([[1.0]]),
        V=np.array([[float(V)]]),
        W=np.array([[float(W_level)]]),
        m0=np.array([float(m0)]),
        C0=np.array([[float(C0)]]),
    )
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_models.py::TestMakeLocalLevel -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/models.py tests/test_models.py
git commit -m "feat(engine): make_local_level builder"
```

---

### Task 5: Model builder — `make_local_linear_trend`

**Files:**
- Modify: `engine/models.py` (append)
- Modify: `tests/test_models.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
from engine.models import make_local_linear_trend


class TestMakeLocalLinearTrend:
    def test_shapes_and_matrices(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        assert spec.p == 1 and spec.d == 2
        np.testing.assert_array_equal(spec.F, [[1.0, 0.0]])
        np.testing.assert_array_equal(spec.G, [[1.0, 1.0], [0.0, 1.0]])

    def test_W_is_diagonal_of_level_and_slope(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        np.testing.assert_array_equal(spec.W, [[0.05, 0.0], [0.0, 0.01]])

    def test_default_prior_is_diffuse(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        np.testing.assert_array_equal(spec.m0, [0.0, 0.0])
        np.testing.assert_array_equal(spec.C0, 1e3 * np.eye(2))

    def test_custom_prior_shapes(self):
        spec = make_local_linear_trend(
            V=0.5, W_level=0.05, W_slope=0.01,
            m0=np.array([100.0, 1.0]),
            C0=np.diag([10.0, 0.5]),
        )
        np.testing.assert_array_equal(spec.m0, [100.0, 1.0])
        np.testing.assert_array_equal(spec.C0, np.diag([10.0, 0.5]))
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_models.py::TestMakeLocalLinearTrend -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `make_local_linear_trend`**

Append to `engine/models.py`:

```python
def make_local_linear_trend(
    V: float,
    W_level: float,
    W_slope: float,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
) -> DLMSpec:
    """Local-linear-trend DLM (level + slope).

    State theta_t = (mu_t, beta_t); d = 2.
        y_t = mu_t + v_t
        mu_t = mu_{t-1} + beta_{t-1} + w1_t
        beta_t = beta_{t-1} + w2_t
    with w1 ~ N(0, W_level), w2 ~ N(0, W_slope), independent.
    """
    F = np.array([[1.0, 0.0]])
    G = np.array([[1.0, 1.0], [0.0, 1.0]])
    W = np.diag([float(W_level), float(W_slope)])
    if m0 is None:
        m0 = np.zeros(2)
    if C0 is None:
        C0 = 1e3 * np.eye(2)
    return DLMSpec(
        F=F, G=G,
        V=np.array([[float(V)]]),
        W=W,
        m0=np.asarray(m0, dtype=float),
        C0=np.asarray(C0, dtype=float),
    )
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_models.py::TestMakeLocalLinearTrend -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/models.py tests/test_models.py
git commit -m "feat(engine): make_local_linear_trend builder"
```

---

### Task 6: Model builder — `make_seasonal_factor`

**Files:**
- Modify: `engine/models.py` (append)
- Modify: `tests/test_models.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
from engine.models import make_seasonal_factor


class TestMakeSeasonalFactor:
    def test_quarterly_shapes(self):
        spec = make_seasonal_factor(period=4, V=0.5, W_season=0.02)
        # State has dim period - 1 = 3 (sum-to-zero constraint implicit in G)
        assert spec.d == 3
        assert spec.F.shape == (1, 3)
        # Only the first state component contributes to observation
        np.testing.assert_array_equal(spec.F, [[1.0, 0.0, 0.0]])

    def test_G_is_permutation_companion(self):
        """G rotates state components and enforces sum-to-zero.

        For period s, G is the (s-1)x(s-1) companion matrix:
            top row:  [-1, -1, ..., -1]   # next season = -sum of previous s-1
            below:    identity shifted right (picks up previous entries)
        """
        spec = make_seasonal_factor(period=4, V=0.5, W_season=0.02)
        expected_G = np.array(
            [[-1.0, -1.0, -1.0],
             [ 1.0,  0.0,  0.0],
             [ 0.0,  1.0,  0.0]]
        )
        np.testing.assert_array_equal(spec.G, expected_G)

    def test_W_only_on_first_component(self):
        """Only the newly-generated seasonal effect carries innovation noise."""
        spec = make_seasonal_factor(period=4, V=0.5, W_season=0.02)
        # Diagonal: first entry is W_season; others must be strictly positive
        # (nugget) so W is PD and passes __post_init__.
        assert spec.W[0, 0] == 0.02
        # The nugget on the other diagonal entries is small
        assert spec.W[1, 1] < 1e-6 and spec.W[1, 1] > 0
        assert spec.W[2, 2] < 1e-6 and spec.W[2, 2] > 0
        # Off-diagonals are zero
        assert spec.W[0, 1] == 0 and spec.W[1, 2] == 0

    def test_period_2_monthly_degenerate(self):
        spec = make_seasonal_factor(period=2, V=0.5, W_season=0.02)
        assert spec.d == 1
        np.testing.assert_array_equal(spec.G, [[-1.0]])

    def test_period_below_2_raises(self):
        with pytest.raises(ValueError, match="period"):
            make_seasonal_factor(period=1, V=0.5, W_season=0.02)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_models.py::TestMakeSeasonalFactor -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `make_seasonal_factor`**

Append to `engine/models.py`:

```python
_SEASONAL_NUGGET = 1e-8


def make_seasonal_factor(
    period: int,
    V: float,
    W_season: float,
    m0: np.ndarray | None = None,
    C0: np.ndarray | None = None,
) -> DLMSpec:
    """Seasonal DLM in factor form with sum-to-zero constraint.

    For a period-s seasonal pattern, the state has dim d = s - 1 and encodes
    the last s-1 seasonal effects; the current-time seasonal factor is their
    negated sum (enforcing sum-to-zero across a full cycle).

        F = [1, 0, ..., 0]                   # observation reads first slot
        G = companion form:
            [[-1, -1, ..., -1],              # new factor = -sum of previous
             [ 1,  0, ...,  0],
             [ 0,  1, ...,  0],
              ...
             [ 0,  0, ..., 1, 0]]

    Only the top-left entry of W carries the innovation variance W_season; a
    small nugget on the remaining diagonal keeps W strictly PD for validation.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    d = period - 1

    F = np.zeros((1, d))
    F[0, 0] = 1.0

    G = np.zeros((d, d))
    G[0, :] = -1.0
    if d > 1:
        # Identity-like shift: row i (>=1) picks up column i-1
        for i in range(1, d):
            G[i, i - 1] = 1.0

    W = np.diag([float(W_season)] + [_SEASONAL_NUGGET] * (d - 1))

    if m0 is None:
        m0 = np.zeros(d)
    if C0 is None:
        C0 = 1e3 * np.eye(d)

    return DLMSpec(
        F=F, G=G,
        V=np.array([[float(V)]]),
        W=W,
        m0=np.asarray(m0, dtype=float),
        C0=np.asarray(C0, dtype=float),
    )
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_models.py::TestMakeSeasonalFactor -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/models.py tests/test_models.py
git commit -m "feat(engine): make_seasonal_factor builder"
```

---

### Task 7: Model builder — `combine` (block-diagonal superposition)

**Files:**
- Modify: `engine/models.py` (append)
- Modify: `tests/test_models.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
from scipy.linalg import block_diag

from engine.models import combine


class TestCombine:
    def test_trend_plus_seasonal_shapes(self):
        trend = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        season = make_seasonal_factor(period=4, V=0.5, W_season=0.02)
        # Must pass one V; combine uses V of the first spec (all specs share the
        # same observation noise in a component DLM; enforced by an assertion).
        combined = combine(trend, season)
        assert combined.d == trend.d + season.d  # 2 + 3 = 5
        assert combined.p == 1
        np.testing.assert_array_equal(
            combined.F,
            np.hstack([trend.F, season.F]),
        )
        np.testing.assert_array_equal(
            combined.G,
            block_diag(trend.G, season.G),
        )
        np.testing.assert_array_equal(
            combined.W,
            block_diag(trend.W, season.W),
        )

    def test_rejects_single_arg(self):
        trend = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        with pytest.raises(ValueError, match="at least two"):
            combine(trend)

    def test_rejects_mismatched_V(self):
        a = make_local_level(V=0.5, W_level=0.1)
        b = make_local_level(V=1.0, W_level=0.1)
        with pytest.raises(ValueError, match="observation"):
            combine(a, b)

    def test_rejects_mismatched_p(self):
        # Hand-construct a p=2 spec
        p2 = DLMSpec(
            F=np.ones((2, 1)),
            G=np.eye(1),
            V=np.eye(2),
            W=np.eye(1),
            m0=np.zeros(1),
            C0=np.eye(1),
        )
        p1 = make_local_level(V=1.0, W_level=0.1)
        with pytest.raises(ValueError, match="observation dimension"):
            combine(p1, p2)

    def test_priors_block_diagonal(self):
        a = make_local_level(V=0.5, W_level=0.1, m0=10.0, C0=5.0)
        b = make_local_level(V=0.5, W_level=0.2, m0=-3.0, C0=1.0)
        c = combine(a, b)
        np.testing.assert_array_equal(c.m0, [10.0, -3.0])
        np.testing.assert_array_equal(c.C0, np.diag([5.0, 1.0]))
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_models.py::TestCombine -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `combine`**

Append to `engine/models.py`:

```python
from scipy.linalg import block_diag as _block_diag


def combine(*specs: DLMSpec) -> DLMSpec:
    """Superpose component DLMs into a single block-diagonal DLM.

    Components must share the same observation dimension p and observation
    covariance V. The state vector is the concatenation of each component's
    state; G and W become block-diagonal; F is horizontally stacked.
    """
    if len(specs) < 2:
        raise ValueError(f"combine requires at least two specs, got {len(specs)}")
    p0 = specs[0].p
    V0 = specs[0].V
    for s in specs[1:]:
        if s.p != p0:
            raise ValueError(f"all specs must share observation dimension; got {p0} and {s.p}")
        if not np.allclose(s.V, V0):
            raise ValueError("all specs must share observation noise covariance V")

    F = np.hstack([s.F for s in specs])
    G = _block_diag(*[s.G for s in specs])
    W = _block_diag(*[s.W for s in specs])
    m0 = np.concatenate([s.m0 for s in specs])
    C0 = _block_diag(*[s.C0 for s in specs])
    return DLMSpec(F=F, G=G, V=V0, W=W, m0=m0, C0=C0)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_models.py -v`
Expected: all model tests (Tasks 3-7) pass.

- [ ] **Step 5: Commit**

```bash
git add engine/models.py tests/test_models.py
git commit -m "feat(engine): combine component DLMs via block-diagonal superposition"
```

---

### Task 8: Simulator — forward-simulate y and theta from a DLMSpec

**Files:**
- Create: `engine/simulate.py`
- Create: `tests/test_simulate.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_simulate.py`:

```python
"""Tests for forward simulation."""

import numpy as np
import pytest

from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import SimulatedSeries, simulate


class TestSimulateBasics:
    def test_returns_simulated_series(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=50, seed=42)
        assert isinstance(series, SimulatedSeries)
        assert series.y.shape == (50, 1)
        assert series.theta_true.shape == (50, 1)
        assert series.seed == 42
        assert series.spec is spec

    def test_multivariate_shapes(self):
        # Hand-construct a p=2, d=1 spec
        from engine.models import DLMSpec
        spec = DLMSpec(
            F=np.array([[1.0], [0.5]]),
            G=np.eye(1),
            V=np.eye(2),
            W=np.eye(1),
            m0=np.zeros(1),
            C0=np.eye(1),
        )
        series = simulate(spec, n=30, seed=0)
        assert series.y.shape == (30, 2)
        assert series.theta_true.shape == (30, 1)


class TestSimulateReproducibility:
    def test_same_seed_bit_identical(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        s1 = simulate(spec, n=100, seed=7)
        s2 = simulate(spec, n=100, seed=7)
        np.testing.assert_array_equal(s1.y, s2.y)
        np.testing.assert_array_equal(s1.theta_true, s2.theta_true)

    def test_different_seed_different_output(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        s1 = simulate(spec, n=100, seed=7)
        s2 = simulate(spec, n=100, seed=8)
        assert not np.array_equal(s1.y, s2.y)


class TestSimulateDistribution:
    def test_local_level_empirical_variance(self):
        """For local level with V=1, W=0, y_t = theta_0 for all t, so Var(y_t)=0
        across realizations given the same prior mean. With a diffuse C0, y_t
        becomes very variable, which masks the noise. Use a tight C0 here."""
        from engine.models import DLMSpec
        spec = DLMSpec(
            F=np.array([[1.0]]),
            G=np.array([[1.0]]),
            V=np.array([[1.0]]),
            W=np.array([[1e-12]]),  # effectively zero
            m0=np.array([0.0]),
            C0=np.array([[1e-12]]),  # effectively deterministic prior
        )
        rng_outputs = [simulate(spec, n=1, seed=s).y[0, 0] for s in range(5000)]
        emp_var = float(np.var(rng_outputs))
        # Expected Var(y_0) = V + C0 ≈ 1.0
        assert 0.9 < emp_var < 1.1

    def test_invalid_n_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        with pytest.raises(ValueError, match="n"):
            simulate(spec, n=0, seed=0)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_simulate.py -v`
Expected: `ImportError: cannot import name 'simulate'`.

- [ ] **Step 3: Implement simulator**

Create `/Users/jameshenson/Documents/tutorials/DLM/engine/simulate.py`:

```python
"""Forward-simulate observations and latent states from a DLMSpec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.models import DLMSpec


@dataclass(frozen=True)
class SimulatedSeries:
    """One realization of a DLM.

    Attributes
    ----------
    y : (T, p) ndarray — observations
    theta_true : (T, d) ndarray — true latent states
    spec : DLMSpec — the generating specification
    seed : int — RNG seed used
    """

    y: np.ndarray
    theta_true: np.ndarray
    spec: DLMSpec
    seed: int


def simulate(spec: DLMSpec, n: int, seed: int) -> SimulatedSeries:
    """Forward-simulate n steps from a DLMSpec.

    Draws theta_0 ~ N(m0, C0), then for t = 1..n:
        theta_t = G theta_{t-1} + w_t,   w_t ~ N(0, W)
        y_t = F theta_t + v_t,           v_t ~ N(0, V)

    Returns a SimulatedSeries with y of shape (n, p) and theta_true (n, d).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    rng = np.random.default_rng(seed)
    p, d = spec.p, spec.d

    theta = np.empty((n, d))
    y = np.empty((n, p))

    # Draw theta_0 from the prior and advance one step to get theta_1; by
    # convention, t=1 is the first observed time.
    theta_prev = rng.multivariate_normal(spec.m0, spec.C0)
    for t in range(n):
        w_t = rng.multivariate_normal(np.zeros(d), spec.W)
        theta_t = spec.G @ theta_prev + w_t
        v_t = rng.multivariate_normal(np.zeros(p), spec.V)
        y_t = spec.F @ theta_t + v_t
        theta[t] = theta_t
        y[t] = y_t
        theta_prev = theta_t

    return SimulatedSeries(y=y, theta_true=theta, spec=spec, seed=seed)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_simulate.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/simulate.py tests/test_simulate.py
git commit -m "feat(engine): forward simulator with reproducible seeding"
```

---

### Task 9: Kalman filter — the teaching artifact

**Files:**
- Create: `engine/filter.py`
- Create: `tests/test_filter.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_filter.py`:

```python
"""Tests for the Kalman filter."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from engine.filter import FilterResult, kalman_filter
from engine.models import DLMSpec, make_local_level
from engine.simulate import simulate


def _ll_from_joint(spec: DLMSpec, y: np.ndarray) -> float:
    """Reference log-likelihood by direct computation of the marginal joint
    N(0, Sigma_y) covariance of y_{1:T}.

    For a stationary-prior local level this is tractable for small T.
    """
    T = y.shape[0]
    p = spec.p
    d = spec.d
    # Compute the full (T*p, T*p) covariance of y_{1:T}.
    F, G, V, W, m0, C0 = spec.F, spec.G, spec.V, spec.W, spec.m0, spec.C0
    # Mean of y_t: F @ G^t @ m0
    mean = np.concatenate([(F @ np.linalg.matrix_power(G, t + 1) @ m0) for t in range(T)])
    # Cov(theta_s, theta_t) = G^s C0 (G^t).T + sum_{k=1}^{min(s,t)} G^{s-k} W (G^{t-k}).T
    Sigma_theta = np.zeros((T * d, T * d))
    for s in range(T):
        Gs = np.linalg.matrix_power(G, s + 1)
        for t in range(T):
            Gt = np.linalg.matrix_power(G, t + 1)
            cov = Gs @ C0 @ Gt.T
            for k in range(1, min(s, t) + 2):
                Gsk = np.linalg.matrix_power(G, s + 1 - k)
                Gtk = np.linalg.matrix_power(G, t + 1 - k)
                cov = cov + Gsk @ W @ Gtk.T
            Sigma_theta[s * d:(s + 1) * d, t * d:(t + 1) * d] = cov
    # Full F_block and V_block
    F_block = np.kron(np.eye(T), F)
    V_block = np.kron(np.eye(T), V)
    Sigma_y = F_block @ Sigma_theta @ F_block.T + V_block
    # Symmetrize to guard fp drift
    Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)
    y_flat = y.reshape(-1)
    return float(multivariate_normal.logpdf(y_flat, mean=mean, cov=Sigma_y))


class TestKalmanFilterBasics:
    def test_returns_shapes(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=50, seed=0)
        res = kalman_filter(spec, series.y)
        assert isinstance(res, FilterResult)
        assert res.m.shape == (50, 1)
        assert res.C.shape == (50, 1, 1)
        assert res.a.shape == (50, 1)
        assert res.R.shape == (50, 1, 1)
        assert res.f.shape == (50, 1)
        assert res.Q.shape == (50, 1, 1)
        assert res.e.shape == (50, 1)
        assert np.isfinite(res.loglik)

    def test_innovations_consistency(self):
        """e_t must equal y_t - f_t."""
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=30, seed=1)
        res = kalman_filter(spec, series.y)
        np.testing.assert_allclose(res.e, series.y - res.f)


class TestKalmanFilterAgainstReference:
    def test_loglik_matches_joint_small_T(self):
        """For T=5, filter loglik must match direct joint-normal logpdf."""
        spec = make_local_level(V=1.0, W_level=0.3, m0=0.0, C0=2.0)
        series = simulate(spec, n=5, seed=42)
        res = kalman_filter(spec, series.y)
        ref = _ll_from_joint(spec, series.y)
        np.testing.assert_allclose(res.loglik, ref, atol=1e-10)


class TestKalmanFilterConvergence:
    def test_local_level_steady_state(self):
        """For constant local level, the filtered variance C_t converges to
        the positive root of the Riccati:  C_inf such that
            C_inf = (C_inf + W) * V / (C_inf + W + V)
        (see West & Harrison §2.3 / Petris §2.6)."""
        V = 1.0
        W = 0.1
        spec = make_local_level(V=V, W_level=W, m0=0.0, C0=1e6)
        # Long enough to reach steady state
        series = simulate(spec, n=500, seed=0)
        res = kalman_filter(spec, series.y)
        # Analytic fixed point: solve C = (C+W)V/(C+W+V)
        # => C(C+W+V) = V(C+W)
        # => C^2 + CW + CV = VC + VW
        # => C^2 + CW - VW = 0
        # => C = (-W + sqrt(W^2 + 4VW)) / 2
        C_inf = 0.5 * (-W + np.sqrt(W ** 2 + 4 * V * W))
        np.testing.assert_allclose(res.C[-1, 0, 0], C_inf, rtol=1e-3)


class TestKalmanFilterEdgeCases:
    def test_length_mismatch_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        y = np.zeros((10, 2))  # p=2 but spec has p=1
        with pytest.raises(ValueError, match="observation dimension"):
            kalman_filter(spec, y)

    def test_zero_length_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        with pytest.raises(ValueError, match="at least one"):
            kalman_filter(spec, np.zeros((0, 1)))
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_filter.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement the Kalman filter**

Create `/Users/jameshenson/Documents/tutorials/DLM/engine/filter.py`:

```python
"""Kalman filter for time-invariant Gaussian DLMs.

This module is a teaching artifact. Each update step is annotated with the
corresponding equation number in West & Harrison, *Bayesian Forecasting and
Dynamic Models* (2nd ed., 1997), chapter 4. Reading this file alongside that
chapter should make the correspondence obvious.

Notation (West-Harrison):
    m_t, C_t       filtered  posterior     theta_t | y_{1:t}   ~ N(m_t, C_t)
    a_t, R_t       predictive state prior  theta_t | y_{1:t-1} ~ N(a_t, R_t)
    f_t, Q_t       predictive obs          y_t    | y_{1:t-1} ~ N(f_t, Q_t)
    e_t            innovation (forecast error)  e_t = y_t - f_t
    A_t            Kalman gain             A_t = R_t F' Q_t^{-1}

For numerical stability we use the Joseph form for the covariance update.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from engine.models import DLMSpec


@dataclass(frozen=True)
class FilterResult:
    """Output of the Kalman filter.

    Attributes
    ----------
    m : (T, d)   filtered state means          E[theta_t | y_{1:t}]
    C : (T, d, d) filtered state covariances
    a : (T, d)   one-step predictive state means E[theta_t | y_{1:t-1}]
    R : (T, d, d) one-step predictive state covariances
    f : (T, p)   one-step predictive obs means   E[y_t | y_{1:t-1}]
    Q : (T, p, p) one-step predictive obs covariances
    e : (T, p)   innovations y_t - f_t
    loglik : total log-marginal-likelihood sum_t log p(y_t | y_{1:t-1})
    """

    m: np.ndarray
    C: np.ndarray
    a: np.ndarray
    R: np.ndarray
    f: np.ndarray
    Q: np.ndarray
    e: np.ndarray
    loglik: float


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _solve_psd(Q: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve Q X = B for X using Cholesky; fall back to pinv if singular."""
    try:
        c, low = cho_factor(Q, lower=True)
        return cho_solve((c, low), B)
    except LinAlgError:
        warnings.warn("Q_t singular in Kalman filter; falling back to pinv", RuntimeWarning)
        return np.linalg.pinv(Q) @ B


def _logdet_psd(Q: np.ndarray) -> float:
    """log|Q| for symmetric PSD Q via Cholesky (or pinv fallback)."""
    try:
        c, _ = cho_factor(Q, lower=True)
        return 2.0 * float(np.sum(np.log(np.diag(c))))
    except LinAlgError:
        sign, logabsdet = np.linalg.slogdet(Q)
        return float(logabsdet)


def kalman_filter(spec: DLMSpec, y: np.ndarray) -> FilterResult:
    """Run the Kalman filter on observations `y` using `spec`.

    `y` has shape (T, p). Returns a FilterResult.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D (T, p), got shape {y.shape}")
    T, p = y.shape
    if T < 1:
        raise ValueError("y must have at least one observation")
    if p != spec.p:
        raise ValueError(f"y observation dimension {p} does not match spec.p {spec.p}")

    d = spec.d
    F, G, V, W = spec.F, spec.G, spec.V, spec.W
    m_prev = spec.m0
    C_prev = spec.C0

    m = np.empty((T, d))
    C = np.empty((T, d, d))
    a = np.empty((T, d))
    R = np.empty((T, d, d))
    f = np.empty((T, p))
    Q = np.empty((T, p, p))
    e = np.empty((T, p))
    loglik = 0.0

    # Constant used by the log-likelihood: -p/2 * log(2 pi)
    log2pi = float(np.log(2.0 * np.pi))

    for t in range(T):
        # --- Prior for theta_t given y_{1:t-1}   (West-Harrison eq 4.3) ----
        a_t = G @ m_prev                                       # E[theta_t | y_{1:t-1}]
        R_t = _symmetrize(G @ C_prev @ G.T + W)                # Cov[theta_t | y_{1:t-1}]

        # --- One-step-ahead obs forecast         (West-Harrison eq 4.4) ----
        f_t = F @ a_t                                          # E[y_t | y_{1:t-1}]
        Q_t = _symmetrize(F @ R_t @ F.T + V)                   # Cov[y_t | y_{1:t-1}]

        # --- Innovation ----
        e_t = y[t] - f_t                                        # forecast error

        # --- Kalman gain and posterior update    (West-Harrison eq 4.5-6) ---
        #   A_t = R_t F' Q_t^{-1}
        # Use _solve_psd to compute Q_t^{-1} (F R_t)'  stably.
        FRt = F @ R_t                                          # (p, d)
        A_t = _solve_psd(Q_t, FRt).T                           # (d, p)

        m_t = a_t + A_t @ e_t                                  # posterior mean
        # Joseph form for PSD stability:  C_t = (I - A F) R (I - A F)' + A V A'
        I = np.eye(d)
        IAF = I - A_t @ F
        C_t = _symmetrize(IAF @ R_t @ IAF.T + A_t @ V @ A_t.T)

        # --- Accumulate log-likelihood  log N(y_t; f_t, Q_t) ---
        logdet_Q = _logdet_psd(Q_t)
        quad = float(e_t @ _solve_psd(Q_t, e_t))
        loglik += -0.5 * (p * log2pi + logdet_Q + quad)

        m[t], C[t] = m_t, C_t
        a[t], R[t] = a_t, R_t
        f[t], Q[t] = f_t, Q_t
        e[t] = e_t

        m_prev, C_prev = m_t, C_t

    return FilterResult(m=m, C=C, a=a, R=R, f=f, Q=Q, e=e, loglik=loglik)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_filter.py -v`
Expected: all pass (loglik match within 1e-10, steady-state convergence within rtol 1e-3).

- [ ] **Step 5: Commit**

```bash
git add engine/filter.py tests/test_filter.py
git commit -m "feat(engine): annotated Kalman filter with Joseph form"
```

---

### Task 10: RTS smoother

**Files:**
- Create: `engine/smoother.py`
- Create: `tests/test_smoother.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_smoother.py`:

```python
"""Tests for the RTS smoother."""

import numpy as np

from engine.filter import kalman_filter
from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import simulate
from engine.smoother import SmoothResult, rts_smoother


class TestSmootherBasics:
    def test_returns_shapes(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=40, seed=0)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        assert isinstance(sr, SmoothResult)
        assert sr.s.shape == (40, 1)
        assert sr.S.shape == (40, 1, 1)

    def test_terminal_equals_filtered(self):
        """At t=T, smoother output equals filter output."""
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        series = simulate(spec, n=30, seed=1)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        np.testing.assert_allclose(sr.s[-1], fr.m[-1])
        np.testing.assert_allclose(sr.S[-1], fr.C[-1])

    def test_covariance_symmetric(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        series = simulate(spec, n=30, seed=1)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        for t in range(sr.S.shape[0]):
            np.testing.assert_allclose(sr.S[t], sr.S[t].T, atol=1e-10)

    def test_smoother_variance_not_greater_than_filter(self):
        """Smoothed variance <= filtered variance at each t (smoother uses more info)."""
        spec = make_local_level(V=1.0, W_level=0.1)
        series = simulate(spec, n=100, seed=2)
        fr = kalman_filter(spec, series.y)
        sr = rts_smoother(spec, fr)
        for t in range(99):
            assert sr.S[t, 0, 0] <= fr.C[t, 0, 0] + 1e-10
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_smoother.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement RTS smoother**

Create `/Users/jameshenson/Documents/tutorials/DLM/engine/smoother.py`:

```python
"""Rauch-Tung-Striebel (RTS) smoother for Gaussian DLMs.

Standard backward pass:
    s_T = m_T,  S_T = C_T
    for t = T-1 down to 1:
        B_t = C_t G' R_{t+1}^{-1}
        s_t = m_t + B_t (s_{t+1} - a_{t+1})
        S_t = C_t + B_t (S_{t+1} - R_{t+1}) B_t'

See West & Harrison chapter 4.8 and Petris chapter 2.4.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.filter import FilterResult, _solve_psd, _symmetrize
from engine.models import DLMSpec


@dataclass(frozen=True)
class SmoothResult:
    """Output of the RTS smoother.

    Attributes
    ----------
    s : (T, d) smoothed state means  E[theta_t | y_{1:T}]
    S : (T, d, d) smoothed state covariances
    """

    s: np.ndarray
    S: np.ndarray


def rts_smoother(spec: DLMSpec, fr: FilterResult) -> SmoothResult:
    T, d = fr.m.shape
    G = spec.G

    s = np.empty((T, d))
    S = np.empty((T, d, d))
    s[-1] = fr.m[-1]
    S[-1] = fr.C[-1]

    for t in range(T - 2, -1, -1):
        # B_t = C_t G' R_{t+1}^{-1}
        C_t = fr.C[t]
        R_next = fr.R[t + 1]
        # Solve R_next X = (C_t G')'  =>  X = R_next^{-1} G C_t
        # then B_t = C_t G' R_next^{-1} = X.T
        CG_T = C_t @ G.T                     # (d, d)
        B_t = _solve_psd(R_next, CG_T.T).T   # (d, d)
        s[t] = fr.m[t] + B_t @ (s[t + 1] - fr.a[t + 1])
        S[t] = _symmetrize(C_t + B_t @ (S[t + 1] - R_next) @ B_t.T)

    return SmoothResult(s=s, S=S)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_smoother.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/smoother.py tests/test_smoother.py
git commit -m "feat(engine): RTS smoother"
```

---

### Task 11: h-step forecast with credible bands

**Files:**
- Create: `engine/forecast.py`
- Create: `tests/test_forecast.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_forecast.py`:

```python
"""Tests for the DLM h-step-ahead forecast."""

import numpy as np
import pytest

from engine.filter import kalman_filter
from engine.forecast import Forecast, forecast_horizon
from engine.models import make_local_level, make_local_linear_trend
from engine.simulate import simulate


class TestForecastBasics:
    def test_shapes(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=50, seed=0)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=10)
        assert isinstance(fc, Forecast)
        assert fc.horizon == 10
        assert fc.means.shape == (10, 1)
        assert fc.lower.shape == (10, 1)
        assert fc.upper.shape == (10, 1)

    def test_horizon_zero_empty(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=20, seed=0)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=0)
        assert fc.means.shape == (0, 1)
        assert fc.lower.shape == (0, 1)
        assert fc.upper.shape == (0, 1)

    def test_negative_horizon_raises(self):
        spec = make_local_level(V=0.5, W_level=0.1)
        series = simulate(spec, n=20, seed=0)
        fr = kalman_filter(spec, series.y)
        with pytest.raises(ValueError, match="horizon"):
            forecast_horizon(spec, fr, h=-1)

    def test_constant_level_forecast_is_flat(self):
        """With W=0 (tiny nugget), forecast mean is constant = last filtered mean."""
        # Use tiny W to stay within the positive-definite constraint.
        spec = make_local_level(V=0.5, W_level=1e-10)
        series = simulate(spec, n=60, seed=2)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=5)
        for t in range(5):
            np.testing.assert_allclose(fc.means[t], fr.m[-1], atol=1e-6)


class TestForecastBandsMonotone:
    def test_band_width_increases_with_horizon(self):
        spec = make_local_linear_trend(V=0.5, W_level=0.05, W_slope=0.01)
        series = simulate(spec, n=80, seed=3)
        fr = kalman_filter(spec, series.y)
        fc = forecast_horizon(spec, fr, h=20)
        widths = fc.upper[:, 0] - fc.lower[:, 0]
        # Strictly non-decreasing
        assert np.all(np.diff(widths) >= -1e-10)
        # And strictly increasing over the whole horizon
        assert widths[-1] > widths[0]
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_forecast.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement forecast**

Create `/Users/jameshenson/Documents/tutorials/DLM/engine/forecast.py`:

```python
"""Forward-propagate filtered posterior to produce h-step-ahead forecasts.

For k = 1..h, the predictive distribution is
    theta_{T+k} | y_{1:T} ~ N(a^{(k)}, R^{(k)})
    y_{T+k}     | y_{1:T} ~ N(F a^{(k)}, F R^{(k)} F' + V)
with recursion
    a^{(1)} = G m_T,   R^{(1)} = G C_T G' + W
    a^{(k)} = G a^{(k-1)},  R^{(k)} = G R^{(k-1)} G' + W

Marginal credible bands at level 1 - alpha (default 95%) are
    [f_k - z * sqrt(diag(Q_k)),  f_k + z * sqrt(diag(Q_k))]
where z = Phi^{-1}(1 - alpha/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from engine.filter import FilterResult, _symmetrize
from engine.models import DLMSpec


@dataclass(frozen=True)
class Forecast:
    """h-step-ahead forecast, marginal per observation component.

    Attributes
    ----------
    horizon : int
    means   : (h, p) predictive means
    lower   : (h, p) predictive 1-alpha/2 lower bound
    upper   : (h, p) predictive 1-alpha/2 upper bound
    """

    horizon: int
    means: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def forecast_horizon(
    spec: DLMSpec,
    fr: FilterResult,
    h: int,
    alpha: float = 0.05,
) -> Forecast:
    if h < 0:
        raise ValueError(f"horizon h must be >= 0, got {h}")
    p = spec.p

    if h == 0:
        zeros = np.empty((0, p))
        return Forecast(horizon=0, means=zeros, lower=zeros, upper=zeros)

    F, G, V, W = spec.F, spec.G, spec.V, spec.W
    a = fr.m[-1].copy()
    R = fr.C[-1].copy()

    means = np.empty((h, p))
    lower = np.empty((h, p))
    upper = np.empty((h, p))
    z = float(norm.ppf(1 - alpha / 2))

    for k in range(h):
        a = G @ a
        R = _symmetrize(G @ R @ G.T + W)
        f_k = F @ a                             # predictive mean of y_{T+k+1}
        Q_k = _symmetrize(F @ R @ F.T + V)      # predictive covariance
        sd = np.sqrt(np.diag(Q_k))
        means[k] = f_k
        lower[k] = f_k - z * sd
        upper[k] = f_k + z * sd

    return Forecast(horizon=h, means=means, lower=lower, upper=upper)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_forecast.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/forecast.py tests/test_forecast.py
git commit -m "feat(engine): h-step forecast with credible bands"
```

---

### Task 12: Diagnostics — ACF/PACF and residual statistics

**Files:**
- Create: `engine/diagnostics.py`
- Create: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_diagnostics.py`:

```python
"""Tests for ACF/PACF and residual diagnostics."""

import numpy as np
import pytest

from engine.diagnostics import (
    AcfPacfResult,
    ResidualDiagnostics,
    acf_pacf,
    residual_diagnostics,
)


class TestAcfPacf:
    def test_returns_expected_shapes(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(500, 1))
        res = acf_pacf(y, nlags=20)
        assert isinstance(res, AcfPacfResult)
        assert res.lags.shape == (21,)       # lag 0 .. 20
        assert res.acf.shape == (21,)
        assert res.pacf.shape == (21,)
        assert res.lags[0] == 0
        assert res.lags[-1] == 20

    def test_white_noise_acf_within_band(self):
        rng = np.random.default_rng(7)
        y = rng.normal(size=(2000, 1))
        res = acf_pacf(y, nlags=20)
        # 95% band ~ 1.96 / sqrt(n) ≈ 0.0438; some small exceedances are OK
        n_exceed = int(np.sum(np.abs(res.acf[1:]) > 1.96 / np.sqrt(2000)))
        assert n_exceed <= 2     # at most 2 of 20 outside band

    def test_rejects_multivariate(self):
        y = np.zeros((50, 2))
        with pytest.raises(ValueError, match="univariate"):
            acf_pacf(y, nlags=10)


class TestResidualDiagnostics:
    def test_shapes_and_fields(self):
        rng = np.random.default_rng(0)
        e = rng.normal(size=(200, 1))        # innovations
        Q = np.ones((200, 1, 1))             # unit predictive variance
        res = residual_diagnostics(e, Q, nlags=15)
        assert isinstance(res, ResidualDiagnostics)
        assert res.standardized.shape == (200, 1)
        assert res.acf_pacf.lags.shape == (16,)
        assert 0.0 <= res.ljung_box_pvalue <= 1.0

    def test_white_noise_ljung_box_not_rejected(self):
        rng = np.random.default_rng(11)
        e = rng.normal(size=(500, 1))
        Q = np.ones((500, 1, 1))
        res = residual_diagnostics(e, Q, nlags=10)
        assert res.ljung_box_pvalue > 0.05

    def test_correlated_residuals_ljung_box_rejects(self):
        """AR(1) residuals should be flagged."""
        rng = np.random.default_rng(12)
        phi = 0.7
        eps = rng.normal(size=600)
        e = np.zeros(600)
        for t in range(1, 600):
            e[t] = phi * e[t - 1] + eps[t]
        e2d = e[:, None]
        Q = np.ones((600, 1, 1))
        res = residual_diagnostics(e2d, Q, nlags=10)
        assert res.ljung_box_pvalue < 0.01
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_diagnostics.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement diagnostics**

Create `/Users/jameshenson/Documents/tutorials/DLM/engine/diagnostics.py`:

```python
"""ACF/PACF plotting data and residual diagnostics.

Uses statsmodels for ACF, PACF, and Ljung-Box as these are well-tested
reference implementations that would be wasteful to reinvent for a tutorial.
Everything else (standardization, assembly into result dataclasses) is
pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf


@dataclass(frozen=True)
class AcfPacfResult:
    """ACF and PACF values at lags 0..nlags."""

    lags: np.ndarray   # (nlags+1,)
    acf: np.ndarray    # (nlags+1,)
    pacf: np.ndarray   # (nlags+1,)


@dataclass(frozen=True)
class ResidualDiagnostics:
    """One-step-forecast residual diagnostics.

    Attributes
    ----------
    standardized : (T, p) standardized innovations e_t / sqrt(diag(Q_t))
    acf_pacf     : ACF/PACF of univariate standardized residuals
                   (only populated for p=1; for p>1 acf_pacf is of the first
                   component for visualization convenience)
    ljung_box_pvalue : float — portmanteau p-value (null: white noise)
    """

    standardized: np.ndarray
    acf_pacf: AcfPacfResult
    ljung_box_pvalue: float


def acf_pacf(y: np.ndarray, nlags: int = 20) -> AcfPacfResult:
    """Compute ACF and PACF for a univariate series of shape (T, 1) or (T,)."""
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(
                f"acf_pacf requires univariate series; got shape {arr.shape}"
            )
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"acf_pacf requires a 1D or (T,1) array; got {arr.shape}")
    a = _acf(arr, nlags=nlags, fft=False)
    p = _pacf(arr, nlags=nlags, method="yw")
    return AcfPacfResult(
        lags=np.arange(nlags + 1),
        acf=np.asarray(a),
        pacf=np.asarray(p),
    )


def residual_diagnostics(
    e: np.ndarray,
    Q: np.ndarray,
    nlags: int = 20,
) -> ResidualDiagnostics:
    """Diagnostics for innovations `e` with predictive covariances `Q`.

    Parameters
    ----------
    e : (T, p) innovations from the Kalman filter
    Q : (T, p, p) predictive covariances
    nlags : number of ACF/PACF lags to compute for the first component
    """
    if e.ndim != 2:
        raise ValueError(f"e must be 2D (T, p), got {e.shape}")
    T, p = e.shape
    if Q.shape != (T, p, p):
        raise ValueError(f"Q shape {Q.shape} must match (T, p, p) = ({T}, {p}, {p})")

    # Standardize each component by its marginal predictive sd.
    sd = np.sqrt(np.diagonal(Q, axis1=1, axis2=2))   # (T, p)
    standardized = e / sd

    # Ljung-Box on the first component (for p=1, this is the only component;
    # for p>1, per-component test is sufficient for the Beginner tier).
    first = standardized[:, 0]
    lb = acorr_ljungbox(first, lags=[nlags], return_df=True)
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    ap = acf_pacf(first[:, None], nlags=nlags)

    return ResidualDiagnostics(
        standardized=standardized,
        acf_pacf=ap,
        ljung_box_pvalue=lb_pvalue,
    )
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_diagnostics.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add engine/diagnostics.py tests/test_diagnostics.py
git commit -m "feat(engine): ACF/PACF and residual diagnostics"
```

---

## Phase 2 — Lesson framework

### Task 13: Workflow types — `ParamSpec`, `ChallengeQuestion`, `WorkflowStep`, `Lesson`

**Files:**
- Create: `lessons/workflow.py`
- Create: `tests/test_lessons.py` (partial — only type tests)

- [ ] **Step 1: Write failing tests**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_lessons.py`:

```python
"""Tests for lesson content types and lesson content integrity."""

import numpy as np
import pytest

from engine.models import DLMSpec, make_local_level
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    WorkflowStep,
    validate_lesson,
)


class TestParamSpec:
    def test_valid_paramspec(self):
        p = ParamSpec(
            name="V", label="Obs variance", min=1e-6, max=5.0,
            default=0.5, step=0.01, help="blah",
        )
        assert p.name == "V"
        assert p.min < p.default < p.max

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="min"):
            ParamSpec(name="x", label="x", min=1.0, max=0.5,
                      default=0.7, step=0.01, help="")

    def test_default_outside_range_raises(self):
        with pytest.raises(ValueError, match="default"):
            ParamSpec(name="x", label="x", min=0.0, max=1.0,
                      default=2.0, step=0.01, help="")


class TestChallengeQuestion:
    def test_multiple_choice_correct_must_be_str(self):
        with pytest.raises(ValueError, match="multiple_choice"):
            ChallengeQuestion(
                kind="multiple_choice", correct=42,
                feedback_correct="", feedback_incorrect="",
            )

    def test_numeric_range_correct_must_be_tuple(self):
        with pytest.raises(ValueError, match="numeric_range"):
            ChallengeQuestion(
                kind="numeric_range", correct="string",
                feedback_correct="", feedback_incorrect="",
            )

    def test_component_toggle_correct_must_be_dict(self):
        with pytest.raises(ValueError, match="component_toggle"):
            ChallengeQuestion(
                kind="component_toggle", correct=[True, False],
                feedback_correct="", feedback_incorrect="",
            )


class TestWorkflowStepAndLesson:
    def test_lesson_rejects_empty_workflow(self):
        with pytest.raises(ValueError, match="workflow"):
            Lesson(
                id="x", title="x", tier="beginner", description="",
                model_builder=lambda p: make_local_level(V=1.0, W_level=0.1),
                param_schema=[],
                workflow_steps=[],
            )

    def test_validate_lesson_accepts_good_lesson(self):
        params = [
            ParamSpec(name="V", label="V", min=1e-6, max=5.0,
                      default=0.5, step=0.01, help=""),
            ParamSpec(name="W_level", label="W", min=1e-6, max=1.0,
                      default=0.1, step=0.01, help=""),
        ]
        steps = [
            WorkflowStep(
                id="inspect", title="Inspect", prompt_md="Look.",
                plot_fn="time_series", hints=[], challenge=None,
            ),
        ]
        lesson = Lesson(
            id="test", title="Test", tier="beginner", description="",
            model_builder=lambda p: make_local_level(V=p["V"], W_level=p["W_level"]),
            param_schema=params,
            workflow_steps=steps,
        )
        # Should build a valid DLMSpec with default params
        spec = lesson.model_builder(
            {p.name: p.default for p in lesson.param_schema}
        )
        assert isinstance(spec, DLMSpec)
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_lessons.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement workflow types**

Create `/Users/jameshenson/Documents/tutorials/DLM/lessons/workflow.py`:

```python
"""Declarative types that describe a lesson's content and structure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from engine.models import DLMSpec

ChallengeKind = Literal["multiple_choice", "numeric_range", "component_toggle"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    min: float
    max: float
    default: float
    step: float
    help: str

    def __post_init__(self) -> None:
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"default ({self.default}) must be within [min, max] = "
                f"[{self.min}, {self.max}]"
            )
        if self.step <= 0:
            raise ValueError(f"step must be > 0, got {self.step}")


@dataclass(frozen=True)
class ChallengeQuestion:
    kind: ChallengeKind
    correct: Any
    feedback_correct: str
    feedback_incorrect: str

    def __post_init__(self) -> None:
        if self.kind == "multiple_choice":
            if not isinstance(self.correct, str):
                raise ValueError(
                    f"multiple_choice 'correct' must be str, got {type(self.correct).__name__}"
                )
        elif self.kind == "numeric_range":
            ok = (
                isinstance(self.correct, tuple)
                and len(self.correct) == 2
                and all(isinstance(x, (int, float)) for x in self.correct)
                and self.correct[0] <= self.correct[1]
            )
            if not ok:
                raise ValueError(
                    f"numeric_range 'correct' must be (low, high) tuple, got {self.correct!r}"
                )
        elif self.kind == "component_toggle":
            if not (isinstance(self.correct, dict)
                    and all(isinstance(v, bool) for v in self.correct.values())):
                raise ValueError(
                    f"component_toggle 'correct' must be dict[str, bool], got {self.correct!r}"
                )
        else:
            raise ValueError(f"unknown ChallengeQuestion kind {self.kind!r}")


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    title: str
    prompt_md: str
    plot_fn: str
    hints: list[str] = field(default_factory=list)
    challenge: ChallengeQuestion | None = None


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    tier: str
    description: str
    model_builder: Callable[[dict[str, float]], DLMSpec]
    param_schema: list[ParamSpec]
    workflow_steps: list[WorkflowStep]

    def __post_init__(self) -> None:
        if not self.workflow_steps:
            raise ValueError(f"Lesson {self.id!r} has empty workflow_steps")
        if not self.param_schema:
            raise ValueError(f"Lesson {self.id!r} has empty param_schema")
        names = [p.name for p in self.param_schema]
        if len(set(names)) != len(names):
            raise ValueError(f"Lesson {self.id!r} has duplicate param names {names}")
        step_ids = [s.id for s in self.workflow_steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError(f"Lesson {self.id!r} has duplicate step ids {step_ids}")


def validate_lesson(lesson: Lesson, allowed_plot_fns: set[str]) -> None:
    """Validate a lesson against the set of plot function names available in the UI.

    Raises ValueError on any of:
    - model_builder fails with default param values
    - any WorkflowStep.plot_fn is not in allowed_plot_fns
    """
    defaults = {p.name: p.default for p in lesson.param_schema}
    try:
        spec = lesson.model_builder(defaults)
    except Exception as e:  # pragma: no cover — behavior surfaced via test_lessons
        raise ValueError(
            f"Lesson {lesson.id!r}: model_builder failed on defaults {defaults}: {e}"
        ) from e
    if not isinstance(spec, DLMSpec):
        raise ValueError(
            f"Lesson {lesson.id!r}: model_builder returned {type(spec).__name__}, not DLMSpec"
        )
    for step in lesson.workflow_steps:
        if step.plot_fn not in allowed_plot_fns:
            raise ValueError(
                f"Lesson {lesson.id!r} step {step.id!r}: unknown plot_fn {step.plot_fn!r}; "
                f"available: {sorted(allowed_plot_fns)}"
            )
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_lessons.py -v`
Expected: type tests pass.

- [ ] **Step 5: Commit**

```bash
git add lessons/__init__.py lessons/workflow.py tests/test_lessons.py
git commit -m "feat(lessons): workflow dataclasses and validator"
```

---

### Task 14: The canonical 9-step workflow skeleton (step ids, titles, default prompts)

**Files:**
- Modify: `lessons/workflow.py` (append canonical step builders)
- Modify: `tests/test_lessons.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_lessons.py`:

```python
from lessons.workflow import canonical_step_ids, make_default_workflow_steps


class TestCanonicalWorkflow:
    def test_nine_step_ids(self):
        assert canonical_step_ids() == [
            "inspect_data",
            "decompose",
            "quantify",
            "pick_components",
            "specify",
            "fit",
            "diagnose",
            "forecast",
            "reveal",
        ]

    def test_default_steps_have_all_nine(self):
        steps = make_default_workflow_steps(has_seasonal=False)
        assert len(steps) == 9
        assert [s.id for s in steps] == canonical_step_ids()

    def test_seasonal_flag_changes_quantify_plot(self):
        no = make_default_workflow_steps(has_seasonal=False)
        yes = make_default_workflow_steps(has_seasonal=True)
        quant_no = next(s for s in no if s.id == "quantify")
        quant_yes = next(s for s in yes if s.id == "quantify")
        assert quant_no.plot_fn == "acf_pacf"
        assert quant_yes.plot_fn == "acf_pacf_and_seasonal_subseries"
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_lessons.py::TestCanonicalWorkflow -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement canonical step builders**

Append to `lessons/workflow.py`:

```python
_CANONICAL_STEP_IDS: tuple[str, ...] = (
    "inspect_data",
    "decompose",
    "quantify",
    "pick_components",
    "specify",
    "fit",
    "diagnose",
    "forecast",
    "reveal",
)


def canonical_step_ids() -> list[str]:
    return list(_CANONICAL_STEP_IDS)


def make_default_workflow_steps(has_seasonal: bool) -> list[WorkflowStep]:
    """The 9-step workflow with default parameter-agnostic prompts.

    Lessons customize steps 4 (pick_components), 5 (specify), and 9 (reveal)
    by passing in `challenge` questions at construction time. The defaults
    here are info-only (challenge=None) and serve the Lesson-mode narrative.
    """
    quantify_plot_fn = "acf_pacf_and_seasonal_subseries" if has_seasonal else "acf_pacf"
    return [
        WorkflowStep(
            id="inspect_data",
            title="Step 1 — Inspect the data",
            prompt_md=(
                "Any time series analysis starts with looking at the data. "
                "What patterns stand out? A persistent level? A drift? "
                "Anything periodic? Jot a mental answer before moving on."
            ),
            plot_fn="time_series",
        ),
        WorkflowStep(
            id="decompose",
            title="Step 2 — Decompose visually",
            prompt_md=(
                "Visual decomposition is the first hypothesis step. "
                "A shaded overlay of a moving-average trend helps surface a drift "
                "even when the noise is large. [More detail]"
                "(/reference#baseline-procedure)."
            ),
            plot_fn="visual_decomposition",
        ),
        WorkflowStep(
            id="quantify",
            title="Step 3 — Quantify autocorrelation",
            prompt_md=(
                "The sample ACF and PACF let us *measure* what the eye suggested. "
                "A slow ACF decay is a trend signature; a spike at lag s is seasonal. "
                "[Reference](/reference#baseline-procedure)."
            ),
            plot_fn=quantify_plot_fn,
        ),
        WorkflowStep(
            id="pick_components",
            title="Step 4 — Pick components",
            prompt_md=(
                "Decide which DLM components you need: a level, a slope, a seasonal. "
                "This is where your answers to steps 2 and 3 crystallize into a model."
            ),
            plot_fn="blank",
        ),
        WorkflowStep(
            id="specify",
            title="Step 5 — Specify the DLM",
            prompt_md=(
                "Write down F, G, V, W for the components you chose. "
                "[Reference on specification](/reference#baseline-procedure)."
            ),
            plot_fn="spec_preview",
        ),
        WorkflowStep(
            id="fit",
            title="Step 6 — Fit (Kalman filter)",
            prompt_md=(
                "Run the Kalman filter. The filtered state means E[theta_t|y_{1:t}] "
                "track the underlying components, with 95% credible bands showing the "
                "posterior uncertainty."
            ),
            plot_fn="filter_state",
        ),
        WorkflowStep(
            id="diagnose",
            title="Step 7 — Diagnose (residuals + smoothing)",
            prompt_md=(
                "One-step forecast residuals should look like white noise. "
                "We also compare the smoothed state (using all of y_{1:T}) against "
                "the filtered state. [Reference on diagnostics](/reference#diagnostics)."
            ),
            plot_fn="diagnostics",
        ),
        WorkflowStep(
            id="forecast",
            title="Step 8 — Forecast",
            prompt_md=(
                "h-step-ahead forecasts with 95% credible bands. Bands widen "
                "with horizon — that is the price of uncertainty."
            ),
            plot_fn="forecast",
        ),
        WorkflowStep(
            id="reveal",
            title="Step 9 — Review",
            prompt_md=(
                "Lesson mode: recap of the method. "
                "Challenge mode: your fitted DLM overlaid against the ground-truth DLM."
            ),
            plot_fn="reveal_overlay",
        ),
    ]
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_lessons.py -v`
Expected: all lesson tests pass.

- [ ] **Step 5: Commit**

```bash
git add lessons/workflow.py tests/test_lessons.py
git commit -m "feat(lessons): canonical 9-step workflow skeleton"
```

---

### Task 15: Lesson 1 — Local level

**Files:**
- Create: `lessons/local_level.py`
- Modify: `tests/test_lessons.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_lessons.py`:

```python
from lessons.local_level import LESSON as LOCAL_LEVEL_LESSON


class TestLocalLevelLesson:
    def test_builds_valid_spec_on_defaults(self):
        params = {p.name: p.default for p in LOCAL_LEVEL_LESSON.param_schema}
        spec = LOCAL_LEVEL_LESSON.model_builder(params)
        assert spec.d == 1 and spec.p == 1

    def test_has_nine_steps(self):
        assert len(LOCAL_LEVEL_LESSON.workflow_steps) == 9

    def test_pick_components_question_is_toggle(self):
        step4 = next(s for s in LOCAL_LEVEL_LESSON.workflow_steps if s.id == "pick_components")
        assert step4.challenge is not None
        assert step4.challenge.kind == "component_toggle"
        # Correct answer: level=True, slope=False, seasonal=False
        assert step4.challenge.correct == {"level": True, "slope": False, "seasonal": False}

    def test_specify_variance_order_of_magnitude(self):
        step5 = next(s for s in LOCAL_LEVEL_LESSON.workflow_steps if s.id == "specify")
        assert step5.challenge is not None
        assert step5.challenge.kind == "numeric_range"
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_lessons.py::TestLocalLevelLesson -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement lesson 1**

Create `/Users/jameshenson/Documents/tutorials/DLM/lessons/local_level.py`:

```python
"""Beginner lesson 1: local level (random walk plus observation noise)."""

from __future__ import annotations

from engine.models import DLMSpec, make_local_level
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    WorkflowStep,
    make_default_workflow_steps,
)


def _build(params: dict[str, float]) -> DLMSpec:
    return make_local_level(V=params["V"], W_level=params["W_level"])


_PARAMS = [
    ParamSpec(
        name="V",
        label="Observation variance V",
        min=1e-6, max=5.0, default=0.5, step=0.01,
        help="Variance of the measurement noise v_t ~ N(0, V).",
    ),
    ParamSpec(
        name="W_level",
        label="Level innovation variance W",
        min=1e-6, max=1.0, default=0.05, step=0.001,
        help="Variance of the level innovation w_t ~ N(0, W).",
    ),
    ParamSpec(
        name="n",
        label="Series length n",
        min=20, max=2000, default=200, step=10,
        help="Number of observations to simulate.",
    ),
]


def _attach_challenges(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    """Add challenge questions to steps 4 and 5 for Challenge mode."""
    out: list[WorkflowStep] = []
    for s in steps:
        if s.id == "pick_components":
            s = WorkflowStep(
                id=s.id, title=s.title, prompt_md=s.prompt_md,
                plot_fn=s.plot_fn, hints=s.hints,
                challenge=ChallengeQuestion(
                    kind="component_toggle",
                    correct={"level": True, "slope": False, "seasonal": False},
                    feedback_correct=(
                        "Right. A local level is the simplest DLM: only an "
                        "unobserved level that drifts, plus observation noise."
                    ),
                    feedback_incorrect=(
                        "Not quite. Look at the ACF — a slow monotone decay "
                        "without a spike is the local-level signature. No slope, "
                        "no seasonal."
                    ),
                ),
            )
        elif s.id == "specify":
            s = WorkflowStep(
                id=s.id, title=s.title, prompt_md=s.prompt_md,
                plot_fn=s.plot_fn, hints=s.hints,
                challenge=ChallengeQuestion(
                    kind="numeric_range",
                    # Order-of-magnitude match for W_level (accept within factor of 3x)
                    correct=(0.001, 0.5),
                    feedback_correct="Good — within the expected order of magnitude.",
                    feedback_incorrect=(
                        "Check the series variability. W controls how quickly the "
                        "level drifts; larger W means the level changes more per step."
                    ),
                ),
            )
        out.append(s)
    return out


LESSON = Lesson(
    id="local_level",
    title="Local level",
    tier="beginner",
    description=(
        "The simplest DLM: an unobserved level that evolves by a random walk, "
        "observed with Gaussian noise."
    ),
    model_builder=_build,
    param_schema=_PARAMS,
    workflow_steps=_attach_challenges(make_default_workflow_steps(has_seasonal=False)),
)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_lessons.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lessons/local_level.py tests/test_lessons.py
git commit -m "feat(lessons): local level lesson content"
```

---

### Task 16: Lesson 2 — Local linear trend

**Files:**
- Create: `lessons/local_linear_trend.py`
- Modify: `tests/test_lessons.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_lessons.py`:

```python
from lessons.local_linear_trend import LESSON as LLT_LESSON


class TestLocalLinearTrendLesson:
    def test_builds_valid_spec_on_defaults(self):
        params = {p.name: p.default for p in LLT_LESSON.param_schema}
        spec = LLT_LESSON.model_builder(params)
        assert spec.d == 2 and spec.p == 1

    def test_pick_components_correct_has_slope(self):
        step4 = next(s for s in LLT_LESSON.workflow_steps if s.id == "pick_components")
        assert step4.challenge.correct == {"level": True, "slope": True, "seasonal": False}

    def test_param_schema_has_slope(self):
        names = [p.name for p in LLT_LESSON.param_schema]
        assert "W_slope" in names
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_lessons.py::TestLocalLinearTrendLesson -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement lesson 2**

Create `/Users/jameshenson/Documents/tutorials/DLM/lessons/local_linear_trend.py`:

```python
"""Beginner lesson 2: local linear trend (level + slope)."""

from __future__ import annotations

from engine.models import DLMSpec, make_local_linear_trend
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    WorkflowStep,
    make_default_workflow_steps,
)


def _build(params: dict[str, float]) -> DLMSpec:
    return make_local_linear_trend(
        V=params["V"],
        W_level=params["W_level"],
        W_slope=params["W_slope"],
    )


_PARAMS = [
    ParamSpec(name="V", label="Observation variance V",
              min=1e-6, max=5.0, default=0.5, step=0.01,
              help="Variance of the measurement noise."),
    ParamSpec(name="W_level", label="Level innovation variance",
              min=1e-6, max=1.0, default=0.01, step=0.001,
              help="Variance on mu_t innovation."),
    ParamSpec(name="W_slope", label="Slope innovation variance",
              min=1e-8, max=0.1, default=0.001, step=1e-4,
              help="Variance on beta_t innovation — typically much smaller than W_level."),
    ParamSpec(name="n", label="Series length n",
              min=20, max=2000, default=200, step=10, help=""),
]


def _attach_challenges(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    out: list[WorkflowStep] = []
    for s in steps:
        if s.id == "pick_components":
            s = WorkflowStep(
                id=s.id, title=s.title, prompt_md=s.prompt_md,
                plot_fn=s.plot_fn, hints=s.hints,
                challenge=ChallengeQuestion(
                    kind="component_toggle",
                    correct={"level": True, "slope": True, "seasonal": False},
                    feedback_correct=(
                        "Right. The series is drifting over time — a slope component "
                        "captures the persistent direction."
                    ),
                    feedback_incorrect=(
                        "Watch the trajectory. Does the series return to a long-run "
                        "level, or does it keep moving? If it keeps moving, you need "
                        "a slope component."
                    ),
                ),
            )
        elif s.id == "specify":
            s = WorkflowStep(
                id=s.id, title=s.title, prompt_md=s.prompt_md,
                plot_fn=s.plot_fn, hints=s.hints,
                challenge=ChallengeQuestion(
                    kind="numeric_range",
                    correct=(0.0001, 0.1),  # W_slope order-of-magnitude
                    feedback_correct=(
                        "Good. W_slope is almost always smaller than W_level: "
                        "slopes usually change more slowly than levels."
                    ),
                    feedback_incorrect=(
                        "W_slope should be orders of magnitude smaller than W_level. "
                        "Too large and the forecast will be wildly unstable."
                    ),
                ),
            )
        out.append(s)
    return out


LESSON = Lesson(
    id="local_linear_trend",
    title="Local linear trend",
    tier="beginner",
    description=(
        "Level + slope. The state carries both the current level mu_t and the "
        "local slope beta_t; both evolve by random walks."
    ),
    model_builder=_build,
    param_schema=_PARAMS,
    workflow_steps=_attach_challenges(make_default_workflow_steps(has_seasonal=False)),
)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_lessons.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lessons/local_linear_trend.py tests/test_lessons.py
git commit -m "feat(lessons): local linear trend lesson content"
```

---

### Task 17: Lesson 3 — Simple seasonal

**Files:**
- Create: `lessons/seasonal.py`
- Modify: `tests/test_lessons.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_lessons.py`:

```python
from lessons.seasonal import LESSON as SEASONAL_LESSON


class TestSeasonalLesson:
    def test_defaults_build_quarterly_spec(self):
        params = {p.name: p.default for p in SEASONAL_LESSON.param_schema}
        spec = SEASONAL_LESSON.model_builder(params)
        # With default period=4, d = period - 1 = 3
        assert spec.d == 3 and spec.p == 1

    def test_quantify_step_uses_seasonal_plot(self):
        step3 = next(s for s in SEASONAL_LESSON.workflow_steps if s.id == "quantify")
        assert step3.plot_fn == "acf_pacf_and_seasonal_subseries"

    def test_pick_components_correct_has_seasonal(self):
        step4 = next(s for s in SEASONAL_LESSON.workflow_steps if s.id == "pick_components")
        assert step4.challenge.correct == {"level": False, "slope": False, "seasonal": True}

    def test_specify_asks_period(self):
        step5 = next(s for s in SEASONAL_LESSON.workflow_steps if s.id == "specify")
        assert step5.challenge is not None
        assert step5.challenge.kind == "multiple_choice"
        # Correct must be string (period label)
        assert step5.challenge.correct in {"2", "4", "7", "12"}
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/test_lessons.py::TestSeasonalLesson -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement lesson 3**

Create `/Users/jameshenson/Documents/tutorials/DLM/lessons/seasonal.py`:

```python
"""Beginner lesson 3: simple seasonal (factor form)."""

from __future__ import annotations

from engine.models import DLMSpec, make_seasonal_factor
from lessons.workflow import (
    ChallengeQuestion,
    Lesson,
    ParamSpec,
    WorkflowStep,
    make_default_workflow_steps,
)


def _build(params: dict[str, float]) -> DLMSpec:
    return make_seasonal_factor(
        period=int(params["period"]),
        V=params["V"],
        W_season=params["W_season"],
    )


_PARAMS = [
    ParamSpec(name="V", label="Observation variance V",
              min=1e-6, max=5.0, default=0.5, step=0.01, help=""),
    ParamSpec(name="W_season", label="Seasonal innovation variance",
              min=1e-6, max=1.0, default=0.02, step=0.001,
              help="How quickly the seasonal pattern drifts over cycles."),
    ParamSpec(name="period", label="Period (e.g. 4 quarterly, 12 monthly)",
              min=2, max=12, default=4, step=1,
              help="Seasonal period in observation units."),
    ParamSpec(name="n", label="Series length n",
              min=20, max=2000, default=200, step=10, help=""),
]


def _attach_challenges(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    out: list[WorkflowStep] = []
    for s in steps:
        if s.id == "pick_components":
            s = WorkflowStep(
                id=s.id, title=s.title, prompt_md=s.prompt_md,
                plot_fn=s.plot_fn, hints=s.hints,
                challenge=ChallengeQuestion(
                    kind="component_toggle",
                    correct={"level": False, "slope": False, "seasonal": True},
                    feedback_correct=(
                        "Right. The repeating pattern at a fixed frequency is the "
                        "seasonal signature — clear spikes at multiples of the period "
                        "in the ACF."
                    ),
                    feedback_incorrect=(
                        "Look at the ACF spikes. Non-decaying peaks at multiples of "
                        "a fixed lag are the seasonal signature."
                    ),
                ),
            )
        elif s.id == "specify":
            s = WorkflowStep(
                id=s.id, title=s.title, prompt_md=s.prompt_md,
                plot_fn=s.plot_fn, hints=s.hints,
                challenge=ChallengeQuestion(
                    kind="multiple_choice",
                    correct="4",   # matches default period; lesson builder reads slider
                    feedback_correct="Right — the ACF spike location gives the period.",
                    feedback_incorrect=(
                        "The period is the lag at which the ACF first shows a large "
                        "non-decaying spike. Try matching that to one of the options."
                    ),
                ),
            )
        out.append(s)
    return out


LESSON = Lesson(
    id="seasonal",
    title="Simple seasonal",
    tier="beginner",
    description=(
        "A sum-to-zero seasonal factor model. The state carries the last "
        "period-1 seasonal effects; observations read the current one."
    ),
    model_builder=_build,
    param_schema=_PARAMS,
    workflow_steps=_attach_challenges(make_default_workflow_steps(has_seasonal=True)),
)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_lessons.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lessons/seasonal.py tests/test_lessons.py
git commit -m "feat(lessons): seasonal lesson content"
```

---

### Task 18: Lesson registry and content-integrity tests

**Files:**
- Modify: `lessons/__init__.py`
- Modify: `tests/test_lessons.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_lessons.py`:

```python
from lessons import ALL_LESSONS, get_lesson


class TestLessonRegistry:
    def test_contains_three_beginner_lessons(self):
        ids = [l.id for l in ALL_LESSONS]
        assert set(ids) == {"local_level", "local_linear_trend", "seasonal"}

    def test_get_lesson_by_id(self):
        l = get_lesson("local_level")
        assert l.id == "local_level"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown"):
            get_lesson("nonexistent")


class TestAllLessonsIntegrity:
    """Per spec §12 test_lessons.py content-integrity checks."""

    def test_all_model_builders_return_valid_spec_on_defaults(self):
        for lesson in ALL_LESSONS:
            defaults = {p.name: p.default for p in lesson.param_schema}
            spec = lesson.model_builder(defaults)
            assert isinstance(spec, DLMSpec)

    def test_all_lessons_have_nine_steps(self):
        from lessons.workflow import canonical_step_ids
        ids = canonical_step_ids()
        for lesson in ALL_LESSONS:
            step_ids = [s.id for s in lesson.workflow_steps]
            assert step_ids == ids, f"Lesson {lesson.id} has {step_ids}"

    def test_all_plot_fns_declared(self):
        """Every plot_fn referenced must be in the known plot-fn set.

        The actual set is declared in ui/plots.py. We hard-code it here since
        the UI layer isn't imported for this test (keeps tests runnable
        without Streamlit).
        """
        allowed = {
            "time_series",
            "visual_decomposition",
            "acf_pacf",
            "acf_pacf_and_seasonal_subseries",
            "blank",
            "spec_preview",
            "filter_state",
            "diagnostics",
            "forecast",
            "reveal_overlay",
        }
        for lesson in ALL_LESSONS:
            for step in lesson.workflow_steps:
                assert step.plot_fn in allowed, (
                    f"Lesson {lesson.id} step {step.id}: unknown plot_fn {step.plot_fn}"
                )
```

- [ ] **Step 2: Run tests; expect ImportError on `ALL_LESSONS`**

Run: `pytest tests/test_lessons.py::TestLessonRegistry -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement lesson registry**

Replace `/Users/jameshenson/Documents/tutorials/DLM/lessons/__init__.py` with:

```python
"""Lesson registry.

Import this module to access the list of all lessons and look them up by id.
"""

from __future__ import annotations

from lessons.local_level import LESSON as _local_level
from lessons.local_linear_trend import LESSON as _llt
from lessons.seasonal import LESSON as _seasonal
from lessons.workflow import Lesson

ALL_LESSONS: list[Lesson] = [_local_level, _llt, _seasonal]

_BY_ID: dict[str, Lesson] = {l.id: l for l in ALL_LESSONS}


def get_lesson(lesson_id: str) -> Lesson:
    if lesson_id not in _BY_ID:
        raise KeyError(f"unknown lesson id {lesson_id!r}; available: {list(_BY_ID)}")
    return _BY_ID[lesson_id]
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_lessons.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lessons/__init__.py tests/test_lessons.py
git commit -m "feat(lessons): registry of all beginner lessons"
```

---

## Phase 3 — UI (Streamlit)

### Task 19: Streamlit session state helpers

**Files:**
- Create: `ui/state.py`

- [ ] **Step 1: Note — no unit test for this module**

This module is a thin wrapper around `st.session_state`, which cannot be tested outside the Streamlit runtime. It's verified by manually running the app in Phase 3. Proceed directly to implementation.

- [ ] **Step 2: Implement state helpers**

Create `/Users/jameshenson/Documents/tutorials/DLM/ui/state.py`:

```python
"""Streamlit session-state helpers.

All state keys live in `st.session_state` with no disk persistence. Helpers
below keep key names consistent and centralize initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

_KEYS = (
    "tier",            # "beginner"
    "lesson_id",       # e.g. "local_level"
    "mode",            # "lesson" | "challenge"
    "sim_params",      # dict[str, float]
    "seed",            # int
    "spec_true",       # DLMSpec
    "series",          # SimulatedSeries
    "step_idx",        # int
    "answers",         # dict[step_id, Any]
    "filter_result",   # FilterResult | None
    "smooth_result",   # SmoothResult | None
    "user_spec",       # DLMSpec | None
    "completed_lessons",   # set[str] (in-session only)
)


@dataclass
class SessionState:
    tier: str = "beginner"
    lesson_id: str | None = None
    mode: str | None = None
    sim_params: dict[str, float] | None = None
    seed: int = 42
    spec_true: Any = None
    series: Any = None
    step_idx: int = 0
    answers: dict[str, Any] | None = None
    filter_result: Any = None
    smooth_result: Any = None
    user_spec: Any = None
    completed_lessons: set[str] | None = None


def init_state() -> None:
    """Initialize session state on first render (idempotent)."""
    defaults = SessionState().__dict__
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v if not isinstance(v, (dict, set)) else (
                type(v)() if v is None else v
            )
    # Replace None mutable defaults with fresh containers
    if st.session_state.get("answers") is None:
        st.session_state["answers"] = {}
    if st.session_state.get("completed_lessons") is None:
        st.session_state["completed_lessons"] = set()


def reset_lesson_state() -> None:
    """Clear everything that depends on the chosen lesson/mode/params."""
    for k in ("sim_params", "spec_true", "series", "filter_result",
              "smooth_result", "user_spec"):
        st.session_state[k] = None
    st.session_state["step_idx"] = 0
    st.session_state["answers"] = {}


def mark_lesson_completed(lesson_id: str) -> None:
    completed = st.session_state.get("completed_lessons") or set()
    completed.add(lesson_id)
    st.session_state["completed_lessons"] = completed


def is_lesson_unlocked(lesson_id: str, ordered_ids: list[str]) -> bool:
    """Linear unlock: index i is unlocked iff all prior indices are completed.

    First lesson (index 0) is always unlocked.
    """
    if lesson_id == ordered_ids[0]:
        return True
    completed = st.session_state.get("completed_lessons") or set()
    idx = ordered_ids.index(lesson_id)
    return all(prev in completed for prev in ordered_ids[:idx])
```

- [ ] **Step 3: Commit**

```bash
git add ui/state.py
git commit -m "feat(ui): session-state helpers with linear-unlock logic"
```

---

### Task 20: Sidebar controls — sliders from `param_schema`, mode toggle

**Files:**
- Create: `ui/controls.py`

- [ ] **Step 1: Implement controls (no unit test — requires Streamlit runtime)**

Create `/Users/jameshenson/Documents/tutorials/DLM/ui/controls.py`:

```python
"""Shared Streamlit widget panels."""

from __future__ import annotations

import streamlit as st

from engine.models import DLMSpec
from engine.simulate import SimulatedSeries, simulate
from lessons.workflow import Lesson


def render_sidebar(lesson: Lesson) -> tuple[dict[str, float], int]:
    """Render the simulation parameter sliders and seed input.

    Returns the current (sim_params, seed) tuple.
    """
    st.sidebar.header("Simulation parameters")
    params: dict[str, float] = {}
    for p in lesson.param_schema:
        if p.step >= 1:
            params[p.name] = st.sidebar.slider(
                p.label,
                min_value=int(p.min), max_value=int(p.max),
                value=int(p.default), step=int(p.step),
                help=p.help, key=f"slider_{lesson.id}_{p.name}",
            )
        else:
            params[p.name] = st.sidebar.slider(
                p.label,
                min_value=float(p.min), max_value=float(p.max),
                value=float(p.default), step=float(p.step),
                help=p.help, key=f"slider_{lesson.id}_{p.name}",
            )
    seed = st.sidebar.number_input(
        "Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
        key=f"seed_{lesson.id}",
    )
    return params, int(seed)


@st.cache_data(show_spinner=False)
def _cached_simulate(
    spec_hash: int, spec: DLMSpec, n: int, seed: int
) -> SimulatedSeries:
    # spec_hash included so cache invalidates on any spec change
    return simulate(spec, n=n, seed=seed)


def simulate_from_params(lesson: Lesson, params: dict[str, float], seed: int) -> tuple[DLMSpec, SimulatedSeries]:
    """Build spec from lesson params and simulate a series, with caching."""
    n = int(params["n"])
    # Params that affect the spec exclude n
    spec_params = {k: v for k, v in params.items() if k != "n"}
    spec = lesson.model_builder(spec_params)
    series = _cached_simulate(hash(spec), spec, n=n, seed=seed)
    return spec, series


def render_mode_selector() -> str:
    return st.radio(
        "Mode",
        options=["lesson", "challenge"],
        format_func=lambda x: "Lesson (transparent walkthrough)"
                              if x == "lesson"
                              else "Challenge (hidden ground truth)",
        horizontal=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add ui/controls.py
git commit -m "feat(ui): sidebar controls and cached simulation"
```

---

### Task 21: Plotly renderers (all 10 plot functions)

**Files:**
- Create: `ui/plots.py`

- [ ] **Step 1: Implement plots**

Create `/Users/jameshenson/Documents/tutorials/DLM/ui/plots.py`:

```python
"""Plotly renderers used by workflow steps.

Each function takes the current session objects (series, filter_result, etc.)
and returns a `plotly.graph_objects.Figure`. The set of function names
registered in `PLOT_FN_REGISTRY` is the source of truth for what lessons can
reference via WorkflowStep.plot_fn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.diagnostics import acf_pacf, residual_diagnostics
from engine.filter import FilterResult
from engine.forecast import Forecast
from engine.models import DLMSpec
from engine.simulate import SimulatedSeries
from engine.smoother import SmoothResult


def _blank() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="No plot for this step.",
                          showarrow=False, font=dict(color="#888"))],
    )
    return fig


def time_series(series: SimulatedSeries, **_: Any) -> go.Figure:
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", name="y_t",
                             marker=dict(size=4)))
    fig.update_layout(title="Observed series", xaxis_title="t", yaxis_title="y",
                      height=350)
    return fig


def visual_decomposition(series: SimulatedSeries, **_: Any) -> go.Figure:
    """y_t plus a simple centered moving average to suggest a trend."""
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    window = max(5, len(y) // 20)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    trend = np.convolve(y, kernel, mode="same")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="y_t",
                             line=dict(color="#888", width=1)))
    fig.add_trace(go.Scatter(x=t, y=trend, mode="lines",
                             name=f"MA({window})", line=dict(width=3)))
    fig.update_layout(title="Visual decomposition", xaxis_title="t", height=350)
    return fig


def acf_pacf_fig(series: SimulatedSeries, **_: Any) -> go.Figure:
    ap = acf_pacf(series.y, nlags=min(40, len(series.y) // 4))
    n = len(series.y)
    band = 1.96 / np.sqrt(n)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    fig.add_trace(go.Bar(x=ap.lags, y=ap.acf, name="acf"), row=1, col=1)
    fig.add_trace(go.Bar(x=ap.lags, y=ap.pacf, name="pacf"), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=band, line_dash="dash", line_color="#aaa", row=1, col=col)
        fig.add_hline(y=-band, line_dash="dash", line_color="#aaa", row=1, col=col)
    fig.update_layout(height=350, showlegend=False)
    return fig


def acf_pacf_and_seasonal_subseries(
    series: SimulatedSeries, period: int = 4, **_: Any
) -> go.Figure:
    """ACF/PACF plus a seasonal subseries grid underneath.

    Caller passes `period` via kwargs from the sim_params dict.
    """
    ap = acf_pacf(series.y, nlags=min(40, len(series.y) // 4))
    y = series.y[:, 0]
    n = len(y)
    band = 1.96 / np.sqrt(n)
    period = int(period)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("ACF", "PACF", "Seasonal subseries", ""),
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )
    fig.add_trace(go.Bar(x=ap.lags, y=ap.acf, name="acf"), row=1, col=1)
    fig.add_trace(go.Bar(x=ap.lags, y=ap.pacf, name="pacf"), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=band, line_dash="dash", line_color="#aaa", row=1, col=col)
        fig.add_hline(y=-band, line_dash="dash", line_color="#aaa", row=1, col=col)
    for k in range(period):
        sub = y[k::period]
        x = np.arange(len(sub))
        fig.add_trace(
            go.Scatter(x=x, y=sub, mode="lines+markers",
                       name=f"season {k + 1}"),
            row=2, col=1,
        )
    fig.update_layout(height=600, showlegend=False)
    return fig


def spec_preview(spec: DLMSpec | None, **_: Any) -> go.Figure:
    """Render the matrices F, G, V, W as a simple annotated heatmap grid."""
    if spec is None:
        return _blank()
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("F", "G", "V", "W"),
    )
    for (r, c, M, name) in [
        (1, 1, spec.F, "F"), (1, 2, spec.G, "G"),
        (2, 1, spec.V, "V"), (2, 2, spec.W, "W"),
    ]:
        fig.add_trace(
            go.Heatmap(z=M, colorscale="Greys", showscale=False,
                       hovertemplate=f"{name}"
                                     "[%{y}, %{x}] = %{z:.4g}<extra></extra>"),
            row=r, col=c,
        )
    fig.update_layout(height=500)
    return fig


def filter_state(
    series: SimulatedSeries, fr: FilterResult, **_: Any
) -> go.Figure:
    """Filtered state means with 95% credible bands, alongside the observed series."""
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    # For multi-dim state, plot the first-dim projection F @ m_t for comparability
    F = series.spec.F
    mean_y = (fr.m @ F.T)[:, 0]
    var_y = np.einsum("ij,tjk,ik->t", F, fr.C, F)
    sd_y = np.sqrt(var_y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="markers", name="y_t",
                             marker=dict(color="#888", size=4)))
    fig.add_trace(go.Scatter(x=t, y=mean_y, mode="lines",
                             name="filtered E[y|past]"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([mean_y + 1.96 * sd_y, (mean_y - 1.96 * sd_y)[::-1]]),
        fill="toself", line=dict(color="rgba(0,0,0,0)"),
        fillcolor="rgba(31, 119, 180, 0.2)",
        name="95% band", hoverinfo="skip",
    ))
    fig.update_layout(title="Filtered state", xaxis_title="t", height=400)
    return fig


def diagnostics_fig(
    fr: FilterResult, sr: SmoothResult | None = None, **_: Any
) -> go.Figure:
    diag = residual_diagnostics(fr.e, fr.Q, nlags=min(20, len(fr.e) // 4))
    t = np.arange(1, diag.standardized.shape[0] + 1)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Standardized residuals",
            f"Ljung-Box p = {diag.ljung_box_pvalue:.3f}",
            "Residual ACF",
            "Residual PACF",
        ),
    )
    fig.add_trace(go.Scatter(x=t, y=diag.standardized[:, 0], mode="lines+markers",
                             marker=dict(size=3), name="std resid"), row=1, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="#f66", row=1, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="#f66", row=1, col=1)

    # Histogram of standardized residuals for normality check
    fig.add_trace(go.Histogram(x=diag.standardized[:, 0], nbinsx=30,
                               name="hist"), row=1, col=2)
    fig.add_trace(go.Bar(x=diag.acf_pacf.lags, y=diag.acf_pacf.acf,
                         name="acf"), row=2, col=1)
    fig.add_trace(go.Bar(x=diag.acf_pacf.lags, y=diag.acf_pacf.pacf,
                         name="pacf"), row=2, col=2)
    fig.update_layout(height=600, showlegend=False)
    return fig


def forecast_fig(
    series: SimulatedSeries, fr: FilterResult, fc: Forecast, **_: Any
) -> go.Figure:
    y = series.y[:, 0]
    T = len(y)
    t_hist = np.arange(1, T + 1)
    t_fc = np.arange(T + 1, T + 1 + fc.horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_hist, y=y, mode="lines+markers",
                             name="observed", marker=dict(size=3)))
    if fc.horizon > 0:
        fig.add_trace(go.Scatter(x=t_fc, y=fc.means[:, 0], mode="lines",
                                 name="forecast"))
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_fc, t_fc[::-1]]),
            y=np.concatenate([fc.upper[:, 0], fc.lower[:, 0][::-1]]),
            fill="toself", line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(214, 39, 40, 0.25)", name="95% band",
            hoverinfo="skip",
        ))
    fig.update_layout(title=f"{fc.horizon}-step forecast",
                      xaxis_title="t", height=400)
    return fig


def reveal_overlay(
    series: SimulatedSeries,
    fr_true: FilterResult,
    fr_user: FilterResult | None = None,
    **_: Any,
) -> go.Figure:
    """Overlay the true DLM's filtered state against the user's (challenge)."""
    y = series.y[:, 0]
    t = np.arange(1, len(y) + 1)
    F = series.spec.F
    true_mean_y = (fr_true.m @ F.T)[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="markers",
                             name="y_t", marker=dict(color="#aaa", size=3)))
    fig.add_trace(go.Scatter(x=t, y=true_mean_y, mode="lines",
                             name="ground-truth filtered"))
    if fr_user is not None:
        user_mean_y = (fr_user.m @ F.T)[:, 0]
        fig.add_trace(go.Scatter(x=t, y=user_mean_y, mode="lines",
                                 name="your filtered",
                                 line=dict(dash="dash")))
    fig.update_layout(title="Reveal: your DLM vs ground truth",
                      height=400, xaxis_title="t")
    return fig


PLOT_FN_REGISTRY: dict[str, Callable[..., go.Figure]] = {
    "time_series": time_series,
    "visual_decomposition": visual_decomposition,
    "acf_pacf": acf_pacf_fig,
    "acf_pacf_and_seasonal_subseries": acf_pacf_and_seasonal_subseries,
    "blank": lambda **_: _blank(),
    "spec_preview": spec_preview,
    "filter_state": filter_state,
    "diagnostics": diagnostics_fig,
    "forecast": forecast_fig,
    "reveal_overlay": reveal_overlay,
}
```

- [ ] **Step 2: Commit**

```bash
git add ui/plots.py
git commit -m "feat(ui): Plotly renderers for all 10 plot functions"
```

---

### Task 22: Streamlit app skeleton — routing and lesson selector

**Files:**
- Create: `ui/app.py`

- [ ] **Step 1: Implement app skeleton**

Create `/Users/jameshenson/Documents/tutorials/DLM/ui/app.py`:

```python
"""Streamlit entry point.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from lessons import ALL_LESSONS, get_lesson
from lessons.workflow import validate_lesson
from ui.controls import render_mode_selector, render_sidebar, simulate_from_params
from ui.plots import PLOT_FN_REGISTRY
from ui.state import (
    init_state,
    is_lesson_unlocked,
    mark_lesson_completed,
    reset_lesson_state,
)
from ui.workflow_render import render_workflow_step

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"

st.set_page_config(page_title="DLM Tutorial — Beginner", layout="wide")


def _validate_lessons_once() -> None:
    """Validate lessons against the plot-fn registry at startup."""
    if "lessons_validated" in st.session_state:
        return
    allowed = set(PLOT_FN_REGISTRY.keys())
    for lesson in ALL_LESSONS:
        validate_lesson(lesson, allowed_plot_fns=allowed)
    st.session_state["lessons_validated"] = True


def _render_home() -> None:
    st.title("DLM Intuition Tutorial — Beginner tier")
    st.markdown(
        "Pick a lesson. Lessons unlock in order — complete Challenge mode to "
        "unlock the next. Completed lessons remain revisitable in this session."
    )
    ordered_ids = [l.id for l in ALL_LESSONS]
    for lesson in ALL_LESSONS:
        unlocked = is_lesson_unlocked(lesson.id, ordered_ids)
        completed = lesson.id in (st.session_state.get("completed_lessons") or set())
        badge = "✅ completed" if completed else ("🔓 available" if unlocked else "🔒 locked")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.subheader(f"{lesson.title}  — {badge}")
            st.markdown(lesson.description)
        with col_b:
            if unlocked and st.button("Open", key=f"open_{lesson.id}"):
                st.session_state["lesson_id"] = lesson.id
                st.session_state["mode"] = None
                reset_lesson_state()
                st.rerun()


def _render_lesson_page(lesson_id: str) -> None:
    lesson = get_lesson(lesson_id)
    st.title(f"{lesson.title}")
    st.caption(lesson.description)

    # Mode (radio at top)
    mode = render_mode_selector()
    if st.session_state.get("mode") != mode:
        st.session_state["mode"] = mode
        reset_lesson_state()
        st.rerun()

    params, seed = render_sidebar(lesson)
    st.session_state["sim_params"] = params
    st.session_state["seed"] = seed

    spec, series = simulate_from_params(lesson, params, seed)
    st.session_state["spec_true"] = spec
    st.session_state["series"] = series

    step_idx = st.session_state.get("step_idx") or 0
    step = lesson.workflow_steps[step_idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Progress:** step {step_idx + 1} / {len(lesson.workflow_steps)}")
    if st.sidebar.button("Back to lessons"):
        st.session_state["lesson_id"] = None
        reset_lesson_state()
        st.rerun()

    render_workflow_step(lesson=lesson, step=step, mode=mode)

    # Navigation buttons
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if step_idx > 0 and st.button("◂ Back"):
            st.session_state["step_idx"] = step_idx - 1
            st.rerun()
    with col_next:
        if step_idx < len(lesson.workflow_steps) - 1:
            if st.button("Next ▸"):
                st.session_state["step_idx"] = step_idx + 1
                st.rerun()
        else:
            if st.button("Finish lesson"):
                if mode == "challenge":
                    mark_lesson_completed(lesson.id)
                st.session_state["lesson_id"] = None
                reset_lesson_state()
                st.rerun()


def _render_reference() -> None:
    path = CONTENT_DIR / "modeling-guide.md"
    if not path.exists():
        st.warning("modeling-guide.md not found.")
        return
    st.markdown(path.read_text(encoding="utf-8"))


def main() -> None:
    init_state()
    _validate_lessons_once()

    page = st.sidebar.radio(
        "Page", options=["Home", "Reference"], horizontal=False, key="page_radio",
    )
    if page == "Reference":
        _render_reference()
        return

    lesson_id = st.session_state.get("lesson_id")
    if lesson_id is None:
        _render_home()
    else:
        _render_lesson_page(lesson_id)


main()
```

- [ ] **Step 2: Commit (app skeleton — still needs workflow_render module)**

```bash
git add ui/app.py
git commit -m "feat(ui): Streamlit app skeleton with routing"
```

---

### Task 23: Workflow step renderer — Lesson and Challenge modes

**Files:**
- Create: `ui/workflow_render.py`

- [ ] **Step 1: Implement the step renderer**

Create `/Users/jameshenson/Documents/tutorials/DLM/ui/workflow_render.py`:

```python
"""Render a single WorkflowStep in Lesson or Challenge mode."""

from __future__ import annotations

from typing import Any

import streamlit as st

from engine.filter import kalman_filter
from engine.forecast import forecast_horizon
from engine.smoother import rts_smoother
from lessons.workflow import ChallengeQuestion, Lesson, WorkflowStep
from ui.plots import PLOT_FN_REGISTRY


@st.cache_data(show_spinner=False)
def _cached_filter(spec_hash: int, spec, y_bytes: bytes, y_shape: tuple):
    import numpy as np
    y = np.frombuffer(y_bytes, dtype=np.float64).reshape(y_shape)
    return kalman_filter(spec, y)


@st.cache_data(show_spinner=False)
def _cached_smooth(spec_hash: int, spec, fr):
    return rts_smoother(spec, fr)


@st.cache_data(show_spinner=False)
def _cached_forecast(spec_hash: int, spec, fr, h: int):
    return forecast_horizon(spec, fr, h=h)


def _run_filter(spec) -> Any:
    series = st.session_state["series"]
    y = series.y
    return _cached_filter(hash(spec), spec, y.tobytes(), y.shape)


def _render_plot(step: WorkflowStep, lesson: Lesson, mode: str) -> None:
    series = st.session_state["series"]
    spec_true = st.session_state["spec_true"]
    sim_params = st.session_state.get("sim_params") or {}

    fn = PLOT_FN_REGISTRY[step.plot_fn]
    kwargs: dict[str, Any] = {"series": series}

    if step.id == "acf_pacf_and_seasonal_subseries" or step.plot_fn == "acf_pacf_and_seasonal_subseries":
        kwargs["period"] = int(sim_params.get("period", 4))

    # Steps 6-9 need the filter result (and perhaps smoother/forecast)
    if step.id in ("fit", "diagnose", "forecast", "reveal"):
        spec_for_fit = (
            st.session_state.get("user_spec")
            if mode == "challenge" and st.session_state.get("user_spec") is not None
            else spec_true
        )
        fr = _run_filter(spec_for_fit)
        st.session_state["filter_result"] = fr
        kwargs["fr"] = fr
        if step.id == "diagnose":
            sr = _cached_smooth(hash(spec_for_fit), spec_for_fit, fr)
            st.session_state["smooth_result"] = sr
            kwargs["sr"] = sr
        if step.id == "forecast":
            h = st.slider("Forecast horizon h", min_value=1, max_value=100, value=20, step=1)
            fc = _cached_forecast(hash(spec_for_fit), spec_for_fit, fr, int(h))
            kwargs["fc"] = fc
        if step.id == "reveal":
            fr_true = _run_filter(spec_true)
            kwargs["fr_true"] = fr_true
            if mode == "challenge":
                kwargs["fr_user"] = fr
            else:
                kwargs["fr_user"] = None

    if step.plot_fn == "spec_preview":
        kwargs = {"spec": (
            st.session_state.get("user_spec") if mode == "challenge"
            else spec_true
        )}
    if step.plot_fn == "blank":
        kwargs = {}

    fig = fn(**kwargs)
    st.plotly_chart(fig, use_container_width=True)


def _render_lesson_mode(step: WorkflowStep, lesson: Lesson) -> None:
    st.markdown(step.prompt_md)
    _render_plot(step, lesson, mode="lesson")


def _render_challenge_widget(step: WorkflowStep, challenge: ChallengeQuestion) -> None:
    key = f"answer_{step.id}"
    prior = st.session_state["answers"].get(step.id)
    if challenge.kind == "component_toggle":
        cols = st.columns(3)
        level = cols[0].checkbox("level", value=(prior or {}).get("level", False),
                                 key=f"{key}_level")
        slope = cols[1].checkbox("slope", value=(prior or {}).get("slope", False),
                                 key=f"{key}_slope")
        seasonal = cols[2].checkbox("seasonal",
                                    value=(prior or {}).get("seasonal", False),
                                    key=f"{key}_seasonal")
        answer = {"level": level, "slope": slope, "seasonal": seasonal}
    elif challenge.kind == "numeric_range":
        answer = st.number_input("Your estimate", min_value=0.0, max_value=10.0,
                                 value=float(prior) if prior is not None else 0.1,
                                 step=0.001, format="%.4f", key=key)
    elif challenge.kind == "multiple_choice":
        options = ["2", "4", "7", "12"]
        answer = st.radio("Pick one", options=options,
                          index=options.index(prior) if prior in options else 1,
                          key=key)
    else:
        answer = None

    if st.button("Submit answer", key=f"{key}_submit"):
        st.session_state["answers"][step.id] = answer
        correct = _is_correct(challenge, answer)
        if correct:
            st.success(challenge.feedback_correct)
        else:
            st.warning(challenge.feedback_incorrect)


def _is_correct(q: ChallengeQuestion, answer: Any) -> bool:
    if q.kind == "component_toggle":
        return answer == q.correct
    if q.kind == "multiple_choice":
        return answer == q.correct
    if q.kind == "numeric_range":
        lo, hi = q.correct
        return lo <= float(answer) <= hi
    return False


def _render_challenge_mode(step: WorkflowStep, lesson: Lesson) -> None:
    st.markdown(step.prompt_md)
    if step.challenge is not None:
        _render_plot(step, lesson, mode="challenge")
        _render_challenge_widget(step, step.challenge)
    else:
        _render_plot(step, lesson, mode="challenge")


def render_workflow_step(lesson: Lesson, step: WorkflowStep, mode: str) -> None:
    st.header(step.title)
    if mode == "challenge":
        _render_challenge_mode(step, lesson)
    else:
        _render_lesson_mode(step, lesson)
    if step.hints:
        with st.expander("Hints"):
            for h in step.hints:
                st.markdown(f"- {h}")
```

- [ ] **Step 2: Commit**

```bash
git add ui/workflow_render.py
git commit -m "feat(ui): workflow step renderer for both modes"
```

---

### Task 24: Manual smoke test — run the app end-to-end

**Files:**
- None (manual step)

- [ ] **Step 1: Run the app**

Run:
```bash
cd /Users/jameshenson/Documents/tutorials/DLM
source .venv/bin/activate
streamlit run ui/app.py
```

Expected: a browser tab opens at `http://localhost:8501` with:
- Home page listing three lessons: local level (unlocked), local linear trend (locked), seasonal (locked)
- Reference page accessible from the sidebar (shows a warning since `modeling-guide.md` isn't yet written — fine for now)

- [ ] **Step 2: Walk through lesson 1, Lesson mode**

- Open "Local level"
- Mode = "Lesson (transparent walkthrough)"
- Click Next through all 9 steps
- Expected: each step renders a plot; no crashes
- Change a slider mid-way — the plot should update without crashing

- [ ] **Step 3: Walk through lesson 1, Challenge mode**

- Mode = "Challenge"
- At step 4, toggle components and submit — feedback appears
- At step 5, enter a number and submit — feedback appears
- Click Finish at the end; return to home; lesson 1 should be ✅

- [ ] **Step 4: Verify lesson 2 unlocks**

- Lesson 2 (local linear trend) should show 🔓 available after completing lesson 1 Challenge mode
- Lesson 3 still 🔒

- [ ] **Step 5: Document any manual-test findings**

If any crashes or visual glitches were observed, note them as follow-up tasks in this plan before proceeding.

- [ ] **Step 6: Commit (no code changes — just verifying)**

No commit needed unless step 5 produced fixes.

---

## Phase 4 — Content

### Task 25: `modeling-guide.md` parts 1–4

**Files:**
- Create: `content/modeling-guide.md`

- [ ] **Step 1: Create `content/` directory and write parts 1-4**

Run:
```bash
mkdir -p /Users/jameshenson/Documents/tutorials/DLM/content
```

Create `/Users/jameshenson/Documents/tutorials/DLM/content/modeling-guide.md`:

```markdown
# DLM Modeling Guide

This reference accompanies the interactive tutorial. It covers the DLM framework, the baseline modeling procedure, variance tuning, and diagnostics. The five sections below map directly to the anchors that workflow steps deep-link to.

- [The DLM framework](#the-dlm-framework)
- [Baseline procedure](#baseline-procedure)
- [Tuning](#tuning)
- [Diagnostics](#diagnostics)
- [Worked example](#worked-example)

---

## The DLM framework

A **Dynamic Linear Model** is a pair of linear Gaussian equations that together define a state-space model for a time series:

Observation equation:   y_t = F_t θ_t + v_t,   v_t ~ N(0, V_t)
State equation:         θ_t = G_t θ_{t-1} + w_t,   w_t ~ N(0, W_t)

- y_t ∈ R^p is the observation at time t (p = 1 for all Beginner lessons)
- θ_t ∈ R^d is the unobserved state
- F_t (p × d) maps the state into the observation space
- G_t (d × d) evolves the state one step forward
- V_t (p × p), W_t (d × d) are noise covariances
- Prior: θ_0 ~ N(m_0, C_0)

For the Beginner tier we restrict to **time-invariant** DLMs: F_t, G_t, V_t, W_t do not depend on t. Notation follows West & Harrison (1997) and cross-references Petris, Petrone & Campagnoli (2009) §2.3.

Three canonical components you will meet in the lessons:

| Component | F | G | State dim |
|---|---|---|---|
| Local level | `[1]` | `[1]` | 1 |
| Local linear trend | `[1, 0]` | `[[1, 1], [0, 1]]` | 2 |
| Seasonal factor, period s | `[1, 0, …, 0]` (length s−1) | companion form with first row `[-1, …, -1]` | s−1 |

Components combine by **superposition**: stack F horizontally, block-diagonalize G and W. That's exactly what `combine()` does in `engine/models.py`.

---

## Baseline procedure

The nine-step modeling workflow is the spine of every lesson. Here's the version written out for reference — the app paces you through each step, but sometimes it helps to see the whole thing.

### Step 1 — Inspect the data

Plot y_t on the vertical axis and t on the horizontal. Ask:
- Is there a persistent **level**? (horizontal band, not drifting)
- Is there **drift** (trend)? (long-run direction)
- Any **periodicity**? (waves repeating at fixed lag)
- Any **heteroscedasticity**? (variance changes over time — not handled by constant-V DLMs)

Why it matters: the mental image you form here drives every subsequent step.

Pitfall: noise can disguise a slow trend. Don't commit to a hypothesis in step 1 — we quantify in step 3.

### Step 2 — Decompose visually

Overlay a centered moving average on the raw series. The MA smooths out the noise and surfaces a trend (or its absence). A seasonal signal often appears as a residual wobble after subtracting the MA.

Why it matters: a second, less noisy view of the same data lets you falsify or confirm the step-1 hypotheses.

Pitfall: if the window is too short the MA tracks the noise; if too long, it misses the trend. We pick n/20 (rounded to an odd number) as a starting point.

### Step 3 — Quantify autocorrelation

Compute and plot the ACF and PACF out to at least `n/4` lags. Look for:
- **Slow monotone ACF decay** → trend or local level (persistent shocks)
- **Spike at lag s** not decaying → seasonal period s
- **PACF spike at lag 1** only → AR(1)-like behavior
- **All lags inside the ±1.96/√n band** → white noise (no model to build)

Why it matters: the eye is unreliable; the ACF is reliable.

Pitfall: a spurious spike inside the confidence band is still possible — don't over-interpret a single lag.

### Step 4 — Pick components

Translate steps 1-3 into a set of components. Rules of thumb:
- Slow decay → local **level**
- Slow decay with *directional* drift → local level **and** slope
- Spike at lag s → **seasonal** with period s
- Multiple patterns → combine with superposition

### Step 5 — Specify the model

Write down the matrices. This is where mis-specification most often happens:
- Mismatched dimensions → fitter will error
- Missing seasonality → residuals will spike at lag s
- Missing slope → forecasts will revert to the mean instead of extrapolating

Variance choice is discussed in [Tuning](#tuning).

### Step 6 — Fit

Run the Kalman filter (see `engine/filter.py`). Output:
- **Filtered means** m_t — the current-time posterior mean of θ_t
- **Filtered variances** C_t — the associated uncertainty
- **One-step forecasts** f_t — what the model *would have* predicted before seeing y_t
- **Innovations** e_t = y_t − f_t

### Step 7 — Diagnose

If your specification is right, the innovations e_t look like white noise. Check:
- Residual ACF — spikes inside the band
- Residual normality — histogram roughly bell-shaped
- Portmanteau (Ljung-Box) p-value > 0.05

And compare **filtered vs smoothed** state: the smoothed state uses all of y_{1:T} rather than just y_{1:t}, so it is smoother and less variable. Divergence of the two in low-data regions is expected; divergence in high-data regions suggests misspecification.

### Step 8 — Forecast

h-step-ahead forecasts propagate the filtered posterior forward through G, adding W each step. Credible bands widen with horizon — that is intentional and informative.

### Step 9 — Reveal / review

Lesson mode: recap what you did and why. Challenge mode: compare your fitted DLM against the ground-truth DLM. If they agree, you recovered the true structure. If they disagree, inspect the residuals: that's where the misspecification shows.

---

## Tuning

Variance specification is where DLM practice becomes a craft. For the Beginner tier we treat V as known; W is what we tune.

**Strategy 1 — Fixed prior.** Pick W based on domain knowledge. Good for simulations (where you *know* the generating W) and for preliminary analyses.

**Strategy 2 — Discount factors.** Write W_t = C_t × (1/δ − 1) for some δ ∈ (0.9, 1). A single number controls the effective memory of the filter. δ = 1 means no information discount (W = 0, the state is static); δ < 1 lets the state evolve. Typical: δ = 0.95-0.99 for a level component, 0.98 for a slope. (Intermediate tier only; Beginner works with fixed W directly.)

**Strategy 3 — Empirical prior.** Use maximum-likelihood or REML to estimate W from the data. (Intermediate/Advanced tier.)

### Signal-to-noise intuition

Let r = W / V. For the local-level model:
- r ≈ 0 → y_t ≈ constant + noise (deep averaging)
- r ≈ 1 → y_t tracks the level closely (low averaging)
- r ≫ 1 → the filter chases each observation (little smoothing)

The steady-state filtered variance is C_∞ = (−W + √(W² + 4VW)) / 2. See `engine/filter.py` — the test suite verifies this analytic fixed point.

### V vs W tradeoff

Increasing V *or* decreasing W both push the filter toward heavier smoothing. When both are unknown, the ratio matters more than either absolute value.

### Prior covariance C_0

For a diffuse prior, use C_0 = κ × I with κ large (1e3 is typical). This lets the first few observations "inform themselves" without prior bias. Very small C_0 is informative; use it only when you have strong prior beliefs.

---

## Diagnostics

### Standardized residuals

Define ε_t = e_t / √Q_t. Under correct specification, ε_t is approximately iid N(0, 1). Plot ε_t against t:
- No runs of same sign
- Roughly 95% within ±2
- No trend or pattern

### Residual ACF

All lags inside ±1.96/√n. Any persistent lag signature indicates missing structure:
- Slow decay → missed trend or level
- Spike at lag s → missed seasonal

### Ljung-Box / portmanteau

A single p-value summarizing the first k lags' autocorrelations. Null: white noise. Reject at p < 0.05.

### What "good" looks like

- Ljung-Box p > 0.1
- No residual ACF spikes beyond the band except possibly 1/20 by chance
- MAPE (mean absolute percent error) small relative to problem scale

---

## Worked example

Part 5 — a worked end-to-end example — is developed in the next section.
```

- [ ] **Step 2: Verify with a quick render — run the app, navigate to Reference**

```bash
streamlit run ui/app.py
```
Expected: Reference page renders the markdown with headings and the table. No 404.

- [ ] **Step 3: Commit**

```bash
git add content/modeling-guide.md
git commit -m "docs: modeling-guide parts 1-4"
```

---

### Task 26: `modeling-guide.md` part 5 — worked example

**Files:**
- Modify: `content/modeling-guide.md` (replace the "Worked example" placeholder section)

- [ ] **Step 1: Append part 5 content**

In `/Users/jameshenson/Documents/tutorials/DLM/content/modeling-guide.md`, replace the `## Worked example` section (currently just a pointer) with:

```markdown
## Worked example

We walk through the 9-step procedure on a synthetic quarterly series generated by a local-linear-trend **plus** seasonal model, noting *why* each diagnostic choice happens and *what* each reveals.

### Setup

We simulate T = 120 quarters (30 years) from:
- Level innovation variance W_μ = 0.02
- Slope innovation variance W_β = 0.001
- Seasonal innovation variance W_s = 0.05
- Observation variance V = 0.3
- Initial level μ_0 = 10, slope β_0 = 0.05, seasonal pattern {1.0, −0.5, 0.2, −0.7}

### Step 1 — Inspect

The series wanders upward over the 30-year span, with a regular four-peaked wobble each cycle. Both a trend and seasonal are plausible.

### Step 2 — Decompose

A centered 5-quarter moving average recovers a steadily rising curve. The residual (y_t − MA) shows a clear four-cycle pattern — confirmation of a seasonal.

### Step 3 — Quantify

The ACF decays slowly (trend signature) and shows non-decaying peaks at lags 4, 8, 12 (seasonal with period 4). The PACF has a dominant spike at lag 1 and near-zero beyond.

### Step 4 — Pick components

Level + slope + seasonal(4). Three components, combined by superposition.

### Step 5 — Specify

- Trend: F_1 = [1, 0], G_1 = [[1, 1], [0, 1]], W_1 = diag(0.02, 0.001)
- Seasonal: F_2 = [1, 0, 0], G_2 = companion form with first row [−1, −1, −1], W_2 = diag(0.05, ε, ε)
- Combined: F = [1, 0, 1, 0, 0], G = block-diag, W = block-diag, d = 5
- V = 0.3 (known for this exercise)
- m_0 = [10, 0.05, 1.0, −0.5, 0.2], C_0 = 1e3 · I_5

### Step 6 — Fit

The Kalman filter returns filtered state means m_t that separate into trend and seasonal components. Plotted alongside y_t, the filtered F m_t tracks the observations closely without overfitting.

### Step 7 — Diagnose

- Standardized residuals: 95% within ±2, no runs — ✓
- Residual ACF: all lags inside the band — ✓
- Ljung-Box p ≈ 0.6 — ✓
- Filtered vs smoothed: visible divergence only in the first 4 quarters (diffuse prior regime)

### Step 8 — Forecast

h = 20 quarters ahead. The forecast extrapolates both the linear trend and the four-cycle pattern. Credible bands widen linearly because the dominant source of forecast uncertainty is the slope innovation propagating through G.

### Step 9 — Reveal

Because this is a worked example with known ground truth, we compare the fitted filtered state against the true state. They agree within the 95% credible band at every t.

### What to take away

- The method is mechanical once you can read ACF/PACF.
- The specification step is where understanding pays off — it's the only step that is not algorithmic.
- Diagnostics are how you know you got it right. Residuals that look like noise are your proof.
- The forecast is the product; everything before it is quality assurance.
```

- [ ] **Step 2: Commit**

```bash
git add content/modeling-guide.md
git commit -m "docs: worked example in modeling-guide"
```

---

### Task 27: Anchor-validation at startup

**Files:**
- Modify: `ui/app.py` (add anchor validation before `_validate_lessons_once`)
- Create: `tests/test_content.py`

- [ ] **Step 1: Write failing test for anchor extraction**

Create `/Users/jameshenson/Documents/tutorials/DLM/tests/test_content.py`:

```python
"""Tests for modeling-guide.md integrity."""

from pathlib import Path

import pytest

from ui.content_index import extract_anchors, find_missing_anchors
from lessons import ALL_LESSONS


GUIDE_PATH = Path(__file__).resolve().parent.parent / "content" / "modeling-guide.md"


def test_guide_file_exists():
    assert GUIDE_PATH.exists()


def test_extract_anchors_finds_headings():
    anchors = extract_anchors(GUIDE_PATH.read_text(encoding="utf-8"))
    assert "the-dlm-framework" in anchors
    assert "baseline-procedure" in anchors
    assert "tuning" in anchors
    assert "diagnostics" in anchors
    assert "worked-example" in anchors


def test_all_step_anchors_resolvable():
    anchors = extract_anchors(GUIDE_PATH.read_text(encoding="utf-8"))
    missing = find_missing_anchors(ALL_LESSONS, available_anchors=anchors)
    assert missing == [], f"Missing anchors: {missing}"
```

- [ ] **Step 2: Run test; expect ImportError on `ui.content_index`**

Run: `pytest tests/test_content.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement anchor-extraction utilities**

Create `/Users/jameshenson/Documents/tutorials/DLM/ui/content_index.py`:

```python
"""Utilities for extracting and validating markdown anchors.

GitHub-flavored markdown auto-generates anchors for headings by lowercasing,
stripping punctuation, and replacing spaces with hyphens. This module
duplicates that rule so we can validate deep links at startup.
"""

from __future__ import annotations

import re

from lessons.workflow import Lesson


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_ANCHOR_LINK_RE = re.compile(r"\]\(/reference#([a-z0-9\-]+)\)")


def _slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


def extract_anchors(markdown: str) -> set[str]:
    """Return the set of anchors produced by H1..H6 headings."""
    return {_slugify(m.group(2)) for m in _HEADING_RE.finditer(markdown)}


def extract_referenced_anchors_from_prompts(lessons: list[Lesson]) -> set[str]:
    """Return all /reference#<anchor> targets referenced in any WorkflowStep.prompt_md."""
    found: set[str] = set()
    for lesson in lessons:
        for step in lesson.workflow_steps:
            for m in _ANCHOR_LINK_RE.finditer(step.prompt_md):
                found.add(m.group(1))
    return found


def find_missing_anchors(
    lessons: list[Lesson], available_anchors: set[str]
) -> list[str]:
    referenced = extract_referenced_anchors_from_prompts(lessons)
    return sorted(referenced - available_anchors)
```

- [ ] **Step 4: Run tests; expect PASS**

Run: `pytest tests/test_content.py -v`
Expected: all pass.

- [ ] **Step 5: Wire anchor validation into app startup**

Modify `/Users/jameshenson/Documents/tutorials/DLM/ui/app.py`: in `_validate_lessons_once`, after the plot-fn validation, add:

```python
def _validate_lessons_once() -> None:
    if "lessons_validated" in st.session_state:
        return
    from ui.content_index import extract_anchors, find_missing_anchors

    allowed = set(PLOT_FN_REGISTRY.keys())
    for lesson in ALL_LESSONS:
        validate_lesson(lesson, allowed_plot_fns=allowed)

    guide_path = CONTENT_DIR / "modeling-guide.md"
    if guide_path.exists():
        anchors = extract_anchors(guide_path.read_text(encoding="utf-8"))
        missing = find_missing_anchors(ALL_LESSONS, available_anchors=anchors)
        if missing:
            raise RuntimeError(
                f"modeling-guide.md is missing anchors referenced by workflow steps: {missing}"
            )
    st.session_state["lessons_validated"] = True
```

- [ ] **Step 6: Manual smoke-test — run the app, confirm no startup error**

```bash
streamlit run ui/app.py
```
Expected: app starts without raising.

- [ ] **Step 7: Commit**

```bash
git add ui/content_index.py tests/test_content.py ui/app.py
git commit -m "feat(ui): validate modeling-guide anchors at startup"
```

---

## Phase 5 — CI and docs

### Task 28: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

Create `/Users/jameshenson/Documents/tutorials/DLM/.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e ".[dev]"

      - name: Ruff
        run: ruff check .

      - name: Mypy
        run: mypy engine lessons

      - name: Pytest
        run: pytest -q
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow"
```

Note: there's no step to verify the workflow runs here since this repo isn't yet on GitHub. If/when it is pushed, GitHub will run the workflow on push.

---

### Task 29: README with quickstart

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `/Users/jameshenson/Documents/tutorials/DLM/README.md`:

```markdown
# DLM Intuition Tutorial

An interactive Streamlit tutorial for building intuition about **Dynamic Linear Models (DLMs)**. Pick simulation parameters, watch the series appear, then walk through a 9-step modeling workflow — first as a demonstration, then as a challenge where you try to recover the model yourself.

Beginner-tier content covers three canonical models:

1. **Local level** — random walk plus observation noise
2. **Local linear trend** — level + slope
3. **Simple seasonal** — sum-to-zero factor model

## Quickstart

Requires Python 3.11+.

```bash
git clone <this-repo-url>
cd DLM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run ui/app.py
```

The app opens in your browser at http://localhost:8501.

## Project layout

- `engine/` — pure NumPy Kalman filter, smoother, forecast, diagnostics. No Streamlit imports; reusable from notebooks.
- `lessons/` — declarative `Lesson` objects wiring a model to a 9-step workflow.
- `ui/` — Streamlit app, Plotly renderers, session-state helpers.
- `content/modeling-guide.md` — long-form reference on DLM specification and fitting.
- `tests/` — pytest suite.
- `docs/superpowers/specs/` — design specifications.
- `docs/superpowers/plans/` — implementation plans.

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart and layout"
```

---

### Task 30: Final end-to-end verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/jameshenson/Documents/tutorials/DLM
source .venv/bin/activate
pytest -q
```
Expected: **all tests pass, no warnings raised to errors**. Wall time < 30 s.

- [ ] **Step 2: Run the full lint/type check**

```bash
ruff check .
mypy engine lessons
```
Expected: both clean (no errors).

- [ ] **Step 3: Smoke-test the app end-to-end**

```bash
streamlit run ui/app.py
```

Walk through:
1. Home page shows 3 lessons: 1 unlocked, 2 locked
2. Open lesson 1, Lesson mode, click Next through all 9 steps
3. Switch to Challenge mode, pick wrong answer at step 4 → warning feedback
4. Submit correct answer → success feedback
5. Progress through all 9 steps, click Finish
6. Home: lesson 1 ✅, lesson 2 🔓
7. Open lesson 2, walk through (verify slope component in plot_state and diagnostics)
8. Open lesson 3 after completing lesson 2, verify period slider works, ACF + seasonal subseries plot renders
9. Reference page renders modeling-guide.md correctly

- [ ] **Step 4: Commit any fixes found during verification**

If anything broke during step 3, fix and commit. Otherwise no commit needed.

- [ ] **Step 5: Final tag**

```bash
git tag -a v0.1.0 -m "Beginner tier v1 — all three lessons end-to-end"
```

This marks the v1 completion point for later reference.

---

## Summary of deliverables

| Area | Files | Status |
|---|---|---|
| Engine | `engine/models.py`, `simulate.py`, `filter.py`, `smoother.py`, `forecast.py`, `diagnostics.py` | Tasks 3–12 |
| Lessons | `lessons/workflow.py`, `local_level.py`, `local_linear_trend.py`, `seasonal.py`, `__init__.py` | Tasks 13–18 |
| UI | `ui/state.py`, `controls.py`, `plots.py`, `app.py`, `workflow_render.py`, `content_index.py` | Tasks 19–23, 27 |
| Content | `content/modeling-guide.md` (5 parts) | Tasks 25–26 |
| Tests | 7 test files covering engine + lessons + content | Throughout |
| CI/Docs | `.github/workflows/ci.yml`, `README.md`, `.gitignore`, `pyproject.toml` | Tasks 1, 2, 28, 29 |
| Verification | Manual smoke tests | Tasks 24, 30 |

When all tasks are complete, Beginner-tier v1 is shippable: three lessons end-to-end in both modes, all engine computations tested against analytic references, lint/type clean, CI configured, documented quickstart.
