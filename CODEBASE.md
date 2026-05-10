# infer-ci Codebase Documentation

> Version 0.2.1 · Python 3.10+ · MIT License

## Overview

**infer-ci** is a Python library that computes ML evaluation metrics with **confidence intervals (CIs)**, enabling statistically-sound model release decisions instead of relying on single-point scores. It covers three task domains — binary/multi-class classification, regression, and object detection — and provides multiple CI methods per domain.

**Install**: `pip install infer-ci`  
**Docs**: https://infer.humblebee.ai  
**PyPI**: https://pypi.org/project/infer-ci/

---

## Repository Structure

```
infer-ci/
├── src/infer_ci/                   # Library source (~4,700 lines)
│   ├── __init__.py                 # Public API exports
│   ├── __version__.py              # Version string ("0.2.1")
│   ├── evaluator.py                # Unified MetricEvaluator class + evaluate_metric()
│   ├── binary_metrics.py           # Binary classification metrics + proportion CIs
│   ├── takahashi_methods.py        # Takahashi delta method (F1, Precision, Recall)
│   ├── regression_metrics.py       # Regression metrics
│   ├── object_detection_metrics.py # YOLO-format object detection metrics
│   ├── auc.py                      # ROC AUC with DeLong / bootstrap
│   ├── delong.py                   # Fast DeLong AUC variance computation
│   ├── methods.py                  # bootstrap_ci(), jackknife_ci() core routines
│   ├── classification_report.py    # classification_report_with_ci()
│   ├── visualize.py                # Bootstrap histogram plots
│   └── utils.py                    # Confusion matrix helpers
├── tests/                          # Pytest test suite (9 files)
├── examples/
│   ├── simple/main.py              # Binary classification example
│   └── advanced/                   # Advanced example placeholders
├── datasets/                       # CSV datasets for tests and examples
├── notebooks/                      # Jupyter notebooks
├── docs/                           # MkDocs documentation source
├── scripts/                        # Build, test, release, version scripts
├── requirements/                   # Dependency files by environment
├── .github/workflows/              # GitHub Actions CI/CD pipelines
├── pyproject.toml                  # Build metadata (setuptools)
├── pytest.ini                      # Pytest configuration
├── Makefile                        # Developer shortcuts
└── .pre-commit-config.yaml         # Pre-commit hooks (black, flake8, bandit, pyright, gitleaks)
```

---

## Core Modules

### `evaluator.py` — Unified Interface

**`MetricEvaluator`** is the primary entry point. It routes calls to task-specific metric implementations and exposes a consistent API regardless of task type.

**Key methods:**

| Method | Description |
|--------|-------------|
| `evaluate(y_true, y_pred, task, metric, method, confidence_level, compute_ci, plot, **kwargs)` | Evaluate a single metric with optional CI |
| `evaluate_multiple(y_true, y_pred, task, metrics, ...)` | Evaluate multiple metrics at once |
| `get_available_metrics(task)` | List metrics for a task type |
| `get_available_methods(task)` | List CI methods for a task type |
| `info()` | Dictionary of all available options |
| `print_info()` | Pretty-printed summary |

Instance methods also expose individual metrics directly (`evaluator.accuracy_score()`, `evaluator.mae()`, etc.).

**`evaluate_metric()`** — Module-level convenience function for one-off evaluations without instantiating the class.

---

### `binary_metrics.py` — Classification Metrics

Implements proportion-based confidence intervals for binary classification outcomes.

**Metrics:** `accuracy`, `precision` / `ppv_score`, `recall` / `tpr_score`, `specificity` / `tnr_score`, `npv_score`, `fpr_score`

**CI methods:** Wilson (default), Normal, Agresti-Coull, Beta, Jeffreys, Binomial test, bootstrap variants

---

### `takahashi_methods.py` — F1 / Precision / Recall CI

Implements the Takahashi (2022) delta method for computing confidence intervals on F1, Precision, and Recall across binary and multi-class settings (micro/macro averaging). Handles variance propagation analytically without resampling.

---

### `regression_metrics.py` — Regression Metrics

**Metrics (13 total):**

| Metric | Key |
|--------|-----|
| Mean Absolute Error | `mae` |
| Mean Squared Error | `mse` |
| Root Mean Squared Error | `rmse` |
| R² (Coefficient of Determination) | `r2` |
| Mean Absolute Percentage Error | `mape` |
| Adjusted R² | `adjusted_r2_score` |
| Symmetric MAPE | `sym_mean_abs_per_error` |
| RMSE of log-transformed values | `rmse_log` |
| Median Absolute Error | `med_abs_err` |
| Huber Loss | `huber_loss` |
| Explained Variance Score | `exp_var_score` |
| Mean Bias Deviation | `mean_bia_dev` |
| Intersection over Union | `iou` |

**CI methods:** Bootstrap BCA (default), Bootstrap Percentile, Bootstrap Basic, Jackknife

---

### `object_detection_metrics.py` — YOLO Object Detection

The largest module (~1,467 lines). Accepts ultralytics-format prediction Results objects and YOLO-format ground truth labels.

**Workflow:**
1. Load ground truth from a YOLO label directory or `data.yaml`
2. Extract boxes, class IDs, and confidence scores from ultralytics Results
3. Compute per-prediction IoU against ground truth boxes
4. Apply NMS to remove duplicate detections
5. Compute TP / FP / FN per class at each IoU threshold
6. Calculate per-class Precision, Recall, AP
7. Wrap metrics in bootstrap CIs

**Metrics:** `map` (mAP@0.5:0.95), `map50` (mAP@0.5), `precision`, `recall` — all with per-class breakdowns

---

### `auc.py` + `delong.py` — ROC AUC

`auc.py` exposes ROC AUC with a choice of DeLong (analytical, fast) or bootstrap CI.  
`delong.py` implements the fast DeLong algorithm for AUC variance, supporting both weighted and unweighted cases.

---

### `methods.py` — Core CI Routines

Low-level CI computation used across modules:

- **`bootstrap_ci(data, statistic, confidence_level, n_bootstrap, method)`** — Runs bootstrap resampling; supports `bca`, `percentile`, `basic` methods
- **`jackknife_ci(data, statistic, confidence_level)`** — Leave-one-out jackknife with t-distribution

---

### `classification_report.py`

**`classification_report_with_ci(y_true, y_pred, confidence_level, method)`** — Returns a Pandas DataFrame mirroring sklearn's `classification_report` but with CI columns for each metric per class, plus micro/macro averages.

---

### `visualize.py`

**`create_bootstrap_histogram_plot(bootstrap_samples, metric_name, ci_lower, ci_upper, point_estimate)`** — Saves bootstrap distribution histogram to `results/` directory. Called automatically when `plot=True` is passed to any metric function.

---

### `utils.py`

**`get_positive_negative_counts(y_true, y_pred)`** — Returns `(TP, FP, FN, TN)` from binary predictions. Used internally by classification metric functions.

---

## Confidence Interval Methods Reference

| Category | Method Key | Description |
|----------|-----------|-------------|
| Proportion | `wilson` | Wilson score interval (default for classification) |
| Proportion | `normal` | Normal approximation |
| Proportion | `agresti_coull` | Improved normal approximation |
| Proportion | `beta` | Bayesian beta distribution |
| Proportion | `jeffreys` | Jeffreys' prior |
| Proportion | `binomial_test` | Exact binomial test |
| Bootstrap | `bootstrap_bca` | Bias-corrected accelerated (default, most robust) |
| Bootstrap | `bootstrap_percentile` | Simple percentile method |
| Bootstrap | `bootstrap_basic` | Basic / reflection method |
| Jackknife | `jackknife` | Leave-one-out, uses t-distribution |
| Analytical | `takahashi` | Delta method for F1/Precision/Recall (2022) |
| Analytical | `delong` | Fast AUC variance (for ROC AUC) |

---

## Supported Metrics by Task

### Classification

```
accuracy, precision, ppv_score, recall, tpr_score,
specificity, tnr_score, npv_score, fpr_score,
f1_score, precision_takahashi, recall_takahashi, roc_auc_score
```

### Regression

```
mae, mse, rmse, r2, mape, adjusted_r2_score,
sym_mean_abs_per_error, rmse_log, med_abs_err,
huber_loss, exp_var_score, mean_bia_dev, iou
```

### Object Detection

```
map, map50, precision, recall  (+ per-class breakdowns)
```

---

## Quick Usage

```python
from infer_ci import MetricEvaluator, evaluate_metric

evaluator = MetricEvaluator()

# Single metric with CI
result = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="classification",
    metric="accuracy",
    method="wilson",
    confidence_level=0.95,
    compute_ci=True,
)
print(result)
# {"metric": "accuracy", "value": 0.94, "ci_lower": 0.91, "ci_upper": 0.97}

# Multiple metrics
results = evaluator.evaluate_multiple(
    y_true=y_true,
    y_pred=y_pred,
    task="regression",
    metrics=["mae", "rmse", "r2"],
)

# One-off convenience function
result = evaluate_metric(y_true, y_pred, task="classification", metric="f1_score")

# Classification report with CIs
from infer_ci import classification_report_with_ci
df = classification_report_with_ci(y_true, y_pred, confidence_level=0.95)
```

---

## Test Suite

| File | Coverage |
|------|----------|
| `conftest.py` | Shared fixtures |
| `test_binary_classification_ci.py` | LR & XGBoost on breast cancer dataset |
| `test_classification.py` | Multi-class CI validation |
| `test_regression_ci.py` | California housing regression metrics |
| `test_regression.py` | General regression pipeline |
| `test_sklearn_equality.py` | Validates metrics match sklearn equivalents |
| `test_delong.py` | DeLong AUC variance |
| `test_object_detection_ci.py` | YOLO-format object detection metrics |

**Run tests:**
```bash
pytest                                        # All tests
pytest --cov=src --cov-report=term-missing    # With coverage
./scripts/test.sh -l                          # GitHub Actions equivalent
```

---

## CI/CD Pipelines

| Workflow | Trigger | Action |
|----------|---------|--------|
| `1.bump-version.yml` | Merge to main | Auto-bumps version |
| `2.build-publish.yml` | Merge to main | Tests → build → publish to TestPyPI |
| `3.update-changelog.yml` | Merge to main | Updates `CHANGELOG.md` |
| `publish-docs.yml` | Tag push | Builds & publishes MkDocs docs |

---

## Dependencies

**Core:**
`numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `pandas`, `pydantic`, `pydantic-settings`, `python-dotenv`, `PyYAML`, `tqdm`, `xgboost`, `setuptools`

**Test:** `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`

**Dev:** `pyright`, `pre-commit`

**Build:** `setuptools`, `wheel`, `build`, `twine`

---

## Code Quality

Pre-commit hooks enforce:
- **Black** — formatting
- **Flake8** — linting (max line 120)
- **Pyupgrade** — Python 3.10+ syntax
- **Bandit** — security scanning
- **Pyright** — static type checking
- **Gitleaks / detect-secrets** — secret detection
- **ShellCheck** — bash script linting
- JSON / TOML / YAML validators
- Merge conflict + large file detection

---

## Key Design Decisions

**Unified interface over flat functions** — `MetricEvaluator` provides a single entry point for all tasks and metrics; individual functions are also exported for direct use when only one metric is needed.

**CI separate from metric logic** — Every metric function accepts `compute_ci=True/False`; CI computation is handled by shared helpers (`_compute_confidence_interval`, `bootstrap_ci`, `jackknife_ci`) that wrap the metric function as a statistic callable.

**Analytical methods preferred for proportions** — Wilson score is the default for classification metrics because it has better coverage properties than the normal approximation, especially for extreme probabilities. Bootstrap BCA is the default elsewhere.

**YOLO-native object detection** — Ground truth is loaded directly from YOLO label format or `data.yaml`, and predictions are consumed from ultralytics `Results` objects, making integration with YOLO-based pipelines zero-friction.
