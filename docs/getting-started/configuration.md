---
title: Configuration
---

# ⚙️ Configuration

`infer-ci` is configured entirely through Python — there is no config file required.

Pass arguments directly to `MetricEvaluator` or `evaluate_metric`:

```python
from infer_ci import MetricEvaluator

evaluator = MetricEvaluator(
    confidence_level=0.95,
    method="bootstrap_bca",
    n_resamples=9999,
)
```

## 🌎 Environment Variables

[**`.env.example`**](https://github.com/humblebeeai/infer-ci/blob/main/.env.example):

```sh
--8<-- "./.env.example"
```
