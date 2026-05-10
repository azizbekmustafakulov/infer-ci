"""Confidence intervals for common binary metrics."""

from statsmodels.stats.proportion import proportion_confint
from typing import List, Tuple, Union, Optional
from functools import partial
import numpy as np
from .utils import get_positive_negative_counts
from .methods import bootstrap_methods
from .visualize import bootstrap_with_plot

# Small epsilon to avoid division by zero in proportion-based metrics
_EPSILON: float = 1e-7

proportion_conf_methods: List[str] = [
    'wilson',
    'normal',
    'agresti_coull',
    'beta',
    'jeffreys',
    'binom_test',
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_binary_inputs(
        y_true: List[int],
        y_pred: List[int],
        method: str,
        confidence_level: float,
        check_binary_range: bool = True) -> None:
    """Validate common inputs for binary proportion CI functions."""
    if method not in proportion_conf_methods:
        raise ValueError(f'Proportion CI method {method} not in {proportion_conf_methods}')
    if not 0 <= confidence_level <= 1:
        raise ValueError(f'confidence_level must be between 0 and 1, got {confidence_level}')
    if check_binary_range:
        if np.min(y_true) < 0 or np.max(y_true) > 1:
            raise ValueError('This metric is supported only for binary classification')
        if np.min(y_pred) < 0 or np.max(y_pred) > 1:
            raise ValueError('This metric is supported only for binary classification')


def _run_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        metric_func_no_ci,
        metric_name: str,
        confidence_level: float,
        method: str,
        n_resamples: int,
        random_state: Optional[np.random.RandomState],
        plot: bool) -> Tuple[float, Tuple[float, float]]:
    """Wrap bootstrap_with_plot with a consistent call signature."""
    return bootstrap_with_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_func=metric_func_no_ci,
        metric_name=metric_name,
        confidence_level=confidence_level,
        method=method,
        n_resamples=n_resamples,
        random_state=random_state,
        plot=plot,
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def accuracy_score_binomial_ci(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the accuracy score and the confidence interval.

    The confidence interval is computed as a binomial proportion.
    See https://www.statsmodels.org/devel/generated/statsmodels.stats.proportion.proportion_confint.html

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    method : str, optional
        The statsmodels proportion method, by default 'wilson'
    compute_ci : bool, optional
        If true return the confidence interval as well as the accuracy score, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The accuracy score and optionally the confidence interval.
    """
    _validate_binary_inputs(y_true, y_pred, method, confidence_level, check_binary_range=False)

    correct = np.sum(np.array(y_pred) == np.array(y_true))
    acc = correct / len(y_pred)
    if compute_ci:
        interval = proportion_confint(
            correct, len(y_pred), alpha=1 - confidence_level, method=method)
        return acc, interval
    else:
        return acc


def accuracy_score_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        n_resamples: int = 9999,
        random_state: Optional[np.random.RandomState] = None,
        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    """Compute the accuracy score confidence interval using the bootstrap method."""
    return _run_bootstrap(
        y_true, y_pred,
        partial(accuracy_score_binomial_ci, compute_ci=False),
        "accuracy", confidence_level, method, n_resamples, random_state, plot,
    )


def accuracy_score(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the accuracy score and optionally the confidence interval."""
    if method in bootstrap_methods:
        return accuracy_score_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return accuracy_score_binomial_ci(y_true, y_pred, confidence_level, method, compute_ci)


# ---------------------------------------------------------------------------
# Positive Predictive Value (Precision)
# ---------------------------------------------------------------------------

def ppv_score_binomial_ci(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    _validate_binary_inputs(y_true, y_pred, method, confidence_level)

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FP = TP[1], FP[1]

    result = TP / (TP + FP + _EPSILON)
    if compute_ci:
        interval = proportion_confint(
            TP, TP + FP, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def ppv_score_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        n_resamples: int = 9999,
        random_state: Optional[np.random.RandomState] = None,
        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    return _run_bootstrap(
        y_true, y_pred,
        partial(ppv_score_binomial_ci, compute_ci=False),
        "precision", confidence_level, method, n_resamples, random_state, plot,
    )


def ppv_score(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return ppv_score_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return ppv_score_binomial_ci(y_true, y_pred, confidence_level, method, compute_ci)


# ---------------------------------------------------------------------------
# Negative Predictive Value
# ---------------------------------------------------------------------------

def npv_score_binomial_ci(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    _validate_binary_inputs(y_true, y_pred, method, confidence_level)

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TN, FN = TN[1], FN[1]

    result = TN / (TN + FN + _EPSILON)
    if compute_ci:
        interval = proportion_confint(
            TN, TN + FN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def npv_score_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        n_resamples: int = 9999,
        random_state: Optional[np.random.RandomState] = None,
        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    return _run_bootstrap(
        y_true, y_pred,
        partial(npv_score_binomial_ci, compute_ci=False),
        "npv", confidence_level, method, n_resamples, random_state, plot,
    )


def npv_score(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return npv_score_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return npv_score_binomial_ci(y_true, y_pred, confidence_level, method, compute_ci)


# ---------------------------------------------------------------------------
# True Positive Rate (Recall / Sensitivity)
# ---------------------------------------------------------------------------

def tpr_score_binomial_ci(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    _validate_binary_inputs(y_true, y_pred, method, confidence_level)

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FN = TP[1], FN[1]

    result = TP / (TP + FN + _EPSILON)
    if compute_ci:
        interval = proportion_confint(
            TP, TP + FN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def tpr_score_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        n_resamples: int = 9999,
        random_state: Optional[np.random.RandomState] = None,
        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    return _run_bootstrap(
        y_true, y_pred,
        partial(tpr_score_binomial_ci, compute_ci=False),
        "recall", confidence_level, method, n_resamples, random_state, plot,
    )


def tpr_score(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return tpr_score_bootstrap(y_true, y_pred, confidence_level, method, **kwargs)
    else:
        return tpr_score_binomial_ci(y_true, y_pred, confidence_level, method, compute_ci=compute_ci)


# ---------------------------------------------------------------------------
# False Positive Rate
# ---------------------------------------------------------------------------

def fpr_score_binomial_ci(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    _validate_binary_inputs(y_true, y_pred, method, confidence_level)

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    FP, TN = FP[1], TN[1]

    result = FP / (FP + TN + _EPSILON)
    if compute_ci:
        interval = proportion_confint(
            FP, FP + TN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def fpr_score_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        n_resamples: int = 9999,
        random_state: Optional[np.random.RandomState] = None,
        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    return _run_bootstrap(
        y_true, y_pred,
        partial(fpr_score_binomial_ci, compute_ci=False),
        "fpr", confidence_level, method, n_resamples, random_state, plot,
    )


def fpr_score(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return fpr_score_bootstrap(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, **kwargs)
    else:
        return fpr_score_binomial_ci(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level,
            method=method, compute_ci=compute_ci)


# ---------------------------------------------------------------------------
# True Negative Rate (Specificity)
# ---------------------------------------------------------------------------

def tnr_score_binomial_ci(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    _validate_binary_inputs(y_true, y_pred, method, confidence_level)

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TN, FP = TN[1], FP[1]

    result = TN / (TN + FP + _EPSILON)
    if compute_ci:
        interval = proportion_confint(
            TN, TN + FP, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def tnr_score_bootstrap(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'bootstrap_bca',
        n_resamples: int = 9999,
        random_state: Optional[np.random.RandomState] = None,
        plot: bool = False) -> Tuple[float, Tuple[float, float]]:
    return _run_bootstrap(
        y_true, y_pred,
        partial(tnr_score_binomial_ci, compute_ci=False),
        "specificity", confidence_level, method, n_resamples, random_state, plot,
    )


def tnr_score(
        y_true: List[int],
        y_pred: List[int],
        confidence_level: float = 0.95,
        method: str = 'wilson',
        compute_ci: bool = True,
        **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    if method in bootstrap_methods:
        return tnr_score_bootstrap(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, method=method, **kwargs)
    else:
        return tnr_score_binomial_ci(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level,
            method=method, compute_ci=compute_ci)
