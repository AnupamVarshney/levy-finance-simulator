"""Summary statistics and risk metrics for simulated returns."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def flatten_returns(log_returns: np.ndarray) -> np.ndarray:
    """Flatten path-wise log-returns into a single sample."""
    return np.asarray(log_returns, dtype=float).ravel()


def aggregate_returns(log_returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Sum log-returns over non-overlapping windows to study tail behavior."""
    if window <= 1:
        return flatten_returns(log_returns)

    arr = np.asarray(log_returns, dtype=float)
    n_paths, n_steps = arr.shape
    n_blocks = n_steps // window
    if n_blocks == 0:
        raise ValueError("window must be smaller than the number of simulated steps.")

    trimmed = arr[:, : n_blocks * window].reshape(n_paths, n_blocks, window)
    return trimmed.sum(axis=2).ravel()


def value_at_risk(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Historical VaR at confidence level alpha.

    Returns the loss quantile as a positive number.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    sample = flatten_returns(returns)
    quantile = np.quantile(sample, 1.0 - alpha)
    return float(-quantile)


def expected_shortfall(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Expected shortfall (CVaR) at confidence level alpha."""
    sample = flatten_returns(returns)
    threshold = np.quantile(sample, 1.0 - alpha)
    tail = sample[sample <= threshold]
    if tail.size == 0:
        return value_at_risk(returns, alpha)
    return float(-np.mean(tail))


def summarize_returns(
    log_returns: np.ndarray,
    model_name: str,
    aggregate_window: int | None = None,
) -> dict[str, float | str]:
    """Compute descriptive and tail-risk statistics for log-returns."""
    if aggregate_window is None:
        sample = flatten_returns(log_returns)
    else:
        sample = aggregate_returns(log_returns, window=aggregate_window)
    return {
        "model": model_name,
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample, ddof=1)),
        "skewness": float(stats.skew(sample)),
        "excess_kurtosis": float(stats.kurtosis(sample, fisher=True)),
        "var_95": value_at_risk(sample, alpha=0.95),
        "var_99": value_at_risk(sample, alpha=0.99),
        "es_95": expected_shortfall(sample, alpha=0.95),
    }


def compare_models(
    results: Iterable[tuple[str, np.ndarray]],
    aggregate_window: int | None = 21,
) -> pd.DataFrame:
    """Build a comparison table for multiple simulated return samples."""
    rows = [
        summarize_returns(log_returns, model_name, aggregate_window=aggregate_window)
        for model_name, log_returns in results
    ]
    df = pd.DataFrame(rows).set_index("model")
    return df.round(4)
