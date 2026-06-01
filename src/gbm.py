"""Geometric Brownian Motion (GBM) simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulated price paths and log-returns."""

    paths: np.ndarray
    log_returns: np.ndarray
    times: np.ndarray
    model: str
    parameters: dict


def simulate_gbm(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 1000,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate Geometric Brownian Motion paths using the exact log-normal update.

    S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    z = rng.standard_normal(size=(n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_prices = np.empty((n_paths, n_steps + 1), dtype=float)
    log_prices[:, 0] = np.log(S0)
    log_prices[:, 1:] = log_prices[:, :1] + np.cumsum(log_increments, axis=1)

    paths = np.exp(log_prices)
    log_returns = np.diff(log_prices, axis=1)

    return SimulationResult(
        paths=paths,
        log_returns=log_returns,
        times=times,
        model="GBM",
        parameters={
            "S0": S0,
            "mu": mu,
            "sigma": sigma,
            "T": T,
            "n_steps": n_steps,
            "n_paths": n_paths,
        },
    )
