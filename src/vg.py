"""Variance Gamma (VG) Lévy process simulator."""

from __future__ import annotations

import numpy as np

from .gbm import SimulationResult


def _martingale_correction(theta: float, sigma: float, nu: float) -> float:
    """
    Compensator for the exponential VG model.

    Ensures E[S_t / S_0] = exp(mu * t) when log-price drift mu is applied.
    """
    inner = 1.0 - theta * nu - 0.5 * sigma**2 * nu
    if inner <= 0:
        raise ValueError(
            "Invalid VG parameters: 1 - theta*nu - 0.5*sigma^2*nu must be positive."
        )
    return (1.0 / nu) * np.log(inner)


def simulate_vg(
    S0: float = 100.0,
    mu: float = 0.05,
    theta: float = -0.1,
    sigma: float = 0.2,
    nu: float = 0.5,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 1000,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate stock prices under a Variance Gamma model.

    The VG process is constructed via gamma subordination:
        X_t = theta * G_t + sigma * W(G_t)
    where G_t is a gamma process with mean t and variance nu * t.

    Price dynamics:
        log(S_t / S_0) = mu * t + X_t - omega * t
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive.")

    omega = _martingale_correction(theta, sigma, nu)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    shape = dt / nu
    dG = rng.gamma(shape=shape, scale=nu, size=(n_paths, n_steps))
    z = rng.standard_normal(size=(n_paths, n_steps))
    dX = theta * dG + sigma * np.sqrt(dG) * z

    log_prices = np.empty((n_paths, n_steps + 1), dtype=float)
    log_prices[:, 0] = np.log(S0)

    drift = (mu - omega) * dt
    log_increments = drift + dX
    log_prices[:, 1:] = log_prices[:, :1] + np.cumsum(log_increments, axis=1)

    paths = np.exp(log_prices)
    log_returns = np.diff(log_prices, axis=1)

    return SimulationResult(
        paths=paths,
        log_returns=log_returns,
        times=times,
        model="VG",
        parameters={
            "S0": S0,
            "mu": mu,
            "theta": theta,
            "sigma": sigma,
            "nu": nu,
            "omega": float(omega),
            "T": T,
            "n_steps": n_steps,
            "n_paths": n_paths,
        },
    )
