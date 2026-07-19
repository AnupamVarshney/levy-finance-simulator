"""Maximum-likelihood calibration of Variance Gamma parameters to market data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, kve

from .market_data import fetch_sp500_log_returns

# Bounds for (sigma, theta, nu) used by the optimizer.
PARAM_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.01, 2.0),
    (-2.0, 0.0),
    (0.01, 2.0),
)

# |x| is clipped away from zero: the VG density has an integrable singularity
# at the origin when nu > 2/(1 + 2/nu), which breaks a naive evaluation.
_EPS = 1e-12


@dataclass
class CalibrationResult:
    """Outcome of a VG maximum-likelihood fit."""

    sigma: float
    theta: float
    nu: float
    log_likelihood: float
    n_observations: int
    success: bool

    @property
    def params(self) -> tuple[float, float, float]:
        """Calibrated ``(sigma, theta, nu)`` triple."""
        return self.sigma, self.theta, self.nu


def vg_logpdf(
    x: np.ndarray,
    sigma: float,
    theta: float,
    nu: float,
) -> np.ndarray:
    """
    Log-density of the Variance Gamma distribution at unit time.

    Implements the closed form

        f(x) = 2 * exp(theta * x / sigma^2)
               / (nu^(1/nu) * sqrt(2*pi) * sigma * gamma(1/nu))
               * (|x| / s)^(1/nu - 0.5)
               * kv(1/nu - 0.5, |x| * s / sigma^2)

    with ``s = sqrt(2*sigma^2/nu + theta^2)``, evaluated in log-space so that
    the Bessel factor does not underflow in the tails. ``scipy.special.kve``
    returns ``kv(v, z) * exp(z)``, hence the ``-z`` correction term.
    """
    x = np.asarray(x, dtype=float)
    abs_x = np.maximum(np.abs(x), _EPS)

    s = np.sqrt(2.0 * sigma**2 / nu + theta**2)
    order = 1.0 / nu - 0.5
    z = abs_x * s / sigma**2

    log_norm = (
        np.log(2.0)
        - (1.0 / nu) * np.log(nu)
        - 0.5 * np.log(2.0 * np.pi)
        - np.log(sigma)
        - gammaln(1.0 / nu)
    )

    bessel = kve(order, z)
    with np.errstate(divide="ignore"):
        log_bessel = np.log(bessel) - z

    return (
        log_norm
        + theta * x / sigma**2
        + order * (np.log(abs_x) - np.log(s))
        + log_bessel
    )


def vg_negative_log_likelihood(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """
    Negative log-likelihood of ``returns`` under VG parameters ``(sigma, theta, nu)``.

    Returns ``inf`` for parameter values that produce a non-finite density so the
    optimizer treats them as infeasible rather than failing.
    """
    sigma, theta, nu = params
    if sigma <= 0.0 or nu <= 0.0:
        return np.inf

    log_density = vg_logpdf(returns, sigma, theta, nu)
    if not np.all(np.isfinite(log_density)):
        return np.inf

    return float(-np.sum(log_density))


def calibrate_vg(
    returns: np.ndarray,
    n_starts: int = 5,
    seed: int | None = 42,
) -> CalibrationResult:
    """
    Fit VG parameters to ``returns`` by maximum likelihood.

    Runs ``n_starts`` L-BFGS-B optimizations from random points inside
    :data:`PARAM_BOUNDS` and keeps the fit with the lowest negative
    log-likelihood, which guards against local minima.
    """
    sample = np.asarray(returns, dtype=float).ravel()
    if sample.size < 2:
        raise ValueError("At least two return observations are required.")
    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")

    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in PARAM_BOUNDS])
    upper = np.array([b[1] for b in PARAM_BOUNDS])

    best = None
    for _ in range(n_starts):
        x0 = lower + rng.random(3) * (upper - lower)
        result = minimize(
            vg_negative_log_likelihood,
            x0=x0,
            args=(sample,),
            method="L-BFGS-B",
            bounds=PARAM_BOUNDS,
        )
        if not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise RuntimeError("VG calibration failed from every starting point.")

    sigma, theta, nu = best.x
    return CalibrationResult(
        sigma=float(sigma),
        theta=float(theta),
        nu=float(nu),
        log_likelihood=float(-best.fun),
        n_observations=int(sample.size),
        success=bool(best.success),
    )


def plot_calibrated_fit(
    returns: np.ndarray,
    result: CalibrationResult,
    output_dir: str | Path = "figures",
    filename: str = "calibrated_vg_vs_sp500.png",
) -> Path:
    """Overlay the calibrated VG density on the S&P 500 return histogram."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample = np.asarray(returns, dtype=float).ravel()

    grid = np.linspace(sample.min(), sample.max(), 1000)
    density = np.exp(vg_logpdf(grid, *result.params))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        sample,
        bins=120,
        density=True,
        alpha=0.55,
        color="#2563eb",
        label="S&P 500 daily log-returns",
    )
    ax.plot(
        grid,
        density,
        color="#dc2626",
        linewidth=2,
        label=(
            f"Calibrated VG (σ={result.sigma:.4f}, "
            f"θ={result.theta:.4f}, ν={result.nu:.4f})"
        ),
    )
    ax.set_title("MLE-Calibrated Variance Gamma vs S&P 500")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_calibration(
    start: str = "2010-01-01",
    end: str | None = None,
    ticker: str = "^GSPC",
    output_dir: str | Path = "figures",
    seed: int | None = 42,
) -> CalibrationResult:
    """Fetch S&P 500 returns, calibrate VG by MLE, print and plot the fit."""
    returns = fetch_sp500_log_returns(start=start, end=end, ticker=ticker)
    result = calibrate_vg(returns, seed=seed)

    print(f"VG calibration on {ticker} ({start} to {end or 'today'})")
    print(f"  observations   : {result.n_observations}")
    print(f"  sigma          : {result.sigma:.6f}")
    print(f"  theta          : {result.theta:.6f}")
    print(f"  nu             : {result.nu:.6f}")
    print(f"  log-likelihood : {result.log_likelihood:.4f}")
    print(f"  converged      : {result.success}")

    path = plot_calibrated_fit(returns, result, output_dir=output_dir)
    print(f"  figure         : {path}")
    return result


if __name__ == "__main__":
    run_calibration()
