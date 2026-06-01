"""Plotting utilities for GBM vs VG comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .market_data import fetch_sp500_log_returns
from .metrics import flatten_returns, summarize_returns


def _ensure_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_return_histograms(
    gbm_returns: np.ndarray,
    vg_returns: np.ndarray,
    output_dir: str | Path = "figures",
    filename: str = "return_histograms.png",
) -> Path:
    """Overlay histograms of GBM and VG log-returns."""
    out = _ensure_dir(Path(output_dir))
    gbm = flatten_returns(gbm_returns)
    vg = flatten_returns(vg_returns)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(gbm.min(), vg.min()),
        max(gbm.max(), vg.max()),
        60,
    )
    ax.hist(gbm, bins=bins, alpha=0.55, density=True, label="GBM", color="#2563eb")
    ax.hist(vg, bins=bins, alpha=0.55, density=True, label="VG", color="#dc2626")
    ax.set_title("Log-Return Distribution: GBM vs Variance Gamma")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_qq_panels(
    gbm_returns: np.ndarray,
    vg_returns: np.ndarray,
    output_dir: str | Path = "figures",
    filename: str = "qq_plots.png",
) -> Path:
    """Side-by-side QQ plots against the normal distribution."""
    out = _ensure_dir(Path(output_dir))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, sample, title in zip(
        axes,
        [flatten_returns(gbm_returns), flatten_returns(vg_returns)],
        ["GBM", "Variance Gamma"],
    ):
        stats.probplot(sample, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot: {title}")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sample_paths(
    gbm_paths: np.ndarray,
    vg_paths: np.ndarray,
    times: np.ndarray,
    output_dir: str | Path = "figures",
    n_paths: int = 8,
    filename: str = "sample_paths.png",
) -> Path:
    """Plot sample price paths for both models."""
    out = _ensure_dir(Path(output_dir))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, paths, title, color in zip(
        axes,
        [gbm_paths, vg_paths],
        ["GBM", "Variance Gamma"],
        ["#2563eb", "#dc2626"],
    ):
        for i in range(min(n_paths, paths.shape[0])):
            ax.plot(times, paths[i], color=color, alpha=0.65, linewidth=1.0)
        ax.set_title(f"Sample Paths: {title}")
        ax.set_xlabel("Time (years)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Price")
    fig.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vg_vs_market_histogram(
    vg_returns: np.ndarray,
    market_returns: np.ndarray | None = None,
    market_label: str = "S&P 500 (daily)",
    output_dir: str | Path = "figures",
    filename: str = "vg_vs_sp500_histogram.png",
    market_start: str = "2010-01-01",
) -> Path:
    """Overlay VG simulated log-returns against real S&P 500 daily log-returns."""
    out = _ensure_dir(Path(output_dir))
    vg = flatten_returns(vg_returns)
    if market_returns is None:
        market_returns = fetch_sp500_log_returns(start=market_start)
    market = np.asarray(market_returns, dtype=float).ravel()

    lo = min(vg.min(), market.min())
    hi = max(vg.max(), market.max())
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        market,
        bins=bins,
        alpha=0.55,
        density=True,
        label=market_label,
        color="#16a34a",
    )
    ax.hist(
        vg,
        bins=bins,
        alpha=0.55,
        density=True,
        label="VG (simulated)",
        color="#dc2626",
    )
    ax.set_title("Daily Log-Returns: Variance Gamma vs S&P 500")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_var_comparison(
    gbm_returns: np.ndarray,
    vg_returns: np.ndarray,
    output_dir: str | Path = "figures",
    filename: str = "var_comparison.png",
    aggregate_window: int = 21,
) -> Path:
    """Bar chart comparing VaR levels across models."""
    out = _ensure_dir(Path(output_dir))
    gbm_stats = summarize_returns(
        gbm_returns, "GBM", aggregate_window=aggregate_window
    )
    vg_stats = summarize_returns(
        vg_returns, "VG", aggregate_window=aggregate_window
    )

    labels = ["VaR 95%", "VaR 99%", "ES 95%"]
    gbm_values = [gbm_stats["var_95"], gbm_stats["var_99"], gbm_stats["es_95"]]
    vg_values = [vg_stats["var_95"], vg_stats["var_99"], vg_stats["es_95"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, gbm_values, width, label="GBM", color="#2563eb")
    ax.bar(x + width / 2, vg_values, width, label="VG", color="#dc2626")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Loss magnitude")
    ax.set_title("Tail-Risk Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_figures(
    gbm_paths: np.ndarray,
    gbm_returns: np.ndarray,
    vg_paths: np.ndarray,
    vg_returns: np.ndarray,
    times: np.ndarray,
    output_dir: str | Path = "figures",
    include_market_comparison: bool = True,
    market_start: str = "2010-01-01",
) -> dict[str, Path]:
    """Generate the full figure set used in the README and report."""
    paths: dict[str, Path] = {
        "histograms": plot_return_histograms(gbm_returns, vg_returns, output_dir),
        "qq_plots": plot_qq_panels(gbm_returns, vg_returns, output_dir),
        "sample_paths": plot_sample_paths(gbm_paths, vg_paths, times, output_dir),
        "var_comparison": plot_var_comparison(gbm_returns, vg_returns, output_dir),
    }
    if include_market_comparison:
        paths["vg_vs_sp500"] = plot_vg_vs_market_histogram(
            vg_returns,
            output_dir=output_dir,
            market_start=market_start,
        )
    return paths
