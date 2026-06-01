"""Fetch real-market log-returns for model comparison."""

from __future__ import annotations

import numpy as np


def fetch_sp500_log_returns(
    start: str = "2010-01-01",
    end: str | None = None,
    ticker: str = "^GSPC",
) -> np.ndarray:
    """
    Download S&P 500 daily prices and return log-returns.

    Uses Yahoo Finance via yfinance (ticker ``^GSPC`` by default).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for market data. Install with: pip install yfinance"
        ) from exc

    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError(f"No price data returned for ticker {ticker!r}.")

    close = data["Close"]
    if hasattr(close, "ndim") and close.ndim > 1:
        close = close.iloc[:, 0]
    prices = np.asarray(close.dropna(), dtype=float)
    if prices.size < 2:
        raise ValueError(f"Insufficient price history for ticker {ticker!r}.")

    return np.diff(np.log(prices))
