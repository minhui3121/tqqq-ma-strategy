"""Calculate the constant daily drag for TQQQ over a requested date range.

The script downloads ^NDX and TQQQ data from Yahoo Finance, then solves for the
constant daily drag g such that:

    price_n = price_{n-1} * (1 + 3 * ndx_return_n - g)

matches the observed TQQQ terminal price over the requested period.

It prints the solved daily drag and the annual equivalent drag implied by that
daily value.
"""

from __future__ import annotations

import argparse
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def download_close(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Download close prices from Yahoo Finance for the requested date range."""

    end_inclusive = pd.to_datetime(end_date) + timedelta(days=1)
    data = yf.download(
        ticker,
        start=start_date,
        end=end_inclusive.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Close" not in data.columns:
        raise RuntimeError(f"Yahoo Finance response for {ticker} does not include Close prices")
    return data["Close"].sort_index()


def solve_constant_drag(ndx_returns: pd.Series, first_price: float, target_price: float) -> float:
    """Solve for constant daily drag g so product((1 + 3*r_i - g)) * first_price == target_price."""

    r = ndx_returns.to_numpy(copy=True)
    tiny = 1e-12
    upper = float((1.0 + 3.0 * r).min() - tiny)
    lower = -0.999
    log_target = math.log(target_price / first_price)

    def f_logsum(g: float) -> float:
        factors = 1.0 + 3.0 * r - g
        if (factors <= 0).any():
            return -math.inf
        return float(np.log(factors).sum()) - log_target

    f_lo = f_logsum(lower)
    f_hi = f_logsum(upper)

    if f_lo == 0:
        return lower
    if f_hi == 0:
        return upper

    if f_lo * f_hi > 0:
        total_log_theoretical = float(np.log(1.0 + 3.0 * r).sum())
        ratio = math.exp(log_target - total_log_theoretical)
        n = len(r)
        return 1.0 - ratio ** (1.0 / n)

    lo = lower
    hi = upper
    flo = f_lo
    mid = lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f_logsum(mid)
        if not math.isfinite(fmid):
            lo = mid
            flo = fmid
            continue
        if abs(fmid) < 1e-12:
            return float(mid)
        if fmid * flo > 0:
            lo = mid
            flo = fmid
        else:
            hi = mid
    return float(mid)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve TQQQ daily drag for a date range")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    if start_date > end_date:
        raise ValueError("start-date must be on or before end-date")

    ndx = download_close("^NDX", args.start_date, args.end_date)
    tqqq = download_close("TQQQ", args.start_date, args.end_date)

    common_index = ndx.index.intersection(tqqq.index)
    common_index = common_index[(common_index >= start_date) & (common_index <= end_date)]
    if common_index.empty:
        raise RuntimeError("No overlapping trading dates found for the requested range")

    ndx = ndx.reindex(common_index)
    tqqq = tqqq.reindex(common_index)

    ndx_returns = ndx.pct_change().dropna()
    if ndx_returns.empty:
        raise RuntimeError("Not enough data points to solve for daily drag")

    first_price = float(tqqq.iloc[0])
    target_price = float(tqqq.iloc[-1])
    solved_daily_drag = solve_constant_drag(ndx_returns, first_price, target_price)
    annual_equivalent_drag = 1.0 - (1.0 - solved_daily_drag) ** 252

    print(f"Range used: {common_index[0].date()} -> {common_index[-1].date()}")
    print(f"Solved daily drag: {solved_daily_drag:.10%}")
    print(f"Annual equivalent drag: {annual_equivalent_drag:.10%}")


if __name__ == "__main__":
    main()