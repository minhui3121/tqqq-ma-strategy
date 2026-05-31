"""Calculate TQQQ drag year by year from 2010 through 2025.

This script is self-contained. It downloads ^NDX and TQQQ from Yahoo Finance,
then solves a separate constant daily drag for each calendar year using the same
rebuild model:

    price_n = price_{n-1} * (1 + 3 * ndx_return_n - g)

It writes a CSV with one row per year containing the solved daily drag and the
annual drag implied by that daily value.
"""

from __future__ import annotations

import argparse
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "yearly_tqqq_fee_summary.csv"


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


def solve_yearly_drag(ndx: pd.Series, tqqq: pd.Series, year: int) -> dict[str, object]:
    """Solve the daily drag for a single calendar year."""

    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)

    year_ndx = ndx.loc[(ndx.index >= year_start) & (ndx.index <= year_end)]
    year_tqqq = tqqq.loc[(tqqq.index >= year_start) & (tqqq.index <= year_end)]

    common_index = year_ndx.index.intersection(year_tqqq.index)
    if common_index.empty:
        raise RuntimeError(f"No overlapping trading dates found for {year}")

    year_ndx = year_ndx.reindex(common_index)
    year_tqqq = year_tqqq.reindex(common_index)

    ndx_returns = year_ndx.pct_change().dropna()
    if ndx_returns.empty:
        raise RuntimeError(f"Not enough data points to solve for daily drag in {year}")

    start_price = float(year_tqqq.iloc[0])
    end_price = float(year_tqqq.iloc[-1])
    solved_daily_drag = solve_constant_drag(ndx_returns, start_price, end_price)
    annual_drag = 1.0 - (1.0 - solved_daily_drag) ** 252

    return {
        "year": year,
        "start_date": common_index[0].date(),
        "end_date": common_index[-1].date(),
        "daily_drag": solved_daily_drag,
        "daily_drag_pct": solved_daily_drag * 100.0,
        "annual_drag": annual_drag,
        "annual_drag_pct": annual_drag * 100.0,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build yearly TQQQ drag summary")
    parser.add_argument("--start-year", type=int, default=2010, help="First year to include")
    parser.add_argument("--end-year", type=int, default=2025, help="Last year to include")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must be less than or equal to end-year")

    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"

    ndx = download_close("^NDX", start_date, end_date)
    tqqq = download_close("TQQQ", start_date, end_date)

    rows = []
    for year in range(args.start_year, args.end_year + 1):
        rows.append(solve_yearly_drag(ndx, tqqq, year))

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)

    print(f"Wrote yearly fee summary to: {OUT_CSV}")
    print(result[["year", "daily_drag", "annual_drag"]].to_string(index=False))


if __name__ == "__main__":
    main()