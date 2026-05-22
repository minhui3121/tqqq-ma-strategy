"""Rebuild TQQQ using Nasdaq-100 returns and a constant daily drag.

This minimal script:
- loads `^NDX` and `TQQQ` close prices (prefers local CSVs in `data/`)
- computes the constant daily drag `g` so that rebuilding with
  price_n = price_{n-1} * (1 + 3*ndx_return_n - g) matches actual terminal TQQQ
- reconstructs the full per-day rebuilt price series and exports a CSV

Output: `tqqq_fee_analysis/rebuild_tqqq.csv` with one row per trading day.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = Path(__file__).resolve().parent / "rebuild_tqqq.csv"
START_DATE = "2010-02-11"
END_DATE = "2026-01-01"


def load_close(ticker: str, local_csv: Path | None = None) -> pd.Series:
    """Load raw Close series, prefer local CSV if it covers the requested range."""
    if local_csv is not None and local_csv.exists():
        df = pd.read_csv(local_csv, parse_dates=["Date"]).set_index("Date").sort_index()
        if "Close" in df.columns:
            requested_start = pd.to_datetime(START_DATE)
            requested_end = pd.to_datetime(END_DATE) - pd.Timedelta(days=1)
            if df.index.min() <= requested_start and df.index.max() >= requested_end:
                return df["Close"].loc[requested_start:requested_end]

    data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data["Close"].sort_index()


def solve_constant_drag(ndx_returns: pd.Series, first_price: float, target_price: float) -> float:
    """Solve for constant daily drag g so product((1 + 3*r_i - g)) * first_price == target_price.

    Uses bisection on plausible interval and computes products in log-space.
    """
    r = ndx_returns.to_numpy(copy=True)
    # valid factors must be positive: 1 + 3*r - g > 0
    tiny = 1e-12
    upper = float((1.0 + 3.0 * r).min() - tiny)
    lower = -0.999

    # operate in log-space: we want sum(log(1+3*r - g)) == log(target_price/first_price)
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

    # If both same sign, fall back to ratio-based compound approximation
    if f_lo * f_hi > 0:
        # approximate by ratio of total product
        total_log_theoretical = float(np.log(1.0 + 3.0 * r).sum())
        ratio = math.exp(log_target - total_log_theoretical)
        n = len(r)
        return 1.0 - ratio ** (1.0 / n)

    lo = lower
    hi = upper
    flo = f_lo
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


def rebuild_series(dates: pd.DatetimeIndex, ndx: pd.Series, tqqq: pd.Series, g: float) -> pd.DataFrame:
    """Rebuild TQQQ sequentially using constant daily drag g."""
    ndx = ndx.reindex(dates)
    tqqq = tqqq.reindex(dates)

    ndx_return = ndx.pct_change()

    rows = []
    prev = float(tqqq.iloc[0])
    for i, date in enumerate(dates):
        r = ndx_return.iloc[i] if i < len(ndx_return) else float('nan')
        if pd.isna(r):
            factor = float('nan')
            rebuilt = prev
        else:
            factor = 1.0 + 3.0 * r - g
            rebuilt = prev * factor
            prev = rebuilt

        rows.append({
            "date": date,
            "ndx_close": ndx.iloc[i],
            "ndx_return": r,
            "factor": factor,
            "rebuilt_tqqq": rebuilt,
            "actual_tqqq_close": tqqq.iloc[i],
            "daily_drag": g,
            "diff_rebuilt_minus_actual": rebuilt - tqqq.iloc[i] if not pd.isna(tqqq.iloc[i]) else float('nan'),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ndx = load_close("^NDX", local_csv=ROOT / "data" / "ndx_data.csv")
    tqqq = load_close("TQQQ", local_csv=ROOT / "data" / "tqqq_data.csv")

    # Align on common dates (where both are present)
    common_index = ndx.index.intersection(tqqq.index)
    common_index = common_index[common_index >= pd.to_datetime(START_DATE)]
    ndx = ndx.reindex(common_index)
    tqqq = tqqq.reindex(common_index)

    # Use returns from day 1..n (drop first NaN for pct_change)
    ndx_returns = ndx.pct_change().dropna()

    first_price = float(tqqq.iloc[0])
    target_price = float(tqqq.iloc[-1])

    g = solve_constant_drag(ndx_returns, first_price, target_price)

    df = rebuild_series(common_index, ndx, tqqq, g)
    df.to_csv(OUT_CSV, index=False)

    print(f"Wrote rebuild CSV to: {OUT_CSV}")
    print(f"Solved constant daily drag: {g*100:.6f}%")


if __name__ == "__main__":
    main()
