"""Deep-dive TQQQ fee analysis using Nasdaq-100 returns.

This script reconstructs a theoretical TQQQ path from the first available
TQQQ close on 2010-02-11 through 2025-12-31 by applying 3x the Nasdaq-100
close-to-close return each day, then compares the result to actual TQQQ close
prices to estimate total, daily, and annual drag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = Path(__file__).resolve().parent / "tqqq_fee_calculation.csv"
START_DATE = "2010-02-11"
END_DATE = "2026-01-01"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class FeeSummary:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    rows: int
    actual_start_close: float
    actual_end_close: float
    theoretical_end_close: float
    total_drag_absolute: float
    total_drag_ratio: float
    implied_daily_drag: float
    implied_annual_drag: float


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance."""

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def load_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
    local_csv: Path | None = None,
) -> pd.DataFrame:
    """Load raw close history from a local CSV if present, otherwise yfinance."""

    if local_csv is not None and local_csv.exists():
        df = pd.read_csv(local_csv, parse_dates=["Date"])
        if "Date" not in df.columns:
            raise ValueError(f"Missing Date column in {local_csv}")
        if "Close" not in df.columns:
            raise ValueError(f"Missing Close column in {local_csv}")
        df = df.set_index("Date").sort_index()

        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        covers_range = df.index.min() <= requested_start and df.index.max() >= (requested_end - pd.Timedelta(days=1))

        if covers_range:
            return df.loc[(df.index >= requested_start) & (df.index < requested_end)]

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for {ticker}.")

    data = flatten_columns(data)
    if "Close" not in data.columns:
        raise ValueError(f"Missing Close column for {ticker}.")

    data = data.sort_index()
    data = data.loc[(data.index >= pd.to_datetime(start_date)) & (data.index < pd.to_datetime(end_date))]
    return data


def build_fee_analysis() -> tuple[pd.DataFrame, FeeSummary]:
    """Build the aligned NDX/TQQQ fee analysis table."""

    print("Loading Nasdaq-100 (^NDX) and TQQQ close data...")

    tqqq = load_price_history(
        "TQQQ",
        START_DATE,
        END_DATE,
        local_csv=ROOT / "data" / "tqqq_data.csv",
    )
    ndx = load_price_history(
        "^NDX",
        START_DATE,
        END_DATE,
        local_csv=ROOT / "data" / "ndx_data.csv",
    )

    tqqq = tqqq[["Close"]].rename(columns={"Close": "tqqq_close"})
    ndx = ndx[["Close"]].rename(columns={"Close": "ndx_close"})

    merged = tqqq.join(ndx, how="inner")
    merged = merged.sort_index()

    if merged.empty:
        raise ValueError("No overlapping TQQQ and ^NDX trading days were found.")

    merged = merged.loc[merged.index >= pd.Timestamp(START_DATE)]

    merged["ndx_return"] = merged["ndx_close"].pct_change()
    merged["tqqq_return"] = merged["tqqq_close"].pct_change()
    merged["theoretical_tqqq_return"] = merged["ndx_return"] * 3.0
    merged["return_gap"] = merged["theoretical_tqqq_return"] - merged["tqqq_return"]

    merged["theoretical_growth_factor"] = (1.0 + merged["theoretical_tqqq_return"]).cumprod()
    merged["actual_growth_factor"] = (1.0 + merged["tqqq_return"]).cumprod()

    first_actual_close = float(merged["tqqq_close"].iloc[0])
    merged["theoretical_tqqq_value"] = first_actual_close * merged["theoretical_growth_factor"].fillna(1.0)
    merged["actual_tqqq_value_from_start"] = first_actual_close * merged["actual_growth_factor"].fillna(1.0)
    merged["value_gap"] = merged["theoretical_tqqq_value"] - merged["tqqq_close"]
    merged["drag_ratio"] = 1.0 - (merged["tqqq_close"] / merged["theoretical_tqqq_value"])

    analysis = merged.reset_index()
    if "Date" in analysis.columns:
        analysis = analysis.rename(columns={"Date": "date"})
    elif "index" in analysis.columns:
        analysis = analysis.rename(columns={"index": "date"})
    else:
        first_column = analysis.columns[0]
        analysis = analysis.rename(columns={first_column: "date"})

    analysis["date"] = pd.to_datetime(analysis["date"])

    analysis["ndx_return_pct"] = analysis["ndx_return"] * 100.0
    analysis["tqqq_return_pct"] = analysis["tqqq_return"] * 100.0
    analysis["theoretical_tqqq_return_pct"] = analysis["theoretical_tqqq_return"] * 100.0
    analysis["return_gap_pct"] = analysis["return_gap"] * 100.0
    analysis["drag_ratio_pct"] = analysis["drag_ratio"] * 100.0

    valid_rows = analysis.dropna(subset=["ndx_return", "tqqq_return"]).copy()
    if valid_rows.empty:
        raise ValueError("Not enough rows to compute return-based drag.")

    start_date = analysis["date"].iloc[0]
    end_date = analysis["date"].iloc[-1]
    actual_end_close = float(analysis["tqqq_close"].iloc[-1])
    theoretical_end_close = float(analysis["theoretical_tqqq_value"].iloc[-1])
    total_drag_absolute = theoretical_end_close - actual_end_close
    total_drag_ratio = 1.0 - (actual_end_close / theoretical_end_close)

    trading_steps = max(len(analysis) - 1, 1)
    implied_daily_drag = 1.0 - (actual_end_close / theoretical_end_close) ** (1.0 / trading_steps)
    implied_annual_drag = 1.0 - (actual_end_close / theoretical_end_close) ** (
        TRADING_DAYS_PER_YEAR / trading_steps
    )

    summary = FeeSummary(
        start_date=start_date,
        end_date=end_date,
        rows=len(analysis),
        actual_start_close=first_actual_close,
        actual_end_close=actual_end_close,
        theoretical_end_close=theoretical_end_close,
        total_drag_absolute=total_drag_absolute,
        total_drag_ratio=total_drag_ratio,
        implied_daily_drag=implied_daily_drag,
        implied_annual_drag=implied_annual_drag,
    )

    return analysis, summary


def export_analysis(analysis: pd.DataFrame) -> None:
    """Export the row-level analysis CSV."""

    export_columns = [
        "date",
        "ndx_close",
        "tqqq_close",
        "ndx_return",
        "tqqq_return",
        "theoretical_tqqq_return",
        "return_gap",
        "theoretical_growth_factor",
        "actual_growth_factor",
        "theoretical_tqqq_value",
        "actual_tqqq_value_from_start",
        "value_gap",
        "drag_ratio",
        "ndx_return_pct",
        "tqqq_return_pct",
        "theoretical_tqqq_return_pct",
        "return_gap_pct",
        "drag_ratio_pct",
    ]

    analysis.loc[:, export_columns].to_csv(OUTPUT_CSV, index=False)


def print_summary(summary: FeeSummary) -> None:
    """Print a concise terminal summary of the analysis."""

    print(f"Analyzed {summary.rows} aligned trading days from {summary.start_date.date()} to {summary.end_date.date()}")
    print(f"Starting actual TQQQ close: {summary.actual_start_close:.4f}")
    print(f"Ending actual TQQQ close: {summary.actual_end_close:.4f}")
    print(f"Ending theoretical TQQQ value: {summary.theoretical_end_close:.4f}")
    print(f"Total drag: {summary.total_drag_absolute:.4f} ({summary.total_drag_ratio * 100:.4f}%)")
    print(f"Implied daily drag: {summary.implied_daily_drag * 100:.6f}%")
    print(f"Implied annual drag: {summary.implied_annual_drag * 100:.4f}%")


def main() -> None:
    analysis, summary = build_fee_analysis()
    export_analysis(analysis)
    print_summary(summary)
    print(f"Detailed calculation exported to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
