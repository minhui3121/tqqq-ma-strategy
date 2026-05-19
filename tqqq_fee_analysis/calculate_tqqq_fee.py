"""Calculate TQQQ management fee from actual data.

This script compares actual TQQQ prices with theoretical 3x leveraged QQQ
from 2010 onwards to derive the effective management fee.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    return df


def calculate_fee_from_data() -> None:
    """Calculate TQQQ management fee from actual data."""
    print("Downloading QQQ and TQQQ data from 2010-02-11 onwards...")
    
    # Download both datasets
    qqq = yf.download("QQQ", start="2010-02-10", end="2025-12-31", progress=False, auto_adjust=False)
    tqqq = yf.download("TQQQ", start="2010-02-10", end="2025-12-31", progress=False, auto_adjust=False)
    
    # Flatten columns
    qqq = flatten_columns(qqq)
    tqqq = flatten_columns(tqqq)
    
    # Reset index to get Date as column
    qqq = qqq.reset_index()
    tqqq = tqqq.reset_index()
    
    # Ensure proper date types
    qqq["Date"] = pd.to_datetime(qqq["Date"])
    tqqq["Date"] = pd.to_datetime(tqqq["Date"])
    
    # Ensure Close column exists
    if "Close" not in qqq.columns:
        raise KeyError("QQQ data missing 'Close' column")
    if "Close" not in tqqq.columns:
        raise KeyError("TQQQ data missing 'Close' column")
    
    # Merge on date
    merged = pd.merge(
        qqq[["Date", "Close"]],
        tqqq[["Date", "Close"]],
        on="Date",
        how="inner",
        suffixes=("_QQQ", "_TQQQ")
    )
    
    print(f"Merged data: {len(merged)} common trading days from {merged['Date'].min().date()} to {merged['Date'].max().date()}")
    
    # Calculate returns
    merged["QQQ_Return"] = merged["Close_QQQ"].pct_change()
    merged["TQQQ_Return"] = merged["Close_TQQQ"].pct_change()
    merged["Theoretical_3x_Return"] = 3.0 * merged["QQQ_Return"]
    
    # Calculate daily fee: the difference between theoretical and actual return
    # Actual Return = Theoretical Return - Daily Fee
    # Daily Fee = Theoretical Return - Actual Return
    merged["Daily_Fee"] = merged["Theoretical_3x_Return"] - merged["TQQQ_Return"]
    
    # Remove first row (NaN due to pct_change) and any NaN rows
    merged = merged.dropna()
    
    print(f"\nAnalyzing {len(merged)} returns...")
    
    # Calculate statistics
    mean_daily_fee = merged["Daily_Fee"].mean()
    median_daily_fee = merged["Daily_Fee"].median()
    std_daily_fee = merged["Daily_Fee"].std()
    
    # Convert to annual fee (assuming 252 trading days)
    annual_fee_from_mean = mean_daily_fee * 252
    annual_fee_from_median = median_daily_fee * 252
    
    print("\nDaily Fee Statistics:")
    print(f"  Mean daily fee: {mean_daily_fee:.6f} ({mean_daily_fee*100:.4f}%)")
    print(f"  Median daily fee: {median_daily_fee:.6f} ({median_daily_fee*100:.4f}%)")
    print(f"  Std dev: {std_daily_fee:.6f}")
    print(f"  Min: {merged['Daily_Fee'].min():.6f}")
    print(f"  Max: {merged['Daily_Fee'].max():.6f}")
    
    print("\nAnnualized Fee (252 trading days):")
    print(f"  From mean daily fee: {annual_fee_from_mean:.4f} ({annual_fee_from_mean*100:.2f}%)")
    print(f"  From median daily fee: {annual_fee_from_median:.4f} ({annual_fee_from_median*100:.2f}%)")
    
    # Alternative: use cumulative approach
    # Compare total return over the whole period
    total_qqq_return = (merged["Close_QQQ"].iloc[-1] / merged["Close_QQQ"].iloc[0]) - 1
    total_tqqq_return = (merged["Close_TQQQ"].iloc[-1] / merged["Close_TQQQ"].iloc[0]) - 1
    
    # If there were no fees, TQQQ should return 3x QQQ return
    expected_tqqq_return = (1 + total_qqq_return) ** 3 - 1
    
    print(f"\nCumulative Return Analysis (Period: {merged['Date'].min().date()} to {merged['Date'].max().date()}):")
    print(f"  QQQ total return: {total_qqq_return:.2%}")
    print(f"  TQQQ total return: {total_tqqq_return:.2%}")
    print(f"  Expected TQQQ (3x QQQ, no fees): {expected_tqqq_return:.2%}")
    print(f"  Difference (fee impact): {expected_tqqq_return - total_tqqq_return:.2%}")
    
    # Estimate annual fee from total period
    years = (merged["Date"].iloc[-1] - merged["Date"].iloc[0]).days / 365.25
    
    # Use geometric mean of daily fees for better accuracy
    merged["Return_Factor"] = 1.0 - merged["Daily_Fee"]
    annual_fee_from_geometric = 1.0 - (merged["Return_Factor"].prod() ** (252 / len(merged)))
    
    print(f"\nAnnualized Fee (from geometric mean): {annual_fee_from_geometric:.4f} ({annual_fee_from_geometric*100:.2f}%)")
    # Export detailed calculation to CSV for inspection
    import os
    output_csv = "tqqq_fee_analysis/tqqq_fee_calculation.csv"
    os.makedirs("tqqq_fee_analysis", exist_ok=True)
    
    export_df = merged[["Date", "Close_QQQ", "Close_TQQQ", "QQQ_Return", 
                        "TQQQ_Return", "Theoretical_3x_Return", "Daily_Fee"]].copy()
    export_df.columns = ["Date", "QQQ_Close", "TQQQ_Close", "QQQ_Return", 
                         "TQQQ_Return", "Theoretical_3x_Return", "Daily_Fee"]
    
    # Add percentage columns for readability
    export_df["QQQ_Return_%"] = export_df["QQQ_Return"] * 100
    export_df["TQQQ_Return_%"] = export_df["TQQQ_Return"] * 100
    export_df["Theoretical_3x_Return_%"] = export_df["Theoretical_3x_Return"] * 100
    export_df["Daily_Fee_%"] = export_df["Daily_Fee"] * 100
    
    export_df.to_csv(output_csv, index=False)
    print(f"\nDetailed calculation exported to: {output_csv}")
    print(f"  Showing {len(export_df)} rows with daily breakdown")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print(f"Use TQQQ_ANNUAL_FEE = {annual_fee_from_median:.6f}")
    print(f"(This is the median daily fee * 252, approximately {annual_fee_from_median*100:.2f}% annually)")


if __name__ == "__main__":
    calculate_fee_from_data()
