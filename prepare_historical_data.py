"""Download and store historical QQQ and TQQQ data with synthetic pre-TQQQ prices.

This script:
1. Downloads QQQ data from its earliest available date to 2025-12-31
2. Downloads TQQQ data from its inception (March 2010) to 2025-12-31
3. Synthesizes TQQQ prices for dates before TQQQ existed, accounting for management fees
4. Stores both datasets as CSV files for reuse
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import datetime

# TQQQ management fee (daily) - empirically derived from actual QQQ/TQQQ overlap
TQQQ_DAILY_FEE = 0.000200
# TQQQ leverage ratio
TQQQ_LEVERAGE = 3.0
# TQQQ inception date (first trading date)
TQQQ_INCEPTION = "2010-02-11"

OUTPUT_DIR = "data"
QQQ_COLUMN_ORDER = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
TQQQ_COLUMN_ORDER = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    return df


def download_qqq_data() -> pd.DataFrame:
    """Download QQQ data from earliest available to 2025-12-31."""
    print("Downloading QQQ data from earliest available date to 2025-12-31...")
    qqq = yf.download("QQQ", start="1995-01-01", end="2025-12-31", progress=False, auto_adjust=False)
    qqq = flatten_columns(qqq)
    qqq = qqq.reset_index()
    qqq["Date"] = pd.to_datetime(qqq["Date"])
    qqq = qqq.sort_values("Date").reset_index(drop=True)
    # Add Adj Close if not present
    if "Adj Close" not in qqq.columns:
        qqq["Adj Close"] = qqq["Close"]
    print(f"QQQ data: {len(qqq)} rows from {qqq['Date'].min().date()} to {qqq['Date'].max().date()}")
    return qqq


def download_tqqq_data() -> pd.DataFrame:
    """Download actual TQQQ data from inception to 2025-12-31."""
    print("Downloading TQQQ data from 2010-02-11 to 2025-12-31...")
    tqqq = yf.download("TQQQ", start="2010-02-11", end="2025-12-31", progress=False, auto_adjust=False)
    tqqq = flatten_columns(tqqq)
    tqqq = tqqq.reset_index()
    tqqq["Date"] = pd.to_datetime(tqqq["Date"])
    tqqq = tqqq.sort_values("Date").reset_index(drop=True)
    # Add Adj Close if not present
    if "Adj Close" not in tqqq.columns:
        tqqq["Adj Close"] = tqqq["Close"]
    print(f"TQQQ data: {len(tqqq)} rows from {tqqq['Date'].min().date()} to {tqqq['Date'].max().date()}")
    return tqqq


def synthesize_tqqq_synthetic(qqq_data: pd.DataFrame, tqqq_actual: pd.DataFrame) -> pd.DataFrame:
    """Synthesize TQQQ prices for dates before TQQQ existed, accounting for fees.
    
    For each day, the synthetic TQQQ return is:
    - 3x the QQQ return (leveraged)
    - Minus the daily fee component
    
    Daily fee = TQQQ_DAILY_FEE
    """
    # Anchor synthetic series to the first real TQQQ close and work backwards so
    # the day before inception flows into the real inception price smoothly.
    tqqq_start_date = tqqq_actual["Date"].min()
    print(f"Synthesizing TQQQ data from {qqq_data['Date'].min().date()} to {tqqq_start_date.date()}...")

    # Compute QQQ returns using raw Close (user requested Close, not Adj Close)
    qqq = qqq_data.sort_values("Date").reset_index(drop=True).copy()
    qqq["QQQ_Return"] = qqq["Close"].pct_change()

    # Subset of QQQ dates before inception
    pre_tqqq = qqq[qqq["Date"] < tqqq_start_date].copy()

    # Get the first actual TQQQ close to anchor the synthetic series
    tqqq_first_close = tqqq_actual.loc[tqqq_actual["Date"] == tqqq_start_date, "Close"].iloc[0]

    daily_fee_rate = TQQQ_DAILY_FEE

    # Work backwards: start at the first actual close and invert the per-day
    # leveraged returns so that the synthetic value on the last pre-inception
    # date, when advanced, equals the actual inception close.
    synthetic_prices = {}
    current_price = float(tqqq_first_close)

    # We'll iterate over pre_tqqq indices in reverse (most recent pre-inception first)
    pre_idx = pre_tqqq.index.tolist()
    pre_idx.sort(reverse=True)

    for idx in pre_idx:
        # The QQQ return that moves from this date to the next date is at qqq['QQQ_Return'].loc[idx+1]
        next_idx = idx + 1
        if next_idx in qqq.index and pd.notna(qqq.at[next_idx, "QQQ_Return"]):
            r = qqq.at[next_idx, "QQQ_Return"]
            leveraged_return = TQQQ_LEVERAGE * r - daily_fee_rate
            # Invert forward multiplication to get the previous day's price
            prev_price = current_price / (1 + leveraged_return) if (1 + leveraged_return) != 0 else current_price
        else:
            # If we don't have a return (edge case), carry price backwards unchanged
            prev_price = current_price

        # Ensure price stays positive and record
        prev_price = max(prev_price, 0.01)
        synthetic_prices[idx] = prev_price
        current_price = prev_price

    # Build DataFrame from synthetic_prices, aligning with dates
    synthetic_list = []
    for idx in sorted(pre_tqqq.index):
        synthetic_list.append({
            "Date": pre_tqqq.at[idx, "Date"],
            "Close": synthetic_prices[idx],
        })

    synthetic_df = pd.DataFrame(synthetic_list)
    synthetic_df["Open"] = synthetic_df["Close"]  # Approximate open as close
    synthetic_df["High"] = None
    synthetic_df["Low"] = None
    synthetic_df["Volume"] = 0  # Mark as synthetic
    synthetic_df["Adj Close"] = None

    print(f"Synthesized {len(synthetic_df)} rows of TQQQ data")
    return synthetic_df


def combine_and_store(
    qqq_data: pd.DataFrame,
    tqqq_actual: pd.DataFrame,
    tqqq_synthetic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine actual and synthetic TQQQ data, then store both datasets."""
    import os
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clean up TQQQ actual data to have consistent columns
    tqqq_to_use = tqqq_actual[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    
    # Combine synthetic and actual TQQQ (synthetic first, then actual)
    tqqq_combined = pd.concat([tqqq_synthetic, tqqq_to_use], ignore_index=True)
    tqqq_combined = tqqq_combined.sort_values("Date").reset_index(drop=True)
    
    # Remove duplicates (keep actual if date overlaps)
    tqqq_combined = tqqq_combined.drop_duplicates(subset=["Date"], keep="last")
    
    # Store QQQ data
    qqq_data = qqq_data[QQQ_COLUMN_ORDER].copy()
    qqq_output = f"{OUTPUT_DIR}/qqq_data.csv"
    qqq_data.to_csv(qqq_output, index=False)
    print(f"\nStored QQQ data: {qqq_output}")
    print(f"  Rows: {len(qqq_data)}")
    print(f"  Date range: {qqq_data['Date'].min().date()} to {qqq_data['Date'].max().date()}")
    
    # Store TQQQ combined data
    tqqq_combined = tqqq_combined[TQQQ_COLUMN_ORDER].copy()
    tqqq_output = f"{OUTPUT_DIR}/tqqq_data.csv"
    tqqq_combined.to_csv(tqqq_output, index=False)
    print(f"\nStored TQQQ data (synthetic + actual): {tqqq_output}")
    print(f"  Rows: {len(tqqq_combined)}")
    print(f"  Date range: {tqqq_combined['Date'].min().date()} to {tqqq_combined['Date'].max().date()}")
    print(f"  Synthetic rows: {(tqqq_combined['Volume'] == 0).sum()}")
    print(f"  Actual rows: {(tqqq_combined['Volume'] > 0).sum()}")
    
    return qqq_data, tqqq_combined


def main() -> None:
    """Main execution: download, synthesize, and store data."""
    print("="*70)
    print("Historical Data Preparation")
    print("="*70)
    print()
    
    # Download datasets
    qqq_data = download_qqq_data()
    tqqq_actual = download_tqqq_data()
    
    # Synthesize TQQQ for pre-inception period (anchor using actual TQQQ data)
    tqqq_synthetic = synthesize_tqqq_synthetic(qqq_data, tqqq_actual)
    
    # Combine and store
    qqq_stored, tqqq_stored = combine_and_store(qqq_data, tqqq_actual, tqqq_synthetic)
    
    print()
    print("="*70)
    print("Data preparation complete!")
    print("="*70)
    print()
    print("Summary:")
    print(f"  QQQ: {len(qqq_stored)} trading days")
    print(f"  TQQQ (with synthesis): {len(tqqq_stored)} trading days")
    print(f"  TQQQ analysis period: {tqqq_stored['Date'].min().date()} to {tqqq_stored['Date'].max().date()}")
    print()


if __name__ == "__main__":
    main()
