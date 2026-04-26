"""Data loading utilities for the TQQQ backtesting framework."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf


def download_single_ticker(
	ticker: str,
	start_date: str | datetime,
	end_date: str | datetime | None,
	price_column: str = "Close",
) -> pd.DataFrame:
	"""Download daily historical data for a single ticker."""

	data = yf.download(
		ticker,
		start=start_date,
		end=end_date,
		interval="1d",
		auto_adjust=False,
		progress=False,
	)

	if data.empty:
		raise ValueError(f"No data returned from yfinance for {ticker}.")

	if isinstance(data.columns, pd.MultiIndex):
		data.columns = data.columns.get_level_values(0)

	required_columns = {price_column}
	missing_columns = required_columns.difference(data.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns from {ticker}: {sorted(missing_columns)}")

	cleaned = data.sort_index().copy()
	cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]
	cleaned = cleaned.dropna(subset=[price_column])

	return cleaned


def download_qqq_and_tqqq_data(
	start_date: str | datetime,
	end_date: str | datetime | None,
	short_window: int = 100,
	long_window: int = 190,
	warmup_days: int = 250,
) -> pd.DataFrame:
	"""Download QQQ and TQQQ data with pre-warmed QQQ moving averages.

	QQQ prices are used for signal generation (with SMAs from full history).
	TQQQ prices are used for portfolio execution.
	
	Strategy: SMAs are calculated on QQQ going back as far as possible, so that
	on the day TQQQ starts trading (Feb 11, 2010), the SMAs are already "warm" 
	and reflect months of QQQ history. Portfolio execution begins when TQQQ data 
	becomes available.

	Parameters
	----------
	start_date:
		Requested start date for backtest. Data is downloaded from earlier.
	end_date:
		Last date to include.
	short_window:
		Short SMA window used to compute ``sma100``.
	long_window:
		Long SMA window used to compute ``sma190``.
	warmup_days:
		Initial calendar-day lookback before start_date. If insufficient to warm
		``long_window`` before the first TQQQ date, the function automatically
		extends QQQ history further back.

	Returns
	-------
	pandas.DataFrame
		Merged daily data with QQQ_Close, TQQQ_Close, sma100, sma190 columns.
		SMAs calculated on full QQQ history.
		Index starts from the user-requested start_date (or earliest TQQQ date if later).
	"""

	from datetime import timedelta
	
	if isinstance(start_date, str):
		start_date_dt = pd.to_datetime(start_date)
	else:
		start_date_dt = start_date

	if short_window <= 0 or long_window <= 0:
		raise ValueError("SMA windows must be positive integers.")
	if short_window >= long_window:
		raise ValueError("short_window must be smaller than long_window.")
	if warmup_days <= 0:
		raise ValueError("warmup_days must be a positive integer.")
	
	extended_start = start_date_dt - timedelta(days=warmup_days)

	# Download TQQQ first to identify the earliest executable trade date.
	tqqq = download_single_ticker("TQQQ", extended_start, end_date, price_column="Close")
	if tqqq.empty:
		raise ValueError("No TQQQ data available.")

	first_tqqq_date = tqqq.index.min()

	# Download QQQ history and ensure we have enough observations before TQQQ starts
	# so the long SMA is already valid on day one of executable trading.
	qqq = download_single_ticker("QQQ", extended_start, end_date, price_column="Close")
	qqq_obs_before_tqqq = int((qqq.index < first_tqqq_date).sum())
	if qqq_obs_before_tqqq < long_window:
		missing_obs = long_window - qqq_obs_before_tqqq
		extra_days = int(missing_obs * 2.0) + 30
		qqq_start = extended_start - timedelta(days=extra_days)
		qqq = download_single_ticker("QQQ", qqq_start, end_date, price_column="Close")

	# Create DataFrame with full QQQ history
	data = pd.DataFrame({"QQQ_Close": qqq["Close"]})
	
	# Add TQQQ where it exists (from Feb 11, 2010 onwards)
	if "Open" not in tqqq.columns:
		raise ValueError("Missing required columns from TQQQ: ['Open']")
	data["TQQQ_Open"] = tqqq["Open"]
	data["TQQQ_Close"] = tqqq["Close"]
	
	# Calculate SMAs on full QQQ history prior to trimming to executable dates.
	data["sma100"] = data["QQQ_Close"].rolling(window=short_window, min_periods=short_window).mean()
	data["sma190"] = data["QQQ_Close"].rolling(window=long_window, min_periods=long_window).mean()
	
	# Filter to dates where TQQQ exists (this is where trading can actually happen)
	data = data.dropna(subset=["TQQQ_Open", "TQQQ_Close"])
	data = data.loc[data.index >= start_date_dt]
	
	if data.empty:
		raise ValueError(f"No data available on or after {start_date_dt.date()}.")
	
	earliest_tqqq = data.index.min()
	if earliest_tqqq > start_date_dt:
		import warnings
		warnings.warn(
			f"Requested start date ({start_date_dt.date()}) is before TQQQ trading began ({earliest_tqqq.date()}). "
			f"Backtest will start from {earliest_tqqq.date()} with pre-warmed SMAs.",
			UserWarning
		)

	return data


def download_tqqq_data(
	start_date: str | datetime,
	end_date: str | datetime | None,
	ticker: str = "TQQQ",
) -> pd.DataFrame:
	"""Download daily historical data for TQQQ.

	DEPRECATED: Use download_qqq_and_tqqq_data() instead.

	Parameters
	----------
	start_date:
		First date to include.
	end_date:
		Last date to include.
	ticker:
		Yahoo Finance ticker symbol. Defaults to TQQQ.

	Returns
	-------
	pandas.DataFrame
		Cleaned daily price data indexed by date with a ``Close`` column.
	"""

	data = yf.download(
		ticker,
		start=start_date,
		end=end_date,
		interval="1d",
		auto_adjust=False,
		progress=False,
	)

	if data.empty:
		raise ValueError("No data returned from yfinance for the requested date range.")

	if isinstance(data.columns, pd.MultiIndex):
		data.columns = data.columns.get_level_values(0)

	required_columns = {"Close"}
	missing_columns = required_columns.difference(data.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns from downloaded data: {sorted(missing_columns)}")

	cleaned = data.sort_index().copy()
	cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]
	cleaned = cleaned.dropna(subset=["Close"])

	return cleaned


def prepare_price_series(data: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
	"""Validate and clean downloaded data for downstream analysis."""

	if price_column not in data.columns:
		raise ValueError(f"Expected price column '{price_column}' was not found.")

	prepared = data.copy().sort_index()
	prepared = prepared.loc[~prepared.index.duplicated(keep="first")]
	prepared = prepared.dropna(subset=[price_column])
	return prepared


