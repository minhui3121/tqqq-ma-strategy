"""Data loading utilities for the TQQQ backtesting framework."""

from __future__ import annotations

from datetime import datetime, timedelta
import os

import pandas as pd
import yfinance as yf


def _normalize_and_slice_prices(
	data: pd.DataFrame,
	start_date: str | datetime,
	end_date: str | datetime | None,
) -> pd.DataFrame:
	"""Sort ascending and slice to the requested inclusive date range."""

	cleaned = data.sort_index(ascending=True).copy()
	cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]

	start_ts = pd.to_datetime(start_date)
	if end_date is not None:
		end_ts = pd.to_datetime(end_date)
		cleaned = cleaned.loc[(cleaned.index >= start_ts) & (cleaned.index <= end_ts)]
	else:
		cleaned = cleaned.loc[cleaned.index >= start_ts]

	return cleaned


def download_single_ticker(
	ticker: str,
	start_date: str | datetime,
	end_date: str | datetime | None,
	price_column: str = "Close",
) -> pd.DataFrame:
	"""Download daily historical data for a single ticker."""

	# Prefer local CSVs if available. This keeps downstream scripts offline and
	# consistent with previously prepared data.
	local_path = os.path.join("data", f"{ticker.lower()}_data.csv")
	if os.path.exists(local_path):
		df = pd.read_csv(local_path, parse_dates=["Date"]) 
		if "Date" in df.columns:
			df = df.set_index("Date")

		if price_column not in df.columns:
			raise ValueError(f"Missing required columns from {local_path}: {price_column}")

		cleaned = _normalize_and_slice_prices(df, start_date, end_date)
		cleaned = cleaned.dropna(subset=[price_column])
		return cleaned

	# Fall back to yfinance if no local CSV found
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

	cleaned = _normalize_and_slice_prices(data, start_date, end_date)
	cleaned = cleaned.dropna(subset=[price_column])

	return cleaned


def download_qqq_and_tqqq_data(
	start_date: str | datetime,
	end_date: str | datetime | None,
	short_window: int = 80,
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
		Short SMA window used to compute ``sma80``.
	long_window:
		Long SMA window used to compute ``sma190``.
	warmup_days:
		Initial calendar-day lookback before start_date. If insufficient to warm
		``long_window`` before the first TQQQ date, the function automatically
		extends QQQ history further back.

	Returns
	-------
	pandas.DataFrame
		Merged daily data with QQQ_Close, TQQQ_Open, TQQQ_Close, sma80, sma190
		columns. SMAs are calculated on full QQQ history. In real-only mode the
		index starts from the earliest available TQQQ date; in synthetic mode it
		starts from the user-requested start date.
	"""
	
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

	# Load TQQQ first to identify the earliest executable trade date.
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
	data["sma80"] = data["QQQ_Close"].rolling(window=short_window, min_periods=short_window).mean()
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


def generate_annual_deposits(
	data: pd.DataFrame,
	deposit_amount: float = 10_000.0,
	skip_first_year: bool = True,
) -> dict[pd.Timestamp, float]:
	"""Generate annual deposit dates (Jan 1) for continuous investment backtests.

	Parameters
	----------
	data:
		Backtest data frame to extract date range from.
	deposit_amount:
		Amount to deposit on each date.
	skip_first_year:
		If True, skip Jan 1 of the first year (since initial capital covers it).

	Returns
	-------
	dict
		Mapping of deposit dates to amounts.
	"""

	if data.empty:
		return {}

	start_year = data.index.min().year
	end_year = data.index.max().year
	deposits = {}

	for year in range(start_year, end_year + 1):
		if skip_first_year and year == start_year:
			continue

		jan1 = pd.Timestamp(f"{year}-01-01")
		if jan1 in data.index:
			deposits[jan1] = deposit_amount
		else:
			candidates = data.index[data.index.year == year]
			if not candidates.empty:
				first_date = candidates.min()
				deposits[first_date] = deposit_amount

	return deposits


