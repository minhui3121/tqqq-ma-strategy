"""Data loading utilities for the TQQQ backtesting framework."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf


def download_single_ticker(
	ticker: str,
	start_date: str | datetime,
	end_date: str | datetime | None,
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

	required_columns = {"Adj Close"}
	missing_columns = required_columns.difference(data.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns from {ticker}: {sorted(missing_columns)}")

	cleaned = data.sort_index().copy()
	cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]
	cleaned = cleaned.dropna(subset=["Adj Close"])

	return cleaned


def download_qqq_and_tqqq_data(
	start_date: str | datetime,
	end_date: str | datetime | None,
) -> pd.DataFrame:
	"""Download QQQ and TQQQ data, merge on common dates.

	QQQ prices are used for signal generation.
	TQQQ prices are used for portfolio execution.

	Parameters
	----------
	start_date:
		First date to include.
	end_date:
		Last date to include.

	Returns
	-------
	pandas.DataFrame
		Merged daily data with QQQ_Close, TQQQ_Close columns.
	"""

	qqq = download_single_ticker("QQQ", start_date, end_date)
	tqqq = download_single_ticker("TQQQ", start_date, end_date)

	merged = pd.DataFrame({
		"QQQ_Close": qqq["Adj Close"],
		"TQQQ_Close": tqqq["Adj Close"],
	})

	merged = merged.dropna(how="any")

	if merged.empty:
		raise ValueError("No overlapping dates between QQQ and TQQQ data.")

	return merged


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
		Cleaned daily price data indexed by date with an ``Adj Close`` column.
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

	required_columns = {"Adj Close"}
	missing_columns = required_columns.difference(data.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns from downloaded data: {sorted(missing_columns)}")

	cleaned = data.sort_index().copy()
	cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]
	cleaned = cleaned.dropna(subset=["Adj Close"])

	return cleaned


def prepare_price_series(data: pd.DataFrame, price_column: str = "Adj Close") -> pd.DataFrame:
	"""Validate and clean downloaded data for downstream analysis."""

	if price_column not in data.columns:
		raise ValueError(f"Expected price column '{price_column}' was not found.")

	prepared = data.copy().sort_index()
	prepared = prepared.loc[~prepared.index.duplicated(keep="first")]
	prepared = prepared.dropna(subset=[price_column])
	return prepared
