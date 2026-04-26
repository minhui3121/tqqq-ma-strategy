"""Data loading utilities for the TQQQ backtesting framework."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf


def download_tqqq_data(
	start_date: str | datetime,
	end_date: str | datetime | None,
	ticker: str = "TQQQ",
) -> pd.DataFrame:
	"""Download daily historical data for TQQQ.

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
