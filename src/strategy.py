"""Strategy logic for a TQQQ moving-average trend strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
	"""Configuration for the moving-average strategy."""

	short_window: int = 100
	long_window: int = 200
	price_column: str = "Adj Close"


def add_indicators(
	data: pd.DataFrame,
	short_window: int = 100,
	long_window: int = 200,
	price_column: str = "QQQ_Close",
) -> pd.DataFrame:
	"""Add SMA indicators to QQQ price data.

	SMAs are always calculated on QQQ_Close for signal generation.
	"""

	if short_window <= 0 or long_window <= 0:
		raise ValueError("SMA windows must be positive integers.")
	if short_window >= long_window:
		raise ValueError("short_window must be smaller than long_window.")
	if price_column not in data.columns:
		raise ValueError(f"Expected price column '{price_column}' was not found.")

	frame = data.copy().sort_index()
	frame["sma100"] = frame[price_column].rolling(window=short_window, min_periods=short_window).mean()
	frame["sma200"] = frame[price_column].rolling(window=long_window, min_periods=long_window).mean()
	return frame


def generate_signals(
	data: pd.DataFrame,
	short_window: int = 100,
	long_window: int = 200,
	price_column: str = "QQQ_Close",
) -> pd.DataFrame:
	"""Generate buy/sell signals based on QQQ prices and SMAs.

	Buy signal when QQQ price > SMA100(QQQ) AND QQQ price > SMA200(QQQ).
	Sell signal when QQQ price < SMA200(QQQ) * 1.01 (with 1% buffer).
	Trades execute on TQQQ prices.
	The target position is 1 when long and 0 when flat.
	"""

	frame = add_indicators(
		data=data,
		short_window=short_window,
		long_window=long_window,
		price_column=price_column,
	)

	price = frame[price_column]
	buy_signal = (price > frame["sma100"]) & (price > frame["sma200"])
	sell_signal = price < frame["sma200"] * 1.01

	raw_position = pd.Series(
		np.select([buy_signal, sell_signal], [1, 0], default=np.nan),
		index=frame.index,
		dtype="float64",
	)

	frame["buy_signal"] = buy_signal
	frame["sell_signal"] = sell_signal
	frame["signal"] = np.select([buy_signal, sell_signal], [1, -1], default=0)
	frame["target_position"] = raw_position.ffill().fillna(0).astype(int)
	return frame
