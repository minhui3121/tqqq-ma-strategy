"""Backtest engine for the TQQQ moving-average strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
	"""Configuration values for the backtest engine."""

	initial_capital: float = 10_000.0
	transaction_cost: float = 0.001
	price_column: str = "Adj Close"
	position_column: str = "target_position"


@dataclass(frozen=True)
class BacktestResult:
	"""Container for the simulated portfolio history."""

	data: pd.DataFrame
	initial_capital: float
	transaction_cost: float


def run_backtest(
	data: pd.DataFrame,
	initial_capital: float = 10_000.0,
	transaction_cost: float = 0.001,
	price_column: str = "Adj Close",
	position_column: str = "target_position",
) -> BacktestResult:
	"""Simulate a long-only, fully invested backtest with transaction costs.

	The engine marks the portfolio to market at each close and applies
	transaction costs when the target position changes.
	"""

	if price_column not in data.columns:
		raise ValueError(f"Expected price column '{price_column}' was not found.")
	if position_column not in data.columns:
		raise ValueError(f"Expected position column '{position_column}' was not found.")
	if initial_capital <= 0:
		raise ValueError("initial_capital must be positive.")
	if not 0 <= transaction_cost < 1:
		raise ValueError("transaction_cost must be between 0 and 1.")

	frame = data.copy().sort_index()
	frame = frame.dropna(subset=[price_column, position_column])
	frame[position_column] = frame[position_column].astype(int).clip(0, 1)

	cash_history: list[float] = []
	shares_history: list[float] = []
	portfolio_history: list[float] = []
	actual_position_history: list[int] = []
	trade_history: list[int] = []

	cash = float(initial_capital)
	shares = 0.0
	actual_position = 0

	for _, row in frame.iterrows():
		price = float(row[price_column])
		target_position = int(row[position_column])

		portfolio_before_trade = cash + shares * price
		trade_executed = 0

		if target_position != actual_position:
			trade_executed = 1
			if target_position == 1:
				cost = portfolio_before_trade * transaction_cost
				investable_capital = portfolio_before_trade - cost
				shares = investable_capital / price
				cash = 0.0
				actual_position = 1
				portfolio_value = investable_capital
			else:
				gross_proceeds = shares * price
				cost = gross_proceeds * transaction_cost
				cash = gross_proceeds - cost
				shares = 0.0
				actual_position = 0
				portfolio_value = cash
		else:
			portfolio_value = portfolio_before_trade

		cash_history.append(cash)
		shares_history.append(shares)
		portfolio_history.append(portfolio_value)
		actual_position_history.append(actual_position)
		trade_history.append(trade_executed)

	frame["cash"] = cash_history
	frame["shares"] = shares_history
	frame["position"] = actual_position_history
	frame["trade_executed"] = trade_history
	frame["portfolio_value"] = portfolio_history
	return BacktestResult(data=frame, initial_capital=initial_capital, transaction_cost=transaction_cost)
