"""Backtest engine for the TQQQ moving-average strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
	"""Configuration values for the backtest engine."""

	initial_capital: float = 10_000.0
	execution_price_column: str = "TQQQ_Open"
	mark_price_column: str = "TQQQ_Close"
	position_column: str = "target_position"


@dataclass(frozen=True)
class BacktestResult:
	"""Container for the simulated portfolio history."""

	data: pd.DataFrame
	initial_capital: float


def run_backtest(
	data: pd.DataFrame,
	initial_capital: float = 10_000.0,
	execution_price_column: str = "TQQQ_Open",
	mark_price_column: str = "TQQQ_Close",
	position_column: str = "target_position",
	deposits: dict[pd.Timestamp, float] | None = None,
) -> BacktestResult:
	"""Simulate a long-only, fully invested backtest on TQQQ prices.
	
	Signals are generated from QQQ price/SMA data.
	Signals observed at day N close execute at day N+1 open.
	Portfolio valuation uses day N close.
	
	Parameters
	----------
	deposits:
		Optional dict mapping dates to deposit amounts. On each date,
		the specified amount is added to cash.
	"""

	if execution_price_column not in data.columns:
		raise ValueError(f"Expected execution price column '{execution_price_column}' was not found.")
	if mark_price_column not in data.columns:
		raise ValueError(f"Expected mark price column '{mark_price_column}' was not found.")
	if position_column not in data.columns:
		raise ValueError(f"Expected position column '{position_column}' was not found.")
	if initial_capital <= 0:
		raise ValueError("initial_capital must be positive.")

	frame = data.copy().sort_index()
	frame = frame.dropna(subset=[execution_price_column, mark_price_column])
	frame[position_column] = frame[position_column].fillna(0).astype(int).clip(0, 1)
	frame["execution_target_position"] = frame[position_column].shift(1).fillna(0).astype(int)

	cash_history: list[float] = []
	shares_history: list[float] = []
	portfolio_history: list[float] = []
	actual_position_history: list[int] = []
	trade_history: list[int] = []
	trade_price_history: list[float] = []
	deposit_history: list[float] = []
	cumulative_invested_history: list[float] = []

	cash = float(initial_capital)
	shares = 0.0
	actual_position = 0
	cumulative_invested = float(initial_capital)
	deposits_dict = deposits if deposits is not None else {}

	for idx, row in frame.iterrows():
		# Apply any deposit on this date
		deposit_amount = 0.0
		if idx in deposits_dict:
			deposit_amount = float(deposits_dict[idx])
			cash += deposit_amount
			cumulative_invested += deposit_amount
		
		execution_price = float(row[execution_price_column])
		mark_price = float(row[mark_price_column])
		target_position = int(row["execution_target_position"])

		# Recalculate portfolio value after deposit is applied
		portfolio_before_trade = cash + shares * execution_price
		trade_executed = int(target_position != actual_position)
		trade_price = float("nan")

		if target_position != actual_position:
			if target_position == 1:
				shares = portfolio_before_trade / execution_price
				cash = 0.0
				actual_position = 1
				trade_price = execution_price
			else:
				cash = shares * execution_price
				shares = 0.0
				actual_position = 0
				trade_price = execution_price
		elif target_position == 1 and deposit_amount > 0:
			# Already in position and received a deposit: reinvest the deposit immediately
			shares += deposit_amount / execution_price
			cash = 0.0
			trade_price = execution_price  # Record deposit-driven buy execution
			trade_executed = 1  # Mark that a deposit-driven buy occurred

		portfolio_value = cash + shares * mark_price

		cash_history.append(cash)
		shares_history.append(shares)
		portfolio_history.append(portfolio_value)
		actual_position_history.append(actual_position)
		trade_history.append(trade_executed)
		trade_price_history.append(trade_price)
		deposit_history.append(deposit_amount)
		cumulative_invested_history.append(cumulative_invested)

	frame["cash"] = cash_history
	frame["shares"] = shares_history
	frame["position"] = actual_position_history
	frame["trade_executed"] = trade_history
	frame["trade_price"] = trade_price_history
	frame["portfolio_value"] = portfolio_history
	frame["deposit"] = deposit_history
	frame["cumulative_invested"] = cumulative_invested_history
	return BacktestResult(data=frame, initial_capital=initial_capital)
