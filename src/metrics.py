"""Performance metrics for the backtesting framework."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_performance_metrics(
	backtest_data: pd.DataFrame,
	initial_capital: float,
	periods_per_year: int = 252,
	portfolio_column: str = "portfolio_value",
) -> dict[str, float]:
	"""Compute key performance statistics from a portfolio value series."""

	if portfolio_column not in backtest_data.columns:
		raise ValueError(f"Expected portfolio column '{portfolio_column}' was not found.")

	portfolio = backtest_data[portfolio_column].dropna()
	if portfolio.empty:
		raise ValueError("Portfolio series is empty.")

	ending_value = float(portfolio.iloc[-1])
	total_return = ending_value / initial_capital - 1.0

	periods = max(len(portfolio) - 1, 1)
	annualized_return = (ending_value / initial_capital) ** (periods_per_year / periods) - 1.0

	rolling_max = portfolio.cummax()
	drawdown = portfolio / rolling_max - 1.0
	max_drawdown = float(drawdown.min())

	daily_returns = portfolio.pct_change().dropna()
	if daily_returns.empty or np.isclose(daily_returns.std(ddof=0), 0.0):
		sharpe_ratio = 0.0
	else:
		sharpe_ratio = float((daily_returns.mean() / daily_returns.std(ddof=0)) * math.sqrt(periods_per_year))

	return {
		"total_return": float(total_return),
		"annualized_return": float(annualized_return),
		"max_drawdown": max_drawdown,
		"sharpe_ratio": sharpe_ratio,
	}


def format_metrics(metrics: dict[str, float]) -> str:
	"""Format metrics for console output."""

	return (
		f"Total Return: {metrics['total_return']:.2%}\n"
		f"Annualized Return: {metrics['annualized_return']:.2%}\n"
		f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
		f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
	)
