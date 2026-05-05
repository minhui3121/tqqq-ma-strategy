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
	total_invested: float | None = None,
) -> dict[str, float]:
	"""Compute key performance statistics from a portfolio value series.
	
	Parameters
	----------
	total_invested:
		Total amount invested (initial + all deposits). If provided, metrics
		will include effective return on total invested capital.
		"""

	if portfolio_column not in backtest_data.columns:
		raise ValueError(f"Expected portfolio column '{portfolio_column}' was not found.")

	portfolio = backtest_data[portfolio_column].dropna()
	if portfolio.empty:
		raise ValueError("Portfolio series is empty.")

	ending_value = float(portfolio.iloc[-1])
	total_return = ending_value / initial_capital - 1.0

	periods = max(len(portfolio) - 1, 1)
	# By default annualize relative to initial capital
	annualized_return = (ending_value / initial_capital) ** (periods_per_year / periods) - 1.0

	rolling_max = portfolio.cummax()
	drawdown = portfolio / rolling_max - 1.0
	max_drawdown = float(drawdown.min())

	daily_returns = portfolio.pct_change().dropna()
	if daily_returns.empty or np.isclose(daily_returns.std(ddof=0), 0.0):
		sharpe_ratio = 0.0
	else:
		sharpe_ratio = float((daily_returns.mean() / daily_returns.std(ddof=0)) * math.sqrt(periods_per_year))

	metrics = {
		"total_return": float(total_return),
		"annualized_return": float(annualized_return),
		"max_drawdown": max_drawdown,
		"sharpe_ratio": sharpe_ratio,
	}

	if total_invested is not None and total_invested > 0:
		effective_return = ending_value / total_invested - 1.0
		# When using continuous investment, annualized return should be
		# based on effective return (ending / total_invested)
		effective_annualized = (ending_value / total_invested) ** (periods_per_year / periods) - 1.0
		metrics["total_invested"] = float(total_invested)
		metrics["effective_return"] = float(effective_return)
		metrics["annualized_return"] = float(effective_annualized)

	return metrics


def format_metrics(metrics: dict[str, float]) -> str:
	"""Format metrics for console output."""

	return (
		f"Total Return: {metrics['total_return']:.2%}\n"
		f"Annualized Return: {metrics['annualized_return']:.2%}\n"
		f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
		f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
	)


def extract_trades(
	backtest_data: pd.DataFrame,
	price_column: str = "trade_price",
	position_column: str = "position",
) -> list[dict]:
	"""Extract individual trades from backtest results using TQQQ execution prices.

	A trade is a buy followed eventually by a sell.
	Returns list of dicts with entry_date, entry_price, exit_date, exit_price, return, shares.
	"""

	trades = []
	entry_date = None
	entry_price = None
	shares = 0.0

	for idx, row in backtest_data.iterrows():
		price = float(row[price_column]) if price_column in row and not pd.isna(row[price_column]) else float(row.get("TQQQ_Close", 0.0))
		position = int(row[position_column])
		position_changed = row.get("trade_executed", 0)

		if position_changed == 1:
			if position == 1:
				entry_date = idx
				entry_price = price
				shares = float(row.get("shares", 0))
			elif position == 0 and entry_date is not None:
				exit_date = idx
				exit_price = price
				trade_return = (exit_price - entry_price) / entry_price
				trades.append({
					"entry_date": entry_date,
					"entry_price": entry_price,
					"exit_date": exit_date,
					"exit_price": exit_price,
					"trade_return": trade_return,
					"shares": shares,
				})
				entry_date = None
				entry_price = None
				shares = 0.0

	return trades


def find_max_drawdown_point(
	backtest_data: pd.DataFrame,
	portfolio_column: str = "portfolio_value",
) -> tuple[pd.Timestamp, float, float]:
	"""Find the date and values when max drawdown occurred.

	Returns (date_of_max_drawdown, peak_value, trough_value).
	"""

	portfolio = backtest_data[portfolio_column]
	rolling_max = portfolio.cummax()
	drawdown = portfolio / rolling_max - 1.0

	min_drawdown_idx = drawdown.idxmin()
	min_value = portfolio.loc[min_drawdown_idx]

	peak_value = rolling_max.loc[min_drawdown_idx]

	return min_drawdown_idx, peak_value, min_value


def sample_portfolio_evolution(
	backtest_data: pd.DataFrame,
	sample_size: int = 10,
	portfolio_column: str = "portfolio_value",
) -> pd.DataFrame:
	"""Return evenly-spaced samples of portfolio value over time."""

	n = len(backtest_data)
	indices = np.linspace(0, n - 1, sample_size, dtype=int)
	return backtest_data.iloc[indices][[portfolio_column]].copy()


def export_trades_to_csv(
	trades: list[dict],
	initial_capital: float,
	output_file: str = "trades.csv",
) -> None:
	"""Export trades to CSV with extended details."""

	export_data = []
	cumulative_pnl = 0.0

	for i, trade in enumerate(trades, start=1):
		entry_value = trade["shares"] * trade["entry_price"]
		exit_value = trade["shares"] * trade["exit_price"]
		pnl_dollars = exit_value - entry_value
		cumulative_pnl += pnl_dollars

		export_data.append({
			"trade_num": i,
			"entry_date": trade["entry_date"].strftime("%Y-%m-%d"),
			"entry_price": round(trade["entry_price"], 2),
			"shares": round(trade["shares"], 4),
			"entry_value": round(entry_value, 2),
			"exit_date": trade["exit_date"].strftime("%Y-%m-%d"),
			"exit_price": round(trade["exit_price"], 2),
			"exit_value": round(exit_value, 2),
			"trade_return_pct": f"{trade['trade_return']*100:.2f}%",
			"trade_return_decimal": round(trade["trade_return"], 4),
			"pnl_dollars": round(pnl_dollars, 2),
			"cumulative_pnl": round(cumulative_pnl, 2),
			"days_held": (trade["exit_date"] - trade["entry_date"]).days,
		})

	df = pd.DataFrame(export_data)
	df.to_csv(output_file, index=False)
	print(f"Trades exported to {output_file}")


def export_daily_portfolio_to_csv(
	backtest_data: pd.DataFrame,
	initial_capital: float,
	output_file: str = "portfolio_daily.csv",
	total_invested: float | None = None,
) -> None:
	"""Export full daily portfolio snapshot to CSV."""

	export_data = []

	for idx, row in backtest_data.iterrows():
		rolling_max = backtest_data.loc[:idx, "portfolio_value"].max()
		current_val = row["portfolio_value"]
		drawdown = (current_val / rolling_max - 1.0) if rolling_max > 0 else 0.0

		daily_return = 0.0
		if idx != backtest_data.index[0]:
			prev_val = backtest_data.loc[:idx].iloc[-2]["portfolio_value"]
			if prev_val > 0:
				daily_return = (current_val / prev_val) - 1.0

			# Choose basis for percent return: effective (total_invested) when provided,
			# otherwise use initial capital for legacy behavior.
			basis = total_invested if (total_invested is not None and total_invested > 0) else initial_capital
			export_data.append({
			"date": idx.strftime("%Y-%m-%d"),
			"qqq_close": round(row.get("QQQ_Close", 0), 2),
			"tqqq_open": round(row.get("TQQQ_Open", 0), 2),
			"tqqq_close": round(row.get("TQQQ_Close", 0), 2),
			"trade_price": round(row.get("trade_price", 0), 4) if "trade_price" in row and not pd.isna(row["trade_price"]) else "",
			"sma80": round(row.get("sma80", 0), 2) if "sma80" in row and not pd.isna(row["sma80"]) else "",
			"sma190": round(row.get("sma190", 0), 2) if "sma190" in row and not pd.isna(row["sma190"]) else "",
			"position": int(row.get("position", 0)),
			"shares": round(row.get("shares", 0), 4),
			"cash": round(row.get("cash", 0), 2),
			"deposit": round(row.get("deposit", 0), 2),
			"cumulative_invested": round(row.get("cumulative_invested", 0), 2),
			"portfolio_value": round(row["portfolio_value"], 2),
			"portfolio_return_pct": f"{((current_val / basis) - 1) * 100:.2f}%",
			"daily_return_pct": f"{daily_return*100:.2f}%" if daily_return != 0 else "0.00%",
			"drawdown_pct": f"{drawdown*100:.2f}%",
			"trade_executed": int(row.get("trade_executed", 0)),
		})

	df = pd.DataFrame(export_data)
	df.to_csv(output_file, index=False)
	print(f"Daily portfolio exported to {output_file}")


def export_signals_to_csv(
	backtest_data: pd.DataFrame,
	output_file: str = "signals.csv",
) -> None:
	"""Export signal history to CSV."""

	export_data = []

	for idx, row in backtest_data.iterrows():
		export_data.append({
			"date": idx.strftime("%Y-%m-%d"),
			"qqq_close": round(row.get("QQQ_Close", 0), 2),
			"sma80": round(row.get("sma80", 0), 2) if "sma80" in row and not pd.isna(row["sma80"]) else "",
			"sma190": round(row.get("sma190", 0), 2) if "sma190" in row and not pd.isna(row["sma190"]) else "",
			"buy_signal": int(row.get("buy_signal", 0)),
			"sell_signal": int(row.get("sell_signal", 0)),
			"target_position": int(row.get("target_position", 0)),
		})

	df = pd.DataFrame(export_data)
	df.to_csv(output_file, index=False)
	print(f"Signals exported to {output_file}")


def export_backtest_summary(
	metrics: dict[str, float],
	trades: list[dict],
	initial_capital: float,
	start_date: str,
	end_date: str,
	output_file: str = "backtest_summary.txt",
) -> None:
	"""Export a human-readable summary to a text file."""

	final_value = initial_capital * (1 + metrics["total_return"])
	winning_trades = [t for t in trades if t["trade_return"] > 0]
	losing_trades = [t for t in trades if t["trade_return"] < 0]
	win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

	avg_win = sum(t["trade_return"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
	avg_loss = sum(t["trade_return"] for t in losing_trades) / len(losing_trades) if losing_trades else 0

	content = f"""
BACKTEST SUMMARY REPORT
{'='*70}

Period: {start_date} to {end_date}
Initial Capital: ${initial_capital:,.2f}
Final Value: ${final_value:,.2f}

PERFORMANCE METRICS
{'-'*70}
Total Return: {metrics['total_return']:.2%}
Annualized Return: {metrics['annualized_return']:.2%}
Max Drawdown: {metrics['max_drawdown']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

TRADE STATISTICS
{'-'*70}
Total Trades: {len(trades)}
Winning Trades: {len(winning_trades)}
Losing Trades: {len(losing_trades)}
Win Rate: {win_rate:.1f}%
Average Win: {avg_win:.2%}
Average Loss: {avg_loss:.2%}
Profit Factor: {abs(sum(t['trade_return'] for t in winning_trades) / sum(t['trade_return'] for t in losing_trades)) if losing_trades and sum(t['trade_return'] for t in losing_trades) != 0 else 0:.2f}

{'='*70}
"""

	with open(output_file, "w") as f:
		f.write(content)
	print(f"Summary exported to {output_file}")

