from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import generate_annual_deposits
from src.strategy import generate_signals


def test_annual_deposits_generated_correctly() -> None:
	"""Test that deposit dates are generated on Jan 1 of each year."""
	index = pd.date_range("2005-01-01", "2010-12-31", freq="B")  # Business days
	data = pd.DataFrame({"QQQ_Close": np.random.randn(len(index)) + 100}, index=index)

	deposits = generate_annual_deposits(data, deposit_amount=10_000.0, skip_first_year=True)

	# Should have deposits for 2006, 2007, 2008, 2009, 2010 (not 2005)
	assert len(deposits) == 5
	assert pd.Timestamp("2006-01-01") in deposits or pd.Timestamp("2006-01-03") in deposits
	for deposit_amount in deposits.values():
		assert deposit_amount == 10_000.0


def test_deposits_increase_cash_in_backtest() -> None:
	"""Test that deposits add to cash and cumulative_invested increases."""
	index = pd.date_range("2005-01-01", "2010-12-31", freq="B")
	close = 100.0 * (1.0005 ** np.arange(len(index)))
	data = pd.DataFrame({
		"QQQ_Close": close,
		"TQQQ_Open": close * 1.001,
		"TQQQ_Close": close,
	}, index=index)

	signals = generate_signals(data, short_window=80, long_window=190)

	# Create deposits on Jan 1 of years 2006-2010
	deposits = {
		pd.Timestamp("2006-01-01"): 10_000.0,
		pd.Timestamp("2007-01-01"): 10_000.0,
		pd.Timestamp("2008-01-01"): 10_000.0,
		pd.Timestamp("2009-01-01"): 10_000.0,
		pd.Timestamp("2010-01-01"): 10_000.0,
	}

	result = run_backtest(signals, initial_capital=10_000.0, deposits=deposits)

	# Check that cumulative_invested column exists
	assert "cumulative_invested" in result.data.columns
	assert "deposit" in result.data.columns

	# Check that deposits appear on the correct dates
	for deposit_date, amount in deposits.items():
		if deposit_date in result.data.index:
			assert result.data.loc[deposit_date, "deposit"] == amount

	# Check that cumulative_invested increases by ~$10,000 per year (approximately)
	# Get the value at end of each year
	start_value = result.data["cumulative_invested"].iloc[0]
	assert start_value == 10_000.0

	last_value = result.data["cumulative_invested"].iloc[-1]
	# Should be initial $10k + 5 deposits of $10k = $60k
	assert last_value == 60_000.0


def test_backtest_runs_without_deposits() -> None:
	"""Test that backtest still works with deposits=None (default behavior)."""
	index = pd.date_range("2005-01-01", "2010-12-31", freq="B")
	close = 100.0 * (1.0005 ** np.arange(len(index)))
	data = pd.DataFrame({
		"QQQ_Close": close,
		"TQQQ_Open": close * 1.001,
		"TQQQ_Close": close,
	}, index=index)

	signals = generate_signals(data, short_window=80, long_window=190)
	result = run_backtest(signals, initial_capital=10_000.0)

	assert "cumulative_invested" in result.data.columns
	# Without deposits, cumulative_invested should stay at initial capital
	assert result.data["cumulative_invested"].iloc[0] == 10_000.0
	assert result.data["cumulative_invested"].iloc[-1] == 10_000.0
