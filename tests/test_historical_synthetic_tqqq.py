from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import download_qqq_and_tqqq_data
from src.strategy import generate_signals


def _fake_download_single_ticker(ticker: str, start, end, price_column: str = "Close") -> pd.DataFrame:
	start_ts = pd.Timestamp(start)
	end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp("2010-03-01")

	if ticker == "QQQ":
		index = pd.bdate_range("2006-01-03", "2010-03-01")
		index = index[(index >= start_ts) & (index <= end_ts)]
		if index.empty:
			return pd.DataFrame()

		number_of_rows = len(index)
		close = 100.0 * (1.0005 ** np.arange(number_of_rows))
		open_ = close / 1.0002
		return pd.DataFrame({"Open": open_, "Close": close}, index=index)

	if ticker == "TQQQ":
		launch_date = pd.Timestamp("2010-02-11")
		index = pd.bdate_range(max(start_ts, launch_date), min(end_ts, pd.Timestamp("2010-03-01")))
		if index.empty:
			return pd.DataFrame()

		number_of_rows = len(index)
		close = 20.0 * (1.001 ** np.arange(number_of_rows))
		open_ = close / 1.0005
		return pd.DataFrame({"Open": open_, "Close": close}, index=index)

	raise ValueError(f"Unexpected ticker: {ticker}")


def test_synthetic_tqqq_backfills_pre_launch_history(monkeypatch) -> None:
	monkeypatch.setattr("src.data_loader.download_single_ticker", _fake_download_single_ticker)

	real_only = download_qqq_and_tqqq_data(
		start_date="2007-01-01",
		end_date="2010-03-01",
		short_window=80,
		long_window=190,
		use_synthetic_tqqq=False,
	)
	synthetic = download_qqq_and_tqqq_data(
		start_date="2007-01-01",
		end_date="2010-03-01",
		short_window=80,
		long_window=190,
		use_synthetic_tqqq=True,
	)

	assert real_only.index.min() == pd.Timestamp("2010-02-11")
	assert synthetic.index.min() == pd.Timestamp("2007-01-01")
	assert {"QQQ_Close", "TQQQ_Open", "TQQQ_Close"}.issubset(synthetic.columns)
	assert synthetic.loc[pd.Timestamp("2010-02-11"), "TQQQ_Close"] == 20.0
	assert synthetic.loc[pd.Timestamp("2010-02-10"), "TQQQ_Close"] > 0

	signals = generate_signals(synthetic, short_window=80, long_window=190)
	assert {"sma80", "sma190", "target_position"}.issubset(signals.columns)
	result = run_backtest(signals)

	assert result.data.index.min() == pd.Timestamp("2007-01-01")
	assert "portfolio_value" in result.data.columns
	assert result.data["portfolio_value"].notna().all()