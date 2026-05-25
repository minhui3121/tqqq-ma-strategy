from __future__ import annotations

import pandas as pd

from src.data_loader import download_qqq_and_tqqq_data, download_single_ticker


def test_local_csv_loader_returns_chronological_data() -> None:
	qqq = download_single_ticker('QQQ', '2005-01-01', '2010-02-12')
	assert qqq.index.is_monotonic_increasing
	assert qqq.index.min() == pd.Timestamp('2005-01-03')
	assert qqq.index.max() == pd.Timestamp('2010-02-12')


def test_backfilled_tqqq_starts_with_requested_qqq_range() -> None:
	data = download_qqq_and_tqqq_data(
		start_date='2005-01-01',
		end_date='2010-02-12',
		short_window=80,
		long_window=190,
	)

	assert data.index.is_monotonic_increasing
	assert data.index.min() == pd.Timestamp('2005-01-03')
	assert data.index.max() == pd.Timestamp('2010-02-12')
	assert {'QQQ_Close', 'TQQQ_Open', 'TQQQ_Close', 'sma80', 'sma190'}.issubset(data.columns)
