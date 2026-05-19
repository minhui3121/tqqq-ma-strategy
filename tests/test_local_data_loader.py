from __future__ import annotations

import pandas as pd

from src.data_loader import download_qqq_and_tqqq_data, download_single_ticker


def test_local_csv_loader_returns_chronological_data() -> None:
	qqq = download_single_ticker('QQQ', '2005-01-01', '2026-01-01')
	assert qqq.index.is_monotonic_increasing
	assert qqq.index.min() == pd.Timestamp('1999-03-10')
	assert qqq.index.max() == pd.Timestamp('2025-12-30')


def test_real_mode_starts_at_actual_tqqq_inception() -> None:
	data = download_qqq_and_tqqq_data(
		start_date='2005-01-01',
		end_date='2026-01-01',
		short_window=80,
		long_window=190,
	)

	assert data.index.is_monotonic_increasing
	assert data.index.min() == pd.Timestamp('2010-02-11')
	assert {'QQQ_Close', 'TQQQ_Open', 'TQQQ_Close', 'sma80', 'sma190'}.issubset(data.columns)
