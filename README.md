# tqqq-ma-strategy

Python backtesting framework for a leveraged ETF trend-following strategy on TQQQ.

## What it does

- Downloads daily TQQQ history from Yahoo Finance with adjusted close prices.
- Computes SMA 100 and SMA 200.
- Generates long-only signals using the moving-average rules.
- Reports total return, annualized return, max drawdown, and Sharpe ratio.
- Plots price/SMA overlays and portfolio value.

## Run

```bash
python main.py --start 2005-01-01 --end 2026-01-01 --short-window 100 --long-window 200
```

## Structure

- `data/` raw historical data
- `notebooks/` experiments
- `src/` implementation modules
- `tests/` tests
- `main.py` pipeline entry point