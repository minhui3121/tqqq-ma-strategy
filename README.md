# tqqq-ma-strategy

Python backtesting framework for a leveraged ETF trend-following strategy on TQQQ.

## Strategy rules (current)

- Signal asset: QQQ (uses raw `Close` price)
- Execution asset: TQQQ (uses raw `Open` for fills, raw `Close` for end-of-day marking)
- Short SMA: 100 days
- Long SMA: 190 days

### Entry logic

- Buy/hold long (`target_position = 1`) when:
  - `QQQ_Close > SMA100`
  - and `QQQ_Close > SMA190`

### Exit logic

- Exit to cash (`target_position = 0`) when:
  - `QQQ_Close < SMA190`

## Trade timing

Signals are computed from day N close data.
Orders are executed at day N+1 open (next trading day open).

This avoids look-ahead bias from same-day close-based signals and same-day execution.

## Data handling

- QQQ and TQQQ both use `Close` prices for signal/marking series.
- TQQQ `Open` is used only for next-day execution fills.
- The loader fetches additional QQQ history automatically to warm up SMA windows.
- If requested start date is before TQQQ inception (2010-02-11), backtest starts on TQQQ's first available date.

## Assumptions

- Long-only strategy
- Fully invested when in position, fully in cash when out
- No transaction costs

## Run

```bash
python main.py --start 2005-01-01 --end 2026-01-01 --short-window 100 --long-window 190
```

## Output files

- `trades.csv`: executed trades (entry/exit dates, execution prices, returns)
- `portfolio_daily.csv`: daily portfolio state (QQQ close, TQQQ open/close, SMA values, position, PnL fields)
- `signals.csv`: daily signal states and target position
- `backtest_summary.txt`: text summary of key performance metrics
- `backtest_results.png`: chart output

## Structure

- `src/` implementation modules
- `main.py` pipeline entry point
- `tests/` tests
- `notebooks/` experiments
- `data/` raw historical data (optional)
