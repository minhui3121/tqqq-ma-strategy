# tqqq-ma-strategy

Python backtesting framework for a leveraged ETF trend-following strategy on TQQQ.

## Strategy rules (current)

- Signal asset: QQQ (uses raw `Close` price)
- Execution asset: TQQQ (uses raw `Open` for fills, raw `Close` for end-of-day marking)
- Short SMA: 80 days
- Long SMA: 190 days

### Entry logic

- Buy/hold long (`target_position = 1`) when:
  - `QQQ_Close > SMA80`
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
- If requested start date is before TQQQ inception (2010-02-11), backtest starts at TQQQ's first available date in the default real-only mode.
- Use `--synthetic-tqqq` to backfill pre-2010 execution prices with a leveraged QQQ-based proxy and run the strategy over earlier history.

## Assumptions

- Long-only strategy
- Fully invested when in position, fully in cash when out
- No transaction costs

## Run

```bash
python main.py --start 2005-01-01 --end 2026-01-01 --short-window 80 --long-window 190
```

Historical synthetic mode:

```bash
python main.py --start 2005-01-01 --end 2026-01-01 --short-window 80 --long-window 190 --synthetic-tqqq
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

## GitHub Actions email automation

This repo includes a workflow at `.github/workflows/daily-email-report.yml`.

It sends one email at 5:00 PM Montreal time on weekdays and also supports manual runs from the Actions tab.

### Required GitHub repository secrets

Go to **Settings -> Secrets and variables -> Actions -> New repository secret** and create:

- `SMTP_HOST` (example: `smtp.gmail.com`)
- `SMTP_PORT` (example: `587`)
- `SMTP_USER` (your SMTP login email)
- `SMTP_PASSWORD` (SMTP app password)
- `REPORT_TO` (recipient email)

### Enable and test

1. Push your branch with `.github/workflows/daily-email-report.yml` to GitHub.
2. Open the **Actions** tab and select **Daily Email Report**.
3. Click **Run workflow** to test immediately.
4. Check the run logs to confirm email delivery.

