"""Entry point for the TQQQ moving-average backtesting framework."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from src.backtest import run_backtest
from src.data_loader import download_tqqq_data, prepare_price_series
from src.metrics import calculate_performance_metrics, format_metrics
from src.strategy import generate_signals


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Backtest a TQQQ SMA strategy.")
    parser.add_argument("--start", default="2005-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--short-window", type=int, default=100, help="Short SMA window.")
    parser.add_argument("--long-window", type=int, default=200, help="Long SMA window.")
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per trade expressed as a decimal.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Starting portfolio value.",
    )
    return parser.parse_args()


def plot_results(backtest_data, price_column: str = "Adj Close") -> None:
    """Plot price/SMA overlays and the portfolio value series."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(backtest_data.index, backtest_data[price_column], label="Adj Close", linewidth=1.4)
    axes[0].plot(backtest_data.index, backtest_data["sma100"], label="SMA 100", linewidth=1.2)
    axes[0].plot(backtest_data.index, backtest_data["sma200"], label="SMA 200", linewidth=1.2)
    axes[0].set_title("TQQQ Price with Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(backtest_data.index, backtest_data["portfolio_value"], label="Portfolio Value", color="black")
    axes[1].set_title("Portfolio Value Over Time")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full data, strategy, backtest, metrics, and plotting pipeline."""

    raw_data = download_tqqq_data(start_date=args.start, end_date=args.end or None)
    prepared_data = prepare_price_series(raw_data)
    signal_data = generate_signals(
        prepared_data,
        short_window=args.short_window,
        long_window=args.long_window,
    )
    backtest_result = run_backtest(
        signal_data,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
    )

    metrics = calculate_performance_metrics(
        backtest_result.data,
        initial_capital=backtest_result.initial_capital,
    )

    print("Performance Metrics")
    print("===================")
    print(format_metrics(metrics))
    plot_results(backtest_result.data)


def main() -> None:
    """Program entry point."""

    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
