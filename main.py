"""Entry point for the TQQQ moving-average backtesting framework."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from src.backtest import run_backtest
from src.data_loader import download_qqq_and_tqqq_data
from src.metrics import (
    calculate_performance_metrics,
    export_backtest_summary,
    export_daily_portfolio_to_csv,
    export_signals_to_csv,
    export_trades_to_csv,
    extract_trades,
    find_max_drawdown_point,
    format_metrics,
    sample_portfolio_evolution,
)
from src.strategy import generate_signals


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Backtest a TQQQ SMA strategy.")
    parser.add_argument("--start", default="2005-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--short-window", type=int, default=100, help="Short SMA window.")
    parser.add_argument("--long-window", type=int, default=200, help="Long SMA window.")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Starting portfolio value.",
    )
    return parser.parse_args()


def plot_results(backtest_data, price_column: str = "QQQ_Close") -> None:
    """Plot QQQ price/SMA overlays and the TQQQ portfolio value series."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(backtest_data.index, backtest_data[price_column], label="QQQ Close", linewidth=1.4)
    axes[0].plot(backtest_data.index, backtest_data["sma100"], label="QQQ SMA 100", linewidth=1.2)
    axes[0].plot(backtest_data.index, backtest_data["sma200"], label="QQQ SMA 200", linewidth=1.2)
    axes[0].set_title("QQQ Price with Moving Averages (Signal Generation)")
    axes[0].set_ylabel("QQQ Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(backtest_data.index, backtest_data["portfolio_value"], label="TQQQ Portfolio Value", color="black")
    axes[1].set_title("TQQQ Portfolio Value Over Time")
    axes[1].set_ylabel("Portfolio Value ($)")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "backtest_results.png"
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full data, strategy, backtest, metrics, and plotting pipeline."""

    print("Downloading QQQ and TQQQ data...")
    merged_data = download_qqq_and_tqqq_data(start_date=args.start, end_date=args.end or None)
    
    signal_data = generate_signals(
        merged_data,
        short_window=args.short_window,
        long_window=args.long_window,
    )
    backtest_result = run_backtest(
        signal_data,
        initial_capital=args.initial_capital,
    )

    metrics = calculate_performance_metrics(
        backtest_result.data,
        initial_capital=backtest_result.initial_capital,
    )

    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Period: {args.start} to {args.end or 'today'}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Final Value: ${backtest_result.data['portfolio_value'].iloc[-1]:,.2f}")
    print()
    print("Performance Metrics")
    print("-" * 70)
    print(format_metrics(metrics))
    print()

    trades = extract_trades(backtest_result.data)
    print(f"Total Trades: {len(trades)}")
    print("Last 5 Trades:")
    print("-" * 70)
    for i, trade in enumerate(trades[-5:], start=1):
        entry_date = trade["entry_date"].strftime("%Y-%m-%d")
        exit_date = trade["exit_date"].strftime("%Y-%m-%d")
        trade_ret = trade["trade_return"]
        print(
            f"{i}. Entry: {entry_date} @ ${trade['entry_price']:.2f} | "
            f"Exit: {exit_date} @ ${trade['exit_price']:.2f} | "
            f"Return: {trade_ret:+.2%}"
        )
    print()

    max_dd_date, peak_value, trough_value = find_max_drawdown_point(backtest_result.data)
    print("Max Drawdown Analysis")
    print("-" * 70)
    print(f"Date: {max_dd_date.strftime('%Y-%m-%d')}")
    print(f"Peak Value: ${peak_value:,.2f}")
    print(f"Trough Value: ${trough_value:,.2f}")
    print(f"Drawdown: {(trough_value / peak_value - 1):.2%}")
    print()

    print("Portfolio Evolution (Sample)")
    print("-" * 70)
    evolution = sample_portfolio_evolution(backtest_result.data)
    for date, row in evolution.iterrows():
        pct_of_initial = row["portfolio_value"] / args.initial_capital
        print(f"{date.strftime('%Y-%m-%d')}: ${row['portfolio_value']:>12,.2f} ({pct_of_initial:>7.2%})")
    print()
    print("="*70 + "\n")

    plot_results(backtest_result.data)

    print("\nEXPORTING DETAILED RESULTS")
    print("="*70)
    export_trades_to_csv(trades, args.initial_capital, output_file="trades.csv")
    export_daily_portfolio_to_csv(backtest_result.data, args.initial_capital, output_file="portfolio_daily.csv")
    export_signals_to_csv(backtest_result.data, output_file="signals.csv")
    export_backtest_summary(
        metrics,
        trades,
        args.initial_capital,
        args.start,
        args.end or "today",
        output_file="backtest_summary.txt",
    )
    print("="*70 + "\n")


def main() -> None:
    """Program entry point."""

    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
