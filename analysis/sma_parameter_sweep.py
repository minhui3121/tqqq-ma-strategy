from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.data_loader import download_qqq_and_tqqq_data, generate_annual_deposits
from src.metrics import calculate_performance_metrics
from src.strategy import generate_signals


PERIOD_START = "2005-01-01"
PERIOD_END = "2026-01-01"

SHORT_VALUES = list(range(20, 151, 10))
LONG_VALUES = list(range(60, 301, 10))

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "sma_parameter_sweep_results.csv"
SUMMARY_MD = OUTPUT_DIR / "sma_parameter_sweep_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an SMA parameter sweep for the TQQQ strategy.")
    parser.add_argument("--start", default=PERIOD_START, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=PERIOD_END, help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--synthetic-tqqq",
        action="store_true",
        help="Use synthetic leveraged TQQQ prices before real TQQQ history begins.",
    )
    parser.add_argument(
        "--continuous-investment",
        action="store_true",
        help="Add $10,000 to portfolio every January 1st during the backtest period.",
    )
    return parser.parse_args()


def evaluate(data: pd.DataFrame, short_window: int, long_window: int, deposits: dict | None = None) -> dict[str, float | int]:
    """Evaluate a single SMA parameter combination."""
    signal = generate_signals(data, short_window=short_window, long_window=long_window)
    result = run_backtest(signal, deposits=deposits)
    metrics = calculate_performance_metrics(
        result.data,
        initial_capital=result.initial_capital,
        total_invested=result.data["cumulative_invested"].iloc[-1] if deposits else None,
    )

    return_dict = {
        "short": short_window,
        "long": long_window,
        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualized_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "trade_days": int(result.data.get("trade_executed", pd.Series()).sum()) if "trade_executed" in result.data.columns else 0,
        "final_value": float(result.data["portfolio_value"].iloc[-1]),
    }

    if "total_invested" in metrics:
        return_dict["total_invested"] = metrics["total_invested"]
        return_dict["effective_return"] = metrics["effective_return"]

    return return_dict


def main() -> None:
    args = parse_args()
    period_start = args.start
    period_end = args.end

    print(f"Downloading data for {period_start} to {period_end}...")
    if args.synthetic_tqqq:
        print("Synthetic pre-2010 TQQQ mode enabled.")
    if args.continuous_investment:
        print("Continuous investment mode enabled.")

    # Download must be performed for each candidate pair so both SMAs are
    # warmed on the same pre-start history that `main.py` uses.
    candidates: list[tuple[int, int]] = []
    for short_w in SHORT_VALUES:
        for long_w in LONG_VALUES:
            if long_w <= short_w + 20:
                continue
            candidates.append((int(short_w), int(long_w)))

    # Load data once to generate deposits if needed
    sample_data = download_qqq_and_tqqq_data(
        period_start,
        period_end,
        short_window=SHORT_VALUES[0],
        long_window=LONG_VALUES[-1],
        use_synthetic_tqqq=args.synthetic_tqqq,
    )
    deposits = None
    if args.continuous_investment:
        deposits = generate_annual_deposits(sample_data, deposit_amount=10_000.0)

    records: list[dict[str, float | int]] = []

    for short_w, long_w in candidates:
        try:
            print(f"Evaluating short_window={short_w}, long_window={long_w}...")
            base = download_qqq_and_tqqq_data(
                period_start,
                period_end,
                short_window=short_w,
                long_window=long_w,
                use_synthetic_tqqq=args.synthetic_tqqq,
            )
            records.append(evaluate(base, short_w, long_w, deposits=deposits))
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"Combo failed short={short_w} long={long_w}: {exc}")

    df = pd.DataFrame(records)
    df["calmar_proxy"] = df["annualized_return"] / df["max_drawdown"].abs().replace(0, math.nan)
    df.to_csv(RESULTS_CSV, index=False)

    best = df.sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False]).iloc[0]

    summary = [
        "# SMA Full-Period Sweep",
        "",
        f"Period: {period_start} to {period_end}",
        f"Synthetic pre-2010 TQQQ mode: {'enabled' if args.synthetic_tqqq else 'disabled'}",
        f"Continuous investment mode: {'enabled' if args.continuous_investment else 'disabled'}",
        "",
        "## Best Candidate (full period)",
        "",
        best.to_string(),
        "",
        "## Top 10 (by Sharpe)",
        "",
        df.sort_values(["sharpe_ratio", "annualized_return"], ascending=[False, False]).head(10).to_string(index=False),
        "",
        "## Notes",
        "",
        "- This single-period sweep selects the best-performing SMA pair over the entire given range (no train/test split).",
        "- Use with caution: single-period winners can be prone to overfitting to the chosen window.",
    ]

    SUMMARY_MD.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved summary to: {SUMMARY_MD}")
    print("Best candidate:\n", best.to_dict())


if __name__ == "__main__":
    main()
