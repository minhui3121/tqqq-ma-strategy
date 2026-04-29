from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path(__file__).resolve().parent / "sma_parameter_sweep_results.csv"
OUTPUT_PNG = Path(__file__).resolve().parent / "sma_parameter_sweep_heatmaps.png"


def _to_matrix(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[int], list[int]]:
    shorts = sorted(df["short"].astype(int).unique().tolist())
    longs = sorted(df["long"].astype(int).unique().tolist())

    matrix = np.full((len(shorts), len(longs)), np.nan, dtype=float)
    short_to_idx = {value: idx for idx, value in enumerate(shorts)}
    long_to_idx = {value: idx for idx, value in enumerate(longs)}

    for _, row in df.iterrows():
        s = int(row["short"])
        l = int(row["long"])
        matrix[short_to_idx[s], long_to_idx[l]] = float(row[metric])

    return matrix, shorts, longs


def _plot_heatmap(ax: plt.Axes, matrix: np.ndarray, shorts: list[int], longs: list[int], metric: str) -> None:
    im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")

    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Long Window")
    ax.set_ylabel("Short Window")

    ax.set_xticks(np.arange(len(longs)))
    ax.set_xticklabels(longs, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(shorts)))
    ax.set_yticklabels(shorts, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(metric, rotation=270, labelpad=12)

    if np.isfinite(matrix).any():
        best_idx = np.nanargmax(matrix)
        best_row, best_col = np.unravel_index(best_idx, matrix.shape)
        ax.scatter(best_col, best_row, marker="x", s=100, linewidths=2)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"short", "long", "sharpe_ratio", "annualized_return", "max_drawdown", "final_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    metrics = ["sharpe_ratio", "annualized_return", "max_drawdown", "final_value"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    for ax, metric in zip(axes.ravel(), metrics):
        matrix, shorts, longs = _to_matrix(df, metric)
        _plot_heatmap(ax, matrix, shorts, longs, metric)

    fig.suptitle("SMA Parameter Sweep Heatmaps", fontsize=16)
    fig.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight")
    print(f"Saved heatmaps to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
