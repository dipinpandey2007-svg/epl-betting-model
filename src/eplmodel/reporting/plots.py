"""Figures, drawn with matplotlib's object API (no pyplot, no blocking windows)."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def plot_calibration(table: pd.DataFrame, label: str, title: str, path: Path) -> Path:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(table["mean_pred"], table["mean_actual"], marker="o", label=label)
    ax.set_xlabel("Mean predicted P(Home)")
    ax.set_ylabel("Observed home-win frequency")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return Path(path)


def plot_rho_search(grid: pd.DataFrame, best_rho: float, path: Path) -> Path:
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.plot(grid["rho"], grid["log_likelihood"], marker="o")
    ax.axvline(best_rho, color="red", linestyle="--", label=f"Best rho = {best_rho:.2f}")
    ax.set_xlabel("rho")
    ax.set_ylabel("Training log-likelihood")
    ax.set_title("Staged Dixon-Coles rho search (training seasons only)")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return Path(path)
