"""Visualization utilities for federated learning experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt


def _plot_histories(
    histories: Dict[str, Sequence[float]],
    *,
    title: str,
    ylabel: str,
    xlabel: str = "Communication Round",
    save_path: str | None = None,
) -> None:
    if not histories:
        raise ValueError("histories cannot be empty")

    num_rounds = max(len(values) for values in histories.values())
    rounds = list(range(1, num_rounds + 1))

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, values in histories.items():
        ax.plot(rounds[: len(values)], values, marker="o", markersize=2, alpha=0.9, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_accuracy_histories(
    histories: Dict[str, Sequence[float]],
    *,
    title: str = "Global Test Accuracy",
    save_path: str | None = None,
) -> None:
    _plot_histories(histories, title=title, ylabel="Accuracy (%)", save_path=save_path)


def plot_drift_histories(
    histories: Dict[str, Sequence[float]],
    *,
    title: str = "Client Drift",
    save_path: str | None = None,
) -> None:
    _plot_histories(histories, title=title, ylabel="Mean L2 Weight Divergence", save_path=save_path)


__all__ = ["plot_accuracy_histories", "plot_drift_histories"]
