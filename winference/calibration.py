"""
Calibration metrics for pairwise win rate predictions.

When your model says "A beats B with probability 0.65", does A actually
win 65% of the time?  These tools check.
"""

from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray


def expected_calibration_error(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        predicted: Array of predicted win probabilities, shape (n,).
        observed: Array of binary outcomes (0 or 1), shape (n,).
        n_bins: Number of bins for grouping predictions.

    Returns:
        Weighted average |predicted - observed| across bins.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(bin_edges):
        mask = (predicted >= lo) & (predicted < hi)
        if mask.sum() == 0:
            continue
        avg_pred = predicted[mask].mean()
        avg_obs = observed[mask].mean()
        weight = mask.sum() / len(predicted)
        ece += weight * abs(avg_pred - avg_obs)
    return float(ece)


def brier_score(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> float:
    """Brier score: mean squared error of probability predictions.

    Lower is better.  Perfect calibration → 0.  Random guessing → 0.25.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return float(np.mean((predicted - observed) ** 2))


def log_loss(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> float:
    """Binary cross-entropy loss.  Lower is better."""
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-10, 1 - 1e-10)
    observed = np.asarray(observed, dtype=float)
    return float(-np.mean(observed * np.log(predicted) + (1 - observed) * np.log(1 - predicted)))


def reliability_diagram(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
    n_bins: int = 10,
    ax: Any = None,
    label: str = "",
    color: str | None = None,
) -> dict[str, Any]:
    """Reliability diagram data and optional plot.

    Args:
        predicted: Array of predicted probabilities.
        observed: Array of binary outcomes.
        n_bins: Number of bins.
        ax: Matplotlib Axes. If provided, plot on it.
        label: Label for the legend.
        color: Color for the plot line.

    Returns:
        Dict with 'bin_midpoints', 'bin_accuracy', 'bin_counts', 'ece'.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    midpoints: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []

    for lo, hi in pairwise(bin_edges):
        mask = (predicted >= lo) & (predicted < hi)
        c = int(mask.sum())
        counts.append(c)
        midpoints.append((lo + hi) / 2)
        accuracies.append(float(observed[mask].mean()) if c > 0 else float("nan"))

    result: dict[str, Any] = {
        "bin_midpoints": np.array(midpoints),
        "bin_accuracy": np.array(accuracies),
        "bin_counts": np.array(counts),
        "ece": expected_calibration_error(predicted, observed, n_bins),
    }

    if ax is not None:
        _plot_reliability(ax, result, label, color)

    return result


def _plot_reliability(ax: Any, result: dict[str, Any], label: str, color: str | None) -> None:
    mid = result["bin_midpoints"]
    acc = result["bin_accuracy"]
    mask = ~np.isnan(acc)
    lbl = f"{label} (ECE={result['ece']:.3f})" if label else f"ECE={result['ece']:.3f}"
    ax.plot(mid[mask], acc[mask], "o-", label=lbl, color=color, markersize=5)
    existing_labels = [line.get_label() for line in ax.get_lines()]
    if "Perfect" not in " ".join(existing_labels):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")


def compare_calibration(
    methods: dict[str, NDArray[np.float64]],
    observed: NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """Compare calibration across multiple prediction methods.

    Args:
        methods: Dict mapping method name to predicted probabilities.
        observed: Binary outcomes.
        n_bins: Number of bins for ECE calculation.

    Returns:
        Dict of {name: {ece, brier, logloss}}.
    """
    results: dict[str, dict[str, float]] = {}
    for name, pred in methods.items():
        results[name] = {
            "ece": expected_calibration_error(pred, observed, n_bins),
            "brier": brier_score(pred, observed),
            "logloss": log_loss(pred, observed),
        }
    return results
