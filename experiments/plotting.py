

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _facies_cmap(num_classes: int) -> ListedColormap:
    base = plt.cm.get_cmap("tab20", max(num_classes, 3))
    return ListedColormap(base(np.arange(max(num_classes, 3))))


def plot_taxonomy_heatmaps(
    m21: np.ndarray,
    m32: np.ndarray,
    output_path: str | Path,
    primary_names: Sequence[str] | None = None,
    secondary_names: Sequence[str] | None = None,
    tertiary_names: Sequence[str] | None = None,
) -> None:
    output = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, matrix, title, xlabels, ylabels in [
        (axes[0], m21, "Secondary -> Primary", secondary_names, primary_names),
        (axes[1], m32, "Tertiary -> Secondary", tertiary_names, secondary_names),
    ]:
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        if xlabels is not None:
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
        if ylabels is not None:
            ax.set_yticklabels(ylabels, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_well_profile_panel(
    depths: Sequence[float],
    logs: np.ndarray,
    truth: Sequence[int],
    predictions: Mapping[str, Sequence[int]],
    output_path: str | Path,
    well_name: str,
    class_names: Sequence[str] | None = None,
    log_names: Sequence[str] | None = None,
) -> None:
    output = _ensure_parent(output_path)
    depths = np.asarray(depths)
    logs = np.asarray(logs)
    truth = np.asarray(truth, dtype=int)
    method_names = list(predictions.keys())
    num_panels = 2 + len(method_names)
    fig, axes = plt.subplots(1, num_panels, figsize=(2.2 * num_panels, 9), sharey=True)
    if num_panels == 1:
        axes = [axes]

    # Log tracks
    ax = axes[0]
    if logs.ndim == 1:
        logs = logs[:, None]
    for idx in range(logs.shape[1]):
        label = log_names[idx] if log_names and idx < len(log_names) else f"Log {idx + 1}"
        ax.plot(logs[:, idx], depths, lw=1.0, label=label)
    ax.invert_yaxis()
    ax.set_title("Input logs")
    ax.set_xlabel("Scaled value")
    ax.set_ylabel("Depth")
    ax.legend(loc="best", fontsize=7)

    # Truth and predictions
    strip_cmap = _facies_cmap(max(int(truth.max()) + 1 if truth.size else 3, 3))
    panels = [("Ground truth", truth)] + [(name, np.asarray(pred, dtype=int)) for name, pred in predictions.items()]
    for axis, (title, labels) in zip(axes[1:], panels):
        axis.imshow(labels[None, :], aspect="auto", cmap=strip_cmap, interpolation="nearest",
                    vmin=0, vmax=max(strip_cmap.N - 1, 1), extent=[0, len(labels), depths.max(), depths.min()])
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(f"{well_name} lithology reconstruction", y=0.99, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_grid(
    matrices: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> None:
    output = _ensure_parent(output_path)
    method_names = list(matrices.keys())
    ncols = min(3, len(method_names))
    nrows = int(np.ceil(len(method_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, name in zip(axes, method_names):
        matrix = np.asarray(matrices[name], dtype=float)
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j] * 100:.2f}%", ha="center", va="center", fontsize=6, color="white")
    for ax in axes[len(method_names):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(
    table: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: str | Path,
    title: str,
) -> None:
    output = _ensure_parent(output_path)
    table = np.asarray(table, dtype=float)
    x = np.arange(table.shape[0])
    width = 0.8 / max(table.shape[1], 1)
    fig, ax = plt.subplots(figsize=(1.2 * table.shape[0] + 4, 5.5))

    for idx, metric in enumerate(col_labels):
        ax.bar(x + idx * width, table[:, idx], width=width, label=metric)
    ax.set_xticks(x + width * (table.shape[1] - 1) / 2)
    ax.set_xticklabels(row_labels, rotation=30, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_curve(
    table: np.ndarray,
    x_values: Sequence[float],
    metric_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> None:
    output = _ensure_parent(output_path)
    table = np.asarray(table, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for idx, metric in enumerate(metric_names):
        ax.plot(x_values, table[:, idx], marker="o", lw=1.8, label=metric)
    ax.set_xlabel("Setting")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_per_well_metrics(
    frame,
    output_path: str | Path,
) -> None:

    import pandas as pd

    output = _ensure_parent(output_path)
    frame = pd.DataFrame(frame)
    if frame.empty:
        return
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(frame)), 5))
    f1_values = frame["f1_macro"].to_numpy(dtype=float) * 100.0
    ax.plot(x, f1_values, marker="o", lw=1.8, label="Macro-F1")
    for idx, (_, row) in enumerate(frame.iterrows()):
        accuracy = row.get("accuracy", 0.0) * 100.0
        samples = int(row.get("samples", 0))
        ax.annotate(
            f"F1 {f1_values[idx]:.1f}%\nacc {accuracy:.1f}%\nn={samples}",
            (idx, f1_values[idx]),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(item) for item in frame["well_id"]], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Per-well test performance of the proposed model")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_summary(
    frame,
    output_path: str | Path,
) -> None:

    import pandas as pd

    output = _ensure_parent(output_path)
    frame = pd.DataFrame(frame)
    if frame.empty:
        return
    methods = list(dict.fromkeys(frame["method"].tolist()))
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(methods)), 5))
    for idx, method in enumerate(methods):
        part = frame[frame["method"] == method]
        values = part["f1"].to_numpy(dtype=float)
        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter([idx] * len(values) + jitter, values, s=40, alpha=0.85, label=method)
        ax.scatter([idx], [values.mean()], marker="D", s=90, color="black", zorder=5)
        ax.errorbar([idx], [values.mean()], yerr=[values.std()], fmt="none", color="black", lw=1.5, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Leave-one-region-out macro-F1 across sub-basins (A, B, C)")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_window_sensitivity(
    window_sizes: Sequence[float],
    accuracy: Sequence[float],
    macro_f1: Sequence[float],
    output_path: str | Path,
) -> None:

    output = _ensure_parent(output_path)
    window_sizes = np.asarray(window_sizes, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)
    macro_f1 = np.asarray(macro_f1, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(window_sizes, accuracy, marker="o", lw=1.8, label="Accuracy")
    ax.plot(window_sizes, macro_f1, marker="s", lw=1.8, label="Macro-F1")
    if len(accuracy):
        best = int(np.argmax(accuracy))
        ax.annotate(
            f"{accuracy[best]:.1f}% at {int(window_sizes[best])} samples",
            (window_sizes[best], accuracy[best]),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9,
        )
    ax.set_xlabel("Window size (samples)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Sensitivity to the sliding-window size")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_heatmap(
    summary_frame,
    x_values: Sequence[float],
    y_values: Sequence[float],
    output_path: str | Path,
    metric_column: str = "f1_mean",
    x_label: str = "lambda_cons",
    y_label: str = "lambda_auc",
) -> None:
 
    import pandas as pd

    output = _ensure_parent(output_path)
    frame = pd.DataFrame(summary_frame)
    if frame.empty:
        return
    x_values = [float(value) for value in x_values]
    y_values = [float(value) for value in y_values]
    table = np.full((len(y_values), len(x_values)), np.nan)
    name_to_metric = dict(zip(frame["method"].tolist(), frame[metric_column].tolist()))
    for row_idx, y_value in enumerate(y_values):
        for col_idx, x_value in enumerate(x_values):
            key = f"cons_{x_value:.2f}_auc_{y_value:.2f}"
            table[row_idx, col_idx] = name_to_metric.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(0.9 * len(x_values) + 3, 0.8 * len(y_values) + 2))
    im = ax.imshow(table, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{value:g}" for value in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{metric_column} over the regulariser grid")
    for row_idx in range(len(y_values)):
        for col_idx in range(len(x_values)):
            value = table[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
