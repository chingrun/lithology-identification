

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Optional, Sequence, Tuple


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:

    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true).float()
    return correct.mean().item()


def intersectionAndUnion(
    output: torch.Tensor,
    target: torch.Tensor,
    K: int,
    ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape

    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]

    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict:

    acc = accuracy_score(targets, predictions)
    precision_macro = precision_score(targets, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(targets, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(targets, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(targets, predictions, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(targets, predictions)

    return {
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'balanced_accuracy': bal_acc
    }


def absent_class_leakage(
    predictions: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    num_classes: Optional[int] = None,
) -> float:

    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) == 0:
        return 0.0

    if num_classes is None:
        num_classes = int(max(predictions.max(initial=0), targets.max(initial=0))) + 1

    present = set(np.unique(targets).tolist())
    absent = [class_idx for class_idx in range(num_classes) if class_idx not in present]
    if not absent:
        return 0.0

    matrix = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    row_totals = matrix.sum(axis=1)
    row_totals[row_totals == 0] = 1.0
    leaked = matrix[:, absent].sum(axis=1) / row_totals
    present_rows = [class_idx for class_idx in sorted(present) if row_totals[class_idx] > 0 or class_idx < len(leaked)]
    if not present_rows:
        return 0.0
    return float(np.mean([leaked[idx] for idx in present_rows]))


def mean_confidence_interval(
    values: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0

    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0
    # Normal approximation used throughout the manuscript (1.96 at 95%).
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_value = z_scores.get(round(confidence, 2), 1.96)
    half_width = float(z_value * values.std(ddof=1) / np.sqrt(values.size))
    return mean, half_width
