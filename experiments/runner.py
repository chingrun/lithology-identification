

from __future__ import annotations

if __package__ is None or __package__ == "":  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

from losses import DynamicHierarchicalLoss, FocalLoss
from models import (
    DirectTertiaryLithologyModel,
    FlatMultiTaskLithologyModel,
    HierarchicalLithologyModel,
    build_baseline,
)
from data.preprocessing import MinMaxFeatureScaler
from utils.metrics import absent_class_leakage

from experiments.core import (
    WINDOW_SIZE,
    center_profile_view,
    classification_metrics,
    collect_labels,
    collect_window_features,
    build_sampler,
    class_counts,
    TRAIN_WELLS,
    VAL_WELLS,
    compute_leave_one_region_out_splits,
    compute_repeated_well_splits,
    count_parameters,
    fit_scale_split,
    hierarchical_class_weights,
    infer_well_role,
    load_well_datasets,
    make_dataloader,
    measure_inference_time,
    merge_datasets,
    normalized_confusion,
    per_well_metrics,
    predict_dataset,
    scale_windowed_dataset,
    select_center_logits,
    split_datasets,
    summarize_split,
    subset_by_well_ids,
    train_one_epoch,
    evaluate_model,
    well_summary_frame,
)
from experiments.plotting import (
    plot_confusion_grid,
    plot_lambda_heatmap,
    plot_metric_bars,
    plot_per_well_metrics,
    plot_region_summary,
    plot_sensitivity_curve,
    plot_taxonomy_heatmaps,
    plot_well_profile_panel,
    plot_window_sensitivity,
)
from experiments.taxonomy import TaxonomySpec, hierarchy_report



BASELINE_MODEL_NAMES = ("resnet", "bilstm", "bigru", "gat", "transformer", "shap")

BASELINE_RECIPES = [
    {"name": "ResNet", "model": "resnet"},
    {"name": "BiLSTM", "model": "bilstm"},
    {"name": "BiGRU", "model": "bigru"},
    {"name": "GAT", "model": "gat"},
    {"name": "Transformer", "model": "transformer"},
    {"name": "SHAP (contrastive Transformer)", "model": "shap"},
]


DEFAULT_MODEL_KWARGS = {
    "feat_scales": [0, 1, 2, 3],
    "feature_outchannel": [16, 32, 64, 128],
    "time_steps": [0, 0, 0, 0],
    "channel_multiplier": [1, 2, 4, 8],
    "fd": 6,
}


# Table 3 of the manuscript: the six weight strategies S1-S6.
WEIGHT_STRATEGIES = [
    {"name": "S1", "model": "proposed", "task_weights": (1.0, 1.0, 1.0), "lambda_cons": 0.5, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
    {"name": "S2", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.5, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
    {"name": "S3", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.05, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
    {"name": "S4", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 1.5, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
    {"name": "S5", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.5, "lambda_auc": 0.05, "mode": "DHLF", "uniform_weights": False},
    {"name": "S6", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.5, "lambda_auc": 1.5, "mode": "DHLF", "uniform_weights": False},
]

# Table 5 of the manuscript: loss-component ablations I-V. Variants I and IV
# keep the hierarchical encoder-decoder and only replace the objective.
LOSS_ABLATIONS = [
    {"name": "I", "model": "proposed", "task_weights": (1.0, 1.0, 1.0), "lambda_cons": 0.0, "lambda_auc": 0.0, "mode": "NOFOCAL", "uniform_weights": True},
    {"name": "II", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.5, "lambda_auc": 0.0, "mode": "DHLF", "uniform_weights": False},
    {"name": "III", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.0, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
    {"name": "IV", "model": "proposed", "task_weights": (1.0, 1.0, 1.0), "lambda_cons": 0.0, "lambda_auc": 0.0, "mode": "NOFOCAL", "uniform_weights": False},
    {"name": "V", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.5, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
]

TRAINING_STRATEGIES = [
    {"name": "I-direct-tertiary", "model": "direct_tertiary", "focal": True, "uniform_weights": False},
    {"name": "II-hierarchical", "model": "proposed", "task_weights": (1.0, 1.2, 1.5), "lambda_cons": 0.5, "lambda_auc": 0.1, "mode": "DHLF", "uniform_weights": False},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run major-revision lithology experiments.")
    parser.add_argument("command", choices=[
        "taxonomy-report", "loss-sweep", "ablation", "training-strategy",
        "benchmark", "lambda-sweep", "window-sweep", "leave-one-region-out",
        "qualitative",
    ])
    parser.add_argument("--data_path", required=True, help="Path to the input well-log archive or directory.")
    parser.add_argument("--taxonomy_json", help="JSON file defining the primary-secondary-tertiary hierarchy.")
    parser.add_argument("--output_dir", default="revision_outputs", help="Directory for result tables and figures.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--window_sizes", default="5,10,15,20,25,30,35",
                        help="Window sizes evaluated by the window-sweep command (Fig. 17).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--num_primary", type=int, default=6)
    parser.add_argument("--num_secondary", type=int, default=10)
    parser.add_argument("--num_tertiary", type=int, default=26)
    parser.add_argument("--resample_interval", type=float, default=None,
                        help="Depth-grid resampling interval in m (manuscript: 0.125).")
    parser.add_argument("--hampel_window", type=int, default=None,
                        help="Hampel outlier filter window (manuscript: 5).")
    parser.add_argument("--hampel_sigma", type=float, default=3.0)
    parser.add_argument("--lambda_cons_values",
                        default="0.02,0.05,0.1,0.2,0.5,0.8,1.0,1.5",
                        help="Regulariser grid of Fig. 8 (48 configurations by default).")
    parser.add_argument("--lambda_auc_values",
                        default="0.0,0.05,0.1,0.3,0.5,1.5",
                        help="AUC-weight grid of Fig. 8 (48 configurations by default).")
    parser.add_argument("--figure_wells", default="", help="Comma-separated well IDs for profile/confusion figures.")
    parser.add_argument(
        "--decoding_mode",
        choices=["teacher", "self", "constrained"],
        default="constrained",
        help="Parent-conditioning protocol: teacher = GT parents during training (setting A), "
             "self = predicted parents during training/inference (setting B), "
             "constrained = GT parents during training + constrained decoding at inference (setting C, deployed).",
    )
    return parser.parse_args()


def load_taxonomy(args: argparse.Namespace) -> Optional[TaxonomySpec]:
    if not args.taxonomy_json:
        return None
    taxonomy = TaxonomySpec.from_json(args.taxonomy_json)
    taxonomy.validate()
    return taxonomy


def model_kwargs(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    kwargs = dict(DEFAULT_MODEL_KWARGS)
    kwargs.update(
        {
            "device": device,
            "in_channels": args.in_channels,
            "num_primary": args.num_primary,
            "num_secondary": args.num_secondary,
            "num_tertiary": args.num_tertiary,
            "num_points": args.window_size,
        }
    )
    return kwargs


def instantiate_model(recipe: Mapping[str, Any], args: argparse.Namespace, device: torch.device) -> nn.Module:
    kwargs = model_kwargs(args, device)
    if recipe["model"] == "flat_multitask":
        return FlatMultiTaskLithologyModel(**kwargs).to(device)
    if recipe["model"] == "direct_tertiary":
        return DirectTertiaryLithologyModel(**kwargs).to(device)
    if recipe["model"] in BASELINE_MODEL_NAMES:
        return build_baseline(
            recipe["model"],
            in_channels=args.in_channels,
            num_classes=args.num_tertiary,
            window_size=args.window_size,
        ).to(device)
    return HierarchicalLithologyModel(**kwargs).to(device)


def build_class_weights(recipe: Mapping[str, Any], train_dataset, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    if recipe.get("uniform_weights", False):
        return {
            "lvl1": torch.ones(args.num_primary, dtype=torch.float32),
            "lvl2": torch.ones(args.num_secondary, dtype=torch.float32),
            "lvl3": torch.ones(args.num_tertiary, dtype=torch.float32),
        }
    labels_p, labels_s, labels_t = collect_labels(train_dataset)
    return hierarchical_class_weights(
        labels_p,
        labels_s,
        labels_t,
        args.num_primary,
        args.num_secondary,
        args.num_tertiary,
    )


def record_class_distribution(
    rows: List[Dict[str, Any]],
    dataset,
    subset_name: str,
    split: Optional[Any],
    args: argparse.Namespace,
    taxonomy: Optional[TaxonomySpec],
) -> None:
    if hasattr(dataset, "__len__") and len(dataset) == 0:
        return
    labels_p, labels_s, labels_t = collect_labels(dataset)
    level_specs = [
        ("primary", labels_p, args.num_primary, taxonomy.primary if taxonomy is not None else [f"P{i}" for i in range(args.num_primary)]),
        ("secondary", labels_s, args.num_secondary, taxonomy.secondary if taxonomy is not None else [f"S{i}" for i in range(args.num_secondary)]),
        ("tertiary", labels_t, args.num_tertiary, taxonomy.tertiary if taxonomy is not None else [f"T{i}" for i in range(args.num_tertiary)]),
    ]
    split_meta = {"repeat_id": getattr(split, "repeat_id", None), "fold_id": getattr(split, "fold_id", None), "subset": subset_name}
    for level_name, labels, num_classes, class_names in level_specs:
        counts = class_counts(labels, num_classes)
        total = max(sum(counts.values()), 1)
        for class_id, count in counts.items():
            rows.append(
                {
                    **split_meta,
                    "level": level_name,
                    "class_id": class_id,
                    "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                    "count": count,
                    "share": count / total,
                }
            )


def instantiate_hierarchical_loss(
    recipe: Mapping[str, Any],
    args: argparse.Namespace,
    taxonomy: Optional[TaxonomySpec],
    class_weights: Dict[str, torch.Tensor],
) -> DynamicHierarchicalLoss:
    m21 = None
    m32 = None
    if taxonomy is not None:
        m21, m32 = taxonomy.matrices()
    return DynamicHierarchicalLoss(
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        num_tertiary=args.num_tertiary,
        task_weights=recipe.get("task_weights", (1.0, 1.0, 1.0)),
        lambda_consistency=recipe.get("lambda_cons", 0.0),
        lambda_auc=recipe.get("lambda_auc", 0.0),
        class_weights=class_weights,
        M21=m21,
        M32=m32,
        mode=recipe.get("mode", "DHLF"),
    )


def instantiate_direct_loss(
    recipe: Mapping[str, Any],
    class_weights: Dict[str, torch.Tensor],
) -> nn.Module:
    if recipe.get("focal", True):
        return FocalLoss(gamma=2.0, weight=class_weights["lvl3"], reduction="mean", ignore_index=-1, label_smoothing=0.0)
    return nn.CrossEntropyLoss(weight=class_weights["lvl3"])


def fit_hierarchical_model(
    model: nn.Module,
    criterion: nn.Module,
    train_dataset,
    val_dataset,
    args: argparse.Namespace,
    device: torch.device,
    teacher_forcing: bool = False,
    constrained: bool = False,
    taxonomy=None,
    use_curriculum: bool = True,
) -> nn.Module:

    sampler = None
    if len(train_dataset) > 0:
        sampler = build_sampler(train_dataset)
    train_loader = make_dataloader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = make_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if not use_curriculum:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_state = deepcopy(model.state_dict())
        best_score = -float("inf")
        stale_epochs = 0

        for _ in range(args.epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device, teacher_forcing=teacher_forcing)
            _, _, _, metrics_t, _ = evaluate_model(model, val_loader, criterion, device,
                                                   taxonomy=taxonomy, constrained=constrained)
            score = metrics_t["f1_macro"]
            if score > best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break

        model.load_state_dict(best_state)
        return model

    from utils.training_utils import CurriculumStage, run_curriculum_training

    # Scale the Table 2 epoch budgets by the CLI --epochs budget (default 100
    # equals the full schedule: 30/30/40/100).
    scale = max(args.epochs, 1) / 100.0

    def _evaluate(model, dataloader, criterion, device, constrained=True, taxonomy=None):
        result = evaluate_model(model, dataloader, criterion, device,
                                taxonomy=taxonomy, constrained=constrained)
        return result[0], result[1], result[2], result[3]

    stages = [
        CurriculumStage(1, ("encoder", "decoder", "primary"), args.lr,
                        max(1, int(30 * scale)), args.patience, "primary"),
        CurriculumStage(2, ("secondary",), args.lr,
                        max(1, int(30 * scale)), args.patience, "secondary"),
        CurriculumStage(3, ("tertiary",), (max(args.lr / 10.0, 1e-5), args.lr),
                        max(1, int(40 * scale)), args.patience, "tertiary"),
        CurriculumStage(4, ("encoder", "decoder", "primary", "secondary", "tertiary"),
                        args.lr / 2.0, max(1, int(100 * scale)), args.patience, "tertiary"),
    ]
    run_curriculum_training(
        model, criterion, train_loader, val_loader, device,
        stages=stages,
        train_epoch=train_one_epoch,
        evaluate=_evaluate,
        teacher_forcing=teacher_forcing,
        constrained=constrained,
        taxonomy=taxonomy,
        seed=args.seed,
    )
    return model


def train_direct_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in dataloader:
        data, _, _, labels_t = batch[:4]
        data = data.to(device)
        labels_t = labels_t.to(device)

        optimizer.zero_grad()
        logits_t = select_center_logits(model(data))
        loss = criterion(logits_t, labels_t)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate_direct_model(
    model: nn.Module,
    dataset,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    if hasattr(dataset, "__len__") and len(dataset) == 0:
        return {
            "loss": 0.0,
            "metrics": classification_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "predictions": np.array([], dtype=np.int64),
            "labels": np.array([], dtype=np.int64),
            "depths": None,
            "well_ids": None,
        }

    if hasattr(dataset, "datasets"):
        parts = [evaluate_direct_model(model, part, criterion, device, batch_size) for part in dataset.datasets]
        preds = np.concatenate([part["predictions"] for part in parts])
        labels = np.concatenate([part["labels"] for part in parts])
        depths = np.concatenate([part["depths"] for part in parts if part["depths"] is not None]) if any(part["depths"] is not None for part in parts) else None
        wells = np.concatenate([part["well_ids"] for part in parts if part["well_ids"] is not None]) if any(part["well_ids"] is not None for part in parts) else None
        loss = float(np.mean([part["loss"] for part in parts])) if parts else 0.0
        return {"loss": loss, "metrics": classification_metrics(preds, labels), "predictions": preds, "labels": labels, "depths": depths, "well_ids": wells}

    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    batches = 0
    predictions, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            data, _, _, labels_t = batch[:4]
            data = data.to(device)
            labels_t = labels_t.to(device)
            logits_t = select_center_logits(model(data))
            loss = criterion(logits_t, labels_t)
            total_loss += float(loss.item())
            batches += 1
            predictions.append(logits_t.argmax(dim=1).cpu().numpy())
            labels_all.append(labels_t.cpu().numpy())

    preds = np.concatenate(predictions).flatten()
    labels = np.concatenate(labels_all).flatten()
    return {
        "loss": total_loss / max(batches, 1),
        "metrics": classification_metrics(preds, labels),
        "predictions": preds,
        "labels": labels,
        "depths": getattr(dataset, "depths", None),
        "well_ids": np.asarray([dataset.well_id] * len(labels), dtype=object) if hasattr(dataset, "well_id") else None,
    }


def fit_direct_model(
    model: nn.Module,
    criterion: nn.Module,
    train_dataset,
    val_dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:

    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    stale_epochs = 0

    for _ in range(args.epochs):
        train_direct_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_direct_model(model, val_dataset, criterion, device, args.batch_size)["metrics"]
        score = metrics["f1_macro"]
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    return model


def summarize_repeated_results(frame: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    grouped = frame.groupby("method", as_index=False)
    rows: List[Dict[str, Any]] = []
    for method, part in grouped:
        row: Dict[str, Any] = {"method": method, "runs": len(part)}
        for metric in metric_columns:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=0)
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / np.sqrt(max(len(part), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def save_prediction_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = {}
    for key, value in payload.items():
        if value is None:
            continue
        sanitized[key] = np.asarray(value)
    np.savez_compressed(path, **sanitized)


def run_torch_suite(
    recipes: Sequence[Mapping[str, Any]],
    datasets,
    taxonomy: Optional[TaxonomySpec],
    args: argparse.Namespace,
    output_dir: Path,
    profile_wells: Sequence[str],
) -> pd.DataFrame:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = compute_repeated_well_splits(datasets, n_splits=args.folds, n_repeats=args.repeats, val_fraction=args.val_fraction, seed=args.seed)
    raw_rows: List[Dict[str, Any]] = []
    per_well_rows: List[pd.DataFrame] = []
    split_rows: List[pd.DataFrame] = [summarize_split(datasets, split) for split in splits]
    class_rows: List[Dict[str, Any]] = []
    for split in splits:
        train_dataset, val_dataset, test_dataset = split_datasets(datasets, split)
        record_class_distribution(class_rows, train_dataset, "train", split, args, taxonomy)
        record_class_distribution(class_rows, val_dataset, "val", split, args, taxonomy)
        record_class_distribution(class_rows, test_dataset, "test", split, args, taxonomy)
    figure_store: Dict[str, Dict[str, Any]] = {}

    for recipe in recipes:
        for split in splits:
            # Min-Max fitted on the training wells only (Table 1).
            train_dataset, val_dataset, test_dataset, _scaler = fit_scale_split(datasets, split)
            class_weights = build_class_weights(recipe, train_dataset, args)
            model = instantiate_model(recipe, args, device)

            if recipe["model"] == "direct_tertiary":
                criterion = instantiate_direct_loss(recipe, class_weights).to(device)
                model = fit_direct_model(model, criterion, train_dataset, val_dataset, args, device)
                eval_result = evaluate_direct_model(model, test_dataset, criterion, device, args.batch_size)
                metrics_t = eval_result["metrics"]
                inference_ms = measure_inference_time(model, test_dataset, device, batch_size=args.batch_size) * 1000.0
                row = {
                    "method": recipe["name"],
                    "repeat_id": split.repeat_id,
                    "fold_id": split.fold_id,
                    "accuracy": metrics_t["accuracy"] * 100.0,
                    "precision": metrics_t["precision_macro"] * 100.0,
                    "recall": metrics_t["recall_macro"] * 100.0,
                    "f1": metrics_t["f1_macro"] * 100.0,
                    "params": count_parameters(model),
                    "inference_ms": inference_ms,
                    "invalid_path_rate": np.nan,
                    "leakage": absent_class_leakage(
                        eval_result["predictions"], eval_result["labels"], args.num_tertiary
                    ) * 100.0,
                }
                raw_rows.append(row)

                if profile_wells:
                    datasets_to_check = test_dataset.datasets if hasattr(test_dataset, "datasets") else [test_dataset]
                    for well_dataset in datasets_to_check:
                        if well_dataset.well_id not in profile_wells:
                            continue
                        profile = center_profile_view(well_dataset)
                        local = evaluate_direct_model(model, well_dataset, criterion, device, args.batch_size)
                        figure_store.setdefault(well_dataset.well_id, {"depths": profile["depths"], "logs": profile["logs"], "truth": profile["labels_t"], "predictions": {}, "confusions": {}})
                        figure_store[well_dataset.well_id]["predictions"][recipe["name"]] = local["predictions"]
                        figure_store[well_dataset.well_id]["confusions"][recipe["name"]] = normalized_confusion(
                            local["predictions"],
                            local["labels"],
                            args.num_tertiary,
                        )

                save_prediction_artifact(
                    output_dir / "predictions" / f"{recipe['name']}_repeat{split.repeat_id}_fold{split.fold_id}.npz",
                    {
                        "predictions_t": eval_result["predictions"],
                        "labels_t": eval_result["labels"],
                        "well_ids": eval_result["well_ids"],
                        "depths": eval_result["depths"],
                    },
                )
                continue

            if recipe["model"] in BASELINE_MODEL_NAMES:
                criterion = nn.CrossEntropyLoss()
                if recipe["model"] == "shap":
                    features = collect_window_features(train_dataset)
                    labels_t = collect_labels(train_dataset)[2]
                    model.fit_feature_importance(
                        torch.as_tensor(features, dtype=torch.float32),
                        torch.as_tensor(labels_t, dtype=torch.long),
                    )
                model = fit_direct_model(model, criterion, train_dataset, val_dataset, args, device)
                eval_result = evaluate_direct_model(model, test_dataset, criterion, device, args.batch_size)
                metrics_t = eval_result["metrics"]
                inference_ms = measure_inference_time(model, test_dataset, device, batch_size=args.batch_size) * 1000.0
                raw_rows.append(
                    {
                        "method": recipe["name"],
                        "repeat_id": split.repeat_id,
                        "fold_id": split.fold_id,
                        "accuracy": metrics_t["accuracy"] * 100.0,
                        "precision": metrics_t["precision_macro"] * 100.0,
                        "recall": metrics_t["recall_macro"] * 100.0,
                        "f1": metrics_t["f1_macro"] * 100.0,
                        "params": count_parameters(model),
                        "inference_ms": inference_ms,
                        "invalid_path_rate": np.nan,
                        "leakage": absent_class_leakage(
                            eval_result["predictions"], eval_result["labels"], args.num_tertiary
                        ) * 100.0,
                    }
                )
                if profile_wells:
                    datasets_to_check = test_dataset.datasets if hasattr(test_dataset, "datasets") else [test_dataset]
                    for well_dataset in datasets_to_check:
                        if well_dataset.well_id not in profile_wells:
                            continue
                        profile = center_profile_view(well_dataset)
                        local = evaluate_direct_model(model, well_dataset, criterion, device, args.batch_size)
                        figure_store.setdefault(well_dataset.well_id, {"depths": profile["depths"], "logs": profile["logs"], "truth": profile["labels_t"], "predictions": {}, "confusions": {}})
                        figure_store[well_dataset.well_id]["predictions"][recipe["name"]] = local["predictions"]
                        figure_store[well_dataset.well_id]["confusions"][recipe["name"]] = normalized_confusion(
                            local["predictions"], local["labels"], args.num_tertiary,
                        )
                save_prediction_artifact(
                    output_dir / "predictions" / f"{recipe['name']}_repeat{split.repeat_id}_fold{split.fold_id}.npz",
                    {
                        "predictions_t": eval_result["predictions"],
                        "labels_t": eval_result["labels"],
                        "well_ids": eval_result["well_ids"],
                        "depths": eval_result["depths"],
                    },
                )
                continue

            criterion = instantiate_hierarchical_loss(recipe, args, taxonomy, class_weights).to(device)
            mode = getattr(args, "decoding_mode", "self")
            teacher_forcing = mode in ("teacher", "constrained")
            constrained = mode == "constrained"
            model = fit_hierarchical_model(model, criterion, train_dataset, val_dataset, args, device,
                                           teacher_forcing=teacher_forcing, constrained=constrained,
                                           taxonomy=taxonomy)
            split_result = predict_dataset(model, test_dataset, device, batch_size=args.batch_size,
                                           criterion=criterion, taxonomy=taxonomy, constrained=constrained)
            metrics_t = classification_metrics(split_result.predictions_t, split_result.labels_t)
            inference_ms = measure_inference_time(model, test_dataset, device, batch_size=args.batch_size) * 1000.0
            hierarchy_stats = (
                hierarchy_report(
                    split_result.predictions_p,
                    split_result.predictions_s,
                    split_result.predictions_t,
                    split_result.labels_p,
                    split_result.labels_s,
                    split_result.labels_t,
                    taxonomy,
                )
                if taxonomy is not None else {}
            )
            raw_rows.append(
                {
                    "method": recipe["name"],
                    "repeat_id": split.repeat_id,
                    "fold_id": split.fold_id,
                    "accuracy": metrics_t["accuracy"] * 100.0,
                    "precision": metrics_t["precision_macro"] * 100.0,
                    "recall": metrics_t["recall_macro"] * 100.0,
                    "f1": metrics_t["f1_macro"] * 100.0,
                    "params": count_parameters(model),
                    "inference_ms": inference_ms,
                    "invalid_path_rate": hierarchy_stats.get("invalid_path_rate", np.nan) * 100.0 if hierarchy_stats else np.nan,
                    "leakage": absent_class_leakage(
                        split_result.predictions_t, split_result.labels_t, args.num_tertiary
                    ) * 100.0,
                }
            )

            per_well = per_well_metrics(split_result)
            if not per_well.empty:
                per_well["method"] = recipe["name"]
                per_well["repeat_id"] = split.repeat_id
                per_well["fold_id"] = split.fold_id
                per_well_rows.append(per_well)

            if profile_wells:
                datasets_to_check = test_dataset.datasets if hasattr(test_dataset, "datasets") else [test_dataset]
                for well_dataset in datasets_to_check:
                    if well_dataset.well_id not in profile_wells:
                        continue
                    profile = center_profile_view(well_dataset)
                    local_bundle = predict_dataset(model, well_dataset, device, batch_size=args.batch_size, criterion=criterion)
                    figure_store.setdefault(well_dataset.well_id, {"depths": profile["depths"], "logs": profile["logs"], "truth": profile["labels_t"], "predictions": {}, "confusions": {}})
                    figure_store[well_dataset.well_id]["predictions"][recipe["name"]] = local_bundle.predictions_t
                    figure_store[well_dataset.well_id]["confusions"][recipe["name"]] = normalized_confusion(
                        local_bundle.predictions_t,
                        local_bundle.labels_t,
                        args.num_tertiary,
                    )

            save_prediction_artifact(
                output_dir / "predictions" / f"{recipe['name']}_repeat{split.repeat_id}_fold{split.fold_id}.npz",
                {
                    "predictions_p": split_result.predictions_p,
                    "predictions_s": split_result.predictions_s,
                    "predictions_t": split_result.predictions_t,
                    "labels_p": split_result.labels_p,
                    "labels_s": split_result.labels_s,
                    "labels_t": split_result.labels_t,
                    "well_ids": split_result.well_ids,
                    "depths": split_result.depths,
                },
            )

    raw_frame = pd.DataFrame(raw_rows)
    raw_frame.to_csv(output_dir / "raw_metrics.csv", index=False)
    if split_rows:
        pd.concat(split_rows, ignore_index=True).to_csv(output_dir / "split_summary.csv", index=False)
    if class_rows:
        pd.DataFrame(class_rows).to_csv(output_dir / "class_distribution.csv", index=False)
    if per_well_rows:
        pd.concat(per_well_rows, ignore_index=True).to_csv(output_dir / "per_well_metrics.csv", index=False)

    if figure_store:
        class_names = taxonomy.tertiary if taxonomy is not None else [f"T{i}" for i in range(args.num_tertiary)]
        for well_id, payload in figure_store.items():
            plot_well_profile_panel(
                payload["depths"],
                payload["logs"],
                payload["truth"],
                payload["predictions"],
                output_dir / "figures" / f"{well_id}_profile.png",
                well_name=well_id,
                class_names=class_names,
                log_names=["GR", "DEN", "AC", "CNL", "RLLS", "RLLD"][:payload["logs"].shape[1]],
            )
            plot_confusion_grid(
                payload["confusions"],
                class_names=class_names,
                output_path=output_dir / "figures" / f"{well_id}_confusion.png",
                title=f"{well_id} comparison confusion matrices",
            )

    return raw_frame


def summarize_window_features(dataset, summary: str = "mean") -> np.ndarray:
    windows = collect_window_features(dataset)
    if summary == "mean":
        return windows.mean(axis=1)
    if summary == "mean_std":
        return np.concatenate([windows.mean(axis=1), windows.std(axis=1)], axis=1)
    raise ValueError(f"Unsupported summary: {summary}")


class HierarchicalExtraTrees:


    def __init__(self, n_estimators: int = 500, max_depth=None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, x, labels_p, labels_s, labels_t):
        self.model_p = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, class_weight="balanced", n_jobs=-1,
        )
        self.model_s = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, class_weight="balanced", n_jobs=-1,
        )
        self.model_t = ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, class_weight="balanced", n_jobs=-1,
        )
        pred_p = self.model_p.fit(x, labels_p).predict(x)
        x_s = np.concatenate([x, pred_p[:, None]], axis=1)
        pred_s = self.model_s.fit(x_s, labels_s).predict(x_s)
        x_t = np.concatenate([x, pred_p[:, None], pred_s[:, None]], axis=1)
        self.model_t.fit(x_t, labels_t)
        return self

    def predict(self, x):
        pred_p = self.model_p.predict(x)
        pred_s = self.model_s.predict(np.concatenate([x, pred_p[:, None]], axis=1))
        return self.model_t.predict(np.concatenate([x, pred_p[:, None], pred_s[:, None]], axis=1))


def available_sklearn_recipes(seed: int, num_classes: int) -> List[Mapping[str, Any]]:
    recipes: List[Mapping[str, Any]] = [
        {"name": "SVM", "estimator": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=seed))},
        {"name": "RandomForest", "estimator": RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)},
        {"name": "Coarse-to-Fine (ER-trees)", "estimator": "er_trees"},
    ]
    if XGBClassifier is not None:
        recipes.append(
            {"name": "XGBoost", "estimator": XGBClassifier(
                max_depth=6,
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                num_class=num_classes,
                tree_method="hist",
                random_state=seed,
            )}
        )
    return recipes


def run_sklearn_suite(
    datasets,
    args: argparse.Namespace,
    output_dir: Path,
    profile_wells: Sequence[str],
) -> pd.DataFrame:
    splits = compute_repeated_well_splits(datasets, n_splits=args.folds, n_repeats=args.repeats, val_fraction=args.val_fraction, seed=args.seed)
    rows: List[Dict[str, Any]] = []
    figure_store: Dict[str, Dict[str, Any]] = {}

    for recipe in available_sklearn_recipes(args.seed, args.num_tertiary):
        for split in splits:
            # Min-Max fitted on the training wells only (Table 1).
            train_dataset, _, test_dataset, _scaler = fit_scale_split(datasets, split)
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                rows.append(
                    {
                        "method": recipe["name"],
                        "repeat_id": split.repeat_id,
                        "fold_id": split.fold_id,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "params": np.nan,
                        "inference_ms": np.nan,
                        "invalid_path_rate": np.nan,
                    }
                )
                continue

            x_train = summarize_window_features(train_dataset, summary="mean")
            x_test = summarize_window_features(test_dataset, summary="mean")
            labels_p_train, labels_s_train, y_train = collect_labels(train_dataset)
            y_test = collect_labels(test_dataset)[2]

            if recipe["estimator"] == "er_trees":
                estimator = HierarchicalExtraTrees(n_estimators=500, random_state=args.seed)
                estimator.fit(x_train, labels_p_train, labels_s_train, y_train)
            else:
                estimator = deepcopy(recipe["estimator"])
                estimator.fit(x_train, y_train)
            start = perf_counter()
            predictions = estimator.predict(x_test)
            inference_ms = (perf_counter() - start) * 1000.0 / max(len(x_test), 1)
            metrics = classification_metrics(predictions, y_test)

            rows.append(
                {
                    "method": recipe["name"],
                    "repeat_id": split.repeat_id,
                    "fold_id": split.fold_id,
                    "accuracy": metrics["accuracy"] * 100.0,
                    "precision": metrics["precision_macro"] * 100.0,
                    "recall": metrics["recall_macro"] * 100.0,
                    "f1": metrics["f1_macro"] * 100.0,
                    "params": np.nan,
                    "inference_ms": inference_ms,
                    "invalid_path_rate": np.nan,
                    "leakage": absent_class_leakage(predictions, y_test, args.num_tertiary) * 100.0,
                }
            )

            if profile_wells:
                datasets_to_check = test_dataset.datasets if hasattr(test_dataset, "datasets") else [test_dataset]
                offset = 0
                for well_dataset in datasets_to_check:
                    local_features = summarize_window_features(well_dataset, summary="mean")
                    local_labels = collect_labels(well_dataset)[2]
                    local_predictions = estimator.predict(local_features)
                    if well_dataset.well_id not in profile_wells:
                        offset += len(local_labels)
                        continue
                    profile = center_profile_view(well_dataset)
                    figure_store.setdefault(well_dataset.well_id, {"depths": profile["depths"], "logs": profile["logs"], "truth": profile["labels_t"], "predictions": {}, "confusions": {}})
                    figure_store[well_dataset.well_id]["predictions"][recipe["name"]] = local_predictions
                    figure_store[well_dataset.well_id]["confusions"][recipe["name"]] = normalized_confusion(local_predictions, local_labels, args.num_tertiary)
                    offset += len(local_labels)

    raw_frame = pd.DataFrame(rows)
    raw_frame.to_csv(output_dir / "sklearn_metrics.csv", index=False)

    if figure_store:
        class_names = [f"T{i}" for i in range(args.num_tertiary)]
        for well_id, payload in figure_store.items():
            plot_well_profile_panel(
                payload["depths"],
                payload["logs"],
                payload["truth"],
                payload["predictions"],
                output_dir / "figures" / f"{well_id}_classical_profile.png",
                well_name=well_id,
                class_names=class_names,
                log_names=["GR", "DEN", "AC", "CNL", "RLLS", "RLLD"][:payload["logs"].shape[1]],
            )
            plot_confusion_grid(
                payload["confusions"],
                class_names=class_names,
                output_path=output_dir / "figures" / f"{well_id}_classical_confusion.png",
                title=f"{well_id} classical baselines",
            )

    return raw_frame


def run_taxonomy_report(datasets, taxonomy: Optional[TaxonomySpec], args: argparse.Namespace, output_dir: Path) -> None:
    summary = well_summary_frame(datasets)
    summary.to_csv(output_dir / "well_summary.csv", index=False)
    split_frames = [summarize_split(datasets, split) for split in compute_repeated_well_splits(datasets, n_splits=args.folds, n_repeats=1, val_fraction=args.val_fraction, seed=args.seed)]
    if split_frames:
        pd.concat(split_frames, ignore_index=True).to_csv(output_dir / "split_summary.csv", index=False)
    class_rows: List[Dict[str, Any]] = []
    all_dataset = subset_by_well_ids(datasets, summary["well_id"].tolist())
    record_class_distribution(class_rows, all_dataset, "all", None, args, taxonomy)
    if class_rows:
        pd.DataFrame(class_rows).to_csv(output_dir / "class_distribution.csv", index=False)
    if taxonomy is None:
        return

    m21, m32 = taxonomy.matrices()
    plot_taxonomy_heatmaps(
        m21.numpy(),
        m32.numpy(),
        output_dir / "figures" / "taxonomy_matrices.png",
        primary_names=taxonomy.primary,
        secondary_names=taxonomy.secondary,
        tertiary_names=taxonomy.tertiary,
    )

    labels_p, labels_s, labels_t = collect_labels(subset_by_well_ids(datasets, summary["well_id"].tolist()))
    taxonomy.class_table(
        primary_counts={idx: int(val) for idx, val in enumerate(np.bincount(labels_p, minlength=args.num_primary))},
        secondary_counts={idx: int(val) for idx, val in enumerate(np.bincount(labels_s, minlength=args.num_secondary))},
        tertiary_counts={idx: int(val) for idx, val in enumerate(np.bincount(labels_t, minlength=args.num_tertiary))},
        split_name="all",
    ).to_csv(output_dir / "taxonomy_counts.csv", index=False)


def run_lambda_sweep(datasets, taxonomy: Optional[TaxonomySpec], args: argparse.Namespace, output_dir: Path, profile_wells: Sequence[str]) -> None:
    lambda_cons_values = [float(value) for value in args.lambda_cons_values.split(",") if value]
    lambda_auc_values = [float(value) for value in args.lambda_auc_values.split(",") if value]

    recipes = []
    for lambda_cons in lambda_cons_values:
        for lambda_auc in lambda_auc_values:
            recipes.append(
                {
                    "name": f"cons_{lambda_cons:.2f}_auc_{lambda_auc:.2f}",
                    "model": "proposed",
                    "task_weights": (1.0, 1.2, 1.5),
                    "lambda_cons": lambda_cons,
                    "lambda_auc": lambda_auc,
                    "mode": "DHLF",
                    "uniform_weights": False,
                }
            )

    raw = run_torch_suite(recipes, datasets, taxonomy, args, output_dir, profile_wells)
    summary = summarize_repeated_results(raw, ["accuracy", "precision", "recall", "f1", "invalid_path_rate"])
    summary.to_csv(output_dir / "lambda_sweep_summary.csv", index=False)

    if len(lambda_cons_values) > 1 and len(lambda_auc_values) > 1:
        # 2D grid: Fig. 8a heatmap of macro-F1 over lambda_cons x lambda_auc.
        plot_lambda_heatmap(
            summary,
            lambda_cons_values,
            lambda_auc_values,
            output_dir / "figures" / "lambda_grid_heatmap.png",
            metric_column="f1_mean",
            x_label="lambda_cons",
            y_label="lambda_auc",
        )

    if len(lambda_auc_values) == 1:
        ordered = [recipe["name"] for recipe in recipes]
        curve = summary.set_index("method").loc[ordered].reset_index()
        plot_sensitivity_curve(
            np.column_stack([curve["accuracy_mean"], curve["f1_mean"], curve["invalid_path_rate_mean"]]),
            x_values=lambda_cons_values,
            metric_names=["Accuracy", "F1", "Invalid path rate"],
            output_path=output_dir / "figures" / "lambda_cons_sensitivity.png",
            title="Sensitivity to lambda_cons",
        )
    elif len(lambda_cons_values) == 1:
        ordered = [recipe["name"] for recipe in recipes]
        curve = summary.set_index("method").loc[ordered].reset_index()
        plot_sensitivity_curve(
            np.column_stack([curve["accuracy_mean"], curve["f1_mean"], curve["invalid_path_rate_mean"]]),
            x_values=lambda_auc_values,
            metric_names=["Accuracy", "F1", "Invalid path rate"],
            output_path=output_dir / "figures" / "lambda_auc_sensitivity.png",
            title="Sensitivity to lambda_auc",
        )


PROPOSED_RECIPE = {
    "name": "Proposed",
    "model": "proposed",
    "task_weights": (1.0, 1.2, 1.5),
    "lambda_cons": 0.5,
    "lambda_auc": 0.1,
    "mode": "DHLF",
    "uniform_weights": False,
}

FLAT_RECIPE = {
    "name": "Flat multi-task (parameter-matched)",
    "model": "flat_multitask",
    "task_weights": (1.0, 1.0, 1.0),
    "lambda_cons": 0.0,
    "lambda_auc": 0.0,
    "mode": "NOFOCAL",
    "uniform_weights": False,
}


def _scale_by_train_wells(train_dataset, val_dataset, test_dataset):

    if len(train_dataset) == 0:
        return train_dataset, val_dataset, test_dataset, None
    scaler = MinMaxFeatureScaler()
    features = collect_window_features(train_dataset)
    scaler.fit(features.reshape(-1, features.shape[-1]))
    return (
        scale_windowed_dataset(train_dataset, scaler),
        scale_windowed_dataset(val_dataset, scaler),
        scale_windowed_dataset(test_dataset, scaler),
        scaler,
    )


def _fit_recipe(recipe, model, criterion, train_dataset, val_dataset, args, device, taxonomy, class_weights):
   
    if recipe["model"] == "direct_tertiary":
        return fit_direct_model(model, criterion, train_dataset, val_dataset, args, device)
    if recipe["model"] in BASELINE_MODEL_NAMES:
        if recipe["model"] == "shap":
            features = collect_window_features(train_dataset)
            labels_t = collect_labels(train_dataset)[2]
            model.fit_feature_importance(
                torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(labels_t, dtype=torch.long),
            )
        return fit_direct_model(model, criterion, train_dataset, val_dataset, args, device)
    mode = getattr(args, "decoding_mode", "constrained")
    teacher_forcing = mode in ("teacher", "constrained")
    constrained = mode == "constrained"
    return fit_hierarchical_model(
        model, criterion, train_dataset, val_dataset, args, device,
        teacher_forcing=teacher_forcing, constrained=constrained, taxonomy=taxonomy,
    )


def _criterion_for_recipe(recipe, args, taxonomy, class_weights, device):
    if recipe["model"] == "direct_tertiary":
        return instantiate_direct_loss(recipe, class_weights).to(device)
    if recipe["model"] in BASELINE_MODEL_NAMES:
        return nn.CrossEntropyLoss()
    return instantiate_hierarchical_loss(recipe, args, taxonomy, class_weights).to(device)


def _evaluate_recipe(recipe, model, criterion, dataset, args, device, taxonomy):
 
    if recipe["model"] in ("direct_tertiary",) + BASELINE_MODEL_NAMES:
        result = evaluate_direct_model(model, dataset, criterion, device, args.batch_size)
        return (
            result["metrics"],
            result["predictions"],
            result["labels"],
            {},
        )
    mode = getattr(args, "decoding_mode", "constrained")
    constrained = mode == "constrained"
    bundle = predict_dataset(
        model, dataset, device, batch_size=args.batch_size,
        criterion=criterion, taxonomy=taxonomy, constrained=constrained,
    )
    metrics = classification_metrics(bundle.predictions_t, bundle.labels_t)
    hierarchy = (
        hierarchy_report(
            bundle.predictions_p, bundle.predictions_s, bundle.predictions_t,
            bundle.labels_p, bundle.labels_s, bundle.labels_t, taxonomy,
        )
        if taxonomy is not None else {}
    )
    return metrics, bundle.predictions_t, bundle.labels_t, hierarchy


def run_window_sweep(datasets, taxonomy, args, output_dir) -> None:
 
    window_sizes = [int(value) for value in args.window_sizes.split(",") if value]
    rows: List[Dict[str, Any]] = []
    composition: List[Dict[str, Any]] = []

    for window_size in window_sizes:
        if window_size % 2 == 0:
            window_size += 1  # centre-depth labelling needs an odd window
        sweep_args = argparse.Namespace(**{**vars(args), "window_size": window_size})
        window_datasets = load_well_datasets(
            args.data_path, window_size=window_size,
            resample_interval=args.resample_interval,
            hampel_window=args.hampel_window, hampel_sigma=args.hampel_sigma,
        )
        if not window_datasets:
            continue
        sweep_output = output_dir / f"window_{window_size}"
        raw = run_torch_suite([PROPOSED_RECIPE], window_datasets, taxonomy, sweep_args, sweep_output, [])
        if raw.empty:
            continue
        acc = raw["accuracy"].mean()
        f1 = raw["f1"].mean()
        rows.append({"window_size": window_size, "accuracy": acc, "f1": f1})

        predictions_dir = sweep_output / "predictions"
        all_preds, all_labels = [], []
        for artifact in predictions_dir.glob("*.npz"):
            payload = np.load(artifact)
            all_preds.append(payload["predictions_t"])
            all_labels.append(payload["labels_t"])
        if all_preds:
            preds = np.concatenate(all_preds).flatten()
            labels = np.concatenate(all_labels).flatten()
            for class_idx in range(args.num_tertiary):
                composition.append(
                    {
                        "window_size": window_size,
                        "class_id": class_idx,
                        "truth_share": float((labels == class_idx).mean()),
                        "predicted_share": float((preds == class_idx).mean()),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "window_sweep_summary.csv", index=False)
    if composition:
        pd.DataFrame(composition).to_csv(output_dir / "window_sweep_composition.csv", index=False)
    if not frame.empty:
        plot_window_sensitivity(
            frame["window_size"].to_numpy(),
            frame["accuracy"].to_numpy(),
            frame["f1"].to_numpy(),
            output_dir / "figures" / "window_sensitivity.png",
        )


def run_region_suite(datasets, taxonomy, args, output_dir) -> None:

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = compute_leave_one_region_out_splits(datasets, val_fraction=args.val_fraction, seed=args.seed)
    torch_recipes = [PROPOSED_RECIPE, FLAT_RECIPE] + BASELINE_RECIPES
    rows: List[Dict[str, Any]] = []

    for region_split in splits:
        train_dataset, val_dataset, test_dataset = split_datasets(datasets, region_split)
        train_dataset, val_dataset, test_dataset, _scaler = _scale_by_train_wells(
            train_dataset, val_dataset, test_dataset
        )

        # Classical ML baselines on window-averaged features.
        x_train = summarize_window_features(train_dataset, summary="mean")
        labels_p_train, labels_s_train, y_train = collect_labels(train_dataset)
        x_test = summarize_window_features(test_dataset, summary="mean")
        y_test = collect_labels(test_dataset)[2]
        for recipe in available_sklearn_recipes(args.seed, args.num_tertiary):
            if recipe["estimator"] == "er_trees":
                estimator = HierarchicalExtraTrees(n_estimators=500, random_state=args.seed)
                estimator.fit(x_train, labels_p_train, labels_s_train, y_train)
            else:
                estimator = deepcopy(recipe["estimator"])
                estimator.fit(x_train, y_train)
            predictions = estimator.predict(x_test)
            metrics = classification_metrics(predictions, y_test)
            rows.append(
                {
                    "region": region_split.region,
                    "method": recipe["name"],
                    "accuracy": metrics["accuracy"] * 100.0,
                    "precision": metrics["precision_macro"] * 100.0,
                    "recall": metrics["recall_macro"] * 100.0,
                    "f1": metrics["f1_macro"] * 100.0,
                    "leakage": absent_class_leakage(predictions, y_test, args.num_tertiary) * 100.0,
                }
            )

        # Deep baselines and the proposed model.
        for recipe in torch_recipes:
            class_weights = build_class_weights(recipe, train_dataset, args)
            model = instantiate_model(recipe, args, device)
            criterion = _criterion_for_recipe(recipe, args, taxonomy, class_weights, device)
            model = _fit_recipe(
                recipe, model, criterion, train_dataset, val_dataset, args, device, taxonomy, class_weights
            )
            metrics, predictions, labels, hierarchy = _evaluate_recipe(
                recipe, model, criterion, test_dataset, args, device, taxonomy
            )
            rows.append(
                {
                    "region": region_split.region,
                    "method": recipe["name"],
                    "accuracy": metrics["accuracy"] * 100.0,
                    "precision": metrics["precision_macro"] * 100.0,
                    "recall": metrics["recall_macro"] * 100.0,
                    "f1": metrics["f1_macro"] * 100.0,
                    "leakage": absent_class_leakage(predictions, labels, args.num_tertiary) * 100.0,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "region_metrics.csv", index=False)
    if not frame.empty:
        plot_region_summary(
            frame,
            output_dir / "figures" / "region_macro_f1.png",
        )


def run_qualitative_suite(datasets, taxonomy, args, output_dir) -> None:
 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary = well_summary_frame(datasets)
    train_wells = [str(w) for w in summary["well_id"] if infer_well_role(str(w)) == "train"]
    test_wells = [str(w) for w in summary["well_id"] if infer_well_role(str(w)) == "test"]
    qual_wells = [str(w) for w in summary["well_id"] if infer_well_role(str(w)) == "qualitative"]
    if not test_wells and not qual_wells and len(summary) > TRAIN_WELLS + VAL_WELLS:
        # Fallback for datasets whose well names do not follow Table 1:
        # first 30 / next 8 / last 2 wells.
        ordered = [str(w) for w in summary["well_id"].tolist()]
        train_wells = ordered[:TRAIN_WELLS]
        test_wells = ordered[TRAIN_WELLS:TRAIN_WELLS + VAL_WELLS]
        qual_wells = ordered[TRAIN_WELLS + VAL_WELLS:]
    if not train_wells:
        train_wells = [str(w) for w in summary["well_id"].tolist()]

    train_dataset = subset_by_well_ids(datasets, train_wells)
    test_dataset = subset_by_well_ids(datasets, test_wells)
    qual_dataset = subset_by_well_ids(datasets, qual_wells)
    train_dataset, test_dataset, qual_dataset, _scaler = _scale_by_train_wells(
        train_dataset, test_dataset, qual_dataset
    )

    torch_recipes = [PROPOSED_RECIPE, FLAT_RECIPE] + BASELINE_RECIPES
    rows: List[Dict[str, Any]] = []
    figure_store: Dict[str, Dict[str, Any]] = {}

    # Classical ML baselines.
    x_train = summarize_window_features(train_dataset, summary="mean")
    labels_p_train, labels_s_train, y_train = collect_labels(train_dataset)
    for well_dataset in (test_dataset.datasets if hasattr(test_dataset, "datasets") else [test_dataset]):
        if not hasattr(well_dataset, "well_id"):
            continue
        profile = center_profile_view(well_dataset)
        figure_store.setdefault(
            well_dataset.well_id,
            {"depths": profile["depths"], "logs": profile["logs"], "truth": profile["labels_t"],
             "predictions": {}, "confusions": {}},
        )
    for well_dataset in (qual_dataset.datasets if hasattr(qual_dataset, "datasets") else [qual_dataset]):
        if not hasattr(well_dataset, "well_id"):
            continue
        profile = center_profile_view(well_dataset)
        figure_store.setdefault(
            well_dataset.well_id,
            {"depths": profile["depths"], "logs": profile["logs"], "truth": profile["labels_t"],
             "predictions": {}, "confusions": {}},
        )

    def _sklearn_predict(estimator, dataset):
        local_features = summarize_window_features(dataset, summary="mean")
        local_labels = collect_labels(dataset)[2]
        return estimator.predict(local_features), local_labels

    for recipe in available_sklearn_recipes(args.seed, args.num_tertiary):
        if recipe["estimator"] == "er_trees":
            estimator = HierarchicalExtraTrees(n_estimators=500, random_state=args.seed)
            estimator.fit(x_train, labels_p_train, labels_s_train, y_train)
        else:
            estimator = deepcopy(recipe["estimator"])
            estimator.fit(x_train, y_train)
        for dataset, role in ((test_dataset, "test"), (qual_dataset, "qualitative")):
            predictions, labels = _sklearn_predict(estimator, dataset)
            metrics = classification_metrics(predictions, labels)
            rows.append(
                {
                    "role": role,
                    "method": recipe["name"],
                    "accuracy": metrics["accuracy"] * 100.0,
                    "f1": metrics["f1_macro"] * 100.0,
                    "leakage": absent_class_leakage(predictions, labels, args.num_tertiary) * 100.0,
                }
            )
            for well_dataset in (dataset.datasets if hasattr(dataset, "datasets") else [dataset]):
                if not hasattr(well_dataset, "well_id") or well_dataset.well_id not in figure_store:
                    continue
                local_predictions, local_labels = _sklearn_predict(estimator, well_dataset)
                figure_store[well_dataset.well_id]["predictions"][recipe["name"]] = local_predictions
                figure_store[well_dataset.well_id]["confusions"][recipe["name"]] = normalized_confusion(
                    local_predictions, local_labels, args.num_tertiary
                )

    # Deep baselines and the proposed model.
    per_well_rows: List[Dict[str, Any]] = []
    for recipe in torch_recipes:
        class_weights = build_class_weights(recipe, train_dataset, args)
        model = instantiate_model(recipe, args, device)
        criterion = _criterion_for_recipe(recipe, args, taxonomy, class_weights, device)
        model = _fit_recipe(
            recipe, model, criterion, train_dataset, test_dataset, args, device, taxonomy, class_weights
        )
        for dataset, role in ((test_dataset, "test"), (qual_dataset, "qualitative")):
            metrics, predictions, labels, _hierarchy = _evaluate_recipe(
                recipe, model, criterion, dataset, args, device, taxonomy
            )
            rows.append(
                {
                    "role": role,
                    "method": recipe["name"],
                    "accuracy": metrics["accuracy"] * 100.0,
                    "f1": metrics["f1_macro"] * 100.0,
                    "leakage": absent_class_leakage(predictions, labels, args.num_tertiary) * 100.0,
                }
            )
            for well_dataset in (dataset.datasets if hasattr(dataset, "datasets") else [dataset]):
                if not hasattr(well_dataset, "well_id") or well_dataset.well_id not in figure_store:
                    continue
                local_metrics, local_predictions, local_labels, _lh = _evaluate_recipe(
                    recipe, model, criterion, well_dataset, args, device, taxonomy
                )
                figure_store[well_dataset.well_id]["predictions"][recipe["name"]] = local_predictions
                figure_store[well_dataset.well_id]["confusions"][recipe["name"]] = normalized_confusion(
                    local_predictions, local_labels, args.num_tertiary
                )
                if recipe["name"] == PROPOSED_RECIPE["name"] and role == "test":
                    per_well_rows.append(
                        {
                            "well_id": well_dataset.well_id,
                            "f1_macro": local_metrics["f1_macro"],
                            "accuracy": local_metrics["accuracy"],
                            "samples": int(local_labels.size),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "qualitative_metrics.csv", index=False)
    if per_well_rows:
        per_well_frame = pd.DataFrame(per_well_rows)
        per_well_frame.to_csv(output_dir / "per_well_test_metrics.csv", index=False)
        plot_per_well_metrics(
            per_well_frame,
            output_dir / "figures" / "per_well_test_performance.png",
        )

    class_names = taxonomy.tertiary if taxonomy is not None else [f"T{i}" for i in range(args.num_tertiary)]
    for well_id, payload in figure_store.items():
        plot_well_profile_panel(
            payload["depths"], payload["logs"], payload["truth"], payload["predictions"],
            output_dir / "figures" / f"{well_id}_profile.png",
            well_name=well_id,
            class_names=class_names,
            log_names=["GR", "DEN", "AC", "CNL", "RLLS", "RLLD"][:payload["logs"].shape[1]],
        )
        plot_confusion_grid(
            payload["confusions"],
            class_names=class_names,
            output_path=output_dir / "figures" / f"{well_id}_confusion.png",
            title=f"{well_id} comparison confusion matrices",
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_well_datasets(
        args.data_path,
        window_size=args.window_size,
        resample_interval=args.resample_interval,
        hampel_window=args.hampel_window,
        hampel_sigma=args.hampel_sigma,
    )
    if datasets:
        labels_p, labels_s, labels_t = collect_labels(merge_datasets(datasets))
        args.num_primary = int(labels_p.max()) + 1 if len(labels_p) else args.num_primary
        args.num_secondary = int(labels_s.max()) + 1 if len(labels_s) else args.num_secondary
        args.num_tertiary = int(labels_t.max()) + 1 if len(labels_t) else args.num_tertiary
    taxonomy = load_taxonomy(args)
    profile_wells = [item for item in (part.strip() for part in args.figure_wells.split(",")) if item]

    if args.command == "taxonomy-report":
        run_taxonomy_report(datasets, taxonomy, args, output_dir)
        return

    if args.command == "loss-sweep":
        raw = run_torch_suite(WEIGHT_STRATEGIES, datasets, taxonomy, args, output_dir, profile_wells)
        summary = summarize_repeated_results(raw, ["accuracy", "precision", "recall", "f1", "invalid_path_rate"])
        summary.to_csv(output_dir / "loss_sweep_summary.csv", index=False)
        plot_metric_bars(
            summary[["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]].to_numpy(),
            row_labels=summary["method"].tolist(),
            col_labels=["Accuracy", "Precision", "Recall", "F1"],
            output_path=output_dir / "figures" / "loss_sweep_metrics.png",
            title="Weight-strategy comparison",
        )
        return

    if args.command == "ablation":
        raw = run_torch_suite(LOSS_ABLATIONS, datasets, taxonomy, args, output_dir, profile_wells)
        summary = summarize_repeated_results(raw, ["accuracy", "precision", "recall", "f1", "invalid_path_rate"])
        summary.to_csv(output_dir / "ablation_summary.csv", index=False)
        plot_metric_bars(
            summary[["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]].to_numpy(),
            row_labels=summary["method"].tolist(),
            col_labels=["Accuracy", "Precision", "Recall", "F1"],
            output_path=output_dir / "figures" / "ablation_metrics.png",
            title="Loss-component ablation",
        )
        return

    if args.command == "training-strategy":
        raw = run_torch_suite(TRAINING_STRATEGIES, datasets, taxonomy, args, output_dir, profile_wells)
        summary = summarize_repeated_results(raw, ["accuracy", "precision", "recall", "f1"])
        summary.to_csv(output_dir / "training_strategy_summary.csv", index=False)
        plot_metric_bars(
            summary[["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]].to_numpy(),
            row_labels=summary["method"].tolist(),
            col_labels=["Accuracy", "Precision", "Recall", "F1"],
            output_path=output_dir / "figures" / "training_strategy_metrics.png",
            title="Strategy I vs Strategy II",
        )
        return

    if args.command == "benchmark":
        # Table 9 comparison: classical ML, deep baselines, the flat
        # parameter-matched model and the proposed framework.
        torch_recipes = [PROPOSED_RECIPE, FLAT_RECIPE] + BASELINE_RECIPES
        torch_raw = run_torch_suite(
            torch_recipes,
            datasets,
            taxonomy,
            args,
            output_dir / "torch",
            profile_wells,
        )
        sklearn_raw = run_sklearn_suite(datasets, args, output_dir / "classical", profile_wells)
        combined = pd.concat([torch_raw, sklearn_raw], ignore_index=True)
        combined.to_csv(output_dir / "benchmark_raw_metrics.csv", index=False)
        summarize_repeated_results(
            combined, ["accuracy", "precision", "recall", "f1", "leakage"]
        ).to_csv(output_dir / "benchmark_summary.csv", index=False)
        return

    if args.command == "lambda-sweep":
        run_lambda_sweep(datasets, taxonomy, args, output_dir, profile_wells)
        return

    if args.command == "window-sweep":
        run_window_sweep(datasets, taxonomy, args, output_dir)
        return

    if args.command == "leave-one-region-out":
        run_region_suite(datasets, taxonomy, args, output_dir)
        return

    if args.command == "qualitative":
        run_qualitative_suite(datasets, taxonomy, args, output_dir)
        return


if __name__ == "__main__":
    main()
