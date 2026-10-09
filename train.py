

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.preprocessing import (
    MinMaxFeatureScaler,
    extract_record_arrays,
    load_prepared_records,
)
from data import HierarchicalWeightedSampler
from experiments.core import (
    WindowedWellDataset,
    collect_labels,
    hierarchical_class_weights,
    select_center_logits,
)
from experiments.taxonomy import TaxonomySpec, constrained_decode
from losses import DynamicHierarchicalLoss
from models import HierarchicalLithologyModel
from utils.metrics import compute_metrics
from utils.training_utils import (
    CurriculumStage,
    paper_curriculum_stages,
    run_curriculum_training,
    seed_all,
)


WINDOW_SIZE = 15          # 15 samples (~2 m), centre-depth labelling
TRAIN_WELLS = 30          # W09-W38
VAL_WELLS = 8             # W01-W08 (held-out test wells)
QUALITATIVE_WELLS = 2     # Well A, Well B

TEST_WELL_IDS = tuple(f"W{idx:02d}" for idx in range(1, 9))        # W01-W08
QUALITATIVE_WELL_IDS = ("WellA", "WellB", "Well A", "Well B")


def sliding_windows(
    features: np.ndarray,
    labels_p: np.ndarray,
    labels_s: np.ndarray,
    labels_t: np.ndarray,
    depths: Optional[np.ndarray] = None,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

    half = window_size // 2
    windows, lp, ls, lt, centre_depths = [], [], [], [], []

    if depths is None:
        depths = np.arange(len(features))

    for centre in range(half, len(features) - half):
        windows.append(features[centre - half:centre + half + 1])
        lp.append(labels_p[centre])
        ls.append(labels_s[centre])
        lt.append(labels_t[centre])
        centre_depths.append(depths[centre])

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(lp, dtype=np.int64),
        np.asarray(ls, dtype=np.int64),
        np.asarray(lt, dtype=np.int64),
        np.asarray(centre_depths),
    )


def build_well_dataset(
    features: np.ndarray,
    labels_p: np.ndarray,
    labels_s: np.ndarray,
    labels_t: np.ndarray,
    depths: Optional[np.ndarray],
    well_id: str,
    window_size: int = WINDOW_SIZE,
) -> WindowedWellDataset:
    features = np.asarray(features, dtype=np.float32)
    labels_p = np.asarray(labels_p, dtype=np.int64)
    labels_s = np.asarray(labels_s, dtype=np.int64)
    labels_t = np.asarray(labels_t, dtype=np.int64)
    windows, lp, ls, lt, centre_depths = sliding_windows(
        features, labels_p, labels_s, labels_t, depths=depths, window_size=window_size
    )
    return WindowedWellDataset(
        features=windows,
        labels_p=lp,
        labels_s=ls,
        labels_t=lt,
        depths=np.asarray(centre_depths),
        well_id=well_id,
        window_size=window_size,
    )


def extract_well_arrays(records) -> Dict[str, Dict[str, np.ndarray]]:

    wells: Dict[str, Dict[str, np.ndarray]] = {}
    for record in records:
        features, labels_p, labels_s, labels_t, well_ids, depth = extract_record_arrays(record)
        if well_ids is None:
            well_ids = np.asarray(["well_0"] * len(features), dtype=object)
        for well_id in np.unique(well_ids):
            mask = well_ids == well_id
            order = np.argsort(depth[mask], kind="stable") if depth is not None else np.arange(mask.sum())
            wells[str(well_id)] = {
                "features": features[mask][order].astype(np.float32),
                "labels_p": labels_p[mask][order].astype(np.int64),
                "labels_s": labels_s[mask][order].astype(np.int64),
                "labels_t": labels_t[mask][order].astype(np.int64),
                "depth": depth[mask][order] if depth is not None else np.arange(mask.sum(), dtype=np.float64),
            }
    return wells


def infer_well_roles(well_ids: List[str]) -> Dict[str, str]:

    roles: Dict[str, str] = {}
    ordered = sorted(well_ids)
    remaining = list(ordered)
    for well_id in ordered:
        if well_id in TEST_WELL_IDS:
            roles[well_id] = "test"
            remaining.remove(well_id)
        elif well_id in QUALITATIVE_WELL_IDS:
            roles[well_id] = "qualitative"
            remaining.remove(well_id)
    for well_id in remaining:
        roles[well_id] = "train"
    if not any(role == "test" for role in roles.values()) and len(ordered) > TRAIN_WELLS + VAL_WELLS:
        roles = {}
        for idx, well_id in enumerate(ordered):
            if idx < TRAIN_WELLS:
                roles[well_id] = "train"
            elif idx < TRAIN_WELLS + VAL_WELLS:
                roles[well_id] = "test"
            else:
                roles[well_id] = "qualitative"
    return roles


def build_mapping_matrices(
    labels_p: np.ndarray,
    labels_s: np.ndarray,
    labels_t: np.ndarray,
    num_primary: int,
    num_secondary: int,
    num_tertiary: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:

    s2p = [0] * num_secondary
    for s_idx in range(num_secondary):
        parents = labels_p[labels_s == s_idx]
        s2p[s_idx] = int(Counter(parents.tolist()).most_common(1)[0][0]) if len(parents) else 0

    t2s = [0] * num_tertiary
    for t_idx in range(num_tertiary):
        parents = labels_s[labels_t == t_idx]
        t2s[t_idx] = int(Counter(parents.tolist()).most_common(1)[0][0]) if len(parents) else 0

    m21 = torch.zeros(num_primary, num_secondary)
    m32 = torch.zeros(num_secondary, num_tertiary)
    for s_idx, p_idx in enumerate(s2p):
        m21[p_idx, s_idx] = 1.0
    for t_idx, s_idx in enumerate(t2s):
        m32[s_idx, t_idx] = 1.0
    return m21, m32, tuple(s2p), tuple(t2s)


def build_sampler(dataset) -> Optional[object]:

    if len(dataset) == 0:
        return None
    labels_p, labels_s, labels_t = collect_labels(dataset)
    return HierarchicalWeightedSampler(
        labels_p, labels_s, labels_t, alpha=0.6, beta=1.3
    ).get_sampler()


def load_data(
    data_path: str,
    window_size: int = WINDOW_SIZE,
    resample_interval: Optional[float] = None,
    hampel_window: Optional[int] = None,
    hampel_sigma: float = 3.0,
):

    records = load_prepared_records(
        data_path,
        resample_interval=resample_interval,
        hampel_window=hampel_window,
        hampel_sigma=hampel_sigma,
    )
    wells = extract_well_arrays(records)
    if not wells:
        raise ValueError("No wells found in the data; check data_path.")
    return wells


def train_epoch(
    model, dataloader, criterion, optimizer, device, teacher_forcing: bool = True
) -> float:

    model.train()
    total_loss = 0.0
    batches = 0

    for data, labels_p, labels_s, labels_t in dataloader:
        data = data.to(device)
        labels_p = labels_p.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        optimizer.zero_grad()
        if teacher_forcing:
            logits_p, logits_s, logits_t = model(data, teacher_parents=(labels_p, labels_s))
        else:
            logits_p, logits_s, logits_t = model(data)
        logits_p = select_center_logits(logits_p)
        logits_s = select_center_logits(logits_s)
        logits_t = select_center_logits(logits_t)

        loss, _ = criterion(logits_p, logits_s, logits_t, labels_p, labels_s, labels_t)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(
    model, dataloader, criterion, device, constrained: bool = True, taxonomy=None
):

    model.eval()
    total_loss = 0.0
    batches = 0
    preds_p, preds_s, preds_t = [], [], []
    labels_p_all, labels_s_all, labels_t_all = [], [], []

    for data, labels_p, labels_s, labels_t in dataloader:
        data = data.to(device)
        labels_p = labels_p.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        logits_p, logits_s, logits_t = model(data)
        logits_p = select_center_logits(logits_p)
        logits_s = select_center_logits(logits_s)
        logits_t = select_center_logits(logits_t)
        loss, _ = criterion(logits_p, logits_s, logits_t, labels_p, labels_s, labels_t)
        total_loss += float(loss.item())
        batches += 1

        if constrained and taxonomy is not None:
            pred_p, pred_s, pred_t = constrained_decode(logits_p, logits_s, logits_t, taxonomy)
        else:
            pred_p = logits_p.argmax(dim=1)
            pred_s = logits_s.argmax(dim=1)
            pred_t = logits_t.argmax(dim=1)
        preds_p.append(pred_p.cpu().numpy())
        preds_s.append(pred_s.cpu().numpy())
        preds_t.append(pred_t.cpu().numpy())
        labels_p_all.append(labels_p.cpu().numpy())
        labels_s_all.append(labels_s.cpu().numpy())
        labels_t_all.append(labels_t.cpu().numpy())

    if not preds_p:
        empty = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
        return 0.0, empty, empty, empty

    preds_p = np.concatenate(preds_p).flatten()
    preds_s = np.concatenate(preds_s).flatten()
    preds_t = np.concatenate(preds_t).flatten()
    labels_p_all = np.concatenate(labels_p_all).flatten()
    labels_s_all = np.concatenate(labels_s_all).flatten()
    labels_t_all = np.concatenate(labels_t_all).flatten()

    metrics_p = compute_metrics(preds_p, labels_p_all, model.num_primary)
    metrics_s = compute_metrics(preds_s, labels_s_all, model.num_secondary)
    metrics_t = compute_metrics(preds_t, labels_t_all, model.num_tertiary)
    return total_loss / max(batches, 1), metrics_p, metrics_s, metrics_t


def parse_args():

    parser = argparse.ArgumentParser(description="Train the hierarchical lithology model")

    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (16-64 in the search)")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE,
                        help="Sliding-window length (optimum: 15 samples ~2 m)")
    parser.add_argument("--resample_interval", type=float, default=0.125,
                        help="Common depth-grid interval in m (0 = disabled)")
    parser.add_argument("--hampel_window", type=int, default=5,
                        help="Hampel filter window (0 = disabled)")
    parser.add_argument("--hampel_sigma", type=float, default=3.0,
                        help="Hampel filter sigma")

    # Model parameters (manuscript taxonomy: 6 / 10 / 26)
    parser.add_argument("--in_channels", type=int, default=6,
                        help="Number of input log channels (GR, AC, CNL, DEN, RLLD, RLLS)")
    parser.add_argument("--num_primary", type=int, default=6, help="Number of primary classes")
    parser.add_argument("--num_secondary", type=int, default=10, help="Number of secondary classes")
    parser.add_argument("--num_tertiary", type=int, default=26, help="Number of tertiary classes")

    # Loss parameters (adopted S2 strategy, Table 3)
    parser.add_argument("--task_weights", type=str, default="1.0,1.2,1.5",
                        help="w_alpha,w_beta,w_gamma of Eq. 7")
    parser.add_argument("--lambda_cons", type=float, default=0.5,
                        help="Consistency regulariser weight")
    parser.add_argument("--lambda_auc", type=float, default=0.1,
                        help="AUC regulariser weight")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal focusing exponent phi (Eq. 9)")

    # Training parameters
    parser.add_argument("--epochs_override", type=str, default=None,
                        help="Override Table 2 max epochs as stage1,stage2,stage3,stage4")
    parser.add_argument("--lr", type=float, default=1e-3, help="Stage 1/2 learning rate")
    parser.add_argument("--stage3_lr", type=str, default="1e-4,1e-3",
                        help="Stage 3 cyclic learning rate min,max")
    parser.add_argument("--stage4_lr", type=float, default=5e-4, help="Stage 4 learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience (Table 2: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (manuscript: 42)")
    parser.add_argument("--sampler_alpha", type=float, default=0.6, help="eta1 of Eq. 14")
    parser.add_argument("--sampler_beta", type=float, default=1.3, help="eta2 of Eq. 14")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pth",
                        help="Path to save checkpoints")
    parser.add_argument("--decoding_mode", type=str, default="constrained",
                        choices=["constrained", "teacher", "self"],
                        help="Decoding setting of Table 7 (deployed: constrained)")

    return parser.parse_args()


def main():

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    print(f"Using device: {device}")

    wells = load_data(
        args.data_path,
        window_size=args.window_size,
        resample_interval=args.resample_interval or None,
        hampel_window=args.hampel_window or None,
        hampel_sigma=args.hampel_sigma,
    )

    roles = infer_well_roles(list(wells.keys()))
    train_ids = [w for w, role in roles.items() if role == "train"]
    test_ids = [w for w, role in roles.items() if role == "test"]
    qual_ids = [w for w, role in roles.items() if role == "qualitative"]
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test / {len(qual_ids)} qualitative wells")

    # Min-Max scaling fitted on the TRAINING wells only (Table 1).
    scaler = MinMaxFeatureScaler()
    if train_ids:
        scaler.fit(np.concatenate([wells[w]["features"] for w in train_ids], axis=0))
    else:
        scaler.fit(np.concatenate([wells[w]["features"] for w in wells], axis=0))

    def make_datasets(ids: List[str]) -> List[WindowedWellDataset]:
        datasets = []
        for well_id in ids:
            well = wells[well_id]
            features = scaler.transform(well["features"])
            if len(features) < args.window_size:
                print(f"  skipping well {well_id}: only {len(features)} samples")
                continue
            datasets.append(
                build_well_dataset(
                    features, well["labels_p"], well["labels_s"], well["labels_t"],
                    depths=well["depth"], well_id=well_id, window_size=args.window_size,
                )
            )
        return datasets

    train_datasets = make_datasets(train_ids)
    test_datasets = make_datasets(test_ids)
    qual_datasets = make_datasets(qual_ids)

    train_dataset = ConcatDataset(train_datasets) if train_datasets else ConcatDataset(make_datasets(list(wells.keys())))
    val_dataset = ConcatDataset(test_datasets) if test_datasets else ConcatDataset(train_datasets)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Loaded dataset split is empty; check data_path and source files.")

    # Infer class counts and the taxonomy mapping matrices from the labels.
    labels_p, labels_s, labels_t = collect_labels(train_dataset)
    args.num_primary = int(labels_p.max()) + 1 if len(labels_p) else args.num_primary
    args.num_secondary = int(labels_s.max()) + 1 if len(labels_s) else args.num_secondary
    args.num_tertiary = int(labels_t.max()) + 1 if len(labels_t) else args.num_tertiary
    print(f"Taxonomy: {args.num_primary} primary / {args.num_secondary} secondary / {args.num_tertiary} tertiary classes")

    m21, m32, s2p, t2s = build_mapping_matrices(
        labels_p, labels_s, labels_t,
        args.num_primary, args.num_secondary, args.num_tertiary,
    )
    taxonomy = TaxonomySpec(
        primary=tuple(f"P{i}" for i in range(args.num_primary)),
        secondary=tuple(f"S{i}" for i in range(args.num_secondary)),
        tertiary=tuple(f"T{i}" for i in range(args.num_tertiary)),
        secondary_to_primary=s2p,
        tertiary_to_secondary=t2s,
    )

    class_weights = hierarchical_class_weights(
        labels_p, labels_s, labels_t,
        args.num_primary, args.num_secondary, args.num_tertiary,
    )
    task_weights = tuple(float(value) for value in args.task_weights.split(","))
    sampler = HierarchicalWeightedSampler(
        labels_p, labels_s, labels_t,
        alpha=args.sampler_alpha, beta=args.sampler_beta,
    ).get_sampler()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = HierarchicalLithologyModel(
        device=device,
        in_channels=args.in_channels,
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        num_tertiary=args.num_tertiary,
        feat_scales=[0, 1, 2, 3],
        feature_outchannel=[16, 32, 64, 128],
        num_points=args.window_size,
        time_steps=[0, 0, 0, 0],
        channel_multiplier=[1, 2, 4, 8],
        fd=6
    ).to(device)

    # Dynamic hierarchical loss with the adopted S2 configuration.
    criterion = DynamicHierarchicalLoss(
        num_primary=args.num_primary,
        num_secondary=args.num_secondary,
        num_tertiary=args.num_tertiary,
        task_weights=task_weights,
        lambda_consistency=args.lambda_cons,
        lambda_auc=args.lambda_auc,
        tertiary_gamma=args.focal_gamma,
        class_weights=class_weights,
        M21=m21,
        M32=m32,
    )

    # Four-stage curriculum training schedule (Table 2).
    stages = paper_curriculum_stages()
    if args.epochs_override:
        epoch_limits = [int(value) for value in args.epochs_override.split(",")]
        stage3_cyclic = tuple(float(v) for v in args.stage3_lr.split(","))
        stages = [
            CurriculumStage(1, ("encoder", "decoder", "primary"), args.lr,
                            epoch_limits[0] if len(epoch_limits) > 0 else 30, args.patience, "primary"),
            CurriculumStage(2, ("secondary",), args.lr,
                            epoch_limits[1] if len(epoch_limits) > 1 else 30, args.patience, "secondary"),
            CurriculumStage(3, ("tertiary",), stage3_cyclic,
                            epoch_limits[2] if len(epoch_limits) > 2 else 40, args.patience, "tertiary"),
            CurriculumStage(4, ("encoder", "decoder", "primary", "secondary", "tertiary"), args.stage4_lr,
                            epoch_limits[3] if len(epoch_limits) > 3 else 100, args.patience, "tertiary"),
        ]

    teacher_forcing = args.decoding_mode in ("teacher", "constrained")
    constrained = args.decoding_mode == "constrained"
    print("Starting four-stage curriculum training (Table 2)...")
    result = run_curriculum_training(
        model, criterion, train_loader, val_loader, device,
        stages=stages,
        train_epoch=train_epoch,
        evaluate=evaluate,
        teacher_forcing=teacher_forcing,
        constrained=constrained,
        taxonomy=taxonomy,
        seed=args.seed,
        checkpoint_dir=str(Path(args.checkpoint_path).parent),
    )

    history = pd.DataFrame(result["history"])
    history_path = Path(args.checkpoint_path).parent / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    print(f"Stage summary: {result['stages']}")

    # Final evaluation on the held-out test wells (W01-W08).
    val_loss, metrics_p, metrics_s, metrics_t = evaluate(
        model, val_loader, criterion, device, constrained=constrained, taxonomy=taxonomy
    )
    print("Held-out test wells (tertiary macro metrics):")
    print(f"  Accuracy : {metrics_t['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics_t['precision_macro'] * 100:.2f}%")
    print(f"  Recall   : {metrics_t['recall_macro'] * 100:.2f}%")
    print(f"  Macro-F1 : {metrics_t['f1_macro'] * 100:.2f}%")

    # Qualitative wells (Well A / Well B): profiles, leakage, invalid paths.
    for qual_dataset in qual_datasets:
        if len(qual_dataset) == 0:
            continue
        loader = DataLoader(qual_dataset, batch_size=args.batch_size, shuffle=False)
        _, qp, qs, qt = evaluate(
            model, loader, criterion, device, constrained=constrained, taxonomy=taxonomy
        )
        from utils.metrics import absent_class_leakage

        preds_t = []
        labels_t = []
        with torch.no_grad():
            for data, lp, ls, lt in loader:
                data = data.to(device)
                logits_p, logits_s, logits_t = model(data)
                logits_p = select_center_logits(logits_p)
                logits_s = select_center_logits(logits_s)
                logits_t = select_center_logits(logits_t)
                if constrained:
                    _, _, pred_t = constrained_decode(logits_p, logits_s, logits_t, taxonomy)
                else:
                    pred_t = logits_t.argmax(dim=1)
                preds_t.append(pred_t.cpu().numpy())
                labels_t.append(lt.numpy())
        preds_t = np.concatenate(preds_t).flatten()
        labels_t = np.concatenate(labels_t).flatten()
        leak = absent_class_leakage(preds_t, labels_t, args.num_tertiary)
        print(f"Qualitative well {qual_dataset.well_id}: accuracy {qt['accuracy'] * 100:.2f}%, "
              f"leakage {leak * 100:.2f}%")

    # Save the final checkpoint.
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "taxonomy": {
                "secondary_to_primary": s2p,
                "tertiary_to_secondary": t2s,
                "num_primary": args.num_primary,
                "num_secondary": args.num_secondary,
                "num_tertiary": args.num_tertiary,
            },
            "scaler": scaler.to_dict(),
        },
        args.checkpoint_path,
    )
    print(f"Training complete. Checkpoint saved to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
