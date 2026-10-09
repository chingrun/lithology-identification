
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

from data.preprocessing import MinMaxFeatureScaler, extract_record_arrays, load_prepared_records
from data import HierarchicalWeightedSampler
from utils.metrics import absent_class_leakage
from utils.training_utils import EarlyStopping, ModelCheckpoint, History
from experiments.taxonomy import constrained_decode


WINDOW_SIZE = 15
TRAIN_WELLS = 30
VAL_WELLS = 8


SUBBASIN_WELLS = {
    "A": {"W02", "W05", "W10", "W12", "W15", "W18", "W20", "W21", "W25", "W29", "W32", "W34", "WellA", "Well A"},
    "B": {"W03", "W04", "W06", "W13", "W14", "W16", "W19", "W23", "W26", "W28", "W31", "W37", "W38", "WellB", "Well B"},
    "C": {"W01", "W07", "W08", "W09", "W11", "W17", "W22", "W24", "W27", "W30", "W33", "W35", "W36"},
}

TEST_WELL_IDS = tuple(f"W{idx:02d}" for idx in range(1, 9))       # W01-W08
QUALITATIVE_WELL_IDS = ("WellA", "WellB", "Well A", "Well B")


@dataclass(frozen=True)
class SplitAssignment:


    repeat_id: int
    fold_id: int
    train_wells: Tuple[str, ...]
    val_wells: Tuple[str, ...]
    test_wells: Tuple[str, ...]


@dataclass(frozen=True)
class RegionSplitAssignment:

    region: str
    train_wells: Tuple[str, ...]
    val_wells: Tuple[str, ...]
    test_wells: Tuple[str, ...]


def well_region(well_id: str) -> Optional[str]:

    key = str(well_id)
    for region, members in SUBBASIN_WELLS.items():
        if key in members or key.lower() in {item.lower() for item in members}:
            return region
    return None


def infer_well_role(well_id: str) -> str:
    key = str(well_id)
    if key in QUALITATIVE_WELL_IDS:
        return "qualitative"
    if key in TEST_WELL_IDS:
        return "test"
    return "train"



@dataclass
class PredictionBundle:


    predictions_p: np.ndarray
    predictions_s: np.ndarray
    predictions_t: np.ndarray
    labels_p: np.ndarray
    labels_s: np.ndarray
    labels_t: np.ndarray
    depths: Optional[np.ndarray] = None
    well_ids: Optional[np.ndarray] = None
    logits_p: Optional[np.ndarray] = None
    logits_s: Optional[np.ndarray] = None
    logits_t: Optional[np.ndarray] = None


@dataclass
class WindowedWellDataset(Dataset):

    features: np.ndarray
    labels_p: np.ndarray
    labels_s: np.ndarray
    labels_t: np.ndarray
    depths: np.ndarray
    well_id: str
    window_size: int = WINDOW_SIZE
    raw_features: Optional[np.ndarray] = None
    raw_labels_p: Optional[np.ndarray] = None
    raw_labels_s: Optional[np.ndarray] = None
    raw_labels_t: Optional[np.ndarray] = None
    raw_depths: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.features = torch.as_tensor(self.features, dtype=torch.float32)
        self.labels_p = torch.as_tensor(self.labels_p, dtype=torch.long)
        self.labels_s = torch.as_tensor(self.labels_s, dtype=torch.long)
        self.labels_t = torch.as_tensor(self.labels_t, dtype=torch.long)
        self.depths = np.asarray(self.depths)
        self.well_id = str(self.well_id)
        self.window_size = int(self.window_size)
        if self.raw_features is not None:
            self.raw_features = np.asarray(self.raw_features)
        if self.raw_labels_p is not None:
            self.raw_labels_p = np.asarray(self.raw_labels_p)
        if self.raw_labels_s is not None:
            self.raw_labels_s = np.asarray(self.raw_labels_s)
        if self.raw_labels_t is not None:
            self.raw_labels_t = np.asarray(self.raw_labels_t)
        if self.raw_depths is not None:
            self.raw_depths = np.asarray(self.raw_depths)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.features[idx],
            self.labels_p[idx],
            self.labels_s[idx],
            self.labels_t[idx],
        )


class EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError("EmptyDataset has no samples.")


def min_max_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    min_v = arr.min(axis=0, keepdims=True)
    max_v = arr.max(axis=0, keepdims=True)
    return (arr - min_v) / (max_v - min_v + 1e-8)


def sliding_windows_with_meta(
    features: np.ndarray,
    labels_p: np.ndarray,
    labels_s: np.ndarray,
    labels_t: np.ndarray,
    depths: Optional[np.ndarray] = None,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = window_size // 2
    windows, lp, ls, lt, center_depths = [], [], [], [], []

    if depths is None:
        depths = np.arange(len(features))

    for center in range(half, len(features) - half):
        windows.append(features[center - half:center + half + 1])
        lp.append(labels_p[center])
        ls.append(labels_s[center])
        lt.append(labels_t[center])
        center_depths.append(depths[center])

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(lp, dtype=np.int64),
        np.asarray(ls, dtype=np.int64),
        np.asarray(lt, dtype=np.int64),
        np.asarray(center_depths),
    )


def build_window_dataset_from_arrays(
    features: np.ndarray,
    labels_p: np.ndarray,
    labels_s: np.ndarray,
    labels_t: np.ndarray,
    depths: Optional[np.ndarray] = None,
    well_id: str = "well_0",
    window_size: int = WINDOW_SIZE,
    raw_features: Optional[np.ndarray] = None,
    raw_labels_p: Optional[np.ndarray] = None,
    raw_labels_s: Optional[np.ndarray] = None,
    raw_labels_t: Optional[np.ndarray] = None,
    raw_depths: Optional[np.ndarray] = None,
) -> WindowedWellDataset:

    features = np.asarray(features, dtype=np.float32)
    labels_p = np.asarray(labels_p, dtype=np.int64)
    labels_s = np.asarray(labels_s, dtype=np.int64)
    labels_t = np.asarray(labels_t, dtype=np.int64)
    if depths is not None:
        depths = np.asarray(depths)

    features, labels_p, labels_s, labels_t, depths = sliding_windows_with_meta(
        features,
        labels_p,
        labels_s,
        labels_t,
        depths=depths,
        window_size=window_size,
    )
    return WindowedWellDataset(
        features,
        labels_p,
        labels_s,
        labels_t,
        depths,
        well_id,
        raw_features=raw_features,
        raw_labels_p=raw_labels_p,
        raw_labels_s=raw_labels_s,
        raw_labels_t=raw_labels_t,
        raw_depths=raw_depths,
        window_size=window_size,
    )


def build_well_datasets_from_record(
    record: Mapping[str, Any],
    window_size: int = WINDOW_SIZE,
) -> List[WindowedWellDataset]:
    features, labels_p, labels_s, labels_t, well_ids, depth = extract_record_arrays(record)

    if well_ids is None:
        return [
            build_window_dataset_from_arrays(
                features,
                labels_p,
                labels_s,
                labels_t,
                depths=depth,
                well_id="well_0",
                window_size=window_size,
                raw_features=features,
                raw_labels_p=labels_p,
                raw_labels_s=labels_s,
                raw_labels_t=labels_t,
                raw_depths=depth,
            )
        ]

    datasets: List[WindowedWellDataset] = []
    for well_id in np.unique(well_ids):
        mask = well_ids == well_id
        well_features = features[mask]
        well_labels_p = labels_p[mask]
        well_labels_s = labels_s[mask]
        well_labels_t = labels_t[mask]
        well_depth = depth[mask] if depth is not None else None

        if well_depth is not None:
            order = np.argsort(well_depth, kind="stable")
            well_features = well_features[order]
            well_labels_p = well_labels_p[order]
            well_labels_s = well_labels_s[order]
            well_labels_t = well_labels_t[order]
            well_depth = well_depth[order]

        if len(well_features) >= window_size:
            datasets.append(
                build_window_dataset_from_arrays(
                    well_features,
                    well_labels_p,
                    well_labels_s,
                    well_labels_t,
                    depths=well_depth,
                    well_id=str(well_id),
                    window_size=window_size,
                    raw_features=well_features,
                    raw_labels_p=well_labels_p,
                    raw_labels_s=well_labels_s,
                    raw_labels_t=well_labels_t,
                    raw_depths=well_depth,
                )
            )

    return datasets


def load_well_datasets(
    data_path: str | Path,
    window_size: int = WINDOW_SIZE,
    resample_interval: Optional[float] = None,
    hampel_window: Optional[int] = None,
    hampel_sigma: float = 3.0,
) -> List[WindowedWellDataset]:
    records = load_prepared_records(
        data_path,
        resample_interval=resample_interval,
        hampel_window=hampel_window,
        hampel_sigma=hampel_sigma,
    )
    datasets: List[WindowedWellDataset] = []
    for record in records:
        datasets.extend(build_well_datasets_from_record(record, window_size=window_size))
    return datasets


def scale_windowed_dataset(dataset: Dataset, scaler: MinMaxFeatureScaler) -> Dataset:

    if isinstance(dataset, WindowedWellDataset):
        scaled_features = scaler.transform(dataset.features.numpy())
        raw_features = dataset.raw_features
        if raw_features is not None:
            raw_features = scaler.transform(raw_features)
        return WindowedWellDataset(
            scaled_features,
            dataset.labels_p.numpy(),
            dataset.labels_s.numpy(),
            dataset.labels_t.numpy(),
            dataset.depths,
            dataset.well_id,
            raw_features=raw_features,
            raw_labels_p=dataset.raw_labels_p,
            raw_labels_s=dataset.raw_labels_s,
            raw_labels_t=dataset.raw_labels_t,
            raw_depths=dataset.raw_depths,
            window_size=dataset.window_size,
        )
    if isinstance(dataset, ConcatDataset):
        return ConcatDataset([scale_windowed_dataset(part, scaler) for part in dataset.datasets])
    if isinstance(dataset, Subset):
        scaled_base = scale_windowed_dataset(dataset.dataset, scaler)
        return Subset(scaled_base, dataset.indices)
    if isinstance(dataset, EmptyDataset):
        return dataset
    raise TypeError(f"Unsupported dataset type for scaling: {type(dataset)!r}")


def fit_scale_split(
    datasets: Sequence[WindowedWellDataset],
    split: SplitAssignment,
) -> Tuple[Dataset, Dataset, Dataset, Optional[MinMaxFeatureScaler]]:

    train, val, test = split_datasets(datasets, split)
    if len(train) == 0:
        return train, val, test, None
    scaler = MinMaxFeatureScaler()
    window_features = collect_window_features(train)
    scaler.fit(window_features.reshape(-1, window_features.shape[-1]))
    return (
        scale_windowed_dataset(train, scaler),
        scale_windowed_dataset(val, scaler),
        scale_windowed_dataset(test, scaler),
        scaler,
    )


def compute_leave_one_region_out_splits(
    datasets: Sequence[WindowedWellDataset],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> List[RegionSplitAssignment]:

    summary = well_summary_frame(datasets)
    splits: List[RegionSplitAssignment] = []
    regions = sorted({region for region in (well_region(w) for w in summary["well_id"]) if region})
    if not regions and len(summary) >= 3:
        # Fallback for datasets whose well names do not match Table 1: hold
        # out three well groups in turn.
        wells_all = [str(w) for w in summary["well_id"]]
        kf = KFold(n_splits=min(3, len(wells_all)), shuffle=True, random_state=seed)
        for region_idx, (train_idx, test_idx) in enumerate(kf.split(wells_all)):
            test_wells = [wells_all[idx] for idx in test_idx]
            rest = [wells_all[idx] for idx in train_idx]
            train_wells, val_wells = train_test_split(
                rest, test_size=min(val_fraction, 0.5), random_state=seed + region_idx,
            )
            splits.append(
                RegionSplitAssignment(
                    region=f"region{region_idx + 1}",
                    train_wells=tuple(train_wells),
                    val_wells=tuple(val_wells),
                    test_wells=tuple(test_wells),
                )
            )
        return splits
    for held_out in regions:
        test_wells = [str(w) for w in summary["well_id"] if well_region(str(w)) == held_out]
        rest = [str(w) for w in summary["well_id"] if well_region(str(w)) != held_out]
        if len(rest) < 2:
            train_wells, val_wells = rest, []
        else:
            train_wells, val_wells = train_test_split(
                rest, test_size=min(val_fraction, 0.5), random_state=seed,
            )
        splits.append(
            RegionSplitAssignment(
                region=held_out,
                train_wells=tuple(train_wells),
                val_wells=tuple(val_wells),
                test_wells=tuple(test_wells),
            )
        )
    return splits


def merge_datasets(datasets: Sequence[Dataset]) -> Dataset:
    if not datasets:
        return EmptyDataset()
    return datasets[0] if len(datasets) == 1 else ConcatDataset(list(datasets))


def subset_by_well_ids(datasets: Sequence[WindowedWellDataset], selected_wells: Sequence[str]) -> Dataset:
    selected = {str(w) for w in selected_wells}
    matched = [dataset for dataset in datasets if dataset.well_id in selected]
    return merge_datasets(matched)


def collect_labels(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(dataset, WindowedWellDataset):
        return dataset.labels_p.numpy(), dataset.labels_s.numpy(), dataset.labels_t.numpy()
    if isinstance(dataset, ConcatDataset):
        parts_p, parts_s, parts_t = [], [], []
        for part in dataset.datasets:
            p, s, t = collect_labels(part)
            parts_p.append(p)
            parts_s.append(s)
            parts_t.append(t)
        return np.concatenate(parts_p), np.concatenate(parts_s), np.concatenate(parts_t)
    if isinstance(dataset, Subset):
        p, s, t = collect_labels(dataset.dataset)
        idx = np.asarray(dataset.indices)
        return p[idx], s[idx], t[idx]

    
    if all(hasattr(dataset, attr) for attr in ("labels_p", "labels_s", "labels_t")):
        def _to_array(value):
            return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

        return _to_array(dataset.labels_p), _to_array(dataset.labels_s), _to_array(dataset.labels_t)
    raise TypeError(f"Unsupported dataset type for label collection: {type(dataset)!r}")


def collect_window_features(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, WindowedWellDataset):
        return dataset.features.numpy()
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([collect_window_features(part) for part in dataset.datasets], axis=0)
    if isinstance(dataset, Subset):
        features = collect_window_features(dataset.dataset)
        return features[np.asarray(dataset.indices)]
    if hasattr(dataset, "features"):
        features = dataset.features
        return features.numpy() if hasattr(features, "numpy") else np.asarray(features)
    raise TypeError(f"Unsupported dataset type for feature collection: {type(dataset)!r}")


def collect_well_ids(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, WindowedWellDataset):
        return np.asarray([dataset.well_id] * len(dataset), dtype=object)
    if isinstance(dataset, ConcatDataset):
        parts = [collect_well_ids(part) for part in dataset.datasets]
        return np.concatenate(parts)
    if isinstance(dataset, Subset):
        ids = collect_well_ids(dataset.dataset)
        return ids[np.asarray(dataset.indices)]
    raise TypeError(f"Unsupported dataset type for well-id collection: {type(dataset)!r}")


def collect_raw_series(dataset: Dataset) -> Dict[str, Optional[np.ndarray]]:
    if isinstance(dataset, WindowedWellDataset):
        return {
            "raw_features": dataset.raw_features,
            "raw_labels_p": dataset.raw_labels_p,
            "raw_labels_s": dataset.raw_labels_s,
            "raw_labels_t": dataset.raw_labels_t,
            "raw_depths": dataset.raw_depths,
        }
    if isinstance(dataset, ConcatDataset):
        parts = [collect_raw_series(part) for part in dataset.datasets]
        return {
            key: np.concatenate([p[key] for p in parts if p[key] is not None]) if any(p[key] is not None for p in parts) else None
            for key in ["raw_features", "raw_labels_p", "raw_labels_s", "raw_labels_t", "raw_depths"]
        }
    if isinstance(dataset, Subset):
        raw = collect_raw_series(dataset.dataset)
        idx = np.asarray(dataset.indices)
        return {
            key: raw[key][idx] if raw[key] is not None else None
            for key in ["raw_features", "raw_labels_p", "raw_labels_s", "raw_labels_t", "raw_depths"]
        }
    raise TypeError(f"Unsupported dataset type for raw-series collection: {type(dataset)!r}")


def center_profile_view(dataset: WindowedWellDataset) -> Dict[str, np.ndarray]:

    if dataset.raw_features is None or dataset.raw_depths is None:
        return {
            "depths": dataset.depths,
            "logs": dataset.features[:, dataset.features.shape[1] // 2, :].cpu().numpy(),
            "labels_p": dataset.labels_p.numpy(),
            "labels_s": dataset.labels_s.numpy(),
            "labels_t": dataset.labels_t.numpy(),
        }

    half = dataset.window_size // 2
    end = -half if half > 0 else None
    return {
        "depths": dataset.raw_depths[half:end],
        "logs": dataset.raw_features[half:end],
        "labels_p": dataset.raw_labels_p[half:end] if dataset.raw_labels_p is not None else dataset.labels_p.numpy(),
        "labels_s": dataset.raw_labels_s[half:end] if dataset.raw_labels_s is not None else dataset.labels_s.numpy(),
        "labels_t": dataset.raw_labels_t[half:end] if dataset.raw_labels_t is not None else dataset.labels_t.numpy(),
    }


def well_summary_frame(datasets: Sequence[WindowedWellDataset]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        p = dataset.labels_p.numpy()
        s = dataset.labels_s.numpy()
        t = dataset.labels_t.numpy()
        rows.append(
            {
                "well_id": dataset.well_id,
                "samples": len(dataset),
                "primary_dominant": int(np.bincount(p).argmax()) if len(p) else -1,
                "secondary_dominant": int(np.bincount(s).argmax()) if len(s) else -1,
                "tertiary_dominant": int(np.bincount(t).argmax()) if len(t) else -1,
                "primary_unique": int(np.unique(p).size),
                "secondary_unique": int(np.unique(s).size),
                "tertiary_unique": int(np.unique(t).size),
            }
        )
    return pd.DataFrame(rows)


def class_counts(labels: np.ndarray, num_classes: int) -> Dict[int, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    return {idx: int(counts[idx]) for idx in range(num_classes)}


def hierarchical_class_weights(
    labels_p: np.ndarray,
    labels_s: np.ndarray,
    labels_t: np.ndarray,
    num_primary: int,
    num_secondary: int,
    num_tertiary: int,
    power: float = 1.0,
) -> Dict[str, torch.Tensor]:
    def _weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)
        if power != 1.0:
            weights = np.power(weights, power)
        weights = weights / weights.mean()
        return torch.as_tensor(weights, dtype=torch.float32)

    return {
        "lvl1": _weights(labels_p, num_primary),
        "lvl2": _weights(labels_s, num_secondary),
        "lvl3": _weights(labels_t, num_tertiary),
    }


def build_sampler(dataset: Dataset):
    labels_p, labels_s, labels_t = collect_labels(dataset)
    return HierarchicalWeightedSampler(labels_p, labels_s, labels_t, alpha=0.6, beta=1.3).get_sampler()


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler: Optional[Any] = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler)


def select_center_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits[:, logits.shape[1] // 2, :]
    return logits


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher_forcing: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        data, labels_p, labels_s, labels_t = batch[:4]
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
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    taxonomy=None,
    constrained: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], PredictionBundle]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    preds_p, preds_s, preds_t = [], [], []
    labels_p_all, labels_s_all, labels_t_all = [], [], []
    logits_p_all, logits_s_all, logits_t_all = [], [], []

    for batch in dataloader:
        data, labels_p, labels_s, labels_t = batch[:4]
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
        total_batches += 1

        logits_p_all.append(logits_p.cpu().numpy())
        logits_s_all.append(logits_s.cpu().numpy())
        logits_t_all.append(logits_t.cpu().numpy())
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
        empty_bundle = PredictionBundle(
            predictions_p=np.array([], dtype=np.int64),
            predictions_s=np.array([], dtype=np.int64),
            predictions_t=np.array([], dtype=np.int64),
            labels_p=np.array([], dtype=np.int64),
            labels_s=np.array([], dtype=np.int64),
            labels_t=np.array([], dtype=np.int64),
        )
        empty_metrics = {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "balanced_accuracy": 0.0,
        }
        return 0.0, empty_metrics, empty_metrics, empty_metrics, empty_bundle

    preds_p_arr = np.concatenate(preds_p).flatten()
    preds_s_arr = np.concatenate(preds_s).flatten()
    preds_t_arr = np.concatenate(preds_t).flatten()
    labels_p_arr = np.concatenate(labels_p_all).flatten()
    labels_s_arr = np.concatenate(labels_s_all).flatten()
    labels_t_arr = np.concatenate(labels_t_all).flatten()

    metrics_p = classification_metrics(preds_p_arr, labels_p_arr)
    metrics_s = classification_metrics(preds_s_arr, labels_s_arr)
    metrics_t = classification_metrics(preds_t_arr, labels_t_arr)

    bundle = PredictionBundle(
        predictions_p=preds_p_arr,
        predictions_s=preds_s_arr,
        predictions_t=preds_t_arr,
        labels_p=labels_p_arr,
        labels_s=labels_s_arr,
        labels_t=labels_t_arr,
        logits_p=np.concatenate(logits_p_all).reshape(len(preds_p_arr), -1),
        logits_s=np.concatenate(logits_s_all).reshape(len(preds_s_arr), -1),
        logits_t=np.concatenate(logits_t_all).reshape(len(preds_t_arr), -1),
    )
    return total_loss / max(total_batches, 1), metrics_p, metrics_s, metrics_t, bundle


def classification_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if len(targets) == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "balanced_accuracy": 0.0,
        }
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)[0]),
        "recall_macro": float(precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)[1]),
        "f1_macro": float(precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)[2]),
        "f1_weighted": float(precision_recall_fscore_support(targets, predictions, average="weighted", zero_division=0)[2]),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
    }


def normalized_confusion(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    if len(targets) == 0 or num_classes <= 0:
        return np.zeros((max(num_classes, 0), max(num_classes, 0)), dtype=np.float32)
    matrix = confusion_matrix(targets, predictions, labels=list(range(num_classes))).astype(np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def per_well_metrics(bundle: PredictionBundle) -> pd.DataFrame:
    if bundle.well_ids is None:
        return pd.DataFrame()
    rows = []
    for well_id in np.unique(bundle.well_ids):
        mask = bundle.well_ids == well_id
        rows.append(
            {
                "well_id": str(well_id),
                **classification_metrics(bundle.predictions_t[mask], bundle.labels_t[mask]),
                "samples": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def attach_well_ids(bundle: PredictionBundle, well_id: str) -> PredictionBundle:
    ids = np.asarray([well_id] * len(bundle.labels_t), dtype=object)
    bundle.well_ids = ids
    return bundle


def predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    criterion: Optional[torch.nn.Module] = None,
    taxonomy=None,
    constrained: bool = False,
) -> PredictionBundle:
    if isinstance(dataset, ConcatDataset):
        parts = [predict_dataset(model, part, device, batch_size=batch_size, criterion=criterion,
                                 taxonomy=taxonomy, constrained=constrained) for part in dataset.datasets]
        return concat_prediction_bundles(parts)
    if isinstance(dataset, Subset):
        base = predict_dataset(model, dataset.dataset, device, batch_size=batch_size, criterion=criterion,
                               taxonomy=taxonomy, constrained=constrained)
        return select_prediction_bundle(base, np.asarray(dataset.indices))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_p, preds_s, preds_t = [], [], []
    labels_p_all, labels_s_all, labels_t_all = [], [], []
    logits_p_all, logits_s_all, logits_t_all = [], [], []
    total_loss = 0.0
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            data, labels_p, labels_s, labels_t = batch[:4]
            data = data.to(device)
            labels_p = labels_p.to(device)
            labels_s = labels_s.to(device)
            labels_t = labels_t.to(device)
            logits_p, logits_s, logits_t = model(data)
            logits_p = select_center_logits(logits_p)
            logits_s = select_center_logits(logits_s)
            logits_t = select_center_logits(logits_t)
            if criterion is not None:
                loss, _ = criterion(logits_p, logits_s, logits_t, labels_p, labels_s, labels_t)
                total_loss += float(loss.item())
                total_batches += 1

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
            logits_p_all.append(logits_p.cpu().numpy())
            logits_s_all.append(logits_s.cpu().numpy())
            logits_t_all.append(logits_t.cpu().numpy())

    if not preds_p:
        return PredictionBundle(
            predictions_p=np.array([], dtype=np.int64),
            predictions_s=np.array([], dtype=np.int64),
            predictions_t=np.array([], dtype=np.int64),
            labels_p=np.array([], dtype=np.int64),
            labels_s=np.array([], dtype=np.int64),
            labels_t=np.array([], dtype=np.int64),
        )

    bundle = PredictionBundle(
        predictions_p=np.concatenate(preds_p).flatten(),
        predictions_s=np.concatenate(preds_s).flatten(),
        predictions_t=np.concatenate(preds_t).flatten(),
        labels_p=np.concatenate(labels_p_all).flatten(),
        labels_s=np.concatenate(labels_s_all).flatten(),
        labels_t=np.concatenate(labels_t_all).flatten(),
        logits_p=np.concatenate(logits_p_all).reshape(len(np.concatenate(preds_p).flatten()), -1),
        logits_s=np.concatenate(logits_s_all).reshape(len(np.concatenate(preds_s).flatten()), -1),
        logits_t=np.concatenate(logits_t_all).reshape(len(np.concatenate(preds_t).flatten()), -1),
    )
    if hasattr(dataset, "well_id"):
        bundle.well_ids = np.asarray([dataset.well_id] * len(bundle.labels_t), dtype=object)
    if hasattr(dataset, "depths"):
        bundle.depths = np.asarray(dataset.depths)
    return bundle


def concat_prediction_bundles(bundles: Sequence[PredictionBundle]) -> PredictionBundle:
    return PredictionBundle(
        predictions_p=np.concatenate([b.predictions_p for b in bundles]),
        predictions_s=np.concatenate([b.predictions_s for b in bundles]),
        predictions_t=np.concatenate([b.predictions_t for b in bundles]),
        labels_p=np.concatenate([b.labels_p for b in bundles]),
        labels_s=np.concatenate([b.labels_s for b in bundles]),
        labels_t=np.concatenate([b.labels_t for b in bundles]),
        depths=np.concatenate([b.depths for b in bundles if b.depths is not None]) if any(b.depths is not None for b in bundles) else None,
        well_ids=np.concatenate([b.well_ids for b in bundles if b.well_ids is not None]) if any(b.well_ids is not None for b in bundles) else None,
        logits_p=np.concatenate([b.logits_p for b in bundles if b.logits_p is not None]) if any(b.logits_p is not None for b in bundles) else None,
        logits_s=np.concatenate([b.logits_s for b in bundles if b.logits_s is not None]) if any(b.logits_s is not None for b in bundles) else None,
        logits_t=np.concatenate([b.logits_t for b in bundles if b.logits_t is not None]) if any(b.logits_t is not None for b in bundles) else None,
    )


def select_prediction_bundle(bundle: PredictionBundle, indices: np.ndarray) -> PredictionBundle:
    indices = np.asarray(indices)
    return PredictionBundle(
        predictions_p=bundle.predictions_p[indices],
        predictions_s=bundle.predictions_s[indices],
        predictions_t=bundle.predictions_t[indices],
        labels_p=bundle.labels_p[indices],
        labels_s=bundle.labels_s[indices],
        labels_t=bundle.labels_t[indices],
        depths=bundle.depths[indices] if bundle.depths is not None else None,
        well_ids=bundle.well_ids[indices] if bundle.well_ids is not None else None,
        logits_p=bundle.logits_p[indices] if bundle.logits_p is not None else None,
        logits_s=bundle.logits_s[indices] if bundle.logits_s is not None else None,
        logits_t=bundle.logits_t[indices] if bundle.logits_t is not None else None,
    )


def evaluate_split(
    model: torch.nn.Module,
    dataset: Dataset,
    criterion: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss, metrics_p, metrics_s, metrics_t, bundle = evaluate_model(model, loader, criterion, device)
    bundle = predict_dataset(model, dataset, device, batch_size=batch_size, criterion=criterion)
    return {
        "loss": loss,
        "primary": metrics_p,
        "secondary": metrics_s,
        "tertiary": metrics_t,
        "bundle": bundle,
        "per_well": per_well_metrics(bundle),
        "confusion_primary": normalized_confusion(bundle.predictions_p, bundle.labels_p, int(bundle.labels_p.max()) + 1 if len(bundle.labels_p) else 0),
        "confusion_secondary": normalized_confusion(bundle.predictions_s, bundle.labels_s, int(bundle.labels_s.max()) + 1 if len(bundle.labels_s) else 0),
        "confusion_tertiary": normalized_confusion(bundle.predictions_t, bundle.labels_t, int(bundle.labels_t.max()) + 1 if len(bundle.labels_t) else 0),
    }


def compute_repeated_well_splits(
    datasets: Sequence[WindowedWellDataset],
    n_splits: int = 5,
    n_repeats: int = 3,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> List[SplitAssignment]:
    summary = well_summary_frame(datasets)
    if summary.empty:
        return []
    if len(summary) < 2:
        wells = tuple(summary["well_id"].astype(str).tolist())
        return [SplitAssignment(0, 0, wells, tuple(), tuple())]

    labels = summary["tertiary_dominant"].to_numpy()
    wells = summary["well_id"].to_numpy()
    splits: List[SplitAssignment] = []

    for repeat_id in range(n_repeats):
        n_actual_splits = min(n_splits, len(summary))
        if n_actual_splits < 2:
            n_actual_splits = 2
        try:
            split_iter = StratifiedKFold(n_splits=n_actual_splits, shuffle=True, random_state=seed + repeat_id).split(wells, labels)
            fold_pairs = list(split_iter)
        except ValueError:
            fold_pairs = list(KFold(n_splits=n_actual_splits, shuffle=True, random_state=seed + repeat_id).split(wells))

        for fold_id, pair in enumerate(fold_pairs):
            train_val_idx, test_idx = pair
            train_val_wells = wells[train_val_idx]
            test_wells = wells[test_idx]
            train_val_labels = labels[train_val_idx]

            if len(train_val_wells) < 2:
                train_wells, val_wells = train_val_wells, np.asarray([], dtype=train_val_wells.dtype)
            else:
                try:
                    train_wells, val_wells = train_test_split(
                        train_val_wells,
                        test_size=min(val_fraction, 0.5),
                        random_state=seed + repeat_id + fold_id,
                        stratify=train_val_labels if len(np.unique(train_val_labels)) > 1 else None,
                    )
                except ValueError:
                    train_wells, val_wells = train_test_split(
                        train_val_wells,
                        test_size=min(val_fraction, 0.5),
                        random_state=seed + repeat_id + fold_id,
                        stratify=None,
                    )

            splits.append(
                SplitAssignment(
                    repeat_id=repeat_id,
                    fold_id=fold_id,
                    train_wells=tuple(str(w) for w in train_wells),
                    val_wells=tuple(str(w) for w in val_wells),
                    test_wells=tuple(str(w) for w in test_wells),
                )
            )

    return splits


def split_datasets(
    datasets: Sequence[WindowedWellDataset],
    split: SplitAssignment,
) -> Tuple[Dataset, Dataset, Dataset]:
    train = subset_by_well_ids(datasets, split.train_wells)
    val = subset_by_well_ids(datasets, split.val_wells)
    test = subset_by_well_ids(datasets, split.test_wells)
    return train, val, test


def summarize_split(
    datasets: Sequence[WindowedWellDataset],
    split: SplitAssignment,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subset_name, wells in [("train", split.train_wells), ("val", split.val_wells), ("test", split.test_wells)]:
        subset = [ds for ds in datasets if ds.well_id in set(wells)]
        if not subset:
            continue
        frame = well_summary_frame(subset)
        rows.append(
            {
                "repeat_id": split.repeat_id,
                "fold_id": split.fold_id,
                "subset": subset_name,
                "wells": len(subset),
                "samples": int(frame["samples"].sum()),
                "primary_unique": int(frame["primary_dominant"].nunique()),
                "secondary_unique": int(frame["secondary_dominant"].nunique()),
                "tertiary_unique": int(frame["tertiary_dominant"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def infer_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    warmup_batches: int = 1,
    timed_batches: int = 10,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    elapsed = 0.0
    seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx < warmup_batches:
                _ = model(batch[0].to(device))
                continue
            if seen >= timed_batches:
                break
            data = batch[0].to(device)
            start = perf_counter()
            _ = model(data)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += perf_counter() - start
            seen += 1
    return elapsed / max(seen, 1)
