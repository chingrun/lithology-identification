

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # torch is only needed for loading .pt/.pth archives
    import torch
except Exception:  # pragma: no cover
    torch = None


FEATURE_COLUMNS = ("GR", "AC", "CNL", "DEN", "RLLD", "RLLS")
LABEL_COLUMNS = ("primary", "secondary", "tertiary")
RAW_WELL_COLUMN = "dWell"
RAW_DEPTH_COLUMN = "dDepth"
CANONICAL_WELL_COLUMN = "well_id"
CANONICAL_DEPTH_COLUMN = "depth"
MISSING_LABEL_TOKENS = {"", "nan", "none", "null", "<na>"}
SUPPORTED_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".csv"}


def _is_missing_text(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_LABEL_TOKENS


def normalize_label_part(value: Any, fallback: str) -> str:
    return fallback if _is_missing_text(value) else str(value).strip()


def split_hierarchical_lithology(value: Any) -> Tuple[str, str, str]:
    if _is_missing_text(value):
        return "unknown", "None", "None"

    parts = str(value).split(",")
    primary = normalize_label_part(parts[0] if len(parts) > 0 else None, "unknown")
    secondary = normalize_label_part(parts[1] if len(parts) > 1 else None, "None")
    tertiary = normalize_label_part(parts[2] if len(parts) > 2 else None, "None")
    return primary, secondary, tertiary


def detect_feature_columns(frame: pd.DataFrame) -> List[str]:
    candidates = [c for c in frame.columns if c.startswith("log_") or c in FEATURE_COLUMNS]
    if candidates:
        return candidates

    excluded = set(LABEL_COLUMNS) | {
        "LITHOLOGY",
        RAW_WELL_COLUMN,
        RAW_DEPTH_COLUMN,
        CANONICAL_WELL_COLUMN,
        CANONICAL_DEPTH_COLUMN,
    }
    numeric_candidates = [
        c for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if numeric_candidates:
        return numeric_candidates
    raise ValueError("Could not infer feature columns from the input frame.")


def _fill_feature_nans(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    ordered = frame.sort_values([CANONICAL_WELL_COLUMN, CANONICAL_DEPTH_COLUMN], kind="stable").copy()
    ordered.loc[:, feature_cols] = ordered.groupby(CANONICAL_WELL_COLUMN, sort=False)[list(feature_cols)].transform(
        lambda group: group.interpolate(method="linear", limit_direction="both").ffill().bfill()
    )
    return ordered.dropna(subset=list(feature_cols)).copy()


def canonicalize_reference_frame(frame: pd.DataFrame) -> pd.DataFrame:

    normalized = frame.copy()
    well_source = CANONICAL_WELL_COLUMN if CANONICAL_WELL_COLUMN in normalized.columns else RAW_WELL_COLUMN
    depth_source = CANONICAL_DEPTH_COLUMN if CANONICAL_DEPTH_COLUMN in normalized.columns else RAW_DEPTH_COLUMN

    if well_source not in normalized.columns or depth_source not in normalized.columns:
        raise ValueError("CSV input must contain a well identifier and a depth column.")
    if "LITHOLOGY" not in normalized.columns and not all(col in normalized.columns for col in LABEL_COLUMNS):
        raise ValueError("CSV input must contain either LITHOLOGY or the split label columns.")

    normalized = normalized.rename(columns={well_source: CANONICAL_WELL_COLUMN, depth_source: CANONICAL_DEPTH_COLUMN})
    normalized = normalized.dropna(subset=[CANONICAL_WELL_COLUMN, CANONICAL_DEPTH_COLUMN]).copy()
    normalized[CANONICAL_WELL_COLUMN] = normalized[CANONICAL_WELL_COLUMN].astype(str)
    normalized[CANONICAL_DEPTH_COLUMN] = pd.to_numeric(normalized[CANONICAL_DEPTH_COLUMN], errors="coerce")
    normalized = normalized.dropna(subset=[CANONICAL_DEPTH_COLUMN]).copy()

    if "LITHOLOGY" in normalized.columns:
        labels = normalized["LITHOLOGY"].map(split_hierarchical_lithology)
        if len(normalized) > 0:
            primary, secondary, tertiary = zip(*labels)
            normalized["primary"] = list(primary)
            normalized["secondary"] = list(secondary)
            normalized["tertiary"] = list(tertiary)
        else:
            normalized["primary"] = []
            normalized["secondary"] = []
            normalized["tertiary"] = []
    else:
        normalized["primary"] = normalized["primary"].map(lambda v: normalize_label_part(v, "unknown"))
        normalized["secondary"] = normalized["secondary"].map(lambda v: normalize_label_part(v, "None"))
        normalized["tertiary"] = normalized["tertiary"].map(lambda v: normalize_label_part(v, "None"))

    feature_cols = detect_feature_columns(normalized)
    normalized.loc[:, feature_cols] = normalized.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    if normalized.loc[:, feature_cols].isna().any().any():
        normalized = _fill_feature_nans(normalized, feature_cols)

    normalized = normalized.sort_values([CANONICAL_WELL_COLUMN, CANONICAL_DEPTH_COLUMN], kind="stable").reset_index(drop=True)
    ordered_cols = [CANONICAL_WELL_COLUMN, CANONICAL_DEPTH_COLUMN, *feature_cols, *LABEL_COLUMNS]
    return normalized.loc[:, ordered_cols].copy()


def _label_class_order(values: Iterable[str], preferred: Sequence[str]) -> List[str]:
    unique = list(dict.fromkeys(values))
    ordered = [token for token in preferred if token in unique]
    ordered.extend(sorted(token for token in unique if token not in preferred))
    return ordered


def fit_label_maps(frames: Sequence[pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    mappings: Dict[str, Dict[str, int]] = {}
    preferred_tokens = {
        "primary": ("unknown",),
        "secondary": ("None",),
        "tertiary": ("None",),
    }
    for column in LABEL_COLUMNS:
        values: List[str] = []
        for frame in frames:
            if column not in frame.columns:
                raise ValueError(f"Missing label column: {column}")
            values.extend(frame[column].astype(str).tolist())
        ordered = _label_class_order(values, preferred_tokens[column])
        mappings[column] = {value: idx for idx, value in enumerate(ordered)}
    return mappings


def encode_label_frame(frame: pd.DataFrame, label_maps: Mapping[str, Mapping[str, int]]) -> pd.DataFrame:
    encoded = frame.copy()
    for column in LABEL_COLUMNS:
        if column not in encoded.columns:
            raise ValueError(f"Missing label column: {column}")
        mapped = encoded[column].map(label_maps[column])
        if mapped.isna().any():
            missing = sorted(set(encoded.loc[mapped.isna(), column].astype(str).tolist()))
            raise ValueError(f"Unknown labels for {column}: {missing}")
        encoded[column] = mapped.astype(np.int64)
    return encoded


def load_archive(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if suffix == ".npy":
        data = np.load(path, allow_pickle=True).item()
        return dict(data)
    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise ImportError("Loading .pt/.pth archives requires PyTorch (pip install torch).")
        data = torch.load(path, map_location="cpu")
        return data if isinstance(data, dict) else {"data": data}
    if suffix == ".csv":
        return {"frame": pd.read_csv(path)}
    raise ValueError(f"Unsupported data file: {path}")


def load_records_from_path(data_path: str | Path) -> List[Dict[str, Any]]:
    root = Path(data_path)
    if root.is_file():
        return [load_archive(root)]

    files = sorted([p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES])
    if not files:
        raise FileNotFoundError("No supported data files found. Expected npz/npy/pt/pth/csv archives in data_path.")
    return [load_archive(path) for path in files]


def prepare_records_for_model(
    records: Sequence[Mapping[str, Any]],
    resample_interval: float | None = None,
    hampel_window: int | None = None,
    hampel_sigma: float = 3.0,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = [dict(record) for record in records]
    frame_positions: List[int] = []
    canonical_frames: List[pd.DataFrame] = []

    for idx, record in enumerate(prepared):
        if "frame" not in record:
            continue
        canonical = canonicalize_reference_frame(record["frame"])
        feature_cols = detect_feature_columns(canonical)
        if hampel_window is not None and hampel_window > 0:
            canonical = apply_hampel_filter(
                canonical, feature_cols, window_size=hampel_window, n_sigma=hampel_sigma
            )
        if resample_interval is not None and resample_interval > 0:
            canonical = resample_depth_grid(canonical, interval=resample_interval)
        prepared[idx]["frame"] = canonical
        frame_positions.append(idx)
        canonical_frames.append(canonical)

    if canonical_frames:
        label_maps = fit_label_maps(canonical_frames)
        for idx in frame_positions:
            prepared[idx]["frame"] = encode_label_frame(prepared[idx]["frame"], label_maps)

    return prepared


def load_prepared_records(
    data_path: str | Path,
    resample_interval: float | None = None,
    hampel_window: int | None = None,
    hampel_sigma: float = 3.0,
) -> List[Dict[str, Any]]:
    return prepare_records_for_model(
        load_records_from_path(data_path),
        resample_interval=resample_interval,
        hampel_window=hampel_window,
        hampel_sigma=hampel_sigma,
    )


def extract_record_arrays(record: Mapping[str, Any]):
    if "frame" in record:
        frame = record["frame"]
        feature_cols = detect_feature_columns(frame)
        features = frame[feature_cols].to_numpy()
        labels_p = frame["primary"].to_numpy()
        labels_s = frame["secondary"].to_numpy()
        labels_t = frame["tertiary"].to_numpy()
        well_ids = frame[CANONICAL_WELL_COLUMN].to_numpy() if CANONICAL_WELL_COLUMN in frame.columns else None
        depth = frame[CANONICAL_DEPTH_COLUMN].to_numpy() if CANONICAL_DEPTH_COLUMN in frame.columns else None
    else:
        features = _resolve_array(record, "features", "x", "logs")
        labels_p = _resolve_array(record, "labels_p", "primary_labels", "primary")
        labels_s = _resolve_array(record, "labels_s", "secondary_labels", "secondary")
        labels_t = _resolve_array(record, "labels_t", "tertiary_labels", "tertiary")
        well_ids = None
        depth = None
        for key in ("well_ids", "well_id", "wells"):
            if key in record:
                well_ids = np.asarray(record[key])
                break
        if "depth" in record:
            depth = np.asarray(record["depth"])

    return (
        np.asarray(features),
        np.asarray(labels_p),
        np.asarray(labels_s),
        np.asarray(labels_t),
        None if well_ids is None else np.asarray(well_ids),
        None if depth is None else np.asarray(depth),
    )


def _resolve_array(container: Mapping[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in container:
            return np.asarray(container[name])
    raise KeyError(f"Missing required field. Tried: {', '.join(names)}")

def hampel_filter(series: np.ndarray, window_size: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    
    values = np.asarray(series, dtype=np.float64)
    if values.size < window_size:
        return values.copy()
    s = pd.Series(values)
    median = s.rolling(window_size, center=True, min_periods=1).median().to_numpy()
    mad = pd.Series(np.abs(values - median)).rolling(
        window_size, center=True, min_periods=1
    ).median().to_numpy()
    threshold = n_sigma * 1.4826 * mad
    corrected = np.where(np.abs(values - median) > threshold, median, values)
    return corrected.astype(series.dtype) if np.issubdtype(np.asarray(series).dtype, np.integer) else corrected


def apply_hampel_filter(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int = 5,
    n_sigma: float = 3.0,
    per_well: bool = True,
) -> pd.DataFrame:
    
    filtered = frame.copy()
    if per_well and CANONICAL_WELL_COLUMN in filtered.columns:
        parts = []
        for _, group in filtered.groupby(CANONICAL_WELL_COLUMN, sort=False):
            group = group.sort_values(CANONICAL_DEPTH_COLUMN, kind="stable")
            for column in feature_cols:
                group[column] = hampel_filter(group[column].to_numpy(), window_size, n_sigma)
            parts.append(group)
        if parts:
            filtered = pd.concat(parts, ignore_index=True)
    else:
        for column in feature_cols:
            filtered[column] = hampel_filter(filtered[column].to_numpy(), window_size, n_sigma)
    return filtered.sort_values([CANONICAL_WELL_COLUMN, CANONICAL_DEPTH_COLUMN], kind="stable").reset_index(drop=True)


def resample_depth_grid(frame: pd.DataFrame, interval: float = 0.125) -> pd.DataFrame:
   
    if CANONICAL_WELL_COLUMN not in frame.columns or CANONICAL_DEPTH_COLUMN not in frame.columns:
        raise ValueError("Depth-grid resampling requires well_id and depth columns.")
    feature_cols = detect_feature_columns(frame)
    resampled_parts: List[pd.DataFrame] = []
    for well_id, group in frame.groupby(CANONICAL_WELL_COLUMN, sort=False):
        group = group.sort_values(CANONICAL_DEPTH_COLUMN, kind="stable").reset_index(drop=True)
        depths = group[CANONICAL_DEPTH_COLUMN].to_numpy(dtype=np.float64)
        if len(depths) < 2:
            resampled_parts.append(group)
            continue
        start = np.ceil(depths.min() / interval) * interval
        end = np.floor(depths.max() / interval) * interval
        grid = np.arange(start, end + interval * 0.5, interval)
        if len(grid) < 2:
            resampled_parts.append(group)
            continue
        rows: Dict[str, Any] = {CANONICAL_WELL_COLUMN: well_id, CANONICAL_DEPTH_COLUMN: grid}
        for column in feature_cols:
            rows[column] = np.interp(grid, depths, group[column].to_numpy(dtype=np.float64))
        for column in LABEL_COLUMNS:
            if column not in group.columns:
                continue
            nearest = np.searchsorted(depths, grid, side="left")
            nearest = np.clip(nearest, 0, len(depths) - 1)
            left = np.clip(nearest - 1, 0, len(depths) - 1)
            rows[column] = np.where(
                np.abs(grid - depths[left]) <= np.abs(grid - depths[nearest]),
                group[column].to_numpy()[left],
                group[column].to_numpy()[nearest],
            )
        resampled_parts.append(pd.DataFrame(rows))
    if not resampled_parts:
        return frame.copy()
    resampled = pd.concat(resampled_parts, ignore_index=True)
    ordered_cols = [column for column in frame.columns if column in resampled.columns]
    return resampled.loc[:, ordered_cols].copy()


class MinMaxFeatureScaler:
    

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "MinMaxFeatureScaler":
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 3:
            flat = features.reshape(-1, features.shape[-1])
        else:
            flat = features
        self.min_ = flat.min(axis=0)
        self.max_ = flat.max(axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("MinMaxFeatureScaler must be fitted before transform().")
        features = np.asarray(features, dtype=np.float32)
        return (features - self.min_) / (self.max_ - self.min_ + 1e-8)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"min": self.min_, "max": self.max_}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MinMaxFeatureScaler":
        scaler = cls()
        scaler.min_ = np.asarray(payload["min"], dtype=np.float32)
        scaler.max_ = np.asarray(payload["max"], dtype=np.float32)
        return scaler

