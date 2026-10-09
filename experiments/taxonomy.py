

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import json
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class TaxonomySpec:


    primary: Tuple[str, ...]
    secondary: Tuple[str, ...]
    tertiary: Tuple[str, ...]
    secondary_to_primary: Tuple[int, ...]
    tertiary_to_secondary: Tuple[int, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaxonomySpec":
        levels = payload.get("levels", payload)
        primary = tuple(levels.get("primary", ()))
        secondary = tuple(levels.get("secondary", ()))
        tertiary = tuple(levels.get("tertiary", ()))

        mapping = payload.get("mapping", payload)
        s2p = mapping.get("secondary_to_primary", ())
        t2s = mapping.get("tertiary_to_secondary", ())

        return cls(
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            secondary_to_primary=_coerce_index_tuple(s2p),
            tertiary_to_secondary=_coerce_index_tuple(t2s),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TaxonomySpec":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @property
    def num_primary(self) -> int:
        return len(self.primary)

    @property
    def num_secondary(self) -> int:
        return len(self.secondary)

    @property
    def num_tertiary(self) -> int:
        return len(self.tertiary)

    def validate(self) -> None:
        if len(self.secondary_to_primary) != self.num_secondary:
            raise ValueError("secondary_to_primary must match the number of secondary classes.")
        if len(self.tertiary_to_secondary) != self.num_tertiary:
            raise ValueError("tertiary_to_secondary must match the number of tertiary classes.")
        if any(idx < 0 or idx >= self.num_primary for idx in self.secondary_to_primary):
            raise ValueError("secondary_to_primary contains an out-of-range primary index.")
        if any(idx < 0 or idx >= self.num_secondary for idx in self.tertiary_to_secondary):
            raise ValueError("tertiary_to_secondary contains an out-of-range secondary index.")

    def matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:

        self.validate()
        m21 = torch.zeros(self.num_primary, self.num_secondary, dtype=torch.float32)
        m32 = torch.zeros(self.num_secondary, self.num_tertiary, dtype=torch.float32)

        for s_idx, p_idx in enumerate(self.secondary_to_primary):
            m21[p_idx, s_idx] = 1.0
        for t_idx, s_idx in enumerate(self.tertiary_to_secondary):
            m32[s_idx, t_idx] = 1.0
        return m21, m32

    def parent_primary(self, secondary_idx: np.ndarray | Sequence[int]) -> np.ndarray:
        arr = np.asarray(secondary_idx, dtype=np.int64)
        mapping = np.asarray(self.secondary_to_primary, dtype=np.int64)
        return mapping[arr]

    def parent_secondary(self, tertiary_idx: np.ndarray | Sequence[int]) -> np.ndarray:
        arr = np.asarray(tertiary_idx, dtype=np.int64)
        mapping = np.asarray(self.tertiary_to_secondary, dtype=np.int64)
        return mapping[arr]

    def class_table(
        self,
        primary_counts: Mapping[int, int],
        secondary_counts: Mapping[int, int],
        tertiary_counts: Mapping[int, int],
        split_name: str,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for level, names, counts in [
            ("primary", self.primary, primary_counts),
            ("secondary", self.secondary, secondary_counts),
            ("tertiary", self.tertiary, tertiary_counts),
        ]:
            total = float(sum(counts.values()) or 1)
            for idx, name in enumerate(names):
                rows.append(
                    {
                        "split": split_name,
                        "level": level,
                        "class_id": idx,
                        "class_name": name,
                        "count": int(counts.get(idx, 0)),
                        "share": float(counts.get(idx, 0)) / total,
                    }
                )
        return pd.DataFrame(rows)


def _coerce_index_tuple(values: Any) -> Tuple[int, ...]:
    if isinstance(values, Mapping):
        values = [values[k] for k in sorted(values, key=lambda item: int(item))]
    return tuple(int(v) for v in values)


def path_validity(
    pred_primary: Sequence[int] | np.ndarray,
    pred_secondary: Sequence[int] | np.ndarray,
    pred_tertiary: Sequence[int] | np.ndarray,
    taxonomy: TaxonomySpec,
) -> np.ndarray:

    pred_primary = np.asarray(pred_primary, dtype=np.int64)
    pred_secondary = np.asarray(pred_secondary, dtype=np.int64)
    pred_tertiary = np.asarray(pred_tertiary, dtype=np.int64)
    taxonomy.validate()

    parent_primary = taxonomy.parent_primary(pred_secondary)
    parent_secondary = taxonomy.parent_secondary(pred_tertiary)
    return (pred_primary == parent_primary) & (pred_secondary == parent_secondary)


def constrained_decode(
    logits_primary: torch.Tensor,
    logits_secondary: torch.Tensor,
    logits_tertiary: torch.Tensor,
    taxonomy: TaxonomySpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    taxonomy.validate()
    prob_p = torch.softmax(logits_primary, dim=-1)
    prob_s = torch.softmax(logits_secondary, dim=-1)
    prob_t = torch.softmax(logits_tertiary, dim=-1)

    pred_t = prob_t.argmax(dim=-1)
    allowed_s = torch.as_tensor(taxonomy.parent_secondary(pred_t.detach().cpu().numpy()), device=pred_t.device)
    pred_s = _restrict_to_allowed_classes(prob_s, allowed_s)

    allowed_p = torch.as_tensor(taxonomy.parent_primary(pred_s.detach().cpu().numpy()), device=pred_s.device)
    pred_p = _restrict_to_allowed_classes(prob_p, allowed_p)
    return pred_p, pred_s, pred_t


def hierarchy_report(
    pred_primary: Sequence[int] | np.ndarray,
    pred_secondary: Sequence[int] | np.ndarray,
    pred_tertiary: Sequence[int] | np.ndarray,
    true_primary: Sequence[int] | np.ndarray,
    true_secondary: Sequence[int] | np.ndarray,
    true_tertiary: Sequence[int] | np.ndarray,
    taxonomy: TaxonomySpec,
) -> Dict[str, float]:

    pred_primary = np.asarray(pred_primary, dtype=np.int64)
    pred_secondary = np.asarray(pred_secondary, dtype=np.int64)
    pred_tertiary = np.asarray(pred_tertiary, dtype=np.int64)
    true_primary = np.asarray(true_primary, dtype=np.int64)
    true_secondary = np.asarray(true_secondary, dtype=np.int64)
    true_tertiary = np.asarray(true_tertiary, dtype=np.int64)

    valid_mask = path_validity(pred_primary, pred_secondary, pred_tertiary, taxonomy)
    predicted_secondary_parent = taxonomy.parent_secondary(pred_tertiary)
    predicted_primary_parent = taxonomy.parent_primary(pred_secondary)
    true_secondary_parent = taxonomy.parent_secondary(true_tertiary)
    true_primary_parent = taxonomy.parent_primary(true_secondary)

    return {
        "invalid_path_rate": float(1.0 - valid_mask.mean()) if len(valid_mask) else 0.0,
        "secondary_parent_match_rate": float((pred_secondary == predicted_secondary_parent).mean()) if len(pred_secondary) else 0.0,
        "primary_parent_match_rate": float((pred_primary == predicted_primary_parent).mean()) if len(pred_primary) else 0.0,
        "ground_truth_secondary_parent_match_rate": float((true_secondary == true_secondary_parent).mean()) if len(true_secondary) else 0.0,
        "ground_truth_primary_parent_match_rate": float((true_primary == true_primary_parent).mean()) if len(true_primary) else 0.0,
    }


def _restrict_to_allowed_classes(probabilities: torch.Tensor, allowed_indices: torch.Tensor) -> torch.Tensor:
    result = []
    allowed_indices = allowed_indices.to(probabilities.device)
    for row, allowed in zip(probabilities, allowed_indices):
        if allowed.dim() == 0:
            allowed = allowed.unsqueeze(0)
        valid = allowed.unique(sorted=True)
        if valid.numel() == 0:
            result.append(row.argmax(dim=-1))
            continue
        scores = row.index_select(0, valid)
        result.append(valid[scores.argmax(dim=-1)])
    return torch.stack(result, dim=0)

