

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .focal_loss import FocalLoss


class DynamicHierarchicalLoss(nn.Module):

    def __init__(
        self,
        num_primary: int,
        num_secondary: int,
        num_tertiary: int,
        task_weights: Tuple[float, float, float] = (1.0, 1.2, 1.5),
        lambda_consistency: float = 0.5,
        tertiary_gamma: float = 2.0,
        label_smoothing_ter: float = 0.0,
        M32: Optional[torch.Tensor] = None,
        M21: Optional[torch.Tensor] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        lambda_auc: float = 0.1,
        mode: str = "DHLF"
    ):
        super().__init__()
        self.mode = mode
        self.num_primary = num_primary
        self.num_secondary = num_secondary
        self.num_tertiary = num_tertiary

        # Task weights (Eq. 7)
        self.alpha, self.beta, self.gamma = task_weights
        self.lambda_consistency = lambda_consistency
        self.lambda_auc = lambda_auc

        # Class weights
        if class_weights is None:
            class_weights = {}

        self.register_buffer("w1_buf", class_weights.get("lvl1", torch.ones(num_primary)))
        self.register_buffer("w2_buf", class_weights.get("lvl2", torch.ones(num_secondary)))
        self.register_buffer("w3_buf", class_weights.get("lvl3", torch.ones(num_tertiary)))

        # Focal loss for the tertiary level (Eqs. 9, 19): pure focal
        # weighting with exponent phi, WITHOUT class weights.
        if self.mode != "NOFOCAL":
            self.focal3 = FocalLoss(
                gamma=tertiary_gamma,
                weight=None,
                reduction="mean",
                ignore_index=-1,
                label_smoothing=label_smoothing_ter
            )

        # Hierarchy mapping matrices (columns = finer classes, one-hot
        # parent rows, Eq. 10). When no taxonomy is supplied, uniform
        # all-ones aggregations are used as a fallback.
        provided_m32 = M32 is not None
        provided_m21 = M21 is not None
        if M32 is None:
            M32 = torch.ones(num_secondary, num_tertiary)
        if M21 is None:
            M21 = torch.ones(num_primary, num_secondary)

        if provided_m32:
            self._validate_parent_matrix(M32, "M32 (tertiary -> secondary)")
        if provided_m21:
            self._validate_parent_matrix(M21, "M21 (secondary -> primary)")

        self.register_buffer("M32", M32.float())  # [S, T]
        self.register_buffer("M21", M21.float())  # [P, S]

    @staticmethod
    def _validate_parent_matrix(matrix: torch.Tensor, name: str) -> None:
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return
        column_sums = matrix.sum(dim=0)
        if not torch.allclose(column_sums, torch.ones_like(column_sums)):
            raise ValueError(
                f"{name} must have one-hot columns (each finer class maps to "
                f"exactly one parent); got column sums {column_sums.tolist()}"
            )

    def forward(
        self,
        logits_p: torch.Tensor,
        logits_s: torch.Tensor,
        logits_t: torch.Tensor,
        labels_p: torch.Tensor,
        labels_s: torch.Tensor,
        labels_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device = logits_p.device

        # Move weights to correct device
        w1 = self.w1_buf.to(device)
        w2 = self.w2_buf.to(device)

        # Task losses (Eq. 8): class-weighted cross-entropy for the coarse
        # heads; focal weighting for the tertiary head (Eq. 9).
        loss_p = F.cross_entropy(logits_p, labels_p, weight=w1, reduction="mean")
        loss_s = F.cross_entropy(logits_s, labels_s, weight=w2, reduction="mean")

        if self.mode != "NOFOCAL":
            loss_t = self.focal3(logits_t, labels_t)
        else:
            w3 = self.w3_buf.to(device)
            loss_t = F.cross_entropy(logits_t, labels_t, weight=w3, reduction="mean")

        # Consistency regularization (Eqs. 10-11, 23-24): aggregate the
        # fine-level predictions upward into the hierarchy-implied coarse
        # distributions q, and regularise the direct coarse predictions p
        # toward them: L_cons = (1/N) sum_i [KL(q_{i,p} || p_{i,p}) +
        # KL(q_{i,s} || p_{i,s})]. The 0 * log 0 convention keeps the term
        # finite when a class carries no probability mass.
        prob_t = F.softmax(logits_t, dim=1)
        prob_s = F.softmax(logits_s, dim=1)
        prob_p = F.softmax(logits_p, dim=1)

        # Tertiary -> Secondary consistency: q_{i,s} = M_{t->s} p_{i,t}
        prob_s_from_t = torch.matmul(prob_t, self.M32.T.to(device))
        consistency_32 = self._kl_divergence(prob_s_from_t, prob_s)

        # Secondary -> Primary consistency: q_{i,p} = M_{s->p} p_{i,s}
        prob_p_from_s = torch.matmul(prob_s, self.M21.T.to(device))
        consistency_21 = self._kl_divergence(prob_p_from_s, prob_p)

        consistency_loss = consistency_32 + consistency_21

        # AUC surrogate on the tertiary branch (Eqs. 12-13)
        auc_loss = self._multiclass_auc_surrogate(logits_t, labels_t)

        # Total loss (Eq. 7)
        task_loss = self.alpha * loss_p + self.beta * loss_s + self.gamma * loss_t
        total_loss = task_loss + self.lambda_consistency * consistency_loss + self.lambda_auc * auc_loss

        # Return loss components
        loss_dict = {
            'loss_primary': loss_p.item(),
            'loss_secondary': loss_s.item(),
            'loss_tertiary': loss_t.item(),
            'consistency_loss': consistency_loss.item(),
            'auc_loss': auc_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    @staticmethod
    def _kl_divergence(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:

        q = q.clamp_min(1e-9)
        p = p.clamp_min(1e-9)
        per_sample = (q * (q.log() - p.log())).sum(dim=1)
        return per_sample.mean()

    def _multiclass_auc_surrogate(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        probs = F.softmax(logits, dim=1)
        losses = []

        for class_idx in range(probs.shape[1]):
            pos = probs[targets == class_idx, class_idx]
            neg = probs[targets != class_idx, class_idx]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            pairwise = pos.unsqueeze(1) - neg.unsqueeze(0)
            losses.append(F.softplus(-pairwise).mean())

        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()
