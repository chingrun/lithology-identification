


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):


    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        # Registered as a buffer so that criterion.to(device) moves the class
        # weights together with the model (prevents CPU/CUDA mismatches).
        self.register_buffer("weight", weight)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # Compute cross entropy
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

        # Compute focal term: (1 - pt)^gamma
        with torch.no_grad():
            pt = torch.exp(-ce).clamp_min(1e-6)  # Probability of true class

        focal = (1 - pt) ** self.gamma * ce

        # Apply reduction
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal
