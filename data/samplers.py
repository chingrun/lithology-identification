

import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from typing import List


class HierarchicalWeightedSampler:
    

    def __init__(
        self,
        primary_labels: np.ndarray,
        secondary_labels: np.ndarray,
        tertiary_labels: np.ndarray,
        alpha: float = 0.6,
        beta: float = 1.3
    ):
        self.alpha = alpha
        self.beta = beta

        # Compute label frequencies
        self.primary_counts = Counter(primary_labels.flatten())
        self.secondary_counts = Counter(zip(primary_labels.flatten(), secondary_labels.flatten()))
        self.tertiary_counts = Counter(zip(
            primary_labels.flatten(),
            secondary_labels.flatten(),
            tertiary_labels.flatten()
        ))

        # Calculate sample weights
        self.weights = self._calculate_weights(
            primary_labels.flatten(),
            secondary_labels.flatten(),
            tertiary_labels.flatten()
        )

    def _calculate_weights(
        self,
        p_labels: np.ndarray,
        s_labels: np.ndarray,
        t_labels: np.ndarray
    ) -> torch.Tensor:
       
        weights = []

        for p, s, t in zip(p_labels, s_labels, t_labels):
            # Primary weight
            w_p = 1.0 / (self.primary_counts[p] ** (1 / self.beta))

            # Secondary weight (conditioned on primary)
            w_s = 1.0 / (self.secondary_counts[(p, s)] ** (1 / self.beta))

            # Tertiary weight (conditioned on primary and secondary)
            w_t = 1.0 / (self.tertiary_counts[(p, s, t)] ** (1 / self.beta))

            # Hierarchical combination with emphasis on tertiary
            weight = (self.alpha ** 2) * w_p + self.alpha * w_s + 1.2 * w_t
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-12)

        return torch.from_numpy(weights.astype(np.float32)).view(-1)

    def get_sampler(self) -> WeightedRandomSampler:
        
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.weights),
            replacement=True
        )
