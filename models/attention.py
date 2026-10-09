

import torch
import torch.nn as nn
from typing import Tuple


class SpatialAttention(nn.Module):


    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        if kernel_size not in (3, 7):
            raise ValueError('kernel_size must be 3 or 7')

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_tensor = x.clone()

        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and learn attention weights
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        attention_weights = self.sigmoid(x)

        # Apply attention
        output_tensor = torch.mul(input_tensor, attention_weights)
        return output_tensor


class ChannelSELayer(nn.Module):


    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        super(ChannelSELayer, self).__init__()

        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        # Squeeze: Global average pooling
        squeeze_tensor = input_tensor.mean(dim=2)

        # Excitation: Two-layer MLP
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # Reshape and apply channel weights
        batch_size, num_channels = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1))

        return output_tensor
