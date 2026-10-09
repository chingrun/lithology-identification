

import torch
import torch.nn as nn
from typing import Literal


def get_activation(activation: str) -> nn.Module:

    activation = activation.lower()

    activation_map = {
        'gelu': nn.GELU(),
        'rrelu': nn.RReLU(inplace=True),
        'selu': nn.SELU(inplace=True),
        'silu': nn.SiLU(inplace=True),
        'hardswish': nn.Hardswish(inplace=True),
        'leakyrelu': nn.LeakyReLU(inplace=True),
        'relu': nn.ReLU(inplace=True)
    }

    return activation_map.get(activation, nn.ReLU(inplace=True))


class ConvBNReLU1D(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        bias: bool = True,
        activation: str = 'relu'
    ):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding
            ),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)


class ConvBNReLURes1D(nn.Module):


    def __init__(
        self,
        channel: int,
        kernel_size: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
        activation: str = 'relu'
    ):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)

        expanded_channel = int(channel * res_expansion)

        # First convolution with expansion
        self.net1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channel,
                out_channels=expanded_channel,
                kernel_size=kernel_size,
                groups=groups,
                bias=bias
            ),
            nn.BatchNorm1d(expanded_channel),
            self.act
        )

        # Second convolution back to original channels
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=expanded_channel,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias
                ),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    bias=bias
                ),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=expanded_channel,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    bias=bias
                ),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.act(self.net2(self.net1(x)) + x)
