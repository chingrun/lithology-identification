

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .attention import SpatialAttention, ChannelSELayer
from .blocks import ConvBNReLU1D, ConvBNReLURes1D


class AttentionBlock(nn.Module):


    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU1D(dim, dim_out, 1),
            ConvBNReLURes1D(dim_out, 1, 1, True),
            ChannelSELayer(num_channels=dim_out, reduction_ratio=2),
            ConvBNReLURes1D(dim_out, 1, 1, True),
            ConvBNReLU1D(dim_out, dim_out, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Block(nn.Module):


    def __init__(self, dim: int, dim_out: int, inner_channel: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU1D(inner_channel, dim, 1),
            ConvBNReLURes1D(dim, 1, 1, True),
            ConvBNReLU1D(dim, dim_out, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def get_in_channels(feat_scales: List[int], inner_channel: int,
                    channel_multiplier: List[int]) -> int:

    in_channels = 0
    for scale in feat_scales:
        in_channels = inner_channel * channel_multiplier[scale]
    return in_channels


class DecoderStream(nn.Module):


    def __init__(
        self,
        in_channels: int,
        scale_channels: int,
        kernel_size: int,
        use_channel_attention: bool,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [ConvBNReLU1D(in_channels, scale_channels, kernel_size, padding=padding)]
        if use_channel_attention:
            layers.append(ChannelSELayer(num_channels=scale_channels, reduction_ratio=2))
        layers.append(ConvBNReLU1D(scale_channels, scale_channels, 1))
        layers.append(ConvBNReLURes1D(scale_channels, 1, 1, True))
        layers.append(ConvBNReLU1D(scale_channels, scale_channels, 1))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalDecoder(nn.Module):


    def __init__(
        self,
        in_channels: int,
        feat_scales: List[int],
        out_channels: int = 6,
        inner_channel: int = 240,
        channel_multiplier: Optional[List[int]] = None,
        time_steps: Optional[List[int]] = None
    ):
        super().__init__()
        feat_scales = sorted(feat_scales)
        self.feat_scales = feat_scales
        self.time_steps = time_steps
        if channel_multiplier is None:
            channel_multiplier = [1, 2, 4, 8]

        # Four parallel streams at increasing scales (240 / 480 / 960 / 1920).
        # Kernel sizes 1/3/5/7 follow the (Cin -> Cout, k/s/p) annotations of
        # Figure 3; streams 1-3 carry channel attention, stream 4 is
        # attention-free.
        scale_channels = [inner_channel * channel_multiplier[scale] for scale in feat_scales]
        kernel_sizes = [1, 3, 5, 7]
        self.streams = nn.ModuleList()
        for idx, scale in enumerate(scale_channels):
            self.streams.append(
                DecoderStream(
                    in_channels=in_channels,
                    scale_channels=scale,
                    kernel_size=kernel_sizes[min(idx, len(kernel_sizes) - 1)],
                    use_channel_attention=idx < 3,
                )
            )

        concatenated_channels = sum(scale_channels)

        # Classification stage collapsing the multi-scale feature tensor to the
        # shared head embedding fd:
        #   C -> C/2 -> C/8 -> C/16 -> fd   (with average pooling between),
        # mirroring the 1920 -> 960 -> 240 -> 120 -> 30 -> 6 chain of Figure 3.
        self.clfr_stg1 = ConvBNReLU1D(concatenated_channels, int(concatenated_channels / 2), 3, padding=1)
        self.clfr_stg2 = ConvBNReLU1D(int(concatenated_channels / 2), int(concatenated_channels / 8), 3, padding=1)
        self.clfr_stg3 = ConvBNReLU1D(int(concatenated_channels / 8), int(concatenated_channels / 16), 3, padding=1)
        self.clfr_stgn = ConvBNReLU1D(int(concatenated_channels / 16), out_channels, kernel_size=3, padding=1)
        self.average_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, f_A: torch.Tensor) -> torch.Tensor:

        stream_outputs = [stream(f_A) for stream in self.streams]
        pooled = [self.average_pool(out) for out in stream_outputs]
        cm = torch.cat(pooled, dim=1)  # (B, sum(C_i), 1)

        cm = self.clfr_stg1(cm)
        cm = self.average_pool(cm)
        cm = self.clfr_stg2(cm)
        cm = self.average_pool(cm)
        cm = self.clfr_stg3(cm)
        cm = self.average_pool(cm)
        cm = self.clfr_stgn(cm)

        return cm


class HierarchicalLithologyModel(nn.Module):


    def __init__(
        self,
        device: torch.device,
        in_channels: int,
        num_primary: int,
        num_secondary: int,
        num_tertiary: int,
        feat_scales: List[int],
        feature_outchannel: List[int],
        num_points: int,
        time_steps: List[int],
        channel_multiplier: List[int],
        fd: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.num_primary = num_primary
        self.num_secondary = num_secondary
        self.num_tertiary = num_tertiary
        self.fd = fd
        self.dropout = dropout
        self.feat_scales = feat_scales
        self.channel_multiplier = channel_multiplier
        self.feature_outchannel = feature_outchannel
        self.time_steps = time_steps
        self.num_points = num_points

        # Total output channels of each encoder stream
        outchannel = sum(feature_outchannel)

        # Encoder streams (Figure 2). Each stream processes the COMPLETE input
        # log tensor (all in_channels curves) through its own attention setup.
        self.hybrid_head = nn.Sequential(
            ConvBNReLU1D(in_channels, outchannel, 1),
            ChannelSELayer(num_channels=outchannel, reduction_ratio=2),
            SpatialAttention(),
        )
        self.hybrid_fuse = nn.Sequential(
            ConvBNReLU1D(outchannel * 2, outchannel, 1),
            ConvBNReLURes1D(outchannel, 1, 1, True),
            ConvBNReLU1D(outchannel, outchannel, 1),
        )
        self.ca_stream = nn.Sequential(
            ConvBNReLU1D(in_channels, outchannel, 1),
            ConvBNReLU1D(outchannel, outchannel, 1),
            ChannelSELayer(num_channels=outchannel, reduction_ratio=2),
            ConvBNReLU1D(outchannel, outchannel, 1),
        )
        self.sa_stream = nn.Sequential(
            ConvBNReLU1D(in_channels, outchannel, 1),
            ConvBNReLU1D(outchannel, outchannel, 1),
            SpatialAttention(),
            ConvBNReLU1D(outchannel, outchannel, 1),
        )
        self.plain_stream = nn.Sequential(
            ConvBNReLU1D(in_channels, outchannel, 1),
            ConvBNReLURes1D(outchannel, 1, 1, True),
            ConvBNReLU1D(outchannel, outchannel, 1),
        )

        # Adaptive average pooling to fixed-size vectors (Eq. 3).
        self.encoder_pool = nn.AdaptiveAvgPool1d(max(1, num_points // 4))

        # Hierarchical fusion decoder. The concatenated encoder embedding has
        # one 240-channel vector per stream (4 x outchannel channels).
        self.decoder = HierarchicalDecoder(
            outchannel * 4,
            self.feat_scales,
            self.fd,
            inner_channel=240,
            channel_multiplier=self.channel_multiplier,
            time_steps=self.time_steps
        ).to(self.device)

        # Three classification heads, conditioned top-down:
        #   primary:   f_p(F)                      (Eq. 5)
        #   secondary: f_s([F; L_p])               (Eq. 7)
        #   tertiary:  f_t([F; L_p; L_s])          (Eq. 9)
        self.primary_head = nn.Sequential(
            nn.Linear(fd, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_primary)
        )

        self.secondary_head = nn.Sequential(
            nn.Linear(fd + 1, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_secondary)
        )

        # The tertiary head is deeper than the coarser heads and includes
        # additional nonlinear transformations to handle the lower
        # signal-to-noise ratio of the fine-grained classes without overfitting.
        self.tertiary_head = nn.Sequential(
            nn.Linear(fd + 2, 2048),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.75),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.25),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_tertiary)
        )

        # Residual connection for tertiary head
        self.tertiary_residual = nn.Linear(fd + 2, self.num_tertiary)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 3:
            raise ValueError(f"Expected a 3D input tensor, got shape {tuple(x.shape)}")

        # Normalise to channel-first (B, C, L); the windowed datasets emit
        # window-first tensors (B, L, C).
        if x.shape[1] == self.in_channels:
            x = x.float()  # already (B, C, L)
        elif x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1).contiguous().float()  # (B, L, C) -> (B, C, L)
        else:
            raise ValueError(
                f"Input tensor shape {tuple(x.shape)} is incompatible with in_channels={self.in_channels}"
            )

        # Stream 1: hybrid fusion (channel attention + spatial attention).
        hybrid_base = self.hybrid_head[0](x)
        hybrid_ca = self.hybrid_head[1](hybrid_base)
        hybrid_sa = self.hybrid_head[2](hybrid_base)
        h_hybrid = self.hybrid_fuse(torch.cat([hybrid_ca, hybrid_sa], dim=1))

        # Stream 2: channel attention.
        h_ca = self.ca_stream(x)

        # Stream 3: spatial attention.
        h_sa = self.sa_stream(x)

        # Stream 4: attention-free residual baseline.
        h_plain = self.plain_stream(x)

        # Adaptive average pooling to fixed-size vectors, then concatenation
        # into the holistic embedding (Eq. 3).
        ft = torch.cat(
            [self.encoder_pool(stream) for stream in (h_hybrid, h_ca, h_sa, h_plain)],
            dim=1
        )
        return self.decoder(ft)

    def forward(
        self,
        x: torch.Tensor,
        teacher_parents: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ov = self._encode_features(x)
        feats = ov.permute(0, 2, 1)  # (B, T, fd)

        # Primary classification (Eq. 5)
        primary = self.primary_head(feats)
        if teacher_parents is not None:
            labels_p, labels_s = teacher_parents
            p_cond = labels_p.view(-1, 1, 1).expand(-1, feats.shape[1], 1).float()
        else:
            p_cond = primary.argmax(dim=2).unsqueeze(2).float()

        # Secondary classification conditioned on the primary outcome (Eq. 7)
        secondary_features = torch.cat([feats, p_cond], dim=2)
        secondary = self.secondary_head(secondary_features)
        if teacher_parents is not None:
            s_cond = labels_s.view(-1, 1, 1).expand(-1, feats.shape[1], 1).float()
        else:
            s_cond = secondary.argmax(dim=2).unsqueeze(2).float()

        # Tertiary classification conditioned on both coarser levels (Eq. 9)
        tertiary_features = torch.cat([feats, p_cond, s_cond], dim=2)

        # Main path + residual connection
        tertiary = self.tertiary_head(tertiary_features) + self.tertiary_residual(tertiary_features)

        return primary, secondary, tertiary

    def set_trainable_modules(
        self,
        encoder: bool = True,
        decoder: bool = True,
        primary: bool = True,
        secondary: bool = True,
        tertiary: bool = True,
    ) -> "HierarchicalLithologyModel":

        head_groups = {
            "primary_head": primary,
            "secondary_head": secondary,
            "tertiary_head": tertiary,
            "tertiary_residual": tertiary,
            "flat_primary_head": primary,
            "flat_secondary_head": secondary,
            "flat_tertiary_head": tertiary,
            "direct_tertiary_head": tertiary,
        }
        for name, param in self.named_parameters():
            is_head = False
            for prefix, trainable in head_groups.items():
                if name.startswith(prefix):
                    param.requires_grad_(trainable)
                    is_head = True
                    break
            if is_head:
                continue
            if name.startswith("decoder."):
                param.requires_grad_(decoder)
            else:
                param.requires_grad_(encoder)
        return self


class FlatMultiTaskLithologyModel(HierarchicalLithologyModel):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_primary_head = nn.Sequential(
            nn.Linear(self.fd, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_primary)
        )
        self.flat_secondary_head = nn.Sequential(
            nn.Linear(self.fd, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_secondary)
        )
        self.flat_tertiary_head = nn.Sequential(
            nn.Linear(self.fd, 2048),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.75),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.25),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_tertiary)
        )

    def forward(
        self,
        x: torch.Tensor,
        teacher_parents: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # teacher_parents is accepted (and ignored) so that the flat baseline
        # can be trained with the same protocol as the hierarchical model.
        ov = self._encode_features(x)
        pooled = ov.permute(0, 2, 1)
        primary = self.flat_primary_head(pooled)
        secondary = self.flat_secondary_head(pooled)
        tertiary = self.flat_tertiary_head(pooled)
        return primary, secondary, tertiary


class DirectTertiaryLithologyModel(HierarchicalLithologyModel):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_tertiary_head = nn.Sequential(
            nn.Linear(self.fd, 1024),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.num_tertiary)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ov = self._encode_features(x)
        pooled = ov.permute(0, 2, 1)
        return self.direct_tertiary_head(pooled)
