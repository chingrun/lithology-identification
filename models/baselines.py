

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_window_input(x: torch.Tensor) -> torch.Tensor:

    if x.dim() != 3:
        raise ValueError(f"Expected a 3D window tensor, got shape {tuple(x.shape)}")
    return x


class ResNet1DModel(nn.Module):


    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 26,
        base_channels: int = 64,
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        channels = base_channels
        for _ in range(3):
            stage = []
            for _ in range(blocks_per_stage):
                stage.append(_ResidualBlock1D(channels, channels))
            self.stages.append(nn.Sequential(*stage))
            self.stages.append(nn.Sequential(nn.Conv1d(channels, channels * 2, 3, stride=2, padding=1),
                                             nn.BatchNorm1d(channels * 2), nn.ReLU(inplace=True)))
            channels *= 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _check_window_input(x)
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.head(x)


class _ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _RecurrentModel(nn.Module):
    def __init__(
        self,
        cell: str,
        in_channels: int = 6,
        num_classes: int = 26,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.recurrent = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = self.recurrent(
            in_channels, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _check_window_input(x)  # (B, L, C), batch_first
        out, _ = self.rnn(x)
        # Concatenate the final forward/backward hidden states.
        forward = out[:, -1, : out.shape[2] // 2]
        backward = out[:, 0, out.shape[2] // 2:]
        return self.head(torch.cat([forward, backward], dim=1))


class BiLSTMModel(_RecurrentModel):


    def __init__(self, in_channels: int = 6, num_classes: int = 26, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__("lstm", in_channels, num_classes, hidden_size, num_layers, dropout)


class BiGRUModel(_RecurrentModel):


    def __init__(self, in_channels: int = 6, num_classes: int = 26, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__("gru", in_channels, num_classes, hidden_size, num_layers, dropout)


class _GATLayer(nn.Module):


    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F). Adjacency = self loops + adjacent depth neighbours.
        b, length, _ = x.shape
        edge_index = torch.stack(
            [torch.arange(length), torch.arange(length)], dim=1
        ).to(x.device)
        if length > 1:
            neighbours = torch.stack(
                [torch.arange(length - 1), torch.arange(1, length)], dim=1
            ).to(x.device)
            edge_index = torch.cat([edge_index, neighbours], dim=0)

        # Wh: (H, B, L, Fout)
        Wh = torch.einsum("blf,hfo->hblo", x, self.W)
        src = Wh[:, :, edge_index[:, 0], :]  # (H, B, E, Fout)
        dst = Wh[:, :, edge_index[:, 1], :]
        e = self.leaky_relu(torch.einsum("hbeo,ho->hbe", torch.cat([src, dst], dim=-1), self.a))

        # Softmax over incoming edges per destination node.
        out = x.new_zeros(b, length, self.num_heads * self.out_features)
        for head in range(self.num_heads):
            scores = e[head]  # (B, E)
            for node in range(length):
                mask = edge_index[:, 1] == node
                node_scores = scores[:, mask]
                alpha = F.softmax(node_scores, dim=1)
                messages = Wh[head][:, edge_index[:, 0][mask], :]
                agg = torch.einsum("be,bef->bf", alpha, messages)
                out[:, node, head * self.out_features:(head + 1) * self.out_features] = agg
        return self.dropout(out)


class GATModel(nn.Module):


    def __init__(self, in_channels: int = 6, num_classes: int = 26, hidden_size: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        current = in_channels
        for idx in range(num_layers):
            out = hidden_size
            layers.append(_GATLayer(current, out, num_heads=num_heads if idx < num_layers - 1 else 1,
                                    dropout=dropout))
            current = out * (num_heads if idx < num_layers - 1 else 1)
            if idx < num_layers - 1:
                layers.append(nn.ELU())
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _check_window_input(x)  # (B, L, C)
        x = self.layers(x)
        x = x.mean(dim=1)
        return self.head(x)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):


    def __init__(self, in_channels: int = 6, num_classes: int = 26, d_model: int = 96,
                 nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_channels, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _check_window_input(x)  # (B, L, C)
        x = self.pos_encoder(self.embed(x))
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


class ShapTransformerModel(TransformerModel):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("feature_importance", None)

    def fit_feature_importance(self, features: torch.Tensor, labels: torch.Tensor) -> None:

        import numpy as np

        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.feature_importance = torch.as_tensor(
            _estimate_feature_importance(features_np, labels_np), dtype=torch.float32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _check_window_input(x)  # (B, L, C)
        if self.feature_importance is not None:
            weight = self.feature_importance.to(x.device).view(1, 1, -1)
            x = x * weight
        return super().forward(x)


def _estimate_feature_importance(features, labels):

    import numpy as np

    flat = features.reshape(features.shape[0], -1)
    try:
        import shap  # type: ignore

        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(flat, labels)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(flat, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        matrix = np.abs(np.asarray(shap_values))
    except Exception:
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(flat, labels)
        matrix = np.asarray(model.feature_importances_)[None, :]

    # Reduce window-position importances to per-log importance.
    per_position = matrix.reshape(matrix.shape[0], features.shape[1], -1).mean(axis=0)
    importance = per_position.mean(axis=1)
    if importance.sum() <= 0:
        return np.ones_like(importance)
    return importance / importance.sum()


def build_baseline(
    name: str,
    in_channels: int = 6,
    num_classes: int = 26,
    window_size: int = 15,
) -> nn.Module:

    key = name.lower().replace("_", "").replace("-", "")
    if key == "resnet":
        return ResNet1DModel(in_channels=in_channels, num_classes=num_classes)
    if key == "bilstm":
        return BiLSTMModel(in_channels=in_channels, num_classes=num_classes)
    if key == "bigru":
        return BiGRUModel(in_channels=in_channels, num_classes=num_classes)
    if key == "gat":
        return GATModel(in_channels=in_channels, num_classes=num_classes)
    if key == "transformer":
        return TransformerModel(in_channels=in_channels, num_classes=num_classes)
    if key in ("shap", "shaptransformer"):
        return ShapTransformerModel(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown baseline model: {name}")
