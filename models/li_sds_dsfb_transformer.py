"""LiSDS-DSFB Transformer: LiConvFormer-inspired lightweight variant.

This file is intended to be placed in the same folder as refined_dsfb_modules.py.
It keeps the spectral-token-mixer story of DSFB, while borrowing several useful
lightweight ideas from LiConvFormer:
  1) aggressive Li-style input layer: AvgPool + k=15 Conv;
  2) separable multi-scale embedding: 1x1 bottleneck + depthwise kernels 3/5/7/9;
  3) learnable weighted residual addition;
  4) a thin contractive FFN by default;
  5) optional ultra-light broadcast context gate inspired by BSA.

Input:  [B, 1, L]
Output: [B, num_classes]
For L=2048, the default front-end outputs [B, 64, 128].
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # when used inside a package, e.g. models/li_sds_dsfb_transformer.py
    from .refined_dsfb_modules import SpectralGatedFilter1D, ConvSwiGLUFFN
except Exception:  # pragma: no cover
    # when used as a standalone file in the same directory
    from refined_dsfb_modules import SpectralGatedFilter1D, ConvSwiGLUFFN


class WeightedAdd(nn.Module):
    """LiConvFormer-style learnable residual fusion."""

    def __init__(self, epsilon: float = 1e-12) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, transformed: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        w = F.relu(self.w)
        w = w / (w.sum() + self.epsilon)
        return w[0] * transformed + w[1] * identity


class AdaptiveShrinkage1D(nn.Module):
    """Channel-wise adaptive soft-thresholding."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.threshold = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=-1, keepdim=True)
        tau = self.threshold(scale) * scale
        return x.sign() * F.relu(x.abs() - tau)


class ChannelWiseBranchGate1D(nn.Module):
    """Channel-wise branch gate with O(B*C) parameters instead of SK's channel MLP."""

    def __init__(self, channels: int, num_branches: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_branches = int(num_branches)
        self.gate = nn.Conv1d(
            channels,
            channels * num_branches,
            kernel_size=1,
            groups=channels,
            bias=True,
        )

    def forward(self, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(branches) != self.num_branches:
            raise ValueError(f"expected {self.num_branches} branches, got {len(branches)}")
        stacked = torch.stack(list(branches), dim=1)  # [B, N, C, T]
        descriptor = stacked.sum(dim=1).mean(dim=-1, keepdim=True)  # [B, C, 1]
        weights = self.gate(descriptor).view(
            descriptor.shape[0], self.num_branches, self.channels, 1
        )
        weights = torch.softmax(weights, dim=1)
        return (stacked * weights).sum(dim=1)


class SparseDilatedShrinkageBlock1D(nn.Module):
    """Local sparse denoising block used before each downsampling stage."""

    def __init__(
        self,
        channels: int,
        dilations: Iterable[int] = (1, 4, 12),
        dropout: float = 0.0,
        shrink: bool = True,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.dw_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                )
                for d in self.dilations
            ]
        )
        self.lowpass = nn.AvgPool1d(
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )
        self.fusion = ChannelWiseBranchGate1D(channels, num_branches=len(self.dilations) + 2)
        self.shrink = AdaptiveShrinkage1D(channels) if shrink else nn.Identity()
        self.channel_mix = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [x, self.lowpass(x)]
        branches.extend(branch(x) for branch in self.dw_branches)
        y = self.fusion(branches)
        y = self.shrink(y)
        y = self.channel_mix(y)
        return x + y


class LiStem1D(nn.Module):
    """LiConvFormer-style input layer: AvgPool then k=15 stride-2 Conv."""

    def __init__(self, out_channels: int = 16, kernel_size: int = 15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                1,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaveletSeparableEmbeddingDownsample1D(nn.Module):
    """Wavelet-shrink downsampling + LiConvFormer-style separable multiscale embedding.

    Step 1: Haar decomposition creates approximation/detail coefficients.
    Step 2: detail coefficients are adaptively shrunk and gated.
    Step 3: a 1x1 bottleneck projects channels to out_channels / n_branches.
    Step 4: depthwise kernels 3/5/7/9 produce separable multiscale features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int] = (3, 5, 7, 9),
        shrink: bool = True,
    ) -> None:
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if out_channels % len(kernel_sizes) != 0:
            raise ValueError("out_channels must be divisible by number of kernel sizes")
        h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
        self.register_buffer("haar", h.view(2, 1, 2), persistent=False)
        self.detail_shrink = AdaptiveShrinkage1D(in_channels) if shrink else nn.Identity()
        self.detail_gate = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=True,
        )
        hidden = out_channels // len(kernel_sizes)
        self.reduce = nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False)
        self.sconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=k,
                    padding=k // 2,
                    groups=hidden,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        self.act_bn = nn.Sequential(nn.BatchNorm1d(out_channels), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        weight = self.haar.repeat(c, 1, 1)  # [2C, 1, 2]
        y = F.conv1d(x, weight, stride=2, groups=c).view(b, c, 2, -1)
        approx = y[:, :, 0, :]
        detail = y[:, :, 1, :]
        detail = self.detail_shrink(detail)
        gate = torch.sigmoid(self.detail_gate(detail.abs().mean(dim=-1, keepdim=True)))
        z = approx + gate * detail
        z = self.reduce(z)
        z = torch.cat([conv(z) for conv in self.sconvs], dim=1)
        return self.act_bn(z)


class LiSDSFrontEnd1D(nn.Module):
    """Narrow progressive SDS front-end inspired by LiConvFormer.

    Default channels are 16 -> 32 -> 64 -> 128, matching the compact setting of
    LiConvFormer while keeping wavelet shrinkage and sparse local denoising.
    """

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64, 128),
        stem_kernel_size: int = 15,
        dilations: Iterable[int] = (1, 4, 12),
        dropout: float = 0.0,
        use_sds_blocks: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain four stage widths, e.g. (16,32,64,128)")
        c1, c2, c3, c4 = [int(c) for c in channels]
        self.out_channels = c4
        self.reduction = 32  # AvgPool+Conv stem (/4) and three wavelet downsamples (/8)
        self.stem = LiStem1D(out_channels=c1, kernel_size=stem_kernel_size)
        block = SparseDilatedShrinkageBlock1D
        self.block1 = block(c1, dilations, dropout) if use_sds_blocks else nn.Identity()
        self.down2 = WaveletSeparableEmbeddingDownsample1D(c1, c2)
        self.block2 = block(c2, dilations, dropout) if use_sds_blocks else nn.Identity()
        self.down3 = WaveletSeparableEmbeddingDownsample1D(c2, c3)
        self.block3 = block(c3, dilations, dropout) if use_sds_blocks else nn.Identity()
        self.down4 = WaveletSeparableEmbeddingDownsample1D(c3, c4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down2(x)
        x = self.block2(x)
        x = self.down3(x)
        x = self.block3(x)
        x = self.down4(x)
        return x.transpose(1, 2)  # [B, T/32, C]


class BroadcastContextGate1D(nn.Module):
    """Ultra-light BSA-inspired context gate.

    It keeps BSA's central idea, i.e. temporal score -> global context -> broadcast,
    but removes full key/value/projection matrices. Parameters are about 3C+1.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=True)
        self.channel_gate = nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # source/target: [B, L, C]
        weights = torch.softmax(self.score(source), dim=1)  # [B, L, 1]
        context = (source * weights).sum(dim=1).unsqueeze(-1)  # [B, C, 1]
        gate = torch.sigmoid(self.channel_gate(context)).transpose(1, 2)  # [B, 1, C]
        return self.drop(target * (1.0 + gate))


class ThinConvFFN(nn.Module):
    """LiConvFormer-style contractive FFN: C -> C/4 -> C."""

    def __init__(self, dim: int, ffn_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(ffn_dim) if ffn_dim is not None else max(dim // 4, 16)
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class DSFBThinEncoderLayer(nn.Module):
    """DSFB encoder with weighted residuals and a thin FFN."""

    def __init__(
        self,
        dim: int,
        seq_len: int = 64,
        dropout: float = 0.1,
        gate_kernel_size: int = 7,
        ffn_dim: int | None = None,
        use_broadcast_gate: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = SpectralGatedFilter1D(
            dim=dim,
            max_seq_len=seq_len,
            gate_kernel_size=gate_kernel_size,
            dropout=dropout,
        )
        self.context_gate = BroadcastContextGate1D(dim, dropout) if use_broadcast_gate else None
        self.add1 = WeightedAdd()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ThinConvFFN(dim=dim, ffn_dim=ffn_dim, dropout=dropout)
        self.add2 = WeightedAdd()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        y = self.token_mixer(z)
        if self.context_gate is not None:
            y = self.context_gate(z, y)
        x = self.add1(y, x)
        x = self.add2(self.ffn(self.norm2(x)), x)
        return x


class LiSDSDSFBTransformer(nn.Module):
    """Compact LiConvFormer-inspired SDS-DSFB model.

    encoder_type:
      - "thin":   DSFB + BSA-inspired gate + contractive FFN; smallest and most Li-like.
      - "swi05":  DSFB + ConvSwiGLU with mlp_ratio=0.5.
      - "swi15":  DSFB + ConvSwiGLU with mlp_ratio=1.5; higher-capacity version.
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = (16, 32, 64, 128),
        num_layers: int = 1,
        max_len: int = 64,
        dropout: float = 0.1,
        encoder_type: str = "thin",
        use_linear_head: bool = True,
        use_broadcast_gate: bool = True,
        stem_kernel_size: int = 15,
        frontend_dilations: Iterable[int] = (1, 4, 12),
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.frontend = LiSDSFrontEnd1D(
            channels=channels,
            stem_kernel_size=stem_kernel_size,
            dilations=frontend_dilations,
            dropout=dropout,
            use_sds_blocks=True,
        )
        dim = int(channels[-1])
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        layers = []
        for _ in range(int(num_layers)):
            if encoder_type == "thin":
                layers.append(
                    DSFBThinEncoderLayer(
                        dim=dim,
                        seq_len=self.max_len,
                        dropout=dropout,
                        use_broadcast_gate=use_broadcast_gate,
                    )
                )
            elif encoder_type.startswith("swi"):
                ratio = float(encoder_type.replace("swi", "")) / 100.0 if encoder_type[3:].isdigit() else 0.5
                # Better aliases: swi05 -> 0.5, swi15 -> 1.5.
                if encoder_type == "swi05":
                    ratio = 0.5
                elif encoder_type == "swi15":
                    ratio = 1.5
                layers.append(
                    DSFBThinEncoderLayer(
                        dim=dim,
                        seq_len=self.max_len,
                        dropout=dropout,
                        use_broadcast_gate=use_broadcast_gate,
                        ffn_dim=max(dim // 4, 16),
                    )
                )
                # Replace the thin FFN with ConvSwiGLU while preserving weighted add/context gate.
                layers[-1].ffn = ConvSwiGLUFFN(dim=dim, mlp_ratio=ratio, dropout=dropout)
            else:
                raise ValueError(f"Unknown encoder_type={encoder_type!r}")
        self.encoder = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(dim)
        if use_linear_head:
            self.head = nn.Linear(dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

    def _pos(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.max_len:
            return self.pos_embedding[:, :seq_len, :]
        return F.interpolate(
            self.pos_embedding.transpose(1, 2),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)  # [B, T/32, C]
        x = x + self._pos(x.shape[1])
        for layer in self.encoder:
            x = layer(x)
        x = self.final_norm(x).mean(dim=1)
        return self.head(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for enc in ("thin", "swi05", "swi15"):
        model = LiSDSDSFBTransformer(num_classes=10, encoder_type=enc)
        x = torch.randn(2, 1, 2048)
        y = model(x)
        print(enc, "output:", tuple(y.shape), "params:", count_trainable_parameters(model))
