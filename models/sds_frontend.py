"""
Lightweight noise-aware front-end for 1D bearing vibration signals.

This module is intended to replace the current MS-SK + dense downsampling front-end.
It keeps the same output convention as the existing model front-end:
    input:  x [B, 1, L]
    output: x [B, L/16, d_model]
so it can be followed by a spectral token mixer such as DSFB-v2.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F



class SumFusion1D(nn.Module):
    """Parameter-free branch summation used when SK fusion is disabled."""

    def __init__(self, num_branches: int) -> None:
        super().__init__()
        self.num_branches = int(num_branches)

    def forward(self, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(branches) != self.num_branches:
            raise ValueError(f"expected {self.num_branches} branches, got {len(branches)}")
        stacked = torch.stack(list(branches), dim=0)
        return stacked.sum(dim=0)


class SKFusion1D(nn.Module):
    """Lightweight Selective-Kernel branch fusion.

    Migrated from ``mslk_transformer.SelectiveKernelFusion`` with these tweaks
    to keep parameter count low at every stage:
      - default ``reduction=16`` (vs. 4 in the original)
      - bottleneck floor of 8 (vs. 16)
      - BN after the squeeze conv for training stability
    """

    def __init__(self, channels: int, num_branches: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.num_branches = num_branches
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Sequential(
            nn.Conv1d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Conv1d(mid, channels * num_branches, kernel_size=1, bias=False)

    def forward(self, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(branches) != self.num_branches:
            raise ValueError(f"expected {self.num_branches} branches, got {len(branches)}")
        # [B, N, C, T]
        stacked = torch.stack(list(branches), dim=1)
        u = stacked.sum(dim=1)                              # [B, C, T]
        s = u.mean(dim=-1, keepdim=True)                    # [B, C, 1]
        z = self.fc1(s)                                     # [B, mid, 1]
        weights = self.fc2(z).view(
            s.shape[0], self.num_branches, self.channels, 1
        )
        weights = torch.softmax(weights, dim=1)
        return (stacked * weights).sum(dim=1)


class SparseDilatedShrinkageBlock1D(nn.Module):
    """Parameter-efficient replacement for multi-scale large-kernel SK blocks.

    Dense large-kernel depthwise branches such as k={15,31,51} require
    (15+31+51)*C = 97C depthwise weights before fusion. This block uses several
    3-tap dilated depthwise branches plus an optional identity branch, giving a
    large effective receptive field with only 3*num_dilations*C depthwise weights.
    """

    def __init__(
        self,
        channels: int,
        dilations: Iterable[int] = (1, 4, 12),
        dropout: float = 0.0,
        cross_scale: bool = True,
        use_identity_branch: bool = True,
        fusion_mode: str = "sk",
        cross_scale_mode: str = "full",
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.use_identity_branch = bool(use_identity_branch)
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
        self.cross_scale = bool(cross_scale)
        if cross_scale_mode not in ("full", "gap"):
            raise ValueError(
                f"Unsupported cross_scale_mode: {cross_scale_mode}. Use 'full' or 'gap'."
            )
        self.cross_scale_mode = cross_scale_mode
        num_branches = len(self.dilations) + (1 if self.use_identity_branch else 0)
        if fusion_mode == "sk":
            self.fusion = SKFusion1D(channels, num_branches=num_branches)
        elif fusion_mode == "sum":
            self.fusion = SumFusion1D(num_branches=num_branches)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}. Use 'sk' or 'sum'.")
        self.fusion_mode = fusion_mode
        self.channel_mix = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = []
        if self.use_identity_branch:
            branches.append(x)

        # Cross-scale coupled dilated branches.
        #   * cross_scale_mode='gap': add a time-pooled summary of the previous
        #     scale. This is the training-script default and a lightweight
        #     channel-wise prior.
        #   * cross_scale_mode='full': add the entire previous-scale response,
        #     preserving time-axis detail at a higher coupling cost.
        prev = None
        dilated_outputs = []
        for i, branch in enumerate(self.dw_branches):
            branch_input = x
            if i > 0 and self.cross_scale:
                if self.cross_scale_mode == "gap":
                    coupling = prev.mean(dim=-1, keepdim=True)
                else:
                    coupling = prev
                branch_input = branch_input + coupling
            prev = branch(branch_input)
            dilated_outputs.append(prev)

        branches.extend(dilated_outputs)
        y = self.fusion(branches)
        y = self.channel_mix(y)
        return y



class HaarSubbandMixin:
    """Shared fixed Haar analysis used by the learnable sub-band mixers."""

    def _init_haar(self) -> None:
        h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
        self.register_buffer("haar", h.view(2, 1, 2), persistent=False)

    def _haar_subbands(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, _ = x.shape
        weight = self.haar.repeat(c, 1, 1)
        y = F.conv1d(x, weight, stride=2, padding=0, groups=c)
        y = y.view(b, c, 2, y.shape[-1])
        return y[:, :, 0, :], y[:, :, 1, :]


class HaarLowpassResidualDownsample1D(nn.Module, HaarSubbandMixin):
    """Haar low-pass downsampling with trainable residual DWConv refinement.

    The approximation band ``A`` remains the anti-aliased backbone.  The detail
    band uses the original bounded residual before projection, and the projected
    feature is refined by an ordinary-initialized depthwise 3-tap residual:

        y0 = Proj(A + 0.5 * tanh(alpha_c) * D)
        y  = y0 + DWConv(y0)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self._init_haar()
        self.detail_alpha = nn.Parameter(torch.zeros(in_channels, 1))
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.refine_dw = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        approx, detail = self._haar_subbands(x)
        alpha = 0.5 * torch.tanh(self.detail_alpha).unsqueeze(0)
        y = self.proj(approx + alpha * detail)
        return y + self.refine_dw(y)


def build_wavelet_downsample1d(
    in_channels: int,
    out_channels: int,
    mode: str = "haar_lpr",
) -> nn.Module:
    """Build the only retained Haar 2x downsampler: Haar-LPR."""

    if mode != "haar_lpr":
        raise ValueError(f"Unsupported wavelet_downsample: {mode}. Use 'haar_lpr'.")
    return HaarLowpassResidualDownsample1D(in_channels, out_channels)


class StridedConvDownsample1D(nn.Module):
    """Plain strided conv downsampler used as the no-wavelet ablation baseline."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
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
        return self.conv(x)


class AvgPoolDownsample1D(nn.Module):
    """Parameter-free 2x downsampler.

    Used at stages where the channel count does not change, in which case the
    extra wavelet/strided 1x1 projection is largely redundant with the
    downstream DSFB token mixer's channel-side filtering.
    """

    def __init__(self, in_channels: int = 0, out_channels: int = 0) -> None:
        super().__init__()
        if out_channels and in_channels and out_channels != in_channels:
            raise ValueError(
                "AvgPoolDownsample1D requires in_channels == out_channels; "
                f"got {in_channels} -> {out_channels}."
            )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class SDSFrontEnd1D(nn.Module):
    """Subband Denoising Sparse front-end.

    Suggested replacement for:
        stem -> MS-SK1 -> down2 -> MS-SK2 -> down3 -> MS-SK3 -> down4

    Output is [B, T/16, d_model], matching the token layout expected by DSFB.
    """

    def __init__(
        self,
        d_model: int = 128,
        stem_channels: int = 32,
        dilations: Iterable[int] = (1, 4, 12),
        dropout: float = 0.0,
        use_haar_wavelet: bool = True,
        use_identity_branch: bool = True,
        simple_last_down: bool = False,
        use_sk_fusion: bool = True,
        cross_scale_mode: str = "full",
        wavelet_downsample: str = "haar_lpr",
    ) -> None:
        super().__init__()
        k = 512
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=k, stride=2, padding=(k - 1) // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        fusion_mode = "sk" if use_sk_fusion else "sum"
        block_kwargs = {
            "use_identity_branch": use_identity_branch,
            "fusion_mode": fusion_mode,
            "cross_scale_mode": cross_scale_mode,
        }
        self.block1 = SparseDilatedShrinkageBlock1D(stem_channels, dilations, dropout, **block_kwargs)
        self.down2 = (
            build_wavelet_downsample1d(stem_channels, 64, mode=wavelet_downsample)
            if use_haar_wavelet
            else StridedConvDownsample1D(stem_channels, 64)
        )
        self.block2 = SparseDilatedShrinkageBlock1D(64, dilations, dropout, **block_kwargs)
        self.down3 = (
            build_wavelet_downsample1d(64, d_model, mode=wavelet_downsample)
            if use_haar_wavelet
            else StridedConvDownsample1D(64, d_model)
        )
        self.block3 = SparseDilatedShrinkageBlock1D(d_model, dilations, dropout, **block_kwargs)
        # The last 2x downsample keeps channel count unchanged; using a plain
        # AvgPool here removes the redundant 1x1 projection (~16K params at
        # d_model=128) since the downstream DSFB already mixes channels.
        if simple_last_down:
            self.down4 = AvgPoolDownsample1D(d_model, d_model)
        else:
            self.down4 = (
                build_wavelet_downsample1d(d_model, d_model, mode=wavelet_downsample)
                if use_haar_wavelet
                else StridedConvDownsample1D(d_model, d_model)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down2(x)
        x = self.block2(x)
        x = self.down3(x)
        x = self.block3(x)
        x = self.down4(x)
        return x.transpose(1, 2)  # [B, T/16, C]


class PlainConvFrontEnd1D(nn.Module):
    """Minimal strided-conv front-end used as the ``--no_sds_frontend`` (A1) baseline.

    The replacement is intentionally *parameter-light* so the ablation can be
    interpreted as "remove RFE-Stem and put back the simplest
    sensible alternative". Concretely:

        Conv(1   -> 32 , k=15, s=2)  + BN + GELU
        Conv(32  -> 64 , k=3 , s=2)  + BN + GELU
        Conv(64  -> 128, k=3 , s=2)  + BN + GELU
        AvgPool(k=2, s=2)                       # parameter-free 16x downsample

    The last 16x stage uses parameter-free average pooling (mirroring the
    ``simple_down4`` option of RFE-Stem), so A1 ends up *cheaper*
    than the full RFE-Stem (~31K vs ~70K front-end params). This means
    any accuracy improvement of Full over A1 cannot be attributed to A1
    having fewer / weaker capacity than necessary.

    Same downsampling factor (16x) and output channel count (d_model) as
    :class:`SDSFrontEnd1D`, so the rest of the model is unchanged.

    Output is ``[B, T/16, d_model]``.
    """

    def __init__(
        self,
        d_model: int = 128,
        stem_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x.transpose(1, 2)


class ResidualSeparableConvBlock1D(nn.Module):
    """Plain residual depthwise-separable conv block for matched A1B capacity."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MatchedConvFrontEnd1D(nn.Module):
    """Parameter-matched plain Conv front-end for the A1B RFE-Stem ablation.

    This keeps the same 16x downsampling and d_model output as RFE-Stem, but
    avoids all RFE-Stem-specific ideas: no sparse multi-dilation branches, no
    Haar-LPR downsampling, no cross-scale coupling, and no SK fusion. Two plain
    residual depthwise-separable refinement blocks make its parameter count
    close to the full no-SK RFE-Stem, so the comparison is not merely a
    capacity mismatch.
    """

    def __init__(
        self,
        d_model: int = 128,
        stem_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualSeparableConvBlock1D(64, kernel_size=7, dropout=dropout),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            ResidualSeparableConvBlock1D(d_model, kernel_size=3, dropout=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x.transpose(1, 2)


if __name__ == "__main__":
    model = SDSFrontEnd1D(d_model=128)
    x = torch.randn(2, 1, 2048)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", tuple(y.shape))
    print("params:", n_params)
