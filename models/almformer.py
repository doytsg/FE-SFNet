import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCT1d(nn.Module):
    def __init__(self, n_features, type="ortho"):
        super().__init__()
        self.register_buffer("dct_matrix", self._get_dct_matrix(n_features, type))

    def _get_dct_matrix(self, n, type="ortho"):
        idx_n = torch.arange(n).float()
        idx_k = torch.arange(n).float()
        idx_n, idx_k = torch.meshgrid(idx_n, idx_k, indexing="ij")
        matrix = torch.cos((math.pi / n) * (idx_n + 0.5) * idx_k)
        if type == "ortho":
            matrix[:, 0] *= 1.0 / math.sqrt(n) * math.sqrt(1.0)
            matrix[:, 1:] *= 1.0 / math.sqrt(n) * math.sqrt(2.0)
        return matrix.transpose(0, 1)

    def forward(self, x, inverse=False):
        if not inverse:
            return torch.matmul(x, self.dct_matrix)
        return torch.matmul(x, self.dct_matrix.t())


class ExtraLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1025):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class SMDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.branches = nn.ModuleList()
        for kernel_size, dilation in [(3, 1), (5, 2), (7, 3), (9, 4), (11, 5)]:
            padding = (kernel_size - 1) * dilation // 2
            self.branches.append(
                nn.Conv1d(
                    out_channels, out_channels, kernel_size=kernel_size, dilation=dilation,
                    padding=padding, groups=out_channels, bias=False,
                ),
            )

        self.post_conv = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x_pre = self.pre_conv(x)
        fused = sum(branch(x_pre) for branch in self.branches)
        out = self.norm(self.post_conv(fused))
        return out + identity


class GRE_DB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand_conv = nn.Conv1d(in_channels, in_channels * 2, 1)
        self.ctrl_dw = nn.Conv1d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False)
        self.gate_proj = nn.Conv1d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.res_down = nn.Conv1d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False)

        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )

        self.fusion_conv = nn.Conv1d(in_channels * 2, out_channels, 1)
        self.elu = nn.ELU()
        self.final_shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.final_shortcut(x)
        expanded = self.expand_conv(x)
        x_a, x_b = torch.chunk(expanded, 2, dim=1)
        x_b_sub = x_b[:, :, ::2]
        x_a_proc = self.ctrl_dw(x_a)

        f1 = self.gate_proj(x_b_sub * x_a_proc)
        x_down = self.alpha * f1 + self.res_down(x)
        feat_b1 = self.b1(x_down)
        feat_b2 = self.b2(x_down)
        feat_cat = torch.cat([feat_b1, feat_b2], dim=1)

        feat_up = F.interpolate(feat_cat, size=x.shape[-1], mode="linear", align_corners=False)
        out = self.elu(self.fusion_conv(feat_up))
        return out + residual


class AdaptiveFrequencyAttention(nn.Module):
    def __init__(self, dim, patch_size=8):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.scale = dim ** -0.5
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim * 3, 1),
            nn.Conv1d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3),
        )
        self.dct_layer = DCT1d(patch_size)
        self.theta = nn.Parameter(torch.zeros(1, dim, 1))
        self.out_proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        batch_size, channels, length = x.shape
        pad_len = 0
        if length % self.patch_size != 0:
            pad_len = self.patch_size - (length % self.patch_size)
            x = F.pad(x, (0, pad_len))
            length += pad_len

        qkv = self.proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        num_patches = length // self.patch_size
        q_patch = q.view(batch_size, channels, num_patches, self.patch_size)
        k_patch = k.view(batch_size, channels, num_patches, self.patch_size)

        q_dct = self.dct_layer(q_patch)
        k_dct = self.dct_layer(k_patch)
        f_dct = (q_dct * k_dct) * self.scale

        keep_low = max(1, self.patch_size // 3)
        low_freq = torch.clamp(f_dct[..., :keep_low], -50, 50)
        energy = torch.mean(low_freq ** 2, dim=(2, 3), keepdim=True)
        median_e = torch.median(energy.view(batch_size, -1), dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        mask_in = 10 * (energy / (median_e + 1e-5) - self.theta.unsqueeze(-1))
        mask = torch.sigmoid(torch.clamp(mask_in, -10, 10))

        f_enhanced = (1 + mask) * f_dct
        f_idct = self.dct_layer(f_enhanced, inverse=True)
        y = f_idct.view(batch_size, channels, length) * v
        if pad_len > 0:
            y = y[..., :-pad_len]
        return self.out_proj(y)


class MetaFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2, patch_size=8, layer_scale_init=1e-5):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.token_mixer = AdaptiveFrequencyAttention(dim, patch_size)
        self.norm2 = nn.BatchNorm1d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, dim, 1),
        )
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1), requires_grad=True)
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1), requires_grad=True)

    def forward(self, x):
        x = x + self.ls1 * self.token_mixer(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class ALMformer(nn.Module):
    def __init__(self, num_classes=10, depth=4):
        super().__init__()
        self.stage1_elck = ExtraLargeKernelConv(1, 16, kernel_size=1025)
        self.stage2_smd = SMDConv(16, 32)
        self.stage3_gre = GRE_DB(32, 16)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[MetaFormerBlock(dim=16, patch_size=8) for _ in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_norm = nn.BatchNorm1d(16)
        self.head = nn.Linear(16, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stage1_elck(x)
        x = self.stage2_smd(x)
        x = self.stage3_gre(x)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        x = self.fc_norm(x)
        return self.head(x)
