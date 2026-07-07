import torch
import torch.nn as nn
import torch.nn.functional as F


class NSEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y_avg = F.adaptive_avg_pool1d(x, 1).view(b, c)
        w_avg = self.mlp(y_avg).view(b, c, 1)
        y_max = F.adaptive_max_pool1d(x, 1).view(b, c)
        w_max = self.mlp(y_max).view(b, c, 1)
        return (x * w_avg) + (x * w_max)


class SparseModifiedMHSA(nn.Module):
    def __init__(self, dim, num_heads, kv_kernel=2, kv_stride=2):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_downsample = nn.Conv1d(
            dim, dim, kernel_size=kv_kernel, stride=kv_stride, padding=kv_kernel // 2,
        )
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, _, seq_len = x.shape
        x_perm = x.permute(0, 2, 1)

        q = self.q_proj(x_perm).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim,
        ).permute(0, 2, 1, 3)

        x_down = self.kv_downsample(x)
        x_down_perm = x_down.permute(0, 2, 1)
        k = self.k_proj(x_down_perm).reshape(
            batch_size, -1, self.num_heads, self.head_dim,
        ).permute(0, 2, 1, 3)
        v = self.v_proj(x_down_perm).reshape(
            batch_size, -1, self.num_heads, self.head_dim,
        ).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_out = (attn @ v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        return self.o_proj(x_out).permute(0, 2, 1)


class ConvformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 c1_k, c1_s, c2_k, c2_s,
                 heads, kv_k, kv_s,
                 proj_k, proj_s):
        super().__init__()
        self.conv_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=c1_k, stride=c1_s, padding=c1_k // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=c2_k, stride=c2_s, padding=c2_k // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.mhsa = SparseModifiedMHSA(out_channels, heads, kv_k, kv_s)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv_proj = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=proj_k, stride=proj_s, padding=proj_k // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv_module(x)
        x_perm = x.permute(0, 2, 1)
        attn_out = self.mhsa(x)
        x_res1 = self.norm1(x_perm + attn_out.permute(0, 2, 1)).permute(0, 2, 1)
        proj_out = self.conv_proj(x_res1)
        return self.norm2(
            x_res1.permute(0, 2, 1) + proj_out.permute(0, 2, 1),
        ).permute(0, 2, 1)


class ConvformerNSE(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.block1 = ConvformerBlock(
            in_channels=in_channels, out_channels=8,
            c1_k=7, c1_s=1, c2_k=7, c2_s=4,
            heads=2, kv_k=2, kv_s=2,
            proj_k=3, proj_s=1,
        )
        self.block2 = ConvformerBlock(
            in_channels=8, out_channels=16,
            c1_k=5, c1_s=1, c2_k=5, c2_s=2,
            heads=2, kv_k=2, kv_s=2,
            proj_k=1, proj_s=1,
        )
        self.block3 = ConvformerBlock(
            in_channels=16, out_channels=32,
            c1_k=3, c1_s=1, c2_k=3, c2_s=1,
            heads=4, kv_k=2, kv_s=2,
            proj_k=1, proj_s=1,
        )

        self.concat_dim = 8 + 16 + 32
        self.conv10 = nn.Sequential(
            nn.Conv1d(self.concat_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.nse = NSEModule(64, reduction=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)

        target_len = f3.size(2)
        f1_pooled = F.adaptive_avg_pool1d(f1, target_len)
        f2_pooled = F.adaptive_avg_pool1d(f2, target_len)
        x_cat = torch.cat([f1_pooled, f2_pooled, f3], dim=1)

        x_conv = self.conv10(x_cat)
        x_att = self.nse(x_conv)
        out = self.global_pool(x_att).flatten(1)
        return self.classifier(out)
