import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv1d(nn.Module):
    """Depthwise Conv + BN + GELU."""
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SelectiveKernelFusion(nn.Module):
    def __init__(self, channels, num_paths=3, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 16)
        self.fc1 = nn.Sequential(
            nn.Conv1d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Conv1d(mid_channels, channels * num_paths, 1, bias=False)
        self.num_paths = num_paths
        self.channels = channels

    def forward(self, inputs):
        batch_size, _, _ = inputs[0].shape
        stacked = torch.stack(inputs, dim=1)
        u = torch.sum(stacked, dim=1)
        s = u.mean(dim=-1, keepdim=True)
        z = self.fc1(s)
        weights = self.fc2(z).view(batch_size, self.num_paths, self.channels, 1)
        weights = F.softmax(weights, dim=1)
        return torch.sum(stacked * weights, dim=1)


class GatedAdaptiveMSLK_Block(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[15, 31, 51]):
        super().__init__()
        self.branches = nn.ModuleList([DepthwiseConv1d(in_channels, kernel_size=k) for k in kernel_sizes])
        self.fusion = SelectiveKernelFusion(in_channels, num_paths=len(kernel_sizes))
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x_fused = self.fusion(branch_outputs)
        return self.proj(x_fused)


class MultiScaleLargeKernel_Block(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[15, 31, 51]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, k, padding=k // 2, groups=in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.fusion_conv = nn.Conv1d(in_channels * len(kernel_sizes), in_channels, 1)

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        x_feat = torch.cat(outputs, dim=1)
        return self.fusion_conv(x_feat)


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_2 = self.drop(self.act(x1))
        x2 = self.conv2(x)
        x2_2 = self.drop(self.act(x2))
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        return x.transpose(1, 2)


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, seq_len=128, dropout=0.1, channel_mixer_groups: int = 4,
                 use_channel_mixer: bool = True):
        super().__init__()
        self.dim = dim
        self.use_channel_mixer = use_channel_mixer
        self.freq_len = seq_len // 2 + 1

        if use_channel_mixer:
            if (2 * dim) % channel_mixer_groups != 0:
                raise ValueError(
                    f"channel_mixer_groups={channel_mixer_groups} must divide in/out channels={2 * dim}.",
                )
            self.channel_mixer = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * dim,
                    out_channels=2 * dim,
                    kernel_size=1,
                    groups=channel_mixer_groups,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        reduction_dim = max(8, self.freq_len // 4)
        self.spectral_gate = nn.Sequential(
            nn.Linear(self.freq_len, reduction_dim),
            nn.LayerNorm(reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, self.freq_len),
            nn.Sigmoid(),
        )

        self.complex_weight = nn.Parameter(
            torch.randn(self.freq_len, dim, 2, dtype=torch.float32) * 0.02,
        )
        nn.init.trunc_normal_(self.complex_weight, std=0.02)

    def forward(self, x_in):
        batch_size, seq_len, channels = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")

        if self.use_channel_mixer:
            x_fft_permuted = x_fft.permute(0, 2, 1)
            fft_cat = torch.cat([x_fft_permuted.real, x_fft_permuted.imag], dim=1)
            mixed_signal = self.channel_mixer(fft_cat)
            mixed_real, mixed_imag = torch.chunk(mixed_signal, 2, dim=1)
            x_fft_mixed = torch.complex(mixed_real, mixed_imag).permute(0, 2, 1)
        else:
            x_fft_mixed = x_fft

        energy = torch.abs(x_fft_mixed).pow(2).mean(dim=-1)
        soft_mask = self.spectral_gate(energy).unsqueeze(-1)
        weight_global = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft_mixed * soft_mask * weight_global
        x_out = torch.fft.irfft(x_weighted, n=seq_len, dim=1, norm="ortho")
        return x_out.to(dtype).view(batch_size, seq_len, channels)


class ASBEncoderLayer(nn.Module):
    def __init__(self, dim, seq_len=128, mlp_ratio=2, dropout=0.1, drop_path=0.0,
                 use_icb=False, channel_mixer_groups: int = 4,
                 use_channel_mixer: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.asb = Adaptive_Spectral_Block(
            dim,
            seq_len=seq_len,
            dropout=dropout,
            channel_mixer_groups=channel_mixer_groups,
            use_channel_mixer=use_channel_mixer,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if use_icb:
            self.ffn = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(dropout),
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.asb(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class MSLKTransformer(nn.Module):
    def __init__(self, num_classes: int = 10, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 1, dropout: float = 0.1, kernel_sizes=[15, 31, 51],
                 use_gated_sk: bool = True, use_asb: bool = False,
                 use_icb: bool = False, channel_mixer_groups: int = 4,
                 use_channel_mixer: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.use_gated_sk = use_gated_sk
        self.use_asb = use_asb

        block_class = GatedAdaptiveMSLK_Block if use_gated_sk else MultiScaleLargeKernel_Block
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.mslk_1 = block_class(32, kernel_sizes=kernel_sizes)
        self.down_2 = DownsampleBlock(32, 64, kernel_size=7, stride=2, padding=3)
        self.mslk_2 = block_class(64, kernel_sizes=kernel_sizes)
        self.down_3 = DownsampleBlock(64, self.d_model, kernel_size=5, stride=2, padding=2)
        self.mslk_3 = block_class(self.d_model, kernel_sizes=kernel_sizes)
        self.down_4 = DownsampleBlock(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.max_len = 128
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        if use_asb:
            self.encoder = nn.ModuleList([
                ASBEncoderLayer(
                    dim=self.d_model,
                    seq_len=self.max_len,
                    mlp_ratio=2,
                    dropout=dropout,
                    drop_path=0.0,
                    use_icb=use_icb,
                    channel_mixer_groups=channel_mixer_groups,
                    use_channel_mixer=use_channel_mixer,
                )
                for _ in range(self.num_layers)
            ])
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=2 * self.d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.mslk_1(x) + x
        x = self.down_2(x)
        x = self.mslk_2(x) + x
        x = self.down_3(x)
        x = self.mslk_3(x) + x
        x = self.down_4(x)
        x = x.transpose(1, 2)

        seq_len = x.shape[1]
        if seq_len <= self.max_len:
            pos = self.pos_embedding[:, :seq_len, :]
        else:
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        x = x + pos
        if self.use_asb:
            for layer in self.encoder:
                x = layer(x)
        else:
            x = self.encoder(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x)
