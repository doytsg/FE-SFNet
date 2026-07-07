import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import DropPath, trunc_normal_
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(a, b)


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_2 = self.drop(self.act(x1))
        x2 = self.conv2(x)
        x2_2 = self.drop(self.act(x2))
        x = self.conv3((x1 * x2_2) + (x2 * x1_2))
        return x.transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=1, embed_dim=128):
        super().__init__()
        stride = patch_size // 2
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        if TIMM_AVAILABLE:
            trunc_normal_(self.complex_weight_high, std=0.02)
            trunc_normal_(self.complex_weight, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        batch_size, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(batch_size, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(batch_size, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in):
        batch_size, seq_len, channels = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted = x_weighted + x_masked * weight_high

        x = torch.fft.irfft(x_weighted, n=seq_len, dim=1, norm="ortho")
        return x.to(dtype).view(batch_size, seq_len, channels)


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, drop=0.0, drop_path=0.0,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.norm1 = nn.LayerNorm(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.icb = ICB(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(nn.Module):
    def __init__(self, seq_len=2048, num_channels=1, num_classes=10,
                 patch_size=8, emb_dim=128, depth=2, dropout_rate=0.15,
                 use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.patch_embed = PatchEmbed(seq_len=seq_len, patch_size=patch_size, in_chans=num_channels, embed_dim=emb_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(
                dim=emb_dim,
                drop=dropout_rate,
                drop_path=dpr[i],
                use_icb=use_icb,
                use_asb=use_asb,
                adaptive_filter=adaptive_filter,
            )
            for i in range(depth)
        ])
        self.head = nn.Linear(emb_dim, num_classes)

        if TIMM_AVAILABLE:
            trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if TIMM_AVAILABLE:
                trunc_normal_(module.weight, std=0.02)
            else:
                nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.tsla_blocks:
            x = block(x)
        x = x.mean(1)
        return self.head(x)
