import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNGELU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CNNTransformer(nn.Module):
    """
    CNN front-end (downsampling) + standard Transformer encoder.

    Input : (B, 1, 2048)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)

        c1 = max(self.d_model // 2, 32)
        c2 = self.d_model
        c3 = int(self.d_model * 1.5)
        c4 = self.d_model * 2
        self.cnn = nn.Sequential(
            ConvBNGELU(1, c1, kernel_size=9, stride=2, padding=4),
            ConvBNGELU(c1, c1, kernel_size=5, stride=1, padding=2),
            ConvBNGELU(c1, c2, kernel_size=7, stride=2, padding=3),
            ConvBNGELU(c2, c2, kernel_size=5, stride=1, padding=2),
            ConvBNGELU(c2, c3, kernel_size=5, stride=2, padding=2),
            ConvBNGELU(c3, c3, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(c3, c3, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(c3, c4, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(c4, c3, kernel_size=1, stride=1, padding=0),
            ConvBNGELU(c3, c2, kernel_size=1, stride=1, padding=0),
        )

        self.pre_norm = nn.LayerNorm(self.d_model)
        self.max_len = 128
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
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
        x = self.cnn(x)
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

        x = self.pre_norm(x + pos)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)
