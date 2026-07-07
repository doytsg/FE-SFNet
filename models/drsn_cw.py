import torch
import torch.nn as nn


class SoftThreshold(nn.Module):
    def forward(self, x, thresholds):
        return torch.sign(x) * torch.maximum(torch.abs(x) - thresholds, torch.zeros_like(x))


class ChannelWiseThreshold(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _ = x.shape
        abs_mean = torch.mean(torch.abs(x), dim=2, keepdim=True)
        gap_out = self.gap(torch.abs(x)).view(batch_size, channels)
        scales = self.fc(gap_out).view(batch_size, channels, 1)
        return abs_mean * scales


class RSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.threshold_module = ChannelWiseThreshold(out_channels)
        self.soft_threshold = SoftThreshold()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.soft_threshold(out, self.threshold_module(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class DRSN_CW(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, base_channels=64, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks[3], stride=2)
        self.bn_final = nn.BatchNorm1d(base_channels * 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride),
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = [RSBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(RSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn_final(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


class DRSN_CW_Lite(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, base_channels=32, num_blocks=[1, 1, 1, 1]):
        super().__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, num_blocks[3], stride=2)
        self.bn_final = nn.BatchNorm1d(base_channels * 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride),
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = [RSBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(RSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn_final(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)
