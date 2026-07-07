import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_map_reshape12(x):
    x_shape = x.shape
    len_feature_map1d = x_shape[2]
    tmp = np.log2(len_feature_map1d)
    if int(tmp % 2) == 0:
        h = int(2 ** (tmp / 2))
        w = int(len_feature_map1d / h)
    else:
        h = int(2 ** ((tmp - 1) / 2))
        w = int(len_feature_map1d / h)
    return torch.reshape(x, [-1, x_shape[1], h, w])


def gram_blk(x):
    xg1 = torch.einsum("bcij,bcjk->bcik", torch.permute(x, (0, 1, 3, 2)), x)
    xg2 = torch.einsum("bcij,bcjk->bcik", x, torch.permute(x, (0, 1, 3, 2)))
    y = torch.einsum("bcij,bcjk->bcik", xg2, x)
    y = torch.einsum("bcij,bcjk->bcik", y, xg1)
    return y


def feature_map_reshape21(x):
    x_shape = x.shape
    len_feature_map1d = x_shape[2] * x_shape[3]
    return torch.reshape(x, [-1, x_shape[1], len_feature_map1d])


def batch_gram_blk(x):
    x = feature_map_reshape12(x)
    x = gram_blk(x)
    x = feature_map_reshape21(x)
    return F.layer_norm(x, normalized_shape=(x.size()[1], x.size()[2]))


def batch_fft_blk(x):
    data_length = x.shape[2]
    x = torch.abs(torch.fft.rfft(x)) / data_length
    return F.layer_norm(x, normalized_shape=(x.size()[1], x.size()[2]))


class Conv1dBlock(nn.Module):
    def __init__(self, in_channel, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, filters, kernel_size, strides, padding)
        self.bn = nn.BatchNorm1d(filters)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class TripleBranchBlock(nn.Module):
    def __init__(self, in_channel, filters=16, kernel_size=3, strides=1, padding=0):
        super().__init__()
        self.conv_r = Conv1dBlock(in_channel, filters, kernel_size, strides, padding)
        if in_channel == 1:
            self.conv_g = Conv1dBlock(1, filters, kernel_size, strides, padding)
            self.conv_f = Conv1dBlock(2, filters, kernel_size, strides, padding)
        else:
            self.conv_g = Conv1dBlock(2 * in_channel, filters, kernel_size, strides, padding)
            self.conv_f = Conv1dBlock(in_channel, filters, kernel_size, strides, padding)

    def forward(self, xr0, xg0, xf0):
        xr1 = self.conv_r(xr0)
        xr1_g = batch_gram_blk(xr1)
        xg0_c = self.conv_g(xg0)
        xg1 = torch.cat([xr1_g, xg0_c], dim=1)
        xf1 = self.conv_f(xf0)
        return xr1, xg1, xf1


class GTFENET(nn.Module):
    def __init__(self, num_classes=10, input_length=2048):
        super().__init__()
        self.input_length = input_length
        self.blk1 = TripleBranchBlock(in_channel=1, filters=32, kernel_size=31, strides=1, padding=15)
        self.blk2 = TripleBranchBlock(in_channel=32, filters=32, kernel_size=31, strides=2, padding=15)
        self.blk3 = TripleBranchBlock(in_channel=32, filters=64, kernel_size=15, strides=2, padding=7)
        self.blk4 = TripleBranchBlock(in_channel=64, filters=64, kernel_size=15, strides=2, padding=7)
        self.blk5 = TripleBranchBlock(in_channel=64, filters=128, kernel_size=5, strides=2, padding=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        xr0 = x
        xg0 = batch_gram_blk(x)
        xf0 = torch.cat([batch_fft_blk(xg0), batch_fft_blk(xr0)], dim=1)

        xr1, xg1, xf1 = self.blk1(xr0, xg0, xf0)
        xr1, xg1, xf1 = self.blk2(xr1, xg1, xf1)
        xr1, xg1, xf1 = self.blk3(xr1, xg1, xf1)
        xr1, xg1, xf1 = self.blk4(xr1, xg1, xf1)
        xr2, xg2, xf2 = self.blk5(xr1, xg1, xf1)

        x_r = F.adaptive_avg_pool1d(xr2, 1)
        x_g = F.adaptive_avg_pool1d(xg2, 1)
        x_f = F.adaptive_avg_pool1d(xf2, 1)
        x = torch.cat([x_r, x_g, x_f], dim=1).squeeze(-1)
        return self.fc(x)
