"""Residual attention 3D U-Net for CellMap segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSE3D(nn.Module):
    """Squeeze-and-excitation attention for 3D feature maps."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attention = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(self.pool(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            ChannelSE3D(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            ResidualBlock3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = F.pad(
            x,
            [
                dx // 2,
                dx - dx // 2,
                dy // 2,
                dy - dy // 2,
                dz // 2,
                dz - dz // 2,
            ],
        )
        return self.block(torch.cat([skip, x], dim=1))


class AttentionResUNet_3D(nn.Module):
    """
    3D residual attention U-Net.

    Compared with the baseline 3D U-Net, this model adds residual feature
    learning and squeeze-and-excitation channel attention in every block.
    """

    def __init__(self, n_channels, n_classes, base_channels=16):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = ResidualBlock3D(n_channels, c1)
        self.down1 = DownBlock3D(c1, c2)
        self.down2 = DownBlock3D(c2, c3)
        self.down3 = DownBlock3D(c3, c4)
        self.down4 = DownBlock3D(c4, c5)

        self.up1 = UpBlock3D(c5, c4, c4)
        self.up2 = UpBlock3D(c4, c3, c3)
        self.up3 = UpBlock3D(c3, c2, c2)
        self.up4 = UpBlock3D(c2, c1, c1)
        self.outc = nn.Conv3d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
