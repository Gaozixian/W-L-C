import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== 1. 基础组件：Bottleneck ====================
class Bottleneck(nn.Module):
    """ResNet50的瓶颈块 (引用自原文件)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

# ==================== 2. 核心改进：自适应双向引导交互模块 ====================
class AdaptiveBidirectionalInteraction(nn.Module):
    """
    自适应多尺度注意力融合机制：
    - 路径1: 浅层空间 As -> Gsd 引导深层通道
    - 路径2: 深层语义 Ac -> Gds 优化浅层空间
    """
    def __init__(self, s_channels, d_channels, reduction=16):
        super(AdaptiveBidirectionalInteraction, self).__init__()

        # --- 浅层分支 (Spatial Focus) ---
        self.s_conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # As -> Gsd (Spatial to Deep Channel Guidance)
        self.s_to_d_mlp = nn.Sequential(
            nn.Linear(1, d_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, d_channels)
        )

        # --- 深层分支 (Channel Semantic) ---
        self.d_shared_mlp = nn.Sequential(
            nn.Linear(d_channels, d_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, d_channels, bias=False)
        )

        # Ac -> Gds (Semantic to Shallow Spatial Guidance)
        self.d_to_s_mlp = nn.Sequential(
            nn.Linear(d_channels, d_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, 1),
            nn.Sigmoid()
        )

        mid_channels = 256
        self.d_to_s_conv = nn.Sequential(
            nn.Conv2d(d_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, Fs, Fd):
        # 1. 深层通道注意力计算 (Ac)
        avg_p = F.adaptive_avg_pool2d(Fd, 1).view(Fd.size(0), -1)
        max_p = F.adaptive_max_pool2d(Fd, 1).view(Fd.size(0), -1)
        d_stats = self.d_shared_mlp(avg_p) + self.d_shared_mlp(max_p)
        Ac = self.sigmoid(d_stats)
        # 浅层空间使用池化
        s_avg = torch.mean(Fs, dim=1, keepdim=True)
        s_max, _ = torch.max(Fs, dim=1, keepdim=True)
        s_desc = torch.cat([s_avg, s_max], dim=1) # 此时为B,2,H,W

        # 2. 深层对浅层的语义引导 (Ac -> Gds)
        """
        # 在 forward 中修改引导逻辑
        # 假设 Fd 是 (B, 2048, H/8, W/8), Fs 是 (B, 512, H/2, W/2)
        Gds = self.d_to_s_conv(Fd)
        # 将深层引导图上采样到浅层的大小
        Gds_spatial = F.interpolate(Gds, size=(Fs.size(2), Fs.size(3)), mode='bilinear', align_corners=False)
        # 相乘
        s_desc_fused = s_desc * Gds_spatial # 此时 Gds_spatial 是 (B, 1, H, W)
        """
        # Gds = self.d_to_s_mlp(Ac)
        Gds = self.d_to_s_conv(Fd * Ac.view(Fd.size(0), Fd.size(1), 1, 1))

        # 3. 浅层空间注意力计算 (As), 注入 Gds
        Gds_up = F.interpolate(Gds, size=(Fs.size(2), Fs.size(3)), mode='bilinear', align_corners=False)

        #这种是使用了
        # s_desc_fused = s_desc * Gds.view(-1, 1, 1, 1) # 语义引导注入
        s_desc_fused = s_desc *Gds_up # 语义引导注入
        As = self.sigmoid(self.s_conv7x7(s_desc_fused))
        Fs_out = Fs * As

        # 4. 浅层对深层的空间引导 (As -> Gsd)
        s_guide_vec = F.adaptive_avg_pool2d(As, 1).view(As.size(0), -1)
        Gsd = self.s_to_d_mlp(s_guide_vec)

        # 5. 深层最终权重融合
        final_Ac = self.sigmoid(d_stats + Gsd).view(Fd.size(0), Fd.size(1), 1, 1)
        Fd_out = Fd * final_Ac

        return Fs_out, Fd_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, gate_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, ratio)
        self.SpatialGate = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = x * self.ChannelGate(x)
        x_out = x_out * self.SpatialGate(x_out)
        return x_out

