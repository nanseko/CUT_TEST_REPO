""" PyTorch port of the attention modules used in the TensorFlow CUT fork.

Mirrors modules/attention.py (TF, NHWC) but in PyTorch (NCHW):
    - ChannelAttention, SpatialAttention, CBAM
    - CoordinateAttention
    - make_attention(attention_type, channels, reduction) factory

Drop this file into the official CUT repo (taesungp/contrastive-unpaired-translation),
e.g. as models/attention.py, and import from models/networks.py.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention (CBAM) for NCHW tensors."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        # shared MLP applied to GAP and GMP descriptors
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))                 # [B, C]
        mx = x.amax(dim=(2, 3))                  # [B, C]
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return w.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention (CBAM) for NCHW tensors."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)        # [B,1,H,W]
        mx = x.amax(dim=1, keepdim=True)         # [B,1,H,W]
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return w


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial)."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


class CoordinateAttention(nn.Module):
    """Lightweight coordinate attention (directional) for NCHW tensors."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.bottleneck = nn.Conv2d(channels, bottleneck, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(bottleneck, channels, 1)
        self.conv_w = nn.Conv2d(bottleneck, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # pool along W (keep H) and along H (keep W)
        pooled_h = x.mean(dim=3, keepdim=True)               # [B,C,H,1]
        pooled_w = x.mean(dim=2, keepdim=True)               # [B,C,1,W]
        pooled_w = pooled_w.permute(0, 1, 3, 2)              # [B,C,W,1]
        y = torch.cat([pooled_h, pooled_w], dim=2)           # [B,C,H+W,1]
        y = self.act(self.bottleneck(y))                     # [B,bottleneck,H+W,1]
        attn_h, attn_w = torch.split(y, [h, w], dim=2)
        attn_w = attn_w.permute(0, 1, 3, 2)                  # [B,bottleneck,1,W]
        attn_h = torch.sigmoid(self.conv_h(attn_h))          # [B,C,H,1]
        attn_w = torch.sigmoid(self.conv_w(attn_w))          # [B,C,1,W]
        return x * attn_h * attn_w


def make_attention(attention_type, channels, reduction=16):
    """Factory returning an attention module or Identity for 'none'."""
    if attention_type == 'none' or attention_type is None:
        return nn.Identity()
    if attention_type == 'cbam':
        return CBAM(channels, reduction)
    if attention_type == 'coord':
        return CoordinateAttention(channels, reduction)
    raise ValueError(f'Unsupported attention type: {attention_type}')
