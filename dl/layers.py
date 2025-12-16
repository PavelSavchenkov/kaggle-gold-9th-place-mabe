from __future__ import annotations

import math

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


class TemporalConvBlock(nn.Module):
    """
    Simple residual 1D conv block with dilation.

    Input / output: [B, C, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding"
        padding = (kernel_size - 1) * dilation // 2  # symmetric, keeps T

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # residual connection (1x1 conv if channel dim changes)
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.activation(out)
        return out


class ConvBlock1D(nn.Module):
    """
    Simple conv -> BN -> GELU -> Dropout block.

    Input / output: [B, C, T]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same-ish' padding"

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding as in "Attention is All You Need".

    Expects input [B, T, D] and adds pe[:T].
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    BYOL-style cosine loss between predicted and target latents.
    p, z: [B, H]
    """
    p = F.normalize(pred, dim=-1)
    z = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()
