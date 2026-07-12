import torch
import torch.nn as nn

from hiera_2d.hiera.attention import Attention


class Mlp(nn.Module):
    """Simple MLP with one hidden layer and GELU activation used in Hiera blocks."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int | None = None,
    ):
        super().__init__()

        fc1 = nn.Linear(d_in, d_hidden)
        act = nn.GELU()
        # timm-compatible default (as og Hiera uses timm library): project back to d_in if d_out is omitted.
        fc2 = nn.Linear(d_hidden, d_out or d_in)

        self.proj = nn.Sequential(fc1, act, fc2)

    def forward(self, x: torch.Tensor):
        return self.proj(x)


class MaxPoolNd(nn.Module):
    """Applies max pooling with kernel size equal to stride, taken from og Hiera source code."""

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor):
        # taken from https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_utils.py#L81
        return x.view(x.shape[0], self.stride, -1, x.shape[-1]).max(dim=1).values


class HieraBlock(nn.Module):
    """Single Hiera transformer block."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        n_tk_per_mu: int | None,
        stride_q_tk: int,
        mlp_ratio: float,
    ):
        super().__init__()

        self.is_dimension_mismatch = d_in != d_out
        self.is_stride = stride_q_tk > 1

        self.norm1 = nn.LayerNorm(d_in)

        self.attn = Attention(
            d_in=d_in,
            d_out=d_out,
            n_heads=n_heads,
            stride_q_tk=stride_q_tk,
            n_tk_per_mu=n_tk_per_mu,
        )

        self.norm2 = nn.LayerNorm(d_out)

        self.mlp = Mlp(d_out, int(d_out * mlp_ratio))

        self.skip_proj = nn.Linear(d_in, d_out) if self.is_dimension_mismatch else None
        self.skip_pool = MaxPoolNd(stride_q_tk) if self.is_stride or self.is_dimension_mismatch else None

    def forward(self, x: torch.Tensor):
        x_norm = self.norm1(x)

        # Attention and skip connection (if q-pooling applied or dimensions do not match, also pool skip connection as
        # we would have a dimension mismatch else)
        skip = self.skip_proj(x_norm) if self.is_dimension_mismatch else x
        if self.skip_pool is not None:
            skip = self.skip_pool(skip)

        x = skip + self.attn(x_norm)

        # MLP with skip connection
        x = x + self.mlp(self.norm2(x))

        return x
