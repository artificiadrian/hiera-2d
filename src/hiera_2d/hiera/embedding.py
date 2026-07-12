import torch
import torch.nn as nn

from hiera_2d.hiera.types import Int2d, Model


class PatchEmbedConfig(Model):
    kernel_px: Int2d
    stride_px: Int2d
    padding_px: Int2d


class PatchEmbed(nn.Module):
    """Embed image patches into a token sequence using strided conv2d."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        config: PatchEmbedConfig,
    ):
        super().__init__()
        if len(config.kernel_px) != 2 or len(config.stride_px) != 2 or len(config.padding_px) != 2:
            raise ValueError("PatchEmbed expects 2D kernel/stride/padding tuples")

        self.proj = nn.Conv2d(
            d_in, d_out, kernel_size=config.kernel_px, stride=config.stride_px, padding=config.padding_px
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, d_out, H', W')
        x = x.flatten(2)  # (B, d_out, H' * W')
        x = x.transpose(1, 2)  # (B, H' * W', d_out)
        return x
