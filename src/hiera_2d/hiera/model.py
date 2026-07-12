import torch
import torch.nn as nn

from hiera_2d.hiera.blocks import HieraBlock
from hiera_2d.hiera.embedding import PatchEmbed, PatchEmbedConfig
from hiera_2d.hiera.reroll import Reroll, Unroll
from hiera_2d.hiera.token_ops import expand_mask_units
from hiera_2d.hiera.types import HieraShapes, Int2d, Model


class HieraStageConfig(Model):
    n_blocks: int
    is_mask_unit_attn: bool
    pool_q: bool


class HieraConfig(Model):
    sz_in_px: Int2d
    n_channels: int
    d_embed: int
    n_heads: int
    stages: tuple[HieraStageConfig, ...]
    sz_mask_unit_tk: Int2d
    stride_q_tk: Int2d
    patch_embed: PatchEmbedConfig
    dim_mul: float
    head_mul: float
    mlp_ratio: float


class Hiera(nn.Module):
    def __init__(
        self,
        config: HieraConfig,
    ):
        super().__init__()
        self.config = config

        # count how many stages apply q_pool
        self.n_q_pool = sum(1 for s in self.config.stages if s.pool_q)

        self.shapes = HieraShapes(
            sz_in_px=self.config.sz_in_px,
            stride_patch_px=self.config.patch_embed.stride_px,
            sz_mask_unit_tk=self.config.sz_mask_unit_tk,
            stride_q_tk=self.config.stride_q_tk,
            n_q_pool=self.n_q_pool,
        )

        # we omit the original "sep_pos_embed" option and just implement the "False" branch by default
        # thus, pos embeddings are a learnable param
        self.pos_embed = nn.Parameter(torch.zeros(1, self.shapes.n_tokens, self.config.d_embed))

        blocks, stage_ends, d_out = self._build_blocks()

        self.blocks = nn.ModuleList(blocks)
        self.stage_ends = stage_ends
        self.d_encoder_out = d_out  # final output dimension

        self.patch_embed = PatchEmbed(
            d_in=self.config.n_channels,
            d_out=self.config.d_embed,
            config=self.config.patch_embed,
        )
        # build unroll schedule for Unroll/Reroll
        unroll_schedule = [self.config.stride_q_tk] * self.n_q_pool

        self.unroll = Unroll(
            sz_tk=self.shapes.sz_tk,
            unroll_schedule=unroll_schedule,
        )

        self.reroll = Reroll(
            sz_in_px=self.shapes.sz_in_px,
            stride_patch_px=self.config.patch_embed.stride_px,
            unroll_schedule=unroll_schedule,
            stage_ends=stage_ends,
            n_q_pool=self.n_q_pool,
            sz_mask_unit_tk=self.config.sz_mask_unit_tk,
        )

    def _build_blocks(
        self,
    ):
        """Build Hiera encoder blocks, configuring mu-attention and q-pooling at the correct stages."""
        d_in = self.config.d_embed
        d_out = self.config.d_embed
        n_heads = self.config.n_heads
        prev_stage: HieraStageConfig | None = None
        cur_n_tk_per_mu = self.shapes.n_tk_per_mu
        stride_q_flat = self.shapes.stride_q_flat

        blocks: list[HieraBlock] = []
        stage_ends: list[int] = []
        global_block_idx = 0

        for stage_idx, stage in enumerate(self.config.stages):
            for block_idx in range(stage.n_blocks):
                # q-pool is triggered by previous stage, as ops are applied at beginning
                is_q_pooling_block = prev_stage is not None and prev_stage.pool_q and block_idx == 0

                if is_q_pooling_block:
                    # we applied q-pooling, so we need to decrease flat mask unit size
                    cur_n_tk_per_mu //= stride_q_flat

                # this is faithful to the original implementation: d_out and n_heads is updated at the beginning of
                # each block (except for the very first), regardless if the stage applies q-pooling or not
                if block_idx == 0 and stage_idx > 0:
                    d_out = int(d_out * self.config.dim_mul)
                    n_heads = int(n_heads * self.config.head_mul)

                blocks.append(
                    HieraBlock(
                        d_in=d_in,
                        d_out=d_out,
                        n_heads=n_heads,
                        n_tk_per_mu=cur_n_tk_per_mu if stage.is_mask_unit_attn else None,
                        stride_q_tk=stride_q_flat if is_q_pooling_block else 1,
                        mlp_ratio=self.config.mlp_ratio,
                    )
                )

                d_in = d_out
                global_block_idx += 1

            # track end of each stage (block index)
            stage_ends.append(global_block_idx - 1)
            prev_stage = stage

        return blocks, stage_ends, d_out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass through the Hiera encoder to the final token sequence."""
        tokens, _ = self._encode(x, mask, collect_intermediates=False)
        return tokens

    def forward_with_intermediates(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Like `forward`, but also returns the per-stage-end feature maps (rerolled to
        the 2D grid) that the AR head consumes as a multi-scale skip stack."""
        return self._encode(x, mask, collect_intermediates=True)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor | None, *, collect_intermediates: bool):
        # embed input patches -> (B, N, d_out)
        # why unroll? after patch embedding, tokens of the same mask unit are not contiguous
        # (it goes left-to-right top-to-bottom), thus mixing tokens of different mask units.
        # Thus, we need to perform "Unroll" to make sure they are again in a contiguous order
        # to not break the grid, i.e. assume PatchEmbed produces a sequence A A B B A A B B
        # then we would Unroll to get A A A A B B B B

        x = self.patch_embed(x) + self.pos_embed
        x = self.unroll(x)

        if mask is not None:
            # Expand mask from (B, n_mask_units) to (B, n_tokens)
            # Each mask unit contains n_tk_per_mu tokens
            mask_expanded = expand_mask_units(mask, self.shapes.n_tk_per_mu)

            # Keep only visible tokens (where mask is True)
            n_visible = int(mask_expanded[0].sum().item())
            x = x[mask_expanded].reshape(x.shape[0], n_visible, x.shape[2])

        intermediates = list[torch.Tensor]()

        for i, block in enumerate(self.blocks):
            x = block(x)

            if collect_intermediates and i in self.stage_ends:
                # reroll the flat token sequence back to the 2D grid at each stage end
                interm = self.reroll(x, block_idx=i, mask=mask)
                intermediates.append(interm)

        return x, intermediates
