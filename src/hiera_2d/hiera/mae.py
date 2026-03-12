import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hiera_2d.hiera.types import Model

from .blocks import Mlp
from .model import Hiera
from .reroll import undo_windowing
from .token_ops import MASK_KEEP, MASK_MASK, broadcast_mask, compute_patch_stats, patchify, unpatchify


def validate_mae_encoder_compatibility(encoder: Hiera):
    """Ensures that the given Hiera encoder is compatible with a MAE architecture (should always be the case)"""
    config = encoder.config

    if config.patch_embed.stride_px[0] != config.patch_embed.stride_px[1]:
        raise ValueError("MAE requires square patch_embed.stride_px")

    if config.stride_q_tk[0] != config.stride_q_tk[1]:
        raise ValueError("MAE requires square stride_q_tk")


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply fusion conv to intermediate features."""

    if isinstance(head, nn.Identity):
        return x

    B, N = x.shape[0:2]

    x = x.reshape(
        B * N, *x.shape[2:]
    ).movedim(
        -1, 1
    )  # move n_mu into batch dim s.t. convolution can be applied (this processes each mask unit independently as if it were its own batch element), and move channel dim to expected position for conv (B * n_mu, C, mu_h, mu_w)
    x = head(x)

    # move channel dim back and restore batch dim
    x = x.movedim(1, -1)
    x = x.reshape(B, N, *x.shape[1:])
    return x  # (B, n_mu, mu_h', mu_w', d_encoder_out)


class DecoderBlock(nn.Module):
    """Simple transformer block for decoder"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )  # just use standard multihead attention from pytorch

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), d_out=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MAEConfig(Model):
    d_decoder_embed: int
    n_decoder_blocks: int
    n_decoder_heads: int


class HieraMAE(nn.Module):
    def __init__(
        self,
        encoder: Hiera,
        config: MAEConfig,
    ):
        super().__init__()
        validate_mae_encoder_compatibility(encoder)
        self.encoder = encoder
        self.config = config
        self.encoder_config = self.encoder.config

        # Get encoder properties
        self.shapes = self.encoder.shapes

        d_encoder_out = self.encoder.d_encoder_out

        self.encoder_norm = nn.LayerNorm(d_encoder_out)

        # multi-scale fusion heads to fuse intermediates from different stages
        self.multi_scale_fusion_heads = nn.ModuleList()

        # track mu size and embed dim at each stage to configure heads correctly
        curr_mu_size = list(self.encoder_config.sz_mask_unit_tk)
        curr_dim = self.encoder_config.d_embed

        # go over stages
        for stage_idx in range(self.shapes.n_q_pool):
            # Kernel to go from curr_mu_size to final_mu_size
            kernel = tuple(c // f for c, f in zip(curr_mu_size, self.shapes.sz_mask_unit_tk_final, strict=True))

            # Update dimension (dims increase at the start of each stage except first)
            if stage_idx > 0:
                curr_dim = int(curr_dim * self.encoder_config.dim_mul)

            self.multi_scale_fusion_heads.append(
                # fusion head is just a conv that maps from curr_dim to d_encoder_out with a kernel that reduces from curr_mu_size to final mu size
                nn.Conv2d(
                    curr_dim,
                    d_encoder_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )

            # Update MU size for next stage (after q_pool)
            curr_mu_size = [c // s for c, s in zip(curr_mu_size, self.encoder_config.stride_q_tk, strict=True)]

        # final stage already has correct shape
        self.multi_scale_fusion_heads.append(nn.Identity())

        # embedding for decoder input
        self.decoder_embed = nn.Linear(d_encoder_out, self.config.d_decoder_embed)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.d_decoder_embed))

        # Decoder positional embedding (learnable)
        n_decoder_tokens = math.prod(self.shapes.sz_tk_final)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_decoder_tokens, self.config.d_decoder_embed))

        # Decoder transformer blocks (no pooling or anything)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=self.config.d_decoder_embed,
                    n_heads=self.config.n_decoder_heads,
                    mlp_ratio=self.encoder_config.mlp_ratio,
                )
                for _ in range(self.config.n_decoder_blocks)
            ]
        )

        self.decoder_norm = nn.LayerNorm(self.config.d_decoder_embed)

        self.stride_pred_px = self.encoder_config.patch_embed.stride_px[0] * (
            self.encoder_config.stride_q_tk[0] ** self.shapes.n_q_pool
        )

        # Each decoder token predicts stride_pred_px^2 pixels * n_channels
        self.decoder_pred = nn.Linear(
            self.config.d_decoder_embed,
            (self.stride_pred_px**2) * self.encoder_config.n_channels,
        )

        self._initialize_weights()

    @staticmethod
    def _init_module(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _initialize_weights(self):
        """Weight init following original Hiera MAE: xavier for Linear, trunc_normal for embeddings."""
        self.apply(self._init_module)

        nn.init.trunc_normal_(self.encoder.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        w = self.encoder.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float):
        """Generate a per-sample random mask over encoder mask units."""
        B = x.shape[0]
        n_mask_units = self.shapes.n_mu
        n_keep = int(n_mask_units * (1 - mask_ratio))

        # Keep at least one visible and one masked unit for stable reconstruction loss.
        n_keep = min(max(n_keep, 1), n_mask_units - 1)

        noise = torch.rand(B, n_mask_units, device=x.device)

        # sort noise for each sample to get random shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros(B, n_mask_units, device=x.device, dtype=torch.bool)
        mask[:, :n_keep] = MASK_KEEP

        # reorder using the shuffle indices to get final mask
        return torch.gather(mask, dim=1, index=ids_restore)

    def _resolve_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            if not (0.0 < mask_ratio < 1.0):
                raise ValueError(f"mask_ratio must be in (0, 1) when mask is not provided, got {mask_ratio}")

            # generate random mask
            return self.get_random_mask(x, mask_ratio)

        # validate provided mask

        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("mask must be a boolean tensor of shape (B, n_mask_units)")

        expected_shape = (x.shape[0], self.shapes.n_mu)

        if mask.shape != expected_shape:
            raise ValueError(f"mask shape must be {expected_shape}, got {tuple(mask.shape)}")

        keep_counts = mask.sum(dim=1)
        if not torch.equal(keep_counts, keep_counts[:1].expand_as(keep_counts)):
            raise ValueError("all samples in a batch must keep the same number of mask units")

        return mask

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask: torch.Tensor | None = None,
    ):
        """Encode input with masking and multi-scale fusion."""
        mask = self._resolve_mask(x, mask_ratio, mask)

        # Get multi-scale representations from encoder
        _, intermediates = self.encoder.forward(x, mask, return_intermediates=True)

        # Only use features from q_pool stages + final
        # Resolution unchanged after q_pool stages happened, so skip those intermediates
        # IMPORTANT!! we assume that q-pool stages only follow other q-pool stages
        intermediates = intermediates[: self.shapes.n_q_pool] + intermediates[-1:]

        # sum multi-scale features with fusion heads
        x = apply_fusion_head(self.multi_scale_fusion_heads[0], intermediates[0])
        for head, interm in zip(self.multi_scale_fusion_heads[1:], intermediates[1:], strict=True):
            x = x + apply_fusion_head(head, interm)

        x = self.encoder_norm(x)

        return x, mask  # (B, n_visible_mu, mu_h_final, mu_w_final, d_encoder_out), (B, n_mu)

    def forward_decoder(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Decode with mask tokens inserted."""

        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Create full tensor with space for mask tokens
        # x: (B, n_visible_mu, mu_h_final, mu_w_final, d_decoder_embed)
        # mask: (B, n_mu)
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)

        # expand mask to work with decoder shape
        mask_tokens = self.mask_token.view((1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,))
        mask_expanded = broadcast_mask(mask, x.shape[2:])

        # Insert visible tokens and mask tokens
        x_dec[mask_expanded] = x.flatten()  # place visible tokens
        x_dec = ~mask_expanded * mask_tokens + mask_expanded * x_dec  # where mask is False, place mask token

        # Get back spatial order for both x and our mask
        x = undo_windowing(
            x_dec,
            self.shapes.sz_tk_final,
            self.shapes.sz_mask_unit_tk_final,
        )  # (B, H', W', d_decoder_embed) where H' and W' are final token grid dimensions after q-pooling

        pred_mask = undo_windowing(
            mask_expanded[..., 0:1],
            self.shapes.sz_tk_final,
            self.shapes.sz_mask_unit_tk_final,
        )  # (B, H', W', 1) boolean mask

        # Flatten spatial dimensions to get sequences for transformer
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, H'*W', C)
        pred_mask = pred_mask.reshape(x.shape[0], -1)  # (B, H'*W')

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # Predict pixels
        x = self.decoder_pred(x)  # (B, H'*W', stride_pred_px^2 * n_channels)

        return x, pred_mask

    def get_pixel_label(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        normalize: bool = True,
    ):
        """Get pixel-level reconstruction target."""
        ps = self.stride_pred_px

        # patchify to get (B, H'*W', ps^2 * n_channels)
        label = patchify(x, ps)

        # Select masked patches (we only predict those!)
        label = label[mask == MASK_MASK]

        if normalize:
            mean, var = compute_patch_stats(label)
            # normalize labels on a per-patch basis and add small epsilon for stability
            label = (label - mean) / (var + 1e-6).sqrt()

        return label

    def forward_loss(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Compute reconstruction loss on masked regions only."""

        label = self.get_pixel_label(x, mask)  # (n_masked_total, ps^2*C)
        pred_masked = pred[mask == MASK_MASK]  # select masked tokens and their predictions (thus same shape)

        loss = F.mse_loss(pred_masked, label)

        return loss, pred_masked, label

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode flat encoder tokens to pixel space (no masking).

        Args:
            tokens: (B, N, d_encoder_out) where N = H_tk * W_tk
        Returns:
            (B, C, H, W) reconstructed image
        """
        x = self.decoder_embed(tokens)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (B, N, stride_pred_px^2 * C)

        h_tk, w_tk = self.shapes.sz_tk_final
        return unpatchify(x, self.stride_pred_px, h_tk, w_tk, self.encoder_config.n_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: torch.Tensor | None = None,
    ):
        # encode
        latent, mask = self.forward_encoder(x, mask_ratio, mask)

        # decode with latents and mask
        pred, pred_mask = self.forward_decoder(latent, mask)

        # calculate loss
        loss, pred_masked, label = self.forward_loss(x, pred, pred_mask)

        return loss, pred_masked, label, mask
