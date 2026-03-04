from collections.abc import Sequence

import torch

# Mask convention: True = keep/visible, False = mask/reconstruct.
MASK_KEEP = True
MASK_MASK = False


def patchify(x: torch.Tensor, patch_size: int):
    """Convert (B, C, H, W) to (B, H/ps*W/ps, ps*ps*C)."""

    if x.shape[-2] % patch_size != 0 or x.shape[-1] % patch_size != 0:
        # make sure we can divide into patches cleanly, otherwise this would complicate things
        raise ValueError("H and W must be divisible by patch_size")

    x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

    # unfold slides a window of size patch_size across H and W dimensions with stride of patch_size, thus giving us non-overlapping patches
    patches = x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)  # (B, H/ps, W/ps, C, ps, ps)

    # first flatten the H/W grid into a sequence of patches, then flatten the patch dimension (C, ps, ps) into single dim of ps*ps*C size
    # thus, we have a sequence of flat patches, where each patch is a vector of pixel vals
    return patches.flatten(1, 2).flatten(2)


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    h_tokens: int,
    w_tokens: int,
    C: int,
):
    """Convert (B, H/ps*W/ps, ps*ps*C) back to (B, C, H, W). (inverse of patchify)"""
    B = patches.shape[0]

    # h_tokens/w_tokens are number of patches in each spatial dimension

    x = patches.reshape(B, h_tokens, w_tokens, C, patch_size, patch_size)  # (B, H/ps, W/ps, C, ps, ps)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/ps, ps, W/ps, ps)
    return x.reshape(B, C, h_tokens * patch_size, w_tokens * patch_size)  # (B, C, H, W)


def compute_patch_stats(patches: torch.Tensor):
    if patches.numel() == 0:
        raise ValueError("cannot compute patch statistics on an empty tensor")

    mean = patches.mean(dim=-1, keepdim=True)

    # Use population variance to avoid edge-case instability with small sample counts.
    var = patches.var(dim=-1, keepdim=True, unbiased=False)

    return mean, var


def expand_mask_units(mask: torch.Tensor, n_tk_per_mu: int):
    # The mask is defined per mask unit, but the encoder works with a flat token sequence.
    # Expand to per-token so we can index into the sequence to keep only visible tokens.
    # Like broadcast_mask but flattened: (B, n_mu) -> (B, n_mu * n_tk_per_mu)
    return mask.unsqueeze(-1).expand(-1, -1, n_tk_per_mu).reshape(mask.shape[0], -1)


def broadcast_mask(mask: torch.Tensor, tail_shape: Sequence[int]):
    # Same idea as expand_mask_units, but for the decoder where tokens have spatial structure.
    # We need the mask to match the full tensor shape so we can index into it to place
    # visible tokens and mask tokens into the right positions.
    # e.g. (B, n_mu) with tail_shape (mu_h, mu_w, d) -> (B, n_mu, mu_h, mu_w, d)
    expand_shape = mask.shape + (1,) * len(tail_shape)
    return mask.reshape(expand_shape).expand(mask.shape + tuple(tail_shape)).bool()
