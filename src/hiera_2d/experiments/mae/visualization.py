import torch
import torchvision.utils as vutils

from hiera_2d.hiera.token_ops import compute_patch_stats, patchify, unpatchify

DEFAULT_FIXED_N_SAMPLES = 10
PANEL_ORDER = ("original", "composite", "masked", "prediction")


def _normalize_for_vis(reference: torch.Tensor, *tensors: torch.Tensor):
    # normalize all tensors to [0,1] range for viz based on reference tensor stats (e.g. original image)
    reduce_dims = tuple(range(1, reference.ndim))
    ref_min = reference.amin(dim=reduce_dims, keepdim=True)
    ref_max = reference.amax(dim=reduce_dims, keepdim=True)
    denom = (ref_max - ref_min).clamp_min(1e-8)

    normalized = []

    for t in tensors:
        normalized.append(((t - ref_min) / denom).clamp(0.0, 1.0))

    return normalized


def render_reconstruction_grid(
    model,
    x_batch: torch.Tensor,
    mask_ratio: float,
):
    # Forward pass - get full predictions before masking
    latent, mask = model.forward_encoder(x_batch, mask_ratio=mask_ratio)
    pred, pred_mask = model.forward_decoder(latent, mask)

    # pred is (B, H'*W', stride_pred_px^2 * C)
    B = pred.shape[0]
    stride_pred_px = model.stride_pred_px
    n_channels = model.encoder_config.n_channels
    h_tk, w_tk = model.shapes.sz_tk_final  # token grid shape

    # De-normalize predictions using per-patch stats (just undo patch normalization by using true patch stats)
    x_patches = patchify(x_batch, stride_pred_px)
    patch_mean, patch_var = compute_patch_stats(x_patches)
    pred = pred * (patch_var + 1e-6).sqrt() + patch_mean

    # convert flat patch preds to image space
    pred_img = unpatchify(pred, stride_pred_px, h_tk, w_tk, n_channels)

    # create masked input visualization by upscaling mask from (B, n_mu) to input resolution
    mask_vis = mask.float()
    h_mu, w_mu = model.shapes.sz_mu
    mask_vis = mask_vis.reshape(B, h_mu, w_mu)
    # interpolate to input res for viz
    mask_vis = torch.nn.functional.interpolate(mask_vis.unsqueeze(1), size=x_batch.shape[-2:], mode="nearest")
    x_masked = x_batch * mask_vis  # zero out masked regions

    # composite: original where visible, prediction where masked
    pred_mask_vis = pred_mask.float().reshape(B, h_tk, w_tk)
    pred_mask_vis = torch.nn.functional.interpolate(pred_mask_vis.unsqueeze(1), size=x_batch.shape[-2:], mode="nearest")
    x_composite = x_batch * pred_mask_vis + pred_img * (1.0 - pred_mask_vis)

    # average channels to get grayscale for visualization
    x_vis = x_batch.mean(dim=1, keepdim=True)
    x_masked_vis = x_masked.mean(dim=1, keepdim=True)
    pred_vis = pred_img.mean(dim=1, keepdim=True)
    composite_vis = x_composite.mean(dim=1, keepdim=True)

    # normalize again because we have averaged and denormalized, so values may not be in [0,1] range anymore. use original image stats for consistency.
    x_vis, x_masked_vis, pred_vis, composite_vis = _normalize_for_vis(
        x_vis,
        x_vis,
        x_masked_vis,
        pred_vis,
        composite_vis,
    )

    samples = {
        "original": x_vis,
        "masked": x_masked_vis,
        "prediction": pred_vis,
        "composite": composite_vis,
    }

    n_samples = samples[PANEL_ORDER[0]].shape[0]
    tiles = [samples[key][idx : idx + 1] for idx in range(n_samples) for key in PANEL_ORDER]
    return vutils.make_grid(torch.cat(tiles, dim=0), nrow=len(PANEL_ORDER), normalize=False)
