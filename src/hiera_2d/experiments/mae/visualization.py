from dataclasses import dataclass

import torch
import torchvision.utils as vutils

from hiera_2d.hiera.token_ops import compute_patch_stats, patchify, unpatchify

DEFAULT_FIXED_N_SAMPLES = 10
PANEL_ORDER = ("original", "composite", "masked", "prediction")


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """An MAE forward pass unpacked into image-space fields, all `(B, C, H, W)`.

    Values are in the model's input units (i.e. still normalized). `prediction` is
    the decoder output everywhere, including where the encoder could see;
    `composite` keeps the original wherever it was visible and substitutes the
    prediction only where it was masked -- the panel that actually shows what the
    model inferred. `visible` is the mask-unit indicator upsampled to pixels: 1
    where the pixel was fed to the encoder, 0 where it was held out.
    """

    original: torch.Tensor
    prediction: torch.Tensor
    composite: torch.Tensor
    visible: torch.Tensor


def reconstruct(model, x_batch: torch.Tensor, mask_ratio: float) -> Reconstruction:
    """Run the MAE end to end and return its image-space fields.

    The decoder emits per-patch predictions with the patch's own mean/variance
    divided out (the normalized-pixel target), so they are put back on the scale
    of the true patch statistics before unpatchifying -- otherwise every patch
    would come back standardized and the field would be visually flat.
    """
    latent, mask = model.forward_encoder(x_batch, mask_ratio=mask_ratio)
    pred, pred_mask = model.forward_decoder(latent, mask)

    batch = pred.shape[0]
    stride_pred_px = model.stride_pred_px
    n_channels = model.encoder_config.n_channels
    h_tk, w_tk = model.shapes.sz_tk_final

    x_patches = patchify(x_batch, stride_pred_px)
    patch_mean, patch_var = compute_patch_stats(x_patches)
    pred = pred * (patch_var + 1e-6).sqrt() + patch_mean

    prediction = unpatchify(pred, stride_pred_px, h_tk, w_tk, n_channels)

    h_mu, w_mu = model.shapes.sz_mu
    visible = torch.nn.functional.interpolate(
        mask.float().reshape(batch, 1, h_mu, w_mu),
        size=x_batch.shape[-2:],
        mode="nearest",
    )

    # The decoder's own kept-token mask lives on the (coarser) prediction grid, so
    # the composite is stitched with that one rather than with `visible`.
    kept = torch.nn.functional.interpolate(
        pred_mask.float().reshape(batch, 1, h_tk, w_tk),
        size=x_batch.shape[-2:],
        mode="nearest",
    )
    composite = x_batch * kept + prediction * (1.0 - kept)

    return Reconstruction(original=x_batch, prediction=prediction, composite=composite, visible=visible)


def normalize_for_vis(reference: torch.Tensor, *tensors: torch.Tensor):
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
    """Channel-averaged grayscale panel grid for TensorBoard training monitoring.

    The report-quality figure is `hiera_2d.analysis.recon_figure` instead; this one
    is deliberately cheap and unlabelled.
    """
    r = reconstruct(model, x_batch, mask_ratio)

    x_vis = r.original.mean(dim=1, keepdim=True)
    x_masked_vis = (r.original * r.visible).mean(dim=1, keepdim=True)
    pred_vis = r.prediction.mean(dim=1, keepdim=True)
    composite_vis = r.composite.mean(dim=1, keepdim=True)

    # normalize again because we have averaged and denormalized, so values may not be in [0, 1]
    # anymore. use original image stats for consistency.
    x_vis, x_masked_vis, pred_vis, composite_vis = normalize_for_vis(
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
