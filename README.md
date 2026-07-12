# hiera-2d

A from-scratch 2D reimplementation of [Hiera](https://github.com/facebookresearch/hiera) with MAE pretraining and
an autoregressive next-frame head, used as a neural surrogate for 2D Kolmogorov flow. The experiment: does MAE
pretraining improve autoregressive rollout compared to training from scratch, at N = 50, 100, 250 training
trajectories?

Result: at a fixed finetuning budget, pretraining helps, and the advantage grows with N and rollout horizon —
roughly neutral at N=50, a clear win at N=250. The pretrained model also injects less spurious high-wavenumber
energy over long rollouts. So it's a convergence-speed effect, not the low-data sample-efficiency effect originally
hypothesized. All comparisons use bootstrap 95% CIs and paired significance tests.

## Setup

Python 3.12, a CUDA GPU (developed on an 8 GB RTX 2070 SUPER), ~30 GB disk for the dataset.

```bash
uv sync
```

## Data

The dataset is not checked in; it is regenerated deterministically (fixed seeds):

```bash
# ~28 GB, ~2 h on an 8 GB GPU
uv run dg-kolmogorov --grid-size 256 --re-min 3000 --re-max 5000 --n-re-values 20 \
    --dt 5e-4 --num-seeds 625 --collect 100 --keep-every 200 \
    --burn-min 20 --burn-max 40 -o kolmogorov2d_256_variedRe.h5
```

This yields 500 train / 125 val trajectories, each 100 frames of (u, v) velocity at 256×256 saved 0.1 s apart,
with one Reynolds number and burn-in per trajectory. Trajectories that diverge near the CFL limit are retried at
half the timestep (with `keep-every` doubled), so the saved frame spacing is identical for all trajectories.

## Reproducing the results

Run the sweep. For each N it trains an MAE, an AR finetune off it, and a from-scratch baseline; both AR arms get
the same 30-epoch budget, so only the encoder init differs. Finished runs are skipped, so it resumes idempotently:

```bash
uv run run-scaling configs/scaling/kg_scaling.toml
```

Then produce the figures (written under `outputs/kg_scaling/`):

```bash
# scaling curves + improvement table, per rollout horizon
for H in 5 10 40; do
  uv run scaling-curve configs/scaling/kg_scaling.toml --n-steps $H -o outputs/kg_scaling/analysis_h$H
done

# power spectra (ground truth vs. finetune vs. scratch)
for N in 50 100 250; do
  uv run spectral-plot configs/scaling/kg_scaling.toml --n $N \
      -o outputs/kg_scaling/analysis_spectral/spectrum_N$N.png
done

# dataset decorrelation time, MAE loss curve, reconstruction panel
uv run autocorrelation --data-path kolmogorov2d_256_variedRe.h5 --delta-t 0.1 \
    -o outputs/analysis_autocorr/autocorrelation.png
uv run loss-curve outputs/kg_scaling/N250/mae_e120 -o outputs/analysis_loss/loss_curve.png
uv run recon-figure --checkpoint outputs/kg_scaling/N250/mae_e120/checkpoints/best_model.pt \
    --sample 0 -o outputs/analysis_recon/recon_sample0.png
```

Individual arms can also be trained directly with `train-mae` / `train-ar`; see `--help` and the comments in
`configs/scaling/*.toml`.

## Notes

- Training eager-loads the train subset; N=500 needs ~26 GB host RAM, N≤250 is fine on a desktop.
- Architecture configs live in `configs/{hiera,mae,ar}/`, experiment configs in `configs/scaling/`.
- Lint: `uv run ruff check src`

## Examples

MAE pretraining loss and a reconstruction on a validation frame (60% masking):

![Loss curve](assets/kg_loss_curve.png)
![Reconstruction sample](assets/kg_recon_a.png)