"""Autoregressive rollout evaluation for a trained HieraAR model.

Produces the artifacts the thesis needs:
  - metrics.json: per-step and aggregate MSE (1-step and full-rollout).
  - spectrum_timeavg.png: radially-averaged power spectrum, ground truth (dark)
    vs prediction, averaged over the rollout. The headline "frequency layout"
    plot.
  - spectrum_leadtime.png: ground truth vs prediction spectrum at increasing
    lead times, showing where/when high-wavenumber energy is lost.
  - rollout.gif: ground truth | prediction | error over the rollout.

Designed so the same script works for any AR checkpoint, which lets us drop two
runs (pretrained vs from-scratch) into the same figures.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from hiera_2d.analysis.spectra import power_db, radial_energy_spectrum, vorticity
from hiera_2d.experiments.ar.model import ARHeadConfig, HieraAR
from hiera_2d.experiments.data import DatasetType, KolmogorovDataset, Split, get_dataset
from hiera_2d.hiera.model import Hiera, HieraConfig


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """The dataset a checkpoint was trained on: which dataset type, and where it lives."""

    data_path: Path
    dataset: DatasetType


def load_checkpoint_provenance(checkpoint_path: Path) -> tuple[str | None, str | None]:
    """Read the embedded (data_path, dataset) provenance from a checkpoint.

    Either element is None for checkpoints trained before provenance embedding.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return ckpt.get("data_path"), ckpt.get("dataset")


def resolve_dataset_identity(
    checkpoint_path: Path,
    override_data_path: Path | None,
    override_dataset: DatasetType | None,
) -> DatasetIdentity:
    """Decide which dataset to evaluate on, preferring an explicit --data-path override.

    If `override_data_path` is given it wins (a note is printed); otherwise the
    checkpoint's embedded provenance is used. A checkpoint that predates provenance
    embedding, with no override, is a hard error — silently evaluating on the wrong
    data is exactly what this guards against.

    Raises:
        ValueError: the checkpoint lacks embedded provenance and no --data-path
            override was passed.
    """
    ckpt_data_path, ckpt_dataset = load_checkpoint_provenance(checkpoint_path)

    if override_data_path is not None:
        if ckpt_data_path is not None:
            print(f"NOTE: --data-path {override_data_path} overrides checkpoint's recorded data {ckpt_data_path}")

        dataset = override_dataset or (DatasetType(ckpt_dataset) if ckpt_dataset else DatasetType.KOLMOGOROV)
        return DatasetIdentity(data_path=override_data_path, dataset=dataset)

    if ckpt_data_path is None or ckpt_dataset is None:
        msg = (
            f"Checkpoint {checkpoint_path} predates dataset-provenance embedding "
            "(no data_path/dataset recorded). Pass --data-path (and --dataset) explicitly."
        )
        raise ValueError(msg)

    return DatasetIdentity(data_path=Path(ckpt_data_path), dataset=override_dataset or DatasetType(ckpt_dataset))


def load_ar_model(checkpoint_path: Path, device: torch.device) -> HieraAR:
    """Rebuild a HieraAR model from an AR checkpoint.

    The encoder architecture comes from the checkpoint's stored hiera config if
    present, otherwise from the referenced MAE checkpoint (older runs).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ckpt["config"]

    if "hiera" in config:
        hiera_config = HieraConfig.model_validate(config["hiera"])
    else:
        mae_path = Path(config["mae_checkpoint"])
        mae = torch.load(mae_path, map_location=device, weights_only=True)
        hiera_config = HieraConfig.model_validate(mae["train_config"]["hiera"])

    encoder = Hiera(config=hiera_config)
    model = HieraAR(encoder=encoder, config=ARHeadConfig.model_validate(config["ar_config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_trajectory(
    dataset_type: DatasetType, data_path: Path, traj_idx: int, split: Split = Split.VAL
) -> tuple[np.ndarray, float, float]:
    """Return one (T, C, H, W) trajectory in physical units plus its norm stats."""
    dataset = get_dataset(dataset_type, data_path, split=split)
    traj = np.asarray(dataset.sims[traj_idx], dtype=np.float32)
    return traj, dataset.norm_stats["mean"], dataset.norm_stats["std"]


@torch.no_grad()
def rollout(model: HieraAR, x0: np.ndarray, n_steps: int, mean: float, std: float, device: torch.device) -> np.ndarray:
    """Autoregressively roll out n_steps from frame x0 (physical units).

    Returns predictions in physical units, shape (n_steps, C, H, W).
    """
    x = torch.from_numpy((x0[None] - mean) / std).to(device)
    preds = model.rollout(x, n_steps)[0].cpu().numpy()  # (n_steps, C, H, W), normalized
    return preds * std + mean


def per_step_mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """MSE per rollout step, averaged over channels and space."""
    return ((gt - pred) ** 2).mean(axis=(1, 2, 3))


def per_traj_rollout_mse(
    model: HieraAR,
    dataset: KolmogorovDataset,
    start: int,
    n_steps: int,
    mean: float,
    std: float,
    device: torch.device,
    max_traj: int | None = None,
) -> np.ndarray:
    """Full-rollout MSE per trajectory: roll `model` from frame `start` over each
    trajectory (capped at `max_traj`, default all) and return the per-step MSE
    averaged over the horizon — one scalar per trajectory, in trajectory order."""
    n = len(dataset.sims) if max_traj is None else min(max_traj, len(dataset.sims))
    out = np.zeros(n, dtype=np.float64)

    for t in range(n):
        traj = np.asarray(dataset.sims[t], dtype=np.float32)
        steps = min(n_steps, traj.shape[0] - start - 1)
        gt = traj[start + 1 : start + 1 + steps]
        pred = rollout(model, traj[start], steps, mean, std, device)
        out[t] = per_step_mse(gt, pred).mean()

    return out


def plot_spectrum_timeavg(gt: np.ndarray, pred: np.ndarray, out_path: Path, label: str):
    """Rollout-averaged power spectrum: ground truth (dark) vs prediction."""
    k, e_gt = radial_energy_spectrum(gt)
    _, e_pred = radial_energy_spectrum(pred)
    e_gt, e_pred = e_gt.mean(axis=0), e_pred.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k[1:], power_db(e_gt)[1:], color="black", lw=2.5, label="ground truth")
    ax.plot(k[1:], power_db(e_pred)[1:], color="tab:red", lw=1.8, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("power (dB)")
    ax.set_title("Rollout-averaged power spectrum")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrum_leadtime(gt: np.ndarray, pred: np.ndarray, out_path: Path, n_lead: int = 5):
    """Power spectrum at increasing lead times: degradation of the high-k tail."""
    k, e_gt = radial_energy_spectrum(gt)
    _, e_pred = radial_energy_spectrum(pred)
    n_steps = pred.shape[0]
    lead_idx = np.unique(np.linspace(0, n_steps - 1, n_lead).astype(int))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k[1:], power_db(e_gt.mean(axis=0))[1:], color="black", lw=2.5, label="ground truth")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lead_idx)))
    for c, t in zip(colors, lead_idx, strict=True):
        ax.plot(k[1:], power_db(e_pred[t])[1:], color=c, lw=1.4, label=f"pred +{t + 1}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("power (dB)")
    ax.set_title("Predicted spectrum vs lead time")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _to_display_field(frames: np.ndarray, use_vorticity: bool) -> np.ndarray:
    """Map (T, C, H, W) to a (T, H, W) scalar field for visualization."""
    if use_vorticity and frames.shape[1] == 2:
        return vorticity(frames)
    return frames.mean(axis=1)


def save_rollout_gif(gt: np.ndarray, pred: np.ndarray, out_path: Path, use_vorticity: bool, fps: int = 10):
    """Animate ground truth | prediction | error over the rollout."""
    gt_f = _to_display_field(gt, use_vorticity)
    pred_f = _to_display_field(pred, use_vorticity)
    err_f = pred_f - gt_f

    vlim = float(np.abs(gt_f).max())
    elim = float(np.abs(err_f).max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    titles = ["ground truth", "prediction", "error"]
    ims = []
    for ax, field, title, lim, cmap in zip(
        axes, [gt_f, pred_f, err_f], titles, [vlim, vlim, elim], ["RdBu_r", "RdBu_r", "PuOr_r"], strict=True
    ):
        im = ax.imshow(field[0], cmap=cmap, vmin=-lim, vmax=lim, origin="lower", animated=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    txt = fig.suptitle("step 1")
    fig.tight_layout()

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=90):
        for t in range(gt_f.shape[0]):
            ims[0].set_data(gt_f[t])
            ims[1].set_data(pred_f[t])
            ims[2].set_data(err_f[t])
            txt.set_text(f"step {t + 1}")
            writer.grab_frame()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Autoregressive rollout evaluation for HieraAR")
    p.add_argument("--checkpoint", type=Path, required=True, help="AR checkpoint (best_model.pt)")
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the dataset path recorded in the checkpoint (default: use the checkpoint's provenance)",
    )
    p.add_argument(
        "--dataset",
        type=DatasetType,
        choices=list(DatasetType),
        default=None,
        help="Override the dataset type recorded in the checkpoint (default: use the checkpoint's provenance)",
    )
    p.add_argument("--split", type=Split, choices=list(Split), default=Split.VAL)
    p.add_argument("--traj", type=int, default=0, help="Trajectory index for the figures (GIF/spectra)")
    p.add_argument("--start", type=int, default=0, help="Start frame within the trajectory")
    p.add_argument("--n-steps", type=int, default=50, help="Rollout length")
    p.add_argument(
        "--n-traj",
        type=int,
        default=1,
        help="If >1 (or 0=all), metrics.json also reports MSE aggregated over that many "
        "val trajectories (mean+std). Single-trajectory metrics are not representative; "
        "for bootstrapped CIs over all val trajectories, use scaling-curve.",
    )
    p.add_argument("--label", type=str, default="prediction", help="Legend label for the model")
    p.add_argument("--no-gif", action="store_true", help="Skip the (slow) GIF render")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    identity = resolve_dataset_identity(args.checkpoint, args.data_path, args.dataset)
    data_path, dataset = identity.data_path, identity.dataset

    model = load_ar_model(args.checkpoint, device)
    traj, mean, std = load_trajectory(dataset, data_path, args.traj, args.split)

    n_steps = min(args.n_steps, traj.shape[0] - args.start - 1)
    x0 = traj[args.start]
    gt = traj[args.start + 1 : args.start + 1 + n_steps]
    pred = rollout(model, x0, n_steps, mean, std, device)
    print(f"Rolled out {n_steps} steps on trajectory {args.traj} (start={args.start})")

    mse = per_step_mse(gt, pred)
    # persistence ("do nothing") baseline: output stays at the start frame. At Delta t this
    # close one-step MSE is near-trivially low, so persistence is the reference the model must beat.
    mse_persist = per_step_mse(gt, np.broadcast_to(x0[None], gt.shape))
    metrics = {
        "n_steps": int(n_steps),
        "mse_1step": float(mse[0]),
        "mse_full_rollout": float(mse.mean()),
        "mse_final_step": float(mse[-1]),
        "persistence_mse_1step": float(mse_persist[0]),
        "persistence_mse_full_rollout": float(mse_persist.mean()),
        "mse_per_step": mse.tolist(),
    }
    if args.n_traj != 1:
        ds = get_dataset(dataset, data_path, split=args.split)
        fulls = per_traj_rollout_mse(
            model, ds, args.start, args.n_steps, mean, std, device, max_traj=args.n_traj or None
        )
        n_agg = len(fulls)
        metrics["n_traj_aggregated"] = int(n_agg)
        metrics["mse_full_rollout_mean_over_traj"] = float(fulls.mean())
        metrics["mse_full_rollout_std_over_traj"] = float(fulls.std())
        print(f"  aggregated over {n_agg} traj: full-rollout MSE {fulls.mean():.5f} ± {fulls.std():.5f}")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"  1-step MSE: {metrics['mse_1step']:.5f} (persistence {metrics['persistence_mse_1step']:.5f})  "
        f"full-rollout MSE: {metrics['mse_full_rollout']:.5f} "
        f"(persistence {metrics['persistence_mse_full_rollout']:.5f})"
    )

    # per-step error curve, with the persistence baseline for context
    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, mse, color="tab:red", label=args.label)
    ax.plot(steps, mse_persist, color="gray", ls="--", label="persistence")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("MSE")
    ax.set_title("Rollout error growth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "error_growth.png", dpi=150)
    plt.close(fig)

    plot_spectrum_timeavg(gt, pred, args.output_dir / "spectrum_timeavg.png", args.label)
    plot_spectrum_leadtime(gt, pred, args.output_dir / "spectrum_leadtime.png")

    if not args.no_gif:
        use_vorticity = dataset == DatasetType.KOLMOGOROV
        save_rollout_gif(gt, pred, args.output_dir / "rollout.gif", use_vorticity)

    print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
