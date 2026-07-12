"""Log-log radially-averaged power-spectrum figure: E(k) vs k.

The thesis "frequency layout" plot in its final form. Ground truth (dark) is the
reference; the MAE-finetuned and from-scratch AR rollouts are overlaid. On
log-log axes the inertial range reads as a straight line, so its slope is a
geometric quantity you can point at in the methodology prose.

Linear energy on both axes (no decibels), three arms (no frozen arm): the pure
`loglog_spectrum_arrays` turns velocity fields into the averaged E(k) curves, and
`main` does the checkpoint loading / rollout / plotting.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hiera_2d.analysis.spectra import radial_energy_spectrum
from hiera_2d.experiments.data import DatasetType, KolmogorovDataset, Split, get_dataset
from hiera_2d.experiments.scaling.config import finetune_run_name, load_experiment_config, scratch_run_name
from hiera_2d.scripts.eval_rollout import load_ar_model, rollout


@dataclass(frozen=True, slots=True)
class LogLogSpectra:
    """Rollout-averaged E(k) curves for the three arms, k=0 dropped.

    `k` are the strictly-positive integer wavenumbers; `gt`, `finetune`, and
    `scratch` are the corresponding linear energies (NOT dB), each the same
    length as `k`.
    """

    k: np.ndarray
    gt: np.ndarray
    finetune: np.ndarray
    scratch: np.ndarray


def _averaged_spectrum(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`radial_energy_spectrum` averaged over ALL leading (non-spatial) axes -> (k, E(k))."""
    k, spectrum = radial_energy_spectrum(velocity)  # spectrum: (..., K)
    lead_axes = tuple(range(spectrum.ndim - 1))
    return k, spectrum.mean(axis=lead_axes)


def loglog_spectrum_arrays(gt: np.ndarray, pred_finetune: np.ndarray, pred_scratch: np.ndarray) -> LogLogSpectra:
    """Averaged log-log E(k) curves for the three arms, k=0 dropped.

    Each input is a velocity array `(..., 2, H, W)`; every leading (non-spatial)
    axis is averaged out to a single `(K,)` curve per arm, then the k=0 bin is
    dropped (it is the domain-mean energy and undefined on a log axis).
    """
    k, e_gt = _averaged_spectrum(gt)
    _, e_ft = _averaged_spectrum(pred_finetune)
    _, e_sc = _averaged_spectrum(pred_scratch)

    return LogLogSpectra(k=k[1:], gt=e_gt[1:], finetune=e_ft[1:], scratch=e_sc[1:])


def plot_loglog_spectrum(spectra: LogLogSpectra, out_path: Path):
    """Log-log E(k) figure: GT (dark) vs finetune vs scratch, saved to `out_path`."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for e, color, lw, label in (
        (spectra.gt, "black", 2.5, "ground truth"),
        (spectra.finetune, "tab:blue", 1.8, "finetune (MAE pretrained)"),
        (spectra.scratch, "tab:red", 1.8, "scratch"),
    ):
        # log axes choke on non-positive energy; keep only strictly-positive bins per arm
        positive = e > 0
        ax.plot(spectra.k[positive], e[positive], color=color, lw=lw, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("energy $E(k)$")
    ax.set_title("Rollout-averaged power spectrum")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _stream_loglog_spectra(
    ft_ckpt: Path,
    sc_ckpt: Path,
    dataset: KolmogorovDataset,
    mean: float,
    std: float,
    start: int,
    n_steps: int,
    device: torch.device,
) -> LogLogSpectra:
    """Averaged log-log E(k) curves for the three arms, streamed one trajectory at
    a time. Numerically equivalent to `loglog_spectrum_arrays` over the full val
    stack (every frame equally weighted), but accumulates each arm's summed radial
    spectrum instead of materializing three `(T_traj, steps, 2, H, W)` arrays --
    the stacked variant peaks at ~26 GB on the full val set and OOMs.
    """
    ft_model = load_ar_model(ft_ckpt, device)
    sc_model = load_ar_model(sc_ckpt, device)

    k = np.zeros(0)
    # size-1 zeros broadcast against the first (K,) spectrum, so the accumulators stay ndarrays.
    sum_gt = sum_ft = sum_sc = np.zeros(1)
    frames = 0

    for t in range(len(dataset.sims)):
        traj = np.asarray(dataset.sims[t], dtype=np.float32)
        steps = min(n_steps, traj.shape[0] - start - 1)

        gt = traj[start + 1 : start + 1 + steps]
        pred_ft = rollout(ft_model, traj[start], steps, mean, std, device)
        pred_sc = rollout(sc_model, traj[start], steps, mean, std, device)

        k, s_gt = radial_energy_spectrum(gt)
        _, s_ft = radial_energy_spectrum(pred_ft)
        _, s_sc = radial_energy_spectrum(pred_sc)

        sum_gt = sum_gt + s_gt.sum(axis=0)
        sum_ft = sum_ft + s_ft.sum(axis=0)
        sum_sc = sum_sc + s_sc.sum(axis=0)
        frames += steps

    if frames == 0:
        msg = "no validation frames to build a spectrum from"
        raise ValueError(msg)

    # k=0 is the domain-mean energy, undefined on a log axis (matches loglog_spectrum_arrays).
    return LogLogSpectra(
        k=k[1:], gt=(sum_gt / frames)[1:], finetune=(sum_ft / frames)[1:], scratch=(sum_sc / frames)[1:]
    )


def parse_args():
    p = argparse.ArgumentParser(description="Log-log rollout power spectrum (finetune vs scratch vs GT)")
    p.add_argument("config", type=Path, help="Path to the scaling-experiment TOML config (same as run-scaling)")
    p.add_argument(
        "--n", type=int, required=True, help="Training budget N to plot (selects N{n} under cfg.output_root)"
    )
    p.add_argument("--split", type=Split, choices=list(Split), default=Split.VAL)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=99)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <output_root>/N{n}/spectrum_N{n}.png)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)

    if args.n not in cfg.n_trajectories:
        available = ", ".join(str(x) for x in cfg.n_trajectories)
        msg = f"N={args.n} is not in the sweep; available N: {available}"
        raise ValueError(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_dir = cfg.output_root / f"N{args.n}"
    ft_ckpt = n_dir / finetune_run_name(cfg) / "checkpoints" / "best_model.pt"
    sc_ckpt = n_dir / scratch_run_name(cfg) / "checkpoints" / "best_model.pt"

    for ckpt in (ft_ckpt, sc_ckpt):
        if not ckpt.exists():
            msg = f"missing checkpoint: {ckpt}"
            raise FileNotFoundError(msg)

    # Lazy val (Kolmogorov only): the streaming spectrum reads one trajectory at a
    # time, so there is no reason to hold the whole split resident.
    dataset = get_dataset(cfg.dataset, cfg.data_path, split=args.split, lazy=cfg.dataset == DatasetType.KOLMOGOROV)
    mean, std = dataset.norm_stats["mean"], dataset.norm_stats["std"]
    print(f"Rolling out N{args.n} over {len(dataset.sims)} {args.split} trajectories, {args.n_steps} steps")

    spectra = _stream_loglog_spectra(ft_ckpt, sc_ckpt, dataset, mean, std, args.start, args.n_steps, device)

    out_path: Path = args.output or n_dir / f"spectrum_N{args.n}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_loglog_spectrum(spectra, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
