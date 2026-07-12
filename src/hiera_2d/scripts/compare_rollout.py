"""Compare multiple AR models on rollout: the thesis money figure + headline table.

Everything — the results TABLE (1-step / full-rollout / final-step MSE) AND the overlaid
spectrum / error-growth PLOTS — is aggregated over ALL validation trajectories, because a
single trajectory is not representative: trajectory 0 in particular was found to flip the
scratch-vs-pretrained ordering (both in MSE and in the spectrum) relative to the population
(see docs §6c). For bootstrapped confidence intervals on the paired finetune-vs-scratch
difference, use the `scaling-curve` analysis.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hiera_2d.analysis.spectra import power_db, radial_energy_spectrum
from hiera_2d.experiments.data import DatasetType, Split, get_dataset
from hiera_2d.scripts.eval_rollout import (
    load_ar_model,
    load_checkpoint_provenance,
    per_step_mse,
    resolve_dataset_identity,
    rollout,
)

MODEL_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple"]


def check_checkpoints_share_training_data(checkpoints: Sequence[Path]) -> None:
    """Verify every checkpoint was trained on the same data path.

    Guards the core invalidity of a comparison figure: overlaying models trained on
    different data. Bypassed by the caller when an explicit --data-path override is
    forcing a deliberate common dataset.

    Raises:
        ValueError: a checkpoint predates provenance embedding, or the checkpoints
            disagree on their recorded training data path.
    """
    recorded = []
    for ckpt in checkpoints:
        path, _ = load_checkpoint_provenance(ckpt)

        if path is None:
            msg = (
                f"Checkpoint {ckpt} predates dataset-provenance embedding "
                "(no data_path/dataset recorded). Pass --data-path (and --dataset) explicitly."
            )
            raise ValueError(msg)

        recorded.append(path)

    if len(set(recorded)) > 1:
        msg = (
            f"Checkpoints record different training data paths: {sorted(set(recorded))}. "
            "Comparing models trained on different data is invalid; pass --data-path to override deliberately."
        )
        raise ValueError(msg)


def parse_args():
    p = argparse.ArgumentParser(description="Compare AR models on a shared rollout")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--labels", type=str, nargs="+", required=True)
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the dataset path recorded in the checkpoints (default: use the checkpoints' provenance)",
    )
    p.add_argument(
        "--dataset",
        type=DatasetType,
        choices=list(DatasetType),
        default=None,
        help="Override the dataset type recorded in the checkpoints (default: use the checkpoints' provenance)",
    )
    p.add_argument("--split", type=Split, choices=list(Split), default=Split.VAL)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--n-traj", type=int, default=0, help="trajectories aggregated for table+plots (0 = all val)")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.checkpoints) != len(args.labels):
        raise ValueError("--checkpoints and --labels must have equal length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset identity comes from the first checkpoint's provenance; all checkpoints must
    # agree on the training data (comparing models trained on different data is invalid),
    # unless an explicit --data-path override is deliberately forcing a common dataset.
    identity = resolve_dataset_identity(args.checkpoints[0], args.data_path, args.dataset)
    data_path, dataset = identity.data_path, identity.dataset

    if args.data_path is not None:
        print(
            "NOTE: --data-path overrides every checkpoint's recorded data; skipping cross-checkpoint consistency check"
        )
    else:
        check_checkpoints_share_training_data(args.checkpoints)

    ds = get_dataset(dataset, data_path, split=args.split)
    mean, std = ds.norm_stats["mean"], ds.norm_stats["std"]
    n_traj = min(args.n_traj, len(ds.sims)) if args.n_traj else len(ds.sims)

    # uniform rollout length across trajectories (all sims share T)
    n_steps = min(args.n_steps, ds.sims[0].shape[0] - args.start - 1)

    def rollout_curves(model):
        """Trajectory-averaged spectrum + per-step MSE, plus per-trajectory scalar stats.

        model=None gives the persistence baseline. Every curve is averaged over the
        n_traj trajectories, so the figures match the aggregated table (no single-
        trajectory artifacts).
        """
        one, full, final = [], [], []
        step_sum = np.zeros(n_steps)
        spec_sum = None
        k = None
        for t in range(n_traj):
            traj = np.asarray(ds.sims[t], dtype=np.float32)
            x0 = traj[args.start]
            gt = traj[args.start + 1 : args.start + 1 + n_steps]
            if model is None:
                pred = np.broadcast_to(x0[None], gt.shape)
            else:
                pred = rollout(model, x0, n_steps, mean, std, device)
            mse = per_step_mse(gt, pred)
            one.append(mse[0])
            full.append(mse.mean())
            final.append(mse[-1])
            step_sum += mse
            k, e_pred = radial_energy_spectrum(pred)
            spec_sum = e_pred.mean(axis=0) if spec_sum is None else spec_sum + e_pred.mean(axis=0)
        return {
            "k": k,
            "one": np.array(one),
            "full": np.array(full),
            "final": np.array(final),
            "step_mean": step_sum / n_traj,
            "spec_mean": spec_sum / n_traj,
        }

    # trajectory-averaged ground-truth spectrum for the reference line
    k = None
    gt_spec = None
    for t in range(n_traj):
        traj = np.asarray(ds.sims[t], dtype=np.float32)
        gt = traj[args.start + 1 : args.start + 1 + n_steps]
        k, e = radial_energy_spectrum(gt)
        gt_spec = e.mean(axis=0) if gt_spec is None else gt_spec + e.mean(axis=0)
    gt_spec /= n_traj

    fig_s, ax_s = plt.subplots(figsize=(6.5, 5))
    fig_e, ax_e = plt.subplots(figsize=(6.5, 4.5))
    ax_s.plot(k[1:], power_db(gt_spec)[1:], color="black", lw=2.6, label="ground truth", zorder=5)

    steps = np.arange(1, n_steps + 1)
    persist = rollout_curves(None)
    ax_e.plot(steps, persist["step_mean"], color="gray", ls="--", lw=1.5, label="persistence")

    def row(label, c):
        return {
            "label": label,
            "mse_1step": float(np.mean(c["one"])),
            "mse_full_rollout": float(np.mean(c["full"])),
            "mse_full_rollout_std": float(np.std(c["full"])),
            "mse_final_step": float(np.mean(c["final"])),
        }

    table = []
    for i, (ckpt, label) in enumerate(zip(args.checkpoints, args.labels, strict=True)):
        c = rollout_curves(load_ar_model(ckpt, device))
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax_s.plot(c["k"][1:], power_db(c["spec_mean"])[1:], color=color, lw=1.8, label=label)
        ax_e.plot(steps, c["step_mean"], color=color, label=label)
        table.append(row(label, c))
        print(f"{label}: full-rollout MSE {np.mean(c['full']):.5f} ± {np.std(c['full']):.5f} (over {n_traj} traj)")

    table.append(row("persistence", persist))

    ax_s.set_xscale("log", base=2)
    ax_s.set_xlabel("wavenumber $k$")
    ax_s.set_ylabel("power (dB)")
    ax_s.set_title(f"Rollout power spectrum, averaged over {n_traj} trajectories")
    ax_s.legend()
    ax_s.grid(True, which="both", alpha=0.3)
    fig_s.tight_layout()
    fig_s.savefig(args.output_dir / "spectrum_compare.png", dpi=150)

    ax_e.set_xlabel("rollout step")
    ax_e.set_ylabel("MSE")
    ax_e.set_title(f"Rollout error growth, averaged over {n_traj} trajectories")
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)
    fig_e.tight_layout()
    fig_e.savefig(args.output_dir / "error_growth_compare.png", dpi=150)

    (args.output_dir / "results.json").write_text(
        json.dumps({"n_steps": int(n_steps), "n_traj": int(n_traj), "models": table}, indent=2)
    )

    lines = [
        f"Aggregated over {n_traj} {args.split} trajectories (mean; ± = std across trajectories). "
        f"For bootstrapped CIs on the paired difference, use scaling-curve.",
        "",
        "| model | 1-step MSE | full-rollout MSE | final-step MSE |",
        "|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['label']} | {r['mse_1step']:.5f} | "
            f"{r['mse_full_rollout']:.5f} ± {r['mse_full_rollout_std']:.5f} | {r['mse_final_step']:.5f} |"
        )
    (args.output_dir / "results_table.md").write_text("\n".join(lines) + "\n")
    print(f"Saved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
