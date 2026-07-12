"""Data-scaling curve + improvement table for MAE-pretrained vs from-scratch AR.

For each training budget N in a scaling sweep we roll BOTH arms (finetune,
scratch) out over EVERY validation trajectory (one rollout per trajectory, same
start frame), giving a paired per-trajectory full-rollout MSE for each arm. The
paired design removes trajectory-difficulty variance from the comparison.

`assemble_scaling_report` (pure) bootstraps, per N, each arm's mean rollout MSE
and the paired difference (finetune - scratch); `diff.significant` is the
improvement table's verdict on whether the gap excludes 0. `main` does all the
I/O (checkpoint loading + rollout) then hands numpy arrays to the pure function
and writes `scaling_report.json` + `scaling_curve.png` — the headline
"pretraining helps more with less data" figure (two converging curves).
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hiera_2d.analysis.bootstrap import BootstrapCI, bootstrap_ci
from hiera_2d.experiments.data import KolmogorovDataset, Split, get_dataset
from hiera_2d.experiments.scaling.config import finetune_run_name, load_experiment_config, scratch_run_name
from hiera_2d.scripts.eval_rollout import load_ar_model, per_traj_rollout_mse


@dataclass(frozen=True, slots=True)
class ArmMses:
    """Paired per-val-trajectory full-rollout MSE for both arms at one N.

    `finetune` and `scratch` are same-length numpy arrays in the SAME trajectory
    order, so `finetune - scratch` is the paired per-trajectory improvement.
    """

    finetune: np.ndarray
    scratch: np.ndarray


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    """Bootstrapped rollout MSE for both arms and their paired gap at one N.

    `diff` is the paired `finetune - scratch` difference; `diff.significant`
    reports whether its 95% CI excludes 0 (the improvement-table verdict).
    """

    n: int
    finetune: BootstrapCI
    scratch: BootstrapCI
    diff: BootstrapCI


@dataclass(frozen=True, slots=True)
class ScalingReport:
    """The scaling curve: one `ScalingPoint` per N, ordered by ascending N."""

    points: tuple[ScalingPoint, ...]
    n_boot: int


def assemble_scaling_report(per_n: Mapping[int, ArmMses], n_boot: int, rng: np.random.Generator) -> ScalingReport:
    """Bootstrap, per N, each arm's mean rollout MSE and the paired gap.

    Pure: numpy arrays in, report out. No model loading or I/O. Points come out
    sorted by ascending N.
    """
    points: list[ScalingPoint] = []

    for n in sorted(per_n):
        arm = per_n[n]
        finetune = np.asarray(arm.finetune, dtype=np.float64)
        scratch = np.asarray(arm.scratch, dtype=np.float64)

        if finetune.shape != scratch.shape:
            msg = f"N{n}: finetune/scratch MSE arrays must be paired, got {finetune.shape} vs {scratch.shape}"
            raise ValueError(msg)

        points.append(
            ScalingPoint(
                n=n,
                finetune=bootstrap_ci(finetune, n_boot, rng),
                scratch=bootstrap_ci(scratch, n_boot, rng),
                diff=bootstrap_ci(finetune - scratch, n_boot, rng),
            )
        )

    return ScalingReport(points=tuple(points), n_boot=n_boot)


def _ci_to_dict(ci: BootstrapCI) -> dict[str, object]:
    return {"mean": ci.mean, "ci95": [ci.lo, ci.hi]}


def _ci_from_dict(data: Mapping[str, object]) -> BootstrapCI:
    lo, hi = data["ci95"]  # pyright: ignore[reportGeneralTypeIssues]
    return BootstrapCI(float(data["mean"]), float(lo), float(hi))  # pyright: ignore[reportArgumentType]


def report_to_dict(report: ScalingReport) -> dict[str, object]:
    """Serialize to nested dict: per N, finetune/scratch/diff each `{mean, ci95:[lo,hi]}` plus `significant`."""
    return {
        "n_boot": report.n_boot,
        "points": {
            str(p.n): {
                "finetune": _ci_to_dict(p.finetune),
                "scratch": _ci_to_dict(p.scratch),
                "diff": _ci_to_dict(p.diff),
                "significant": p.diff.significant,
            }
            for p in report.points
        },
    }


def report_from_dict(data: Mapping[str, object]) -> ScalingReport:
    """Inverse of `report_to_dict` (drops the derived `significant`, recomputed from `diff`)."""
    raw_points: Mapping[str, Mapping[str, object]] = data["points"]  # pyright: ignore[reportAssignmentType]
    points = tuple(
        ScalingPoint(
            n=int(n),
            finetune=_ci_from_dict(entry["finetune"]),  # pyright: ignore[reportArgumentType]
            scratch=_ci_from_dict(entry["scratch"]),  # pyright: ignore[reportArgumentType]
            diff=_ci_from_dict(entry["diff"]),  # pyright: ignore[reportArgumentType]
        )
        for n, entry in sorted(raw_points.items(), key=lambda kv: int(kv[0]))
    )
    return ScalingReport(points=points, n_boot=int(data["n_boot"]))  # pyright: ignore[reportArgumentType]


def plot_scaling_curve(report: ScalingReport, out_path: Path):
    """Two converging mean lines (finetune vs scratch) with 95% CI bands, x=N on a log scale."""
    ns = [p.n for p in report.points]

    fig, ax = plt.subplots(figsize=(7, 5))

    for get, color, label in (
        (lambda p: p.finetune, "tab:blue", "finetune (MAE pretrained)"),
        (lambda p: p.scratch, "tab:red", "scratch"),
    ):
        means = [get(p).mean for p in report.points]
        lo = [get(p).lo for p in report.points]
        hi = [get(p).hi for p in report.points]
        ax.plot(ns, means, marker="o", color=color, label=label)
        ax.fill_between(ns, lo, hi, color=color, alpha=0.2)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("training trajectories $N$")
    ax.set_ylabel("full-rollout MSE")
    ax.set_title("Data scaling: pretraining helps more with less data")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _rollout_mses(
    ckpt: Path, dataset: KolmogorovDataset, mean: float, std: float, start: int, n_steps: int, device: torch.device
) -> np.ndarray:
    """Per-val-trajectory full-rollout MSE for one checkpoint."""
    model = load_ar_model(ckpt, device)

    return per_traj_rollout_mse(model, dataset, start, n_steps, mean, std, device)


def parse_args():
    p = argparse.ArgumentParser(description="Data-scaling curve + improvement table (finetune vs scratch)")
    p.add_argument("config", type=Path, help="Path to the scaling-experiment TOML config (same as run-scaling)")
    p.add_argument("--split", type=Split, choices=list(Split), default=Split.VAL)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=99)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Where to write artifacts (default: cfg.output_root)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    dataset = get_dataset(cfg.dataset, cfg.data_path, split=args.split)
    mean, std = dataset.norm_stats["mean"], dataset.norm_stats["std"]
    n_traj = len(dataset.sims)
    print(f"Evaluating scaling sweep under {cfg.output_root} on {n_traj} {args.split} trajectories")

    per_n: dict[int, ArmMses] = {}

    for n in cfg.n_trajectories:
        n_dir = cfg.output_root / f"N{n}"
        ft_ckpt = n_dir / finetune_run_name(cfg) / "checkpoints" / "best_model.pt"
        sc_ckpt = n_dir / scratch_run_name(cfg) / "checkpoints" / "best_model.pt"

        if not ft_ckpt.exists() or not sc_ckpt.exists():
            print(f"[skip] N{n}: missing checkpoint(s) (finetune={ft_ckpt.exists()}, scratch={sc_ckpt.exists()})")
            continue

        finetune = _rollout_mses(ft_ckpt, dataset, mean, std, args.start, args.n_steps, device)
        scratch = _rollout_mses(sc_ckpt, dataset, mean, std, args.start, args.n_steps, device)
        per_n[n] = ArmMses(finetune=finetune, scratch=scratch)
        print(f"  N{n}: finetune {finetune.mean():.5f} | scratch {scratch.mean():.5f}")

    if not per_n:
        msg = f"no N* legs with both finetune and scratch checkpoints under {cfg.output_root}"
        raise ValueError(msg)

    report = assemble_scaling_report(per_n, args.n_boot, rng)

    out_dir = args.output_dir or cfg.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaling_report.json").write_text(json.dumps(report_to_dict(report), indent=2))
    plot_scaling_curve(report, out_dir / "scaling_curve.png")

    print("\n=== improvement table (paired finetune - scratch, 95% CI) ===")
    for p in report.points:
        sig = "significant" if p.diff.significant else "NOT significant (CI spans 0)"
        print(
            f"  N{p.n:<5d} finetune {p.finetune.mean:.4f} | scratch {p.scratch.mean:.4f} | "
            f"Δ {p.diff.mean:+.4f} [{p.diff.lo:+.4f}, {p.diff.hi:+.4f}] -> {sig}"
        )

    print(f"\nSaved to {out_dir / 'scaling_report.json'} and {out_dir / 'scaling_curve.png'}")


if __name__ == "__main__":
    main()
