import argparse
import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch

from hiera_2d.experiments.ar.train import train_ar
from hiera_2d.experiments.mae.train import train_mae
from hiera_2d.experiments.scaling.config import (
    ExperimentConfig,
    RunIdentity,
    finetune_run_name,
    load_experiment_config,
    mae_run_name,
    scratch_run_name,
)

_CKPT_RELPATH = ("checkpoints", "best_model.pt")


class RunKind(StrEnum):
    """Which trainer a planned run dispatches to: an MAE pretrain or an AR head."""

    MAE = "mae"
    AR = "ar"


@dataclass(frozen=True, slots=True)
class RunSpec:
    """A single planned training invocation: which trainer it dispatches to
    (`kind`), the typed per-run identity handed to that trainer (`identity`), and
    any upstream checkpoint it depends on (`requires_checkpoint`, set on the
    finetune arm to the MAE best checkpoint it consumes)."""

    kind: RunKind
    identity: RunIdentity
    requires_checkpoint: Path | None = None


def _best_ckpt(run_dir: Path) -> Path:
    return run_dir.joinpath(*_CKPT_RELPATH)


def _run_dir(identity: RunIdentity) -> Path:
    return identity.out_dir / identity.name


def plan_runs(cfg: ExperimentConfig) -> tuple[RunSpec, ...]:
    """Pure: expand the config into the flat, ordered list of runs. Per N the
    order is mae -> finetune -> scratch (finetune consumes the MAE checkpoint).

    Only the per-run identity (n_trajectories, output dir, name, encoder source)
    distinguishes the runs; every run reads the same typed `cfg` for its
    hyperparameters, so the finetune and scratch arms share one epoch budget.
    """
    specs: list[RunSpec] = []

    mae_name = mae_run_name(cfg)
    finetune_name = finetune_run_name(cfg)

    for n in cfg.n_trajectories:
        n_dir = cfg.output_root / f"N{n}"

        mae = RunSpec(
            kind=RunKind.MAE,
            identity=RunIdentity(n_trajectories=n, out_dir=n_dir, name=mae_name),
        )

        mae_ckpt = _best_ckpt(n_dir / mae_name)
        finetune = RunSpec(
            kind=RunKind.AR,
            identity=RunIdentity(n_trajectories=n, out_dir=n_dir, name=finetune_name, mae_checkpoint=mae_ckpt),
            requires_checkpoint=mae_ckpt,
        )

        scratch = RunSpec(
            kind=RunKind.AR,
            identity=RunIdentity(n_trajectories=n, out_dir=n_dir, name=scratch_run_name(cfg)),
        )

        specs += [mae, finetune, scratch]

    return tuple(specs)


def _run_in_process(kind: RunKind, cfg: ExperimentConfig, identity: RunIdentity) -> None:
    """Route one planned run to its trainer inside a freshly spawned process.

    Module-level and picklable-argument so it is a valid `spawn` target: the child
    re-imports this module and calls the trainer with the typed config + identity.
    """
    match kind:
        case RunKind.MAE:
            train_mae(cfg, identity)

        case RunKind.AR:
            train_ar(cfg, identity)

        case _:
            msg = f"unknown run kind: {kind!r}"
            raise ValueError(msg)


def _planned_epochs(cfg: ExperimentConfig, spec: RunSpec) -> int:
    """The epoch budget the plan asks of one run: MAE -> `cfg.mae.n_epochs`;
    AR -> the identity's `n_epochs` override if set, else `cfg.ar.n_epochs`. The
    value recorded into the checkpoint, so the skip check can compare it against
    what an existing checkpoint was trained under."""
    if spec.kind is RunKind.MAE:
        return cfg.mae.n_epochs

    run = spec.identity
    return run.n_epochs if run.n_epochs is not None else cfg.ar.n_epochs


def _recorded_epochs(ckpt_path: Path) -> int | None:
    """The `n_epochs` an existing checkpoint was trained under, or `None` when the
    checkpoint predates budget recording (several such checkpoints exist on disk)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    return ckpt.get("n_epochs")


def _describe(cfg: ExperimentConfig, spec: RunSpec) -> str:
    run = spec.identity
    run_dir = _run_dir(run)

    if spec.kind is RunKind.MAE:
        encoder = "mae-pretrain"
    elif run.mae_checkpoint is not None:
        encoder = f"finetune <- {run.mae_checkpoint}"
    else:
        encoder = "scratch (random init)"

    return f"[{spec.kind.value}] N={run.n_trajectories} -> {run_dir} | {encoder} | {_planned_epochs(cfg, spec)} epochs"


def _parse_args(argv: Sequence[str] | None):
    parser = argparse.ArgumentParser(description="Config-driven Kolmogorov data-scaling sweep driver")
    parser.add_argument("config", type=Path, help="Path to the scaling-experiment TOML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs and exit without training anything",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)
    cfg = load_experiment_config(args.config)
    specs = plan_runs(cfg)

    if args.dry_run:
        for spec in specs:
            print(_describe(cfg, spec))

        return

    # spawn (not fork): forking a CUDA-initialized parent is unsafe. Each run gets
    # a pristine process so GPU memory is fully released between runs. Sequential:
    # join before starting the next — the runs share one GPU.
    ctx = multiprocessing.get_context("spawn")

    for spec in specs:
        run_dir = _run_dir(spec.identity)
        best_ckpt = _best_ckpt(run_dir)

        if best_ckpt.exists():
            planned = _planned_epochs(cfg, spec)
            recorded = _recorded_epochs(best_ckpt)

            if recorded is None:
                print(
                    f"[warn] {run_dir} has {_CKPT_RELPATH[-1]} but no recorded epoch budget "
                    f"(predates provenance); cannot verify it was trained for the planned "
                    f"{planned} epochs — skipping and reusing it"
                )
                continue

            if recorded != planned:
                msg = (
                    f"{run_dir} has a checkpoint trained for {recorded} epochs but the config "
                    f"asks for {planned}; delete the run dir or change the config"
                )
                raise ValueError(msg)

            print(f"[skip] {run_dir} already has {_CKPT_RELPATH[-1]} ({recorded} epochs)")
            continue

        if spec.requires_checkpoint is not None and not spec.requires_checkpoint.exists():
            msg = f"finetune run {run_dir} requires MAE checkpoint {spec.requires_checkpoint}, which is missing"
            raise ValueError(msg)

        print(f"[run] {_describe(cfg, spec)}")
        p = ctx.Process(target=_run_in_process, args=(spec.kind, cfg, spec.identity))
        p.start()
        p.join()

        if p.exitcode != 0:
            msg = f"run {run_dir} ({spec.kind.value}) failed: worker process exited with code {p.exitcode}"
            raise RuntimeError(msg)
