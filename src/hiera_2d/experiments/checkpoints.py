from pathlib import Path

import torch

from hiera_2d.experiments.ar.model import HieraAR
from hiera_2d.hiera.mae import HieraMAE


def delete_old_checkpoints(run_path: Path):
    keep_last = 5

    checkpoints = sorted(
        (run_path / "checkpoints").glob("epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for old_ckpt in checkpoints[keep_last:]:
        old_ckpt.unlink()


def save_mae_training_checkpoint(
    *,
    run_path: Path,
    epoch: int,
    n_epochs: int,
    model: HieraMAE,
    train_loss: float,
    val_loss: float,
    train_config: dict,
    data_path: str,
    dataset: str,
    is_best: bool,
):
    """Write the rolling per-epoch checkpoint (last 5 kept) and, when `is_best`,
    also overwrite best_model.pt. The caller decides `is_best`.

    `n_epochs` is the planned epoch budget for the run, embedded so run-scaling can
    verify (not assume) that an existing checkpoint was trained under the budget the
    current config asks for before it skips. `data_path` (resolved dataset path) and
    `dataset` (DatasetType value) are the dataset provenance, embedded so downstream
    eval reads identity from the checkpoint instead of a hand-passed flag."""
    checkpoint = {
        "epoch": epoch,
        "n_epochs": n_epochs,
        "model_type": "hiera_mae",
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_config": train_config,
        "data_path": data_path,
        "dataset": dataset,
    }

    ckpt_dir = run_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, ckpt_dir / f"epoch_{epoch}.pt")
    delete_old_checkpoints(run_path)

    if is_best:
        torch.save(checkpoint, ckpt_dir / "best_model.pt")


def save_ar_training_checkpoint(
    *,
    run_path: Path,
    epoch: int,
    n_epochs: int,
    model: HieraAR,
    val_loss: float,
    config: dict,
    data_path: str,
    dataset: str,
):
    """Overwrite best_model.pt for an AR run (called only when the epoch is best).

    `n_epochs` is the planned epoch budget for the run, embedded so run-scaling can
    verify (not assume) that an existing checkpoint was trained under the budget the
    current config asks for before it skips. `data_path` (resolved dataset path) and
    `dataset` (DatasetType value) are the dataset provenance, embedded so downstream
    rollout eval reads identity from the checkpoint instead of a hand-passed flag."""
    checkpoint = {
        "epoch": epoch,
        "n_epochs": n_epochs,
        "model_type": "hiera_ar",
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "config": config,
        "data_path": data_path,
        "dataset": dataset,
    }

    ckpt_dir = run_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, ckpt_dir / "best_model.pt")
