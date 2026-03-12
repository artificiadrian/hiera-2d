from pathlib import Path

import torch

from hiera_2d.hiera.mae import HieraMAE


def delete_old_checkpoints(run_path: Path, *, keep_last: int = 5):

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
    model: HieraMAE,
    train_loss: float,
    val_loss: float,
    train_config: dict,
    best_val_loss: float,
):
    checkpoint = {
        "epoch": epoch,
        "model_type": "hiera_mae",
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_config": train_config,
    }

    dir = run_path / "checkpoints"
    dir.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, dir / f"epoch_{epoch}.pt")
    delete_old_checkpoints(run_path)

    is_best = val_loss < best_val_loss

    if is_best:
        torch.save(checkpoint, dir / "best_model.pt")
        best_val_loss = val_loss

    return best_val_loss, is_best
