"""Shared training helpers used by both MAE pretraining and AR finetuning."""

import random
from collections.abc import Callable
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def seed_everything(seed: int):
    """Seed Python, NumPy and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    n_warmup_epochs: int,
    n_epochs: int,
    min_lr: float,
):
    """Linear warmup followed by cosine decay.

    Warmup stabilizes early optimization; cosine provides smooth annealing.
    """
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=n_warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=n_epochs - n_warmup_epochs, eta_min=min_lr),
        ],
        milestones=[n_warmup_epochs],
    )


def run_training_loop(
    *,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    writer: SummaryWriter,
    train_epoch: Callable[[int], float],
    validate_epoch: Callable[[int], float],
    on_epoch_end: Callable[[int, float, float, bool], None],
):
    """Drive the epoch loop shared by MAE and AR training.

    Owns per-epoch timing, the tensorboard scalar keys (`lr`, `loss/*`,
    `time/*`), progress output, best-val tracking and scheduler stepping.
    Callers supply the model-specific hooks: `train_epoch(epoch) -> loss`,
    `validate_epoch(epoch) -> loss`, and `on_epoch_end(epoch, train_loss,
    val_loss, is_best)` to visualize and checkpoint (the caller writes the best
    checkpoint iff `is_best`). Returns the best validation loss seen.
    """
    best_val_loss = float("inf")
    t_training_start = datetime.now()
    print(f"Training started at {t_training_start:%Y-%m-%d %H:%M:%S}")

    with tqdm(range(n_epochs), desc="Training", unit="epoch") as progress:
        for epoch in progress:
            t_epoch_start = datetime.now()
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            train_loss = train_epoch(epoch)
            val_loss = validate_epoch(epoch)

            epoch_seconds = (datetime.now() - t_epoch_start).total_seconds()
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)
            writer.add_scalar("time/elapsed_minutes", (datetime.now() - t_training_start).total_seconds() / 60, epoch)

            progress.write(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f} ({epoch_seconds:.1f}s)")
            progress.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            on_epoch_end(epoch, train_loss, val_loss, is_best)

            if is_best:
                progress.write(f"Epoch {epoch}: saved new best model")

            scheduler.step()

    total_minutes = (datetime.now() - t_training_start).total_seconds() / 60
    print(f"Training complete in {total_minutes:.1f}min. Best validation loss: {best_val_loss:.6f}")

    return best_val_loss
