import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pw_hiera.experiments.checkpoints import save_mae_training_checkpoint
from pw_hiera.experiments.data import GrayScottDataset
from pw_hiera.hiera.mae import HieraMAE
from pw_hiera.hiera.model import Hiera

from .config import get_train_args
from .visualization import DEFAULT_FIXED_N_SAMPLES, render_reconstruction_grid


def seed_everything(seed: int):
    # set all seeds for determinism

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: HieraMAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    mask_ratio: float,
    *,
    training: bool,
):
    if training and optimizer is None:
        raise ValueError("optimizer is required when training=True")

    model.train(mode=training)
    total_loss = 0.0
    n_batches = 0

    mode_label = "Training" if training else "Validating"
    with (
        torch.set_grad_enabled(training),
        tqdm(dataloader, desc=f"-> {mode_label} epoch {epoch}", leave=False, unit="batch") as progress,
    ):
        for batch in progress:
            x = batch["initial"].to(device)
            loss, _, _, _ = model(x, mask_ratio=mask_ratio)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def visualize_reconstructions(
    model: HieraMAE,
    dataset: GrayScottDataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    mask_ratio: float,
    n_fixed_samples: int = DEFAULT_FIXED_N_SAMPLES,
):
    """Evaluate model on fixed set of samples and log to tensorboard"""

    model.eval()
    n_available = len(dataset)
    sample_indices = list(range(min(n_fixed_samples, n_available)))

    x_fixed = torch.stack([dataset[i]["initial"] for i in sample_indices]).to(device)
    grid = render_reconstruction_grid(model, x_fixed, mask_ratio)
    writer.add_image("samples/fixed_grid", grid, epoch)


def main():
    args = get_train_args()
    seed_everything(args.train_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # make sure output directory does not exist
    if args.path.exists():
        raise FileExistsError(f"Output directory already exists: {args.path}")
    args.path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.path)
    (args.path / "args.json").write_text(args.model_dump_json(indent=4))

    # load data
    ds_train = GrayScottDataset(args.data_path, split="train")
    ds_val = GrayScottDataset(args.data_path, split="val")

    ld_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    ld_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    try:
        sample = ds_train[0]["initial"]
        print(f"Input sample shape: ({sample.shape})")

        # setup model and optimizer
        model = HieraMAE(
            encoder=Hiera(config=args.train_config.hiera),
            config=args.train_config.mae,
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=args.train_config.lr,
            betas=(0.9, 0.95),
            weight_decay=args.train_config.weight_decay,
        )
        # Hiera MAE uses linear warmup followed by cosine decay:
        # warmup stabilizes early optimization, cosine provides smooth annealing.
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1e-2,
                    end_factor=1.0,
                    total_iters=args.train_config.n_warmup_epochs,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=args.train_config.n_epochs - args.train_config.n_warmup_epochs,
                    eta_min=args.train_config.min_lr,
                ),
            ],
            milestones=[args.train_config.n_warmup_epochs],
        )

        print(f"Encoder token grid={model.shapes.sz_tk_final}, stride_pred_px={model.stride_pred_px}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        best_val_loss = float("inf")
        (args.path / "checkpoints").mkdir(exist_ok=True)

        # train for n_epochs and validate after each one
        with tqdm(range(args.train_config.n_epochs), desc="Training", unit="epoch") as progress:
            for epoch in progress:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

                train_loss = run_epoch(
                    model,
                    ld_train,
                    optimizer,
                    device,
                    epoch,
                    args.train_config.mask_ratio,
                    training=True,
                )
                val_loss = run_epoch(
                    model,
                    ld_val,
                    None,
                    device,
                    epoch,
                    args.train_config.mask_ratio,
                    training=False,
                )
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)

                progress.write(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
                progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

                visualize_reconstructions(
                    model,
                    ds_val,
                    device,
                    epoch,
                    writer,
                    args.train_config.mask_ratio,
                )

                best_val_loss, is_best = save_mae_training_checkpoint(
                    run_path=args.path,
                    epoch=epoch,
                    model=model,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_config=args.train_config.model_dump(mode="json"),
                    best_val_loss=best_val_loss,
                )
                if is_best:
                    progress.write(f"Epoch {epoch}: saved new best model")

                scheduler.step()

        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    finally:
        writer.close()
        ds_train.close()
        ds_val.close()


if __name__ == "__main__":
    main()
