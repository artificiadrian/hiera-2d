import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hiera_2d.experiments.data import DatasetType, get_dataset
from hiera_2d.experiments.mae.visualization import _normalize_for_vis
from hiera_2d.hiera.model import Hiera, HieraConfig

from .data import to_ar_dataset
from .model import ARHeadConfig, HieraAR, ar_loss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_encoder_from_mae_checkpoint(checkpoint_path: Path, device: torch.device) -> Hiera:
    """Load a pretrained Hiera encoder from an MAE checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hiera_config = HieraConfig.model_validate(ckpt["train_config"]["hiera"])
    encoder = Hiera(config=hiera_config)

    encoder_prefix = "encoder."
    encoder_state = {
        k.removeprefix(encoder_prefix): v for k, v in ckpt["model_state_dict"].items() if k.startswith(encoder_prefix)
    }
    encoder.load_state_dict(encoder_state)
    return encoder.to(device)


def run_epoch(
    model: HieraAR,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    *,
    training: bool,
):
    model.train(mode=training)

    total_loss = 0.0
    n_batches = 0

    label = "Training" if training else "Validating"
    with (
        torch.set_grad_enabled(training),
        tqdm(dataloader, desc=f"-> {label} epoch {epoch}", leave=False, unit="batch") as progress,
    ):
        for batch in progress:
            frames = batch["frames"].to(device)  # (B, seq_len, C, H, W)
            B, S = frames.shape[:2]

            inputs = frames[:, :-1].reshape(B * (S - 1), *frames.shape[2:])
            targets = frames[:, 1:].reshape(B * (S - 1), *frames.shape[2:])

            preds = model(inputs)
            loss = ar_loss(preds, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def visualize_ar_predictions(
    model: HieraAR,
    dataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    n_steps: int = 5,
):
    model.eval()

    sample = dataset[0]
    frames = sample["frames"].to(device)  # (seq_len, C, H, W)

    n_steps = min(n_steps, frames.shape[0] - 1)
    inputs = frames[:n_steps]
    targets = frames[1 : n_steps + 1]
    preds = model(inputs)

    # average channels for grayscale
    target_vis = targets.mean(dim=1, keepdim=True)
    pred_vis = preds.mean(dim=1, keepdim=True)

    # normalize using target range
    pred_vis, target_vis = _normalize_for_vis(target_vis, pred_vis, target_vis)

    # grid: row 1 = target, row 2 = prediction
    tiles = [target_vis[i : i + 1] for i in range(n_steps)]
    tiles += [pred_vis[i : i + 1] for i in range(n_steps)]
    grid = vutils.make_grid(torch.cat(tiles, dim=0), nrow=n_steps, normalize=False)
    writer.add_image("samples/target_vs_pred", grid, epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="AR head training for Hiera")
    parser.add_argument("--mae-checkpoint", type=Path, required=True, help="Path to pretrained MAE checkpoint")
    parser.add_argument("--ar-config", type=Path, help="Path to AR head JSON config")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=10, help="Number of frames per AR sequence")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.GRAY_SCOTT)
    parser.add_argument("--name", type=str, default=f"ar_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_path = Path(args.output_dir / args.name).resolve()
    if run_path.exists():
        raise FileExistsError(f"Output directory already exists: {run_path}")
    run_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=run_path)

    # load pretrained encoder from MAE checkpoint
    encoder = load_encoder_from_mae_checkpoint(args.mae_checkpoint, device)
    print(f"Loaded encoder from {args.mae_checkpoint}")

    # AR head config
    if args.ar_config:
        ar_config = ARHeadConfig.model_validate_json(args.ar_config.read_text())
    else:
        ar_config = ARHeadConfig()

    model = HieraAR(encoder=encoder, config=ar_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,} (all trainable)")

    # data
    ds_train = get_dataset(args.dataset, args.data_path, split="train")
    ds_val = get_dataset(args.dataset, args.data_path, split="val")
    ar_train = to_ar_dataset(ds_train, seq_len=args.seq_len)
    ar_val = to_ar_dataset(ds_val, seq_len=args.seq_len)

    ld_train = DataLoader(
        ar_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    ld_val = DataLoader(
        ar_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=args.n_warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.n_epochs - args.n_warmup_epochs, eta_min=args.min_lr),
        ],
        milestones=[args.n_warmup_epochs],
    )

    config_dump = {
        "ar_config": ar_config.model_dump(mode="json"),
        "mae_checkpoint": str(args.mae_checkpoint),
        "seq_len": args.seq_len,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
    }
    (run_path / "args.json").write_text(json.dumps(config_dump, indent=4))

    best_val_loss = float("inf")
    (run_path / "checkpoints").mkdir(exist_ok=True)

    t_training_start = datetime.now()
    print(f"Training started at {t_training_start:%Y-%m-%d %H:%M:%S}")

    with tqdm(range(args.n_epochs), desc="Training", unit="epoch") as progress:
        for epoch in progress:
            t_epoch_start = datetime.now()
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            train_loss = run_epoch(model, ld_train, optimizer, device, epoch, training=True)
            val_loss = run_epoch(model, ld_val, None, device, epoch, training=False)

            epoch_duration = (datetime.now() - t_epoch_start).total_seconds()

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_duration, epoch)
            writer.add_scalar("time/elapsed_minutes", (datetime.now() - t_training_start).total_seconds() / 60, epoch)

            progress.write(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f} ({epoch_duration:.1f}s)")
            progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

            visualize_ar_predictions(model, ar_val, device, epoch, writer)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss,
                        "config": config_dump,
                    },
                    run_path / "checkpoints" / "best_model.pt",
                )
                progress.write(f"Epoch {epoch}: saved new best model")

            scheduler.step()

    total_duration = datetime.now() - t_training_start
    print(f"Training complete in {total_duration.total_seconds() / 60:.1f}min. Best validation loss: {best_val_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
