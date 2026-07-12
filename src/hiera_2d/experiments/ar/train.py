import json
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hiera_2d.experiments.ar.config import (
    EncoderSource,
    PretrainedEncoderSource,
    ScratchEncoderSource,
    _parse_args,
    build_ar_train_args,
)
from hiera_2d.experiments.ar.data import ARDataset, to_ar_dataset
from hiera_2d.experiments.ar.model import HieraAR, ar_loss
from hiera_2d.experiments.checkpoints import save_ar_training_checkpoint
from hiera_2d.experiments.data import DatasetType, Split, get_dataset
from hiera_2d.experiments.mae.visualization import normalize_for_vis
from hiera_2d.experiments.scaling.config import ExperimentConfig, RunIdentity, load_experiment_config
from hiera_2d.experiments.training_utils import build_warmup_cosine_scheduler, run_training_loop, seed_everything
from hiera_2d.hiera.model import Hiera, HieraConfig


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


def build_encoder(source: EncoderSource, device: torch.device) -> tuple[Hiera, HieraConfig, bool]:
    """Build the encoder from an MAE checkpoint (pretrained) or random init (scratch).

    The from-scratch path is the baseline for the pretrained-vs-random comparison:
    identical architecture, trained on next-frame prediction only. Returns
    (encoder, hiera_config, is_pretrained).
    """
    match source:
        case PretrainedEncoderSource():
            encoder = load_encoder_from_mae_checkpoint(source.mae_checkpoint, device)
            print(f"Loaded pretrained encoder from {source.mae_checkpoint}")
            return encoder, encoder.config, True

        case ScratchEncoderSource():
            encoder = Hiera(config=source.hiera).to(device)
            print("Initialized encoder from scratch")
            return encoder, source.hiera, False

        case _:
            msg = f"unknown encoder source: {source!r}"
            raise ValueError(msg)


def run_epoch(
    model: HieraAR,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    *,
    training: bool,
    unroll_steps: int,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
):
    """One epoch of pushforward rollout training.

    Unrolls the model `unroll_steps` steps from the first frame, feeding its own
    predictions back, and supervises every step against ground truth. During
    training the fed-back prediction is detached, so gradients stay one-step-local
    (the pushforward trick): the model learns to correct its own drift without
    backprop-through-time, and VRAM stays at a single step. unroll_steps=1
    recovers plain teacher-forced one-step training.
    """
    model.train(mode=training)

    total_loss = 0.0
    n_batches = 0

    label = "Training" if training else "Validating"
    with (
        torch.set_grad_enabled(training),
        tqdm(dataloader, desc=f"-> {label} epoch {epoch}", leave=False, unit="batch") as progress,
    ):
        for batch in progress:
            frames = batch["frames"].to(device)  # (B, unroll_steps + 1, C, H, W)

            if training:
                optimizer.zero_grad(set_to_none=True)

            x = frames[:, 0]
            rollout_loss = 0.0
            for t in range(unroll_steps):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(x)
                    loss = ar_loss(pred, frames[:, t + 1])
                if training:
                    scaler.scale(loss / unroll_steps).backward()
                    x = pred.detach().float()
                else:
                    x = pred.float()
                rollout_loss += loss.item()

            if training:
                scaler.step(optimizer)
                scaler.update()

            total_loss += rollout_loss / unroll_steps
            n_batches += 1
            progress.set_postfix(loss=f"{rollout_loss / unroll_steps:.6f}")

    return total_loss / n_batches


@torch.no_grad()
def visualize_ar_predictions(
    model: HieraAR,
    dataset: ARDataset,
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
    pred_vis, target_vis = normalize_for_vis(target_vis, pred_vis, target_vis)

    # grid: row 1 = target, row 2 = prediction
    tiles = [target_vis[i : i + 1] for i in range(n_steps)]
    tiles += [pred_vis[i : i + 1] for i in range(n_steps)]
    grid = vutils.make_grid(torch.cat(tiles, dim=0), nrow=n_steps, normalize=False)
    writer.add_image("samples/target_vs_pred", grid, epoch)


def train_ar(cfg: ExperimentConfig, run: RunIdentity) -> None:
    """Run one AR training to completion: resolve the typed setup from `cfg` + the
    per-run `identity`, build the encoder (pretrained from `run.mae_checkpoint`, or
    from-scratch when it is `None`) and AR head, train, and write the provenance
    dump and best checkpoint under `out_dir / name`.

    The imperative core of the AR trainer — no argparse, no TOML loading. Called
    directly (in a fresh process) by the scaling orchestrator and by the thin
    `main` shell that backs the `train-ar` console script.
    """
    args = build_ar_train_args(cfg, run)
    ar_run = args.run
    seed_everything(ar_run.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.path.exists():
        msg = f"Output directory already exists: {args.path}"
        raise FileExistsError(msg)

    args.path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.path)

    # pretrained (from MAE) or from-scratch (random) encoder
    encoder, hiera_config, is_pretrained = build_encoder(args.encoder, device)
    model = HieraAR(encoder=encoder, config=args.ar_head).to(device)

    # Frozen-encoder probe: train only the AR head on top of fixed (pretrained)
    # features. Set the encoder to eval() so any encoder-internal stochasticity
    # stays off, and exclude its params from the optimizer.
    if ar_run.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

        model.encoder.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total params: {n_params:,} ({n_trainable:,} trainable) | "
        f"freeze_encoder={ar_run.freeze_encoder} "
        f"unroll_steps={ar_run.unroll_steps} predict_residual={args.ar_head.predict_residual}"
    )

    ds_train = get_dataset(args.dataset, args.data_path, split=Split.TRAIN, n_trajectories=args.n_trajectories)
    # Read val lazily (Kolmogorov only): at the largest N its eager copy would sit on
    # top of the full train subset and exhaust host RAM. Val is scanned sequentially,
    # so on-demand reads cost nothing extra.
    ds_val = get_dataset(args.dataset, args.data_path, split=Split.VAL, lazy=args.dataset == DatasetType.KOLMOGOROV)
    seq_len = ar_run.unroll_steps + 1  # each window: 1 input frame + unroll_steps targets
    ar_train = to_ar_dataset(ds_train, seq_len=seq_len)
    ar_val = to_ar_dataset(ds_val, seq_len=seq_len)

    ld_train = DataLoader(
        ar_train,
        batch_size=ar_run.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    ld_val = DataLoader(
        ar_val,
        batch_size=ar_run.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=ar_run.lr,
        betas=(0.9, 0.95),
        weight_decay=ar_run.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        n_warmup_epochs=ar_run.n_warmup_epochs,
        n_epochs=ar_run.n_epochs,
        min_lr=ar_run.min_lr,
    )

    use_amp = ar_run.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    mae_checkpoint = args.encoder.mae_checkpoint if isinstance(args.encoder, PretrainedEncoderSource) else None
    config_dump = {
        "ar_config": args.ar_head.model_dump(mode="json"),
        "hiera": hiera_config.model_dump(mode="json"),
        "is_pretrained": is_pretrained,
        "mae_checkpoint": str(mae_checkpoint) if mae_checkpoint else None,
        "freeze_encoder": ar_run.freeze_encoder,
        "unroll_steps": ar_run.unroll_steps,
        "amp": use_amp,
        "lr": ar_run.lr,
        "n_epochs": ar_run.n_epochs,
        "batch_size": ar_run.batch_size,
        "n_trajectories": args.n_trajectories,
    }
    (args.path / "args.json").write_text(json.dumps(config_dump, indent=4))
    (args.path / "checkpoints").mkdir(exist_ok=True)

    def train_epoch(epoch: int):
        return run_epoch(
            model,
            ld_train,
            optimizer,
            device,
            epoch,
            training=True,
            unroll_steps=ar_run.unroll_steps,
            scaler=scaler,
            use_amp=use_amp,
        )

    def validate_epoch(epoch: int):
        return run_epoch(
            model,
            ld_val,
            None,
            device,
            epoch,
            training=False,
            unroll_steps=ar_run.unroll_steps,
            scaler=scaler,
            use_amp=use_amp,
        )

    def on_epoch_end(epoch: int, _train_loss: float, val_loss: float, is_best: bool):
        visualize_ar_predictions(model, ar_val, device, epoch, writer)
        if is_best:
            save_ar_training_checkpoint(
                run_path=args.path,
                epoch=epoch,
                n_epochs=ar_run.n_epochs,
                model=model,
                val_loss=val_loss,
                config=config_dump,
                data_path=str(args.data_path),
                dataset=args.dataset.value,
            )

    try:
        run_training_loop(
            n_epochs=ar_run.n_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=writer,
            train_epoch=train_epoch,
            validate_epoch=validate_epoch,
            on_epoch_end=on_epoch_end,
        )
    finally:
        writer.close()


def main(argv: list[str] | None = None):
    """Thin shell for the `train-ar` console script: parse the CLI, load the
    experiment config, build the per-run identity (encoder source + epoch
    override travel on it), and hand off to `train_ar`."""
    parsed = _parse_args(argv)
    cfg = load_experiment_config(parsed.config)
    run = RunIdentity(
        n_trajectories=parsed.n_trajectories,
        out_dir=parsed.output_dir,
        name=parsed.name,
        mae_checkpoint=parsed.mae_checkpoint,
        n_epochs=parsed.n_epochs,
    )
    train_ar(cfg, run)


if __name__ == "__main__":
    main()
