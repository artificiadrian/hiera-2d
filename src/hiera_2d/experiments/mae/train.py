import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hiera_2d.experiments.checkpoints import save_mae_training_checkpoint
from hiera_2d.experiments.data import DatasetType, KolmogorovDataset, Split, get_dataset
from hiera_2d.experiments.mae.config import _parse_args, build_mae_train_args
from hiera_2d.experiments.mae.visualization import DEFAULT_FIXED_N_SAMPLES, render_reconstruction_grid
from hiera_2d.experiments.scaling.config import ExperimentConfig, RunIdentity, load_experiment_config
from hiera_2d.experiments.training_utils import build_warmup_cosine_scheduler, run_training_loop, seed_everything
from hiera_2d.hiera.mae import HieraMAE
from hiera_2d.hiera.model import Hiera


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
            progress.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / n_batches


@torch.no_grad()
def visualize_reconstructions(
    model: HieraMAE,
    dataset: KolmogorovDataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    mask_ratio: float,
):
    """Evaluate model on fixed set of samples and log to tensorboard"""

    model.eval()
    n_available = len(dataset)
    sample_indices = list(range(min(DEFAULT_FIXED_N_SAMPLES, n_available)))

    x_fixed = torch.stack([dataset[i]["initial"] for i in sample_indices]).to(device)
    grid = render_reconstruction_grid(model, x_fixed, mask_ratio)
    writer.add_image("samples/fixed_grid", grid, epoch)


def train_mae(cfg: ExperimentConfig, run: RunIdentity) -> None:
    """Run one MAE pretraining to completion: resolve the typed setup from `cfg`
    + the per-run `identity`, build the model/opt/scheduler, train, and write the
    provenance dump and checkpoints under `out_dir / name`.

    The imperative core of the MAE trainer — no argparse, no TOML loading. Called
    directly (in a fresh process) by the scaling orchestrator and by the thin
    `main` shell that backs the `train-mae` console script.
    """
    args = build_mae_train_args(cfg, run)
    mae_run = args.run
    seed_everything(mae_run.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # make sure output directory does not exist
    if args.path.exists():
        raise FileExistsError(f"Output directory already exists: {args.path}")
    args.path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.path)
    (args.path / "args.json").write_text(args.model_dump_json(indent=4))

    ds_train = get_dataset(args.dataset, args.data_path, split=Split.TRAIN, n_trajectories=args.n_trajectories)
    # Read val lazily (Kolmogorov only): at the largest N its eager copy would sit on
    # top of the full train subset and exhaust host RAM. Val is scanned sequentially,
    # so on-demand reads cost nothing extra.
    ds_val = get_dataset(args.dataset, args.data_path, split=Split.VAL, lazy=args.dataset == DatasetType.KOLMOGOROV)

    ld_train = DataLoader(
        ds_train,
        batch_size=mae_run.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    ld_val = DataLoader(
        ds_val,
        batch_size=mae_run.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    try:
        sample = ds_train[0]["initial"]
        print(f"Input sample shape: ({sample.shape})")

        # setup model and optimizer
        model = HieraMAE(
            encoder=Hiera(config=args.hiera),
            config=args.mae,
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=mae_run.lr,
            betas=(0.9, 0.95),
            weight_decay=mae_run.weight_decay,
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            n_warmup_epochs=mae_run.n_warmup_epochs,
            n_epochs=mae_run.n_epochs,
            min_lr=mae_run.min_lr,
        )

        print(f"Encoder token grid={model.shapes.sz_tk_final}, stride_pred_px={model.stride_pred_px}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        (args.path / "checkpoints").mkdir(exist_ok=True)

        # `hiera` is the load-bearing key: the AR trainer reads the encoder
        # architecture back from train_config["hiera"] when finetuning off this MAE.
        train_config = {
            "hiera": args.hiera.model_dump(mode="json"),
            "mae": args.mae.model_dump(mode="json"),
            **mae_run.model_dump(mode="json"),
        }

        def train_epoch(epoch: int):
            return run_epoch(model, ld_train, optimizer, device, epoch, mae_run.mask_ratio, training=True)

        def validate_epoch(epoch: int):
            return run_epoch(model, ld_val, None, device, epoch, mae_run.mask_ratio, training=False)

        def on_epoch_end(epoch: int, train_loss: float, val_loss: float, is_best: bool):
            visualize_reconstructions(model, ds_val, device, epoch, writer, mae_run.mask_ratio)
            save_mae_training_checkpoint(
                run_path=args.path,
                epoch=epoch,
                n_epochs=mae_run.n_epochs,
                model=model,
                train_loss=train_loss,
                val_loss=val_loss,
                train_config=train_config,
                data_path=str(args.data_path),
                dataset=args.dataset.value,
                is_best=is_best,
            )

        run_training_loop(
            n_epochs=mae_run.n_epochs,
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
    """Thin shell for the `train-mae` console script: parse the CLI, load the
    experiment config, build the per-run identity, and hand off to `train_mae`."""
    parsed = _parse_args(argv)
    cfg = load_experiment_config(parsed.config)
    run = RunIdentity(n_trajectories=parsed.n_trajectories, out_dir=parsed.output_dir, name=parsed.name)
    train_mae(cfg, run)


if __name__ == "__main__":
    main()
