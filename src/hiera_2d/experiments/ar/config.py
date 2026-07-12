import argparse
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from hiera_2d.experiments.ar.model import ARHeadConfig
from hiera_2d.experiments.config_io import load_toml
from hiera_2d.experiments.data import DatasetType
from hiera_2d.experiments.scaling.config import ArRun, ExperimentConfig, RunIdentity
from hiera_2d.hiera.model import HieraConfig
from hiera_2d.hiera.types import Model


class PretrainedEncoderSource(Model):
    """Encoder initialized from an MAE checkpoint; its architecture is read from
    the checkpoint at load time."""

    kind: Literal["pretrained"] = "pretrained"
    mae_checkpoint: Path


class ScratchEncoderSource(Model):
    """Randomly initialized encoder — the from-scratch baseline. Its architecture
    is the parsed Hiera config."""

    kind: Literal["scratch"] = "scratch"
    hiera: HieraConfig


type EncoderSource = Annotated[PretrainedEncoderSource | ScratchEncoderSource, Field(discriminator="kind")]


class ArTrainArgs(Model):
    """Fully-resolved inputs to one AR training run: the AR head architecture, the
    `[ar]` hyperparameters (with `--n-epochs` override applied), the encoder source
    (pretrained MAE checkpoint or from-scratch Hiera), dataset identity, and the
    per-run CLI identity (`n_trajectories`, output `path`)."""

    ar_head: ARHeadConfig
    run: ArRun
    encoder: EncoderSource
    dataset: DatasetType
    data_path: Path
    n_trajectories: int
    path: Path


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="AR head training for Hiera (config-driven)")
    parser.add_argument("config", type=Path, help="Path to the scaling-experiment TOML config")
    parser.add_argument("--n-trajectories", type=int, required=True, help="Subset the TRAINING split to this many")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory where the run is saved")
    parser.add_argument(
        "--name",
        type=str,
        default=f"ar_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Run name (subdirectory of --output-dir)",
    )
    parser.add_argument(
        "--mae-checkpoint",
        type=Path,
        default=None,
        help="Pretrained MAE checkpoint to finetune from; omit for the from-scratch baseline (random init)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override the config's AR epoch budget (the scratch-arm fairness bump)",
    )

    return parser.parse_args(argv)


def build_ar_train_args(cfg: ExperimentConfig, run: RunIdentity) -> ArTrainArgs:
    """Resolve the fully-typed inputs for one AR run from the experiment config
    and the per-run identity: apply the `n_epochs` override, parse the AR-head
    config, and select the encoder source — an MAE checkpoint (finetune) when
    `run.mae_checkpoint` is set, else a random-init Hiera (scratch). No CLI here."""
    ar_run = cfg.ar if run.n_epochs is None else cfg.ar.model_copy(update={"n_epochs": run.n_epochs})

    ar_head = load_toml(cfg.ar_config, ARHeadConfig) if cfg.ar_config else ARHeadConfig()
    ar_head = ar_head.model_copy(update={"predict_residual": ar_run.predict_residual})

    if run.mae_checkpoint is not None:
        encoder: EncoderSource = PretrainedEncoderSource(mae_checkpoint=run.mae_checkpoint)
    else:
        encoder = ScratchEncoderSource(hiera=load_toml(cfg.hiera_config, HieraConfig))

    return ArTrainArgs(
        ar_head=ar_head,
        run=ar_run,
        encoder=encoder,
        dataset=cfg.dataset,
        data_path=cfg.data_path.resolve(),
        n_trajectories=run.n_trajectories,
        path=(run.out_dir / run.name).resolve(),
    )
