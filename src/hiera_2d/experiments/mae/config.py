import argparse
from datetime import datetime
from pathlib import Path

from hiera_2d.experiments.data import DatasetType
from hiera_2d.experiments.scaling.config import ExperimentConfig, MaeRun, RunIdentity
from hiera_2d.hiera.mae import MAEConfig
from hiera_2d.hiera.model import HieraConfig
from hiera_2d.hiera.types import Model


class MaeTrainArgs(Model):
    """Fully-resolved inputs to one MAE pretraining run: the encoder/decoder
    architecture, the `[mae]` hyperparameters, dataset identity (from CONFIG), and
    the per-run CLI identity (`n_trajectories`, output `path`). Dumped verbatim as
    the run's provenance record."""

    hiera: HieraConfig
    mae: MAEConfig
    run: MaeRun
    dataset: DatasetType
    data_path: Path
    n_trajectories: int
    path: Path


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="MAE pretraining for Hiera (config-driven)")
    parser.add_argument("config", type=Path, help="Path to the scaling-experiment TOML config")
    parser.add_argument(
        "--n-trajectories",
        type=int,
        required=True,
        help="Subset the TRAINING split to this many trajectories (per-run identity in the data-scaling sweep)",
    )
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory where the run is saved")
    parser.add_argument(
        "--name",
        type=str,
        default=f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Run name (subdirectory of --output-dir)",
    )

    return parser.parse_args(argv)


def build_mae_train_args(cfg: ExperimentConfig, run: RunIdentity) -> MaeTrainArgs:
    """Resolve the fully-typed inputs for one MAE run from the experiment config
    and the per-run identity: parse the architecture JSONs, pin dataset identity,
    and place the run at `out_dir / name`. No CLI, no TOML reading here."""
    hiera = HieraConfig.model_validate_json(cfg.hiera_config.read_text())
    mae = MAEConfig.model_validate_json(cfg.mae_config.read_text())

    return MaeTrainArgs(
        hiera=hiera,
        mae=mae,
        run=cfg.mae,
        dataset=cfg.dataset,
        data_path=cfg.data_path.resolve(),
        n_trajectories=run.n_trajectories,
        path=(run.out_dir / run.name).resolve(),
    )
