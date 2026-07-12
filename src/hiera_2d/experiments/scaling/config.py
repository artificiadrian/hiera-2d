import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, field_validator

from hiera_2d.experiments.data import DatasetType
from hiera_2d.hiera.types import Model


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """The per-run identity that distinguishes one training invocation from
    another sharing the same `ExperimentConfig`: how much data it sees, where it
    lands, and (for AR) which encoder it starts from and any epoch override.

    Picklable and self-contained so it can cross a `spawn`ed process boundary as
    the typed argument to `train_mae`/`train_ar` — no CLI-string round-trip.
    The run directory is `out_dir / name`; `mae_checkpoint` set selects the AR
    finetune arm (encoder loaded from that MAE), `None` the from-scratch arm;
    `n_epochs` set overrides the config's AR epoch budget (the scratch fairness bump).
    """

    n_trajectories: int
    out_dir: Path
    name: str
    mae_checkpoint: Path | None = None
    n_epochs: int | None = None


class MaeRun(Model):
    """MAE pretraining hyperparameters — the `[mae]` block. Single source for
    every MAE run in the sweep and for a standalone `train-mae`."""

    n_epochs: int = Field(gt=0)
    n_warmup_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    mask_ratio: float = Field(gt=0.0, lt=1.0)
    lr: float = Field(gt=0.0)
    min_lr: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    seed: int


class ArRun(Model):
    """AR head hyperparameters — the `[ar]` block. Shared by the finetune and
    scratch arms; the scratch arm scales `n_epochs` by
    `ExperimentConfig.scratch_epoch_mult` (train-ar's `--n-epochs` override)."""

    n_epochs: int = Field(gt=0)
    n_warmup_epochs: int = Field(gt=0)
    unroll_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    min_lr: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    predict_residual: bool
    seed: int
    amp: bool
    freeze_encoder: bool = False


class ExperimentConfig(Model):
    """One TOML file fully determines the Kolmogorov data-scaling sweep: per N in
    `n_trajectories`, an MAE pretrain, an AR finetune from that MAE, and an AR
    from-scratch baseline. The `[mae]`/`[ar]` blocks are the single source of the
    training hyperparameters; the `*_config` paths point at the architecture JSONs.
    Paths are validated to exist at load."""

    dataset: DatasetType = DatasetType.KOLMOGOROV
    data_path: Path
    output_root: Path
    hiera_config: Path
    mae_config: Path
    ar_config: Path | None = None
    n_trajectories: tuple[int, ...]
    mae: MaeRun
    ar: ArRun
    scratch_epoch_mult: float = 1.5

    @field_validator("data_path", "hiera_config", "mae_config", "ar_config")
    @classmethod
    def _path_exists(cls, value: Path | None):
        if value is not None and not value.exists():
            msg = f"path does not exist: {value}"
            raise ValueError(msg)

        return value


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Read a scaling-experiment TOML config (stdlib `tomllib`) and validate it.

    The single loader used by run-scaling, train-mae, train-ar, scaling-curve, and
    spectral-plot so every script parses the experiment identically.
    """
    data = tomllib.loads(path.read_text())

    return ExperimentConfig.model_validate(data)


# Every arm's run/directory name is budget-suffixed (`{arm}_e{epochs}`): the epoch
# budget is a property of every run, not just scratch, and encoding it uniformly
# keeps run-scaling's skip honest — a run is reused only when a same-budget
# checkpoint sits at its exact name, so any budget change lands in a fresh dir
# instead of silently reusing a differently-trained one. Names that happen to be
# equal across two protocols (mae_e120, finetune_e30) are still shared and skipped;
# only the arm whose budget actually differs (scratch_e30 vs scratch_e45) splits.
# Single source for run-scaling and the analysis scripts.
def scratch_epochs(cfg: ExperimentConfig) -> int:
    """The scratch arm's epoch budget: the AR budget scaled by the fairness bump
    `scratch_epoch_mult`, rounded to a whole number of epochs."""
    return round(cfg.ar.n_epochs * cfg.scratch_epoch_mult)


def mae_run_name(cfg: ExperimentConfig) -> str:
    """The MAE pretrain arm's run/directory name, budget-suffixed (e.g. `mae_e120`)."""
    return f"mae_e{cfg.mae.n_epochs}"


def finetune_run_name(cfg: ExperimentConfig) -> str:
    """The AR finetune arm's run/directory name, budget-suffixed (e.g. `finetune_e30`)."""
    return f"finetune_e{cfg.ar.n_epochs}"


def scratch_run_name(cfg: ExperimentConfig) -> str:
    """The scratch arm's run/directory name, budget-suffixed (e.g. `scratch_e45`),
    so two `scratch_epoch_mult` protocols coexist under one `output_root` instead
    of one silently reusing the other's checkpoints."""
    return f"scratch_e{scratch_epochs(cfg)}"
