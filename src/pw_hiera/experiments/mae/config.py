import argparse
from datetime import datetime
from pathlib import Path

from pydantic import Field, field_validator

from pw_hiera.hiera.mae import MAEConfig
from pw_hiera.hiera.model import HieraConfig
from pw_hiera.hiera.types import Model


class TrainConfig(Model):
    mae: MAEConfig
    hiera: HieraConfig
    mask_ratio: float = Field(gt=0.0, lt=1.0)
    n_epochs: int = Field(gt=0)
    n_warmup_epochs: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    min_lr: float = Field(ge=0.0)
    weight_decay: float = Field(ge=0.0)
    seed: int


class TrainArgs(Model):
    train_config: TrainConfig
    path: Path
    data_path: Path
    batch_size: int = Field(gt=0)

    @field_validator("data_path")
    @classmethod
    def _validate_data_path(cls, value: Path):
        if not value.exists():
            raise ValueError(f"data_path does not exist: {value}")
        return value


def _parse_args():
    parser = argparse.ArgumentParser(description="MAE pretraining for Hiera")
    parser.add_argument("--mae-config", type=Path, required=True, help="Path to a MAE decoder JSON config file")
    parser.add_argument("--hiera-config", type=Path, required=True, help="Path to a Hiera encoder JSON config file")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for train/val dataloaders")
    parser.add_argument("--mask-ratio", type=float, default=0.6, help="Mask ratio in (0, 1)")
    parser.add_argument("--n-epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n-warmup-epochs", type=int, required=True, help="Number of linear warmup epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory where outputs are saved", required=True)
    parser.add_argument(
        "--name",
        type=str,
        help="Name for the experiment",
        default=f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    return parser.parse_args()


def get_train_args():
    parsed = _parse_args()
    train_config = TrainConfig(
        mae=MAEConfig.model_validate_json(parsed.mae_config.read_text()),
        hiera=HieraConfig.model_validate_json(parsed.hiera_config.read_text()),
        mask_ratio=parsed.mask_ratio,
        n_epochs=parsed.n_epochs,
        n_warmup_epochs=parsed.n_warmup_epochs,
        lr=parsed.lr,
        min_lr=parsed.min_lr,
        weight_decay=parsed.weight_decay,
        seed=parsed.seed,
    )

    return TrainArgs(
        train_config=train_config,
        path=Path(parsed.output_dir / parsed.name).resolve(),
        data_path=Path(parsed.data_path).resolve(),
        batch_size=parsed.batch_size,
    )
