import tomllib
from pathlib import Path

from pydantic import BaseModel


def load_toml[T: BaseModel](path: Path, model: type[T]) -> T:
    """Read a TOML config file (stdlib `tomllib`) and validate it into `model`.

    The single parse boundary for every config the experiments read — the
    experiment/scaling file and the Hiera / MAE / AR-head architectures — so all of
    them are parsed identically and reject unknown keys the same way.
    """
    return model.model_validate(tomllib.loads(path.read_text()))
