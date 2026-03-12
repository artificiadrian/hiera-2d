import json
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset


class DatasetType(StrEnum):
    GRAY_SCOTT = "gray-scott"
    KOLMOGOROV = "kolmogorov"


class PDEDataset(Dataset):
    """Base dataset for PDE simulations.

    Subclasses populate: sims, sim_keys, sim_cond, norm_stats, n_channels,
    n_cond, n_timesteps.
    """

    sims: dict[int, np.ndarray]
    sim_keys: list[str]
    sim_cond: dict[str, np.ndarray]
    norm_stats: dict[str, float]
    n_channels: int
    n_cond: int
    n_timesteps: int
    bundle: int

    def __len__(self):
        return len(self.sims) * (self.n_timesteps - self.bundle - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n_windows = self.n_timesteps - self.bundle - 1
        time_idx = idx % n_windows
        file_idx = idx // n_windows
        sim = self.sims[file_idx]

        dt_offset = 1

        initial = torch.from_numpy(sim[time_idx]).squeeze()
        time_tgt = time_idx + dt_offset
        tgt_window = slice(time_tgt - self.bundle + 1, time_tgt + self.bundle)
        target = torch.from_numpy(sim[tgt_window])
        target = rearrange(target, "t c h w -> (t c) h w").squeeze()

        cond = torch.from_numpy(self.sim_cond[self.sim_keys[file_idx]])

        initial = (initial - self.norm_stats["mean"]) / self.norm_stats["std"]
        target = (target - self.norm_stats["mean"]) / self.norm_stats["std"]

        return {"initial": initial, "target": target, "cond": cond}

    @staticmethod
    def _compute_norm_stats(sims: dict[int, np.ndarray]) -> dict[str, float]:
        means = np.array([sims[i].mean() for i in range(len(sims))])
        stds = np.array([sims[i].std() for i in range(len(sims))])
        return {
            "mean": float(means.mean()),
            "std": float(np.sqrt(np.mean(stds**2 + (means - means.mean()) ** 2))),
        }


class GrayScottDataset(PDEDataset):
    """https://arxiv.org/abs/2505.24717"""

    def __init__(
        self,
        path: Path,
        split: str = "train",
        bundle: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.bundle = bundle
        self.seed = seed
        self.n_channels = 2

        with open(f"{path}/{split}_stats.json") as f:
            sim_conds = json.load(f)

            def simplify(key):
                if key.endswith("0000"):
                    return "sim0"
                else:
                    return f"sim{int(key.split('_')[1])}"

            self.sim_cond = {
                simplify(k): np.array(
                    [v for k2, v in vals.items() if k2.lower() != "seed"],
                    dtype=np.float32,
                )
                for k, vals in sim_conds.items()
                if "sim" in k
            }

        self.sims = {}
        with h5py.File(f"{path}/{split}.hdf5", "r") as f:
            self.sim_keys = sorted(
                [k for k in f["sims"].keys() if k.startswith("sim")],
                key=lambda x: int(x.replace("sim", "")),
            )
            for idx in range(len(self.sim_keys)):
                self.sims[idx] = f[f"sims/{self.sim_keys[idx]}"][:]

        self.norm_stats = self._compute_norm_stats(self.sims)
        self.n_timesteps = self.sims[0].shape[0]
        self.n_cond = len(self.sim_cond[self.sim_keys[0]])


class KolmogorovDataset(PDEDataset):
    """Kolmogorov 2D flow dataset.

    HDF5 layout: {split}/velocity with shape (n_sims, n_timesteps, 2, H, W).
    Stats stored as root attrs (u_mean, u_std, v_mean, v_std).
    """

    def __init__(
        self,
        path: Path,
        split: str = "train",
        bundle: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.bundle = bundle
        self.seed = seed
        self.n_channels = 2
        self.n_cond = 0

        with h5py.File(path, "r") as f:
            data = f[split]["velocity"][:]  # (n_sims, T, 2, H, W)
            if "u_mean" in f.attrs:
                self.norm_stats = {
                    "mean": float(np.mean([f.attrs["u_mean"], f.attrs["v_mean"]])),
                    "std": float(np.mean([f.attrs["u_std"], f.attrs["v_std"]])),
                }

        self.sims = {i: data[i] for i in range(data.shape[0])}
        self.sim_keys = [f"sim{i}" for i in range(len(self.sims))]
        self.sim_cond = {k: np.zeros(0, dtype=np.float32) for k in self.sim_keys}

        if not hasattr(self, "norm_stats"):
            self.norm_stats = self._compute_norm_stats(self.sims)

        self.n_timesteps = data.shape[1]


def get_dataset(dataset_type: DatasetType, path: Path, split: str = "train", **kwargs) -> PDEDataset:
    match dataset_type:
        case DatasetType.GRAY_SCOTT:
            return GrayScottDataset(path=path, split=split, **kwargs)
        case DatasetType.KOLMOGOROV:
            return KolmogorovDataset(path=path, split=split, **kwargs)
