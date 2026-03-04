import json
from contextlib import suppress
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class GrayScottDataset(Dataset):
    """Gray-Scott MAE dataset using train.hdf5 / val.hdf5."""

    NORM_STATS_SUFFIX = "_norm_stats.json"

    def __init__(
        self,
        path: Path,
        split: Literal["train", "val"] = "train",
        cache_sims: bool = False,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.cache_sims = cache_sims
        self.dataset_path = self.path / f"{split}.hdf5"

        self.sims: dict[int, np.ndarray] = {}
        self._h5: h5py.File | None = None

        norm_stats = self._load_norm_stats()
        need_norm_scan = norm_stats is None
        total_sum, total_sq_sum, total_count = self._index_dataset(need_norm_scan=need_norm_scan)

        if need_norm_scan:
            norm_stats = self._compute_norm_stats(total_sum, total_sq_sum, total_count)
            self._save_norm_stats(norm_stats)

        self.norm_stats = norm_stats
        self.samples_per_sim = self.n_timesteps

    def _index_dataset(self, *, need_norm_scan: bool) -> tuple[float, float, int]:
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        with h5py.File(self.dataset_path, "r") as handle:
            sims_group = handle["sims"]
            self.sim_keys = self._sorted_sim_keys(sims_group)
            if not self.sim_keys:
                raise ValueError(f"No simulation entries found in {self.dataset_path}")

            self.n_timesteps = sims_group[self.sim_keys[0]].shape[0]
            for idx, sim_key in enumerate(self.sim_keys):
                s, ss, c = self._register_sim(idx, sims_group[sim_key], need_norm_scan=need_norm_scan)
                total_sum += s
                total_sq_sum += ss
                total_count += c

        return total_sum, total_sq_sum, total_count

    @staticmethod
    def _sorted_sim_keys(sims_group: h5py.Group) -> list[str]:
        return sorted(
            [key for key in sims_group if key.startswith("sim")],
            key=lambda key: int(key.removeprefix("sim")),
        )

    def _register_sim(
        self,
        idx: int,
        sim: h5py.Dataset,
        *,
        need_norm_scan: bool,
    ) -> tuple[float, float, int]:
        if self.cache_sims:
            sim_arr = sim[:]
            self.sims[idx] = sim_arr
            if need_norm_scan:
                return self._stats_from_array(sim_arr)
            return 0.0, 0.0, 0

        if need_norm_scan:
            return self._stats_from_dataset(sim)
        return 0.0, 0.0, 0

    @staticmethod
    def _compute_norm_stats(total_sum: float, total_sq_sum: float, total_count: int) -> dict[str, float]:
        if total_count == 0:
            raise ValueError("no data found while computing normalization stats")
        mean = total_sum / total_count
        variance = max(total_sq_sum / total_count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        if std <= 0:
            raise ValueError("computed normalization std is non-positive")
        return {"mean": float(mean), "std": std}

    def _norm_stats_path(self) -> Path:
        return self.path / f"{self.split}{self.NORM_STATS_SUFFIX}"

    def _load_norm_stats(self) -> dict[str, float] | None:
        path = self._norm_stats_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if "mean" not in data or "std" not in data:
            return None
        std = float(data["std"])
        if std <= 0:
            return None
        return {"mean": float(data["mean"]), "std": std}

    def _save_norm_stats(self, stats: dict[str, float]) -> None:
        with suppress(OSError):
            self._norm_stats_path().write_text(json.dumps(stats, indent=2))

    @staticmethod
    def _stats_from_array(arr: np.ndarray) -> tuple[float, float, int]:
        arr64 = arr.astype(np.float64, copy=False)
        return float(arr64.sum()), float(np.square(arr64).sum()), int(arr64.size)

    @staticmethod
    def _stats_from_dataset(sim: h5py.Dataset, chunk_size: int = 2) -> tuple[float, float, int]:
        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        n_steps = sim.shape[0]
        for start in range(0, n_steps, chunk_size):
            chunk = sim[start : min(start + chunk_size, n_steps)]
            chunk64 = chunk.astype(np.float64, copy=False)
            total_sum += float(chunk64.sum())
            total_sq_sum += float(np.square(chunk64).sum())
            total_count += int(chunk64.size)

        return total_sum, total_sq_sum, total_count

    def __len__(self):
        return len(self.sim_keys) * self.samples_per_sim

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _get_sims_group(self):
        if self.cache_sims:
            return None
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5["sims"]

    def close(self) -> None:
        if self._h5 is not None:
            with suppress(Exception):
                self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        time_idx = idx % self.samples_per_sim
        sim_idx = idx // self.samples_per_sim

        sims = self._get_sims_group()
        sim = self.sims[sim_idx] if self.cache_sims else sims[self.sim_keys[sim_idx]]
        initial = torch.from_numpy(sim[time_idx]).squeeze()
        initial = (initial - self.norm_stats["mean"]) / self.norm_stats["std"]
        return {"initial": initial}
