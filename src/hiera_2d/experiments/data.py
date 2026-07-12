from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetType(StrEnum):
    KOLMOGOROV = "kolmogorov"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"


class LazyH5Trajectories:
    """Trajectory-indexed view over an open HDF5 ``velocity`` dataset that reads
    each trajectory from disk on access and caches the most recent one.

    Presents the subset of the mapping interface ``KolmogorovDataset`` needs — ``len``
    and integer indexing returning one ``(T, 2, H, W)`` array. Validation runs with
    ``shuffle=False`` and accesses trajectories in order, so the single-entry cache
    serves every frame of a trajectory from one read: no random-access penalty. Used
    for the validation split, whose eager load would otherwise hold the whole val set
    in RAM on top of the (already large) training subset.

    The file handle stays open for the object's lifetime. This is safe only under the
    conditions the training loop guarantees: the dataset is built inside the spawned
    worker (never pickled across the process boundary) and read with ``num_workers=0``
    (never forked). The OS reclaims the handle when the worker exits.
    """

    def __init__(self, path: Path, rows: np.ndarray):
        self._file = h5py.File(path, "r")
        self._dset = self._file["velocity"]
        self._rows = rows  # file row-index for each local trajectory i
        self._cached_i = -1
        self._cached: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> np.ndarray:
        if i != self._cached_i:
            self._cached = self._dset[int(self._rows[i])]
            self._cached_i = i

        return self._cached


class KolmogorovDataset(Dataset):
    """Kolmogorov 2D flow dataset.

    HDF5 layout: a single flat ``velocity`` dataset of shape
    (n_sims, n_timesteps, 2, H, W). The train/val split is applied at load time
    by a fixed permutation (``split_seed``), not stored in the file. Global norm
    stats are stored as root attrs (u_mean, u_std, v_mean, v_std).

    Indexing yields a single normalized frame (``{"initial": (2, H, W)}``), which is
    what MAE pretraining consumes; the AR pipeline builds its own sequences from
    ``sims`` via ``to_ar_dataset``.
    """

    sims: dict[int, np.ndarray] | LazyH5Trajectories
    norm_stats: dict[str, float]
    n_channels: int
    n_timesteps: int
    bundle: int

    def __init__(
        self,
        path: Path,
        split: Split = Split.TRAIN,
        bundle: int = 1,
        *,
        train_frac: float = 0.8,
        split_seed: int = 0,
        n_trajectories: int | None = None,
        subset_seed: int = 0,
        lazy: bool = False,
    ):
        super().__init__()
        self.path = path
        self.split = split
        self.bundle = bundle
        self.n_channels = 2

        if n_trajectories is not None and split != Split.TRAIN:
            msg = f"n_trajectories subsetting is train-only, got split={split}"
            raise ValueError(msg)

        with h5py.File(path, "r") as f:
            dset = f["velocity"]  # (n_sims, T, 2, H, W)
            n_total = dset.shape[0]
            n_timesteps = dset.shape[1]

            # Split into train/val by a fixed permutation so the partition is
            # reproducible and disjoint; the file itself is split-independent.
            perm = np.random.default_rng(split_seed).permutation(n_total)
            n_train = round(n_total * train_frac)
            train_idx = perm[:n_train]
            split_idx = train_idx if split == Split.TRAIN else perm[n_train:]

            if split == Split.TRAIN and n_trajectories is not None:
                if not 0 < n_trajectories <= len(train_idx):
                    msg = f"n_trajectories must be in (0, {len(train_idx)}], got {n_trajectories}"
                    raise ValueError(msg)

                # Nested subset WITHIN the train indices. Fixed subset_seed =>
                # nested subsets AS SETS: {n=2} ⊆ {n=4}, so a smaller N is
                # contained in a larger N.
                sub = np.random.default_rng(subset_seed).permutation(len(train_idx))[:n_trajectories]
                final_idx = train_idx[sub]
            else:
                final_idx = split_idx

            # np.sort is required by h5py fancy indexing (sorted, unique,
            # increasing); intra-subset order is irrelevant since training
            # shuffles. Peak RAM scales with the selection, not the full file.
            rows = np.sort(final_idx)

            if "u_mean" in f.attrs:
                self.norm_stats = {
                    "mean": float(np.mean([f.attrs["u_mean"], f.attrs["v_mean"]])),
                    "std": float(np.mean([f.attrs["u_std"], f.attrs["v_std"]])),
                }

            data = None if lazy else dset[rows]

        # Subsetting keeps the file's global norm stats (loaded above); it does NOT
        # recompute from the subset, so every N shares one normalization. The
        # attr-absent fallback below only fires for files without stored stats.
        if lazy:
            # Lazy reading cannot recompute stats without materializing the data it is
            # avoiding, so the file's global stats must be present.
            if not hasattr(self, "norm_stats"):
                msg = "lazy loading requires global stat attrs (u_mean/u_std/...) in the file"
                raise ValueError(msg)

            self.sims = LazyH5Trajectories(path, rows)
        else:
            self.sims = {i: data[i] for i in range(data.shape[0])}

        if n_trajectories is not None and not hasattr(self, "norm_stats"):
            # A subset relies on the file's global stats; recomputing from the subset
            # would silently break the shared-normalization invariant across N.
            msg = "n_trajectories subset requires global stat attrs (u_mean/u_std/...) in the file"
            raise ValueError(msg)

        if not hasattr(self, "norm_stats"):
            self.norm_stats = self._compute_norm_stats(self.sims)

        self.n_timesteps = n_timesteps

    def __len__(self):
        return len(self.sims) * (self.n_timesteps - self.bundle - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # The trailing margin in n_windows is kept so sample counts stay stable.
        n_windows = self.n_timesteps - self.bundle - 1
        time_idx = idx % n_windows
        file_idx = idx // n_windows

        initial = torch.from_numpy(self.sims[file_idx][time_idx]).squeeze()
        initial = (initial - self.norm_stats["mean"]) / self.norm_stats["std"]

        return {"initial": initial}

    @staticmethod
    def _compute_norm_stats(sims: dict[int, np.ndarray]) -> dict[str, float]:
        means = np.array([sims[i].mean() for i in range(len(sims))])
        stds = np.array([sims[i].std() for i in range(len(sims))])
        return {
            "mean": float(means.mean()),
            "std": float(np.sqrt(np.mean(stds**2 + (means - means.mean()) ** 2))),
        }


def get_dataset(dataset_type: DatasetType, path: Path, split: Split = Split.TRAIN, **kwargs) -> KolmogorovDataset:
    if dataset_type != DatasetType.KOLMOGOROV:
        msg = f"unsupported dataset type: {dataset_type}"
        raise ValueError(msg)

    return KolmogorovDataset(path=path, split=split, **kwargs)
