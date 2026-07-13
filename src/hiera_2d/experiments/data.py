import json
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


def split_rows(n_total: int, train_frac: float, split_seed: int, split: Split) -> np.ndarray:
    """File row indices belonging to `split`.

    The partition is a fixed permutation of the file's trajectories, so it is
    reproducible and disjoint, and the file itself stays split-independent. Single
    source for the dataset and for the normalization statistics, which must agree on
    what "training data" means or the statistics leak.
    """
    perm = np.random.default_rng(split_seed).permutation(n_total)
    n_train = round(n_total * train_frac)

    return perm[:n_train] if split == Split.TRAIN else perm[n_train:]


def train_pool_norm_stats(path: Path, *, train_frac: float = 0.8, split_seed: int = 0) -> dict[str, float]:
    """Normalization statistics computed over the TRAINING split only.

    The validation trajectories must not contribute: statistics are model inputs, so
    deriving them from held-out data leaks it. The file's own `u_mean`/`u_std` root
    attrs are computed by the generator over *every* trajectory and are therefore
    unusable for this purpose -- they are deliberately ignored.

    The statistics come from the whole training *pool*, not from the `n_trajectories`
    subset a given run draws, so every data budget and both splits share one
    normalization; otherwise the loss scale would shift with N and the scaling curves
    would not be comparable.

    Computed by streaming over the training trajectories (the full file does not fit
    in RAM) and cached in a sidecar JSON next to the dataset, since the read is slow
    and the result depends only on (file, train_frac, split_seed).
    """
    cache_path = path.with_suffix(path.suffix + ".trainstats.json")
    key = {"train_frac": train_frac, "split_seed": split_seed, "n_bytes": path.stat().st_size}

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key:
            return {"mean": cached["mean"], "std": cached["std"]}

    with h5py.File(path, "r") as f:
        dset = f["velocity"]
        rows = np.sort(split_rows(dset.shape[0], train_frac, split_seed, Split.TRAIN))

        # Per-channel sums in float64: the frames are float32 and there are ~10^10 of
        # them, so a naive float32 accumulation would lose the low-order bits.
        total = np.zeros(2)
        total_sq = np.zeros(2)
        count = 0

        for row in rows:
            traj = dset[int(row)].astype(np.float64)  # (T, 2, H, W)
            total += traj.sum(axis=(0, 2, 3))
            total_sq += (traj**2).sum(axis=(0, 2, 3))
            count += traj.shape[0] * traj.shape[2] * traj.shape[3]

    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)

    # One scalar mean/std for both velocity components, matching how the frames are
    # normalized downstream (a single shared scale, not per-channel).
    stats = {"mean": float(mean.mean()), "std": float(std.mean())}
    cache_path.write_text(json.dumps({"key": key, **stats}, indent=2))

    return stats


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
    by a fixed permutation (``split_seed``), not stored in the file. Normalization
    comes from ``train_pool_norm_stats`` -- the training split only, never the
    file's all-trajectory root attrs.

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

            train_idx = split_rows(n_total, train_frac, split_seed, Split.TRAIN)
            split_idx = train_idx if split == Split.TRAIN else split_rows(n_total, train_frac, split_seed, Split.VAL)

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

            data = None if lazy else dset[rows]

        # Train-pool statistics, for BOTH splits and every N: they must not see the
        # validation trajectories (leak) and must not vary with the subset (which would
        # make the loss scale N-dependent). NB the checkpoints reported in the write-up
        # predate this and were trained against the file's all-trajectory attrs, whose
        # std is 0.05% larger.
        self.norm_stats = train_pool_norm_stats(path, train_frac=train_frac, split_seed=split_seed)

        if lazy:
            self.sims = LazyH5Trajectories(path, rows)
        else:
            self.sims = {i: data[i] for i in range(data.shape[0])}

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


def get_dataset(dataset_type: DatasetType, path: Path, split: Split = Split.TRAIN, **kwargs) -> KolmogorovDataset:
    if dataset_type != DatasetType.KOLMOGOROV:
        msg = f"unsupported dataset type: {dataset_type}"
        raise ValueError(msg)

    return KolmogorovDataset(path=path, split=split, **kwargs)
