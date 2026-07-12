from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from hiera_2d.experiments.data import KolmogorovDataset, LazyH5Trajectories


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One simulation trajectory: its (T, C, H, W) frame stack."""

    frames: torch.Tensor


class ARDataset(Dataset):
    """Autoregressive dataset: slides a window over trajectories.

    Each sample returns `seq_len` consecutive frames from one trajectory.

    Trajectories are kept in their raw form (sharing memory with the source
    arrays) and normalized per-window in __getitem__. Normalizing the whole
    dataset up front would double its memory footprint, which OOMs for large
    datasets (e.g. Kolmogorov: ~21 GB raw -> ~42 GB if copied).
    """

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        seq_len: int,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        self.trajectories = trajectories
        self.seq_len = seq_len
        self.mean = mean
        self.std = std

        self.windows: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(trajectories):
            n_steps = traj.frames.shape[0]
            for t in range(n_steps - seq_len + 1):
                self.windows.append((traj_idx, t))

        if len(self.windows) == 0:
            msg = f"seq_len={seq_len} exceeds all trajectory lengths"
            raise ValueError(msg)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, t = self.windows[idx]
        traj = self.trajectories[traj_idx]
        frames = traj.frames[t : t + self.seq_len]
        frames = (frames.float() - self.mean) / self.std
        return {"frames": frames}


class _LazyTrajectorySeq(Sequence[Trajectory]):
    """On-demand ``Sequence[Trajectory]`` backed by a lazy trajectory reader.

    Reading `dataset.sims[i]` hits the reader's cache, so an AR dataset built from a
    lazy (validation) source never holds more than one trajectory at a time —
    neither while `ARDataset` scans lengths to build its window index nor during
    the sequential val pass. `torch.from_numpy` returns a view, so no frame is copied
    until per-window normalization.
    """

    def __init__(self, dataset: KolmogorovDataset):
        self._sims = dataset.sims

    def __len__(self) -> int:
        return len(self._sims)

    def __getitem__(self, i: int) -> Trajectory:
        return Trajectory(frames=torch.from_numpy(self._sims[i]))


def to_ar_dataset(dataset: KolmogorovDataset, seq_len: int = 10) -> ARDataset:
    """Convert a KolmogorovDataset into an autoregressive sequence dataset.

    Trajectory frames share memory with `dataset.sims`; normalization is deferred
    to ARDataset.__getitem__ so we never hold a second full copy of the data. When
    the source loads lazily (validation), trajectories are read on demand instead of
    materialized into a list up front.
    """
    trajectories: Sequence[Trajectory]
    if isinstance(dataset.sims, LazyH5Trajectories):
        trajectories = _LazyTrajectorySeq(dataset)
    else:
        trajectories = [
            Trajectory(frames=torch.from_numpy(dataset.sims[sim_idx])) for sim_idx in range(len(dataset.sims))
        ]

    return ARDataset(
        trajectories,
        seq_len=seq_len,
        mean=dataset.norm_stats["mean"],
        std=dataset.norm_stats["std"],
    )
