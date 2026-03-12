import torch
from torch.utils.data import Dataset

from hiera_2d.experiments.data import PDEDataset


class ARDataset(Dataset):
    """Autoregressive dataset: slides a window over trajectories.

    Each sample returns `seq_len` consecutive frames from one trajectory.
    """

    def __init__(self, trajectories: list[torch.Tensor], conds: list[torch.Tensor], seq_len: int):
        self.trajectories = trajectories
        self.conds = conds
        self.seq_len = seq_len

        self.windows: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(trajectories):
            n_steps = traj.shape[0]
            for t in range(n_steps - seq_len + 1):
                self.windows.append((traj_idx, t))

        if len(self.windows) == 0:
            raise ValueError(f"seq_len={seq_len} exceeds all trajectory lengths")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, t = self.windows[idx]
        frames = self.trajectories[traj_idx][t : t + self.seq_len]
        return {"frames": frames, "cond": self.conds[traj_idx]}


def to_ar_dataset(dataset: PDEDataset, seq_len: int = 10) -> ARDataset:
    """Convert any PDEDataset into an autoregressive sequence dataset."""
    mean = dataset.norm_stats["mean"]
    std = dataset.norm_stats["std"]

    trajectories = []
    conds = []
    for sim_idx in range(len(dataset.sims)):
        traj = torch.from_numpy(dataset.sims[sim_idx]).float()
        traj = (traj - mean) / std
        trajectories.append(traj)
        conds.append(torch.from_numpy(dataset.sim_cond[dataset.sim_keys[sim_idx]]))

    return ARDataset(trajectories, conds, seq_len=seq_len)
