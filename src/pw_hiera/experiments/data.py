import json
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset


class GrayScottDataset(Dataset):
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

        dataset_path = f"{path}/{split}.hdf5"
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
        self.norm_stats = {"mean": [], "std": []}
        with h5py.File(dataset_path, "r") as f:
            self.sim_keys = sorted(
                [k for k in f["sims"].keys() if k.startswith("sim")],
                key=lambda x: int(x.replace("sim", "")),
            )
            for idx in range(len(self.sim_keys)):
                self.sims[idx] = f[f"sims/{self.sim_keys[idx]}"][:]
                self.norm_stats["mean"].append(self.sims[idx].mean())
                self.norm_stats["std"].append(self.sims[idx].std())

        means = np.array(self.norm_stats["mean"])
        stds = np.array(self.norm_stats["std"])
        self.norm_stats["mean"] = means.mean()
        self.norm_stats["std"] = np.sqrt(np.mean(stds**2 + (means - means.mean()) ** 2))

        self.n_timesteps = self.sims[0].shape[0]

        self.n_cond = len(self.sim_cond[self.sim_keys[0]])

    def __del__(self):
        if hasattr(self, "_ds"):
            self.sims.close()

    def __len__(self):
        return len(self.sims) * (self.n_timesteps - self.bundle - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        time_idx = idx % (self.n_timesteps - self.bundle - 1)  # + self.bundle - 1
        file_idx = idx // (self.n_timesteps - self.bundle)
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
