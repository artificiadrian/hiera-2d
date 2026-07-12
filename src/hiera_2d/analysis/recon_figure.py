"""Report-quality MAE reconstruction figure.

The TensorBoard grid in `experiments/mae/visualization.py` is a channel-averaged
grayscale strip with no labels -- fine for watching a run, useless in a report.
This module renders the same forward pass as a labelled, diverging-colormap
figure matching the dataset figures, and shows two rows:

  * the velocity component `u`, which is what the model actually predicts, and
  * the vorticity `w`, which is what the eye can read.

Vorticity is a spatial derivative of the prediction, so it amplifies exactly the
small-scale error the velocity row hides. Showing both is the honest choice: the
velocity row says the reconstruction is close, the vorticity row says where it
is not.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hiera_2d.analysis.spectra import vorticity
from hiera_2d.experiments.data import DatasetType, Split, get_dataset
from hiera_2d.experiments.mae.visualization import reconstruct
from hiera_2d.hiera.mae import HieraMAE, MAEConfig
from hiera_2d.hiera.model import Hiera, HieraConfig

# The decoder's raw output is deliberately NOT a column: the MAE loss only ever scores
# masked patches, so on visible ones the decoder was never trained to be accurate and its
# output there shows seams it was never asked to avoid. The composite -- original where
# visible, prediction where hidden -- is what the model was actually optimized to produce.
COLUMNS = ("ground truth", "masked input", "composite (prediction where masked)")


@dataclass(frozen=True, slots=True)
class LoadedMAE:
    """An MAE restored from a checkpoint, with the run settings needed to reproduce its figure."""

    model: HieraMAE
    mask_ratio: float
    data_path: Path
    val_loss: float


def load_mae(checkpoint_path: Path, device: torch.device) -> LoadedMAE:
    """Rebuild the MAE (encoder + decoder) from an MAE training checkpoint.

    The architecture is read back from the checkpoint's `train_config` rather than
    from a config file, so the figure cannot silently be drawn with a different
    model than the one that was trained.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    train_config = ckpt["train_config"]

    model = HieraMAE(
        encoder=Hiera(config=HieraConfig.model_validate(train_config["hiera"])),
        config=MAEConfig.model_validate(train_config["mae"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return LoadedMAE(
        model=model,
        mask_ratio=float(train_config["mask_ratio"]),
        data_path=Path(ckpt["data_path"]),
        val_loss=float(ckpt["val_loss"]),
    )


@dataclass(frozen=True, slots=True)
class ReconPanels:
    """One sample's panels, in physical units, for both display rows.

    Each array is `(3, H, W)` ordered as `COLUMNS`. In `u` and `omega` the masked
    column carries `NaN` wherever the encoder was blind, so it renders as a
    neutral hole rather than as a fake zero velocity.
    """

    u: np.ndarray
    omega: np.ndarray


def build_panels(loaded: LoadedMAE, frame: torch.Tensor, mean: float, std: float, device: torch.device) -> ReconPanels:
    """Run the MAE on one `(2, H, W)` normalized frame and assemble the display panels.

    Fields are de-normalized back to physical velocity before the vorticity is
    taken, so the colorbars carry real units.
    """
    with torch.no_grad():
        r = reconstruct(loaded.model, frame.unsqueeze(0).to(device), loaded.mask_ratio)

    def physical(t: torch.Tensor) -> np.ndarray:
        return t[0].cpu().numpy() * std + mean

    original = physical(r.original)
    composite = physical(r.composite)

    visible = r.visible[0, 0].cpu().numpy().astype(bool)
    masked = np.where(visible, original, np.nan)

    fields = (original, masked, composite)

    # Vorticity is a spectral derivative, so it cannot be taken through the holes:
    # filling them with any value (zero, or the NaNs themselves) rings across the
    # whole domain and the masked panel would show that artifact rather than the
    # flow. The visible region's vorticity is by definition the true field's, so it
    # is differentiated intact and the holes are punched in afterwards.
    omega_true = vorticity(original)
    omega = np.stack([omega_true, np.where(visible, omega_true, np.nan), vorticity(composite)])

    return ReconPanels(u=np.stack([f[0] for f in fields]), omega=omega)


def plot_reconstruction(panels: ReconPanels, out_path: Path, title: str) -> None:
    """Write the labelled reconstruction figure: `COLUMNS` across, velocity and vorticity down."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")  # masked-out regions read as neutral gray, not as a value

    rows = (
        (panels.u, r"velocity $u$ [m/s]"),
        (panels.omega, r"vorticity $\omega$ [1/s]"),
    )

    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(10.5, 7.0))
    for row, (data, label) in enumerate(rows):
        # One symmetric scale per row, taken from ground truth, so the panels are
        # directly comparable and a washed-out prediction stays visibly washed out.
        vlim = float(np.nanmax(np.abs(data[0])))

        for col in range(len(COLUMNS)):
            ax = axes[row, col]
            im = ax.imshow(data[col], cmap=cmap, vmin=-vlim, vmax=vlim, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(COLUMNS[col], fontsize=11)

            if col == 0:
                ax.set_ylabel(label, fontsize=11)

        cbar = fig.colorbar(im, ax=axes[row, :], fraction=0.02, pad=0.01)
        cbar.set_label(label)

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Render a labelled MAE reconstruction figure")
    p.add_argument("--checkpoint", type=Path, required=True, help="MAE best_model.pt")
    p.add_argument("--data-path", type=Path, default=None, help="Defaults to the checkpoint's own dataset")
    p.add_argument("--sample", type=int, default=0, help="Index into the validation split")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/analysis_recon"))
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_mae(args.checkpoint, device)
    data_path = args.data_path or loaded.data_path

    dataset = get_dataset(DatasetType.KOLMOGOROV, data_path, split=Split.VAL, lazy=True)
    frame = dataset[args.sample]["initial"]
    mean, std = dataset.norm_stats["mean"], dataset.norm_stats["std"]

    panels = build_panels(loaded, frame, mean, std, device)
    out_path = args.output_dir / f"recon_sample{args.sample}.png"
    plot_reconstruction(
        panels,
        out_path,
        title=(
            f"MAE reconstruction on a held-out Kolmogorov frame "
            f"({loaded.mask_ratio:.0%} of mask units hidden, val loss {loaded.val_loss:.4f})"
        ),
    )

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
