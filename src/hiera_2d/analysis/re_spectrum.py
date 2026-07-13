"""Reynolds-number signature of the Kolmogorov dataset.

The Re family spans only 3000..5000, so the effect is invisible in a vorticity
snapshot: two trajectories at different Re differ far more by their random
realization than by their Reynolds number. It is, however, clearly visible in the
energy spectrum, where higher Re means weaker viscous dissipation and therefore a
cascade that survives to smaller scales -- a high-k tail that lifts with Re.
Averaging E(k) over many trajectories per bin is what cancels the realization
noise that swamps the single-snapshot comparison.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from hiera_2d.analysis.spectra import radial_energy_spectrum


@dataclass(frozen=True, slots=True)
class ReBin:
    """One Reynolds bin: its Re range, how many trajectories it averages, and the
    resulting mean radial energy spectrum (`spectrum[j]` is the energy at `k[j]`)."""

    re_min: float
    re_max: float
    n_trajectories: int
    spectrum: np.ndarray


@dataclass(frozen=True, slots=True)
class ReSpectra:
    """Radial wavenumbers shared by every bin, and the per-bin mean spectra in
    ascending Re order."""

    k: np.ndarray
    bins: tuple[ReBin, ...]


def bin_by_reynolds(re: np.ndarray, spectra: np.ndarray, n_bins: int) -> tuple[ReBin, ...]:
    """Group per-trajectory spectra into `n_bins` equal-count Reynolds bins and average
    within each.

    Equal-count (not equal-width) bins: the generator draws Re round-robin over a fixed
    ladder, so equal counts keep every bin's mean equally well-resolved. Pure -- the
    caller supplies the spectra.

    Args:
        re: (T,) Reynolds number per trajectory.
        spectra: (T, K) radial energy spectrum per trajectory.
        n_bins: number of bins; must not exceed the number of trajectories.
    """
    if re.shape[0] != spectra.shape[0]:
        msg = f"re has {re.shape[0]} trajectories but spectra has {spectra.shape[0]}"
        raise ValueError(msg)

    if not 0 < n_bins <= re.shape[0]:
        msg = f"n_bins must be in 1..{re.shape[0]}, got {n_bins}"
        raise ValueError(msg)

    order = np.argsort(re, kind="stable")

    return tuple(
        ReBin(
            re_min=float(re[chunk].min()),
            re_max=float(re[chunk].max()),
            n_trajectories=int(chunk.size),
            spectrum=spectra[chunk].mean(axis=0),
        )
        for chunk in np.array_split(order, n_bins)
    )


def load_re_spectra(data_path: Path, max_traj_per_re: int, n_frames: int, n_bins: int, k_max: int) -> ReSpectra:
    """Read the dataset, spectrum-transform a sample of its trajectories, and bin by Re.

    Reads `max_traj_per_re` trajectories for each Re on the generator's ladder and
    `n_frames` evenly spaced frames from each, so the cost is bounded by the sample
    rather than by the (~30 GB) file. Spectra are truncated at `k_max`.
    """
    with h5py.File(data_path, "r") as f:
        re_all = f["re"][:]
        velocity = f["velocity"]
        n_saved = velocity.shape[1]
        frames = np.linspace(0, n_saved - 1, n_frames, dtype=int)

        sampled: list[int] = []
        for value in np.unique(re_all):
            sampled.extend(np.flatnonzero(re_all == value)[:max_traj_per_re].tolist())

        k = None
        spectra = []
        for i in sorted(sampled):
            # h5py fancy-indexes along one axis at a time; frames is sorted, so this is
            # a hyperslab read of just the sampled frames rather than the whole trajectory.
            k, per_frame = radial_energy_spectrum(velocity[i, frames])
            spectra.append(per_frame.mean(axis=0))

    if k is None:
        msg = f"no trajectories found in {data_path}"
        raise ValueError(msg)

    re = re_all[sorted(sampled)]
    bins = bin_by_reynolds(re, np.stack(spectra), n_bins)

    # k=0 is the domain-mean flow (zero by construction) and has no place on a log axis.
    # The upper cut is the solver's 2/3-dealiasing limit: above it the energy is the
    # filter's, not the flow's, and any Re trend read there is a numerical artifact.
    keep = slice(1, k_max + 1)

    return ReSpectra(
        k=k[keep],
        bins=tuple(ReBin(b.re_min, b.re_max, b.n_trajectories, b.spectrum[keep]) for b in bins),
    )


def plot_re_spectra(spectra: ReSpectra, out_path: Path) -> None:
    """Two panels: E(k) per Reynolds bin, and each bin relative to the lowest-Re bin.

    The ratio panel carries the figure. On the raw log-log axis the bins differ by at
    most a factor of two while the spectrum itself falls through six decades, so the
    curves lie on top of one another and the trend is unreadable; dividing by the
    lowest-Re bin puts that factor on a linear axis where it is plain.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("plasma")
    n = len(spectra.bins)
    colours = [cmap(0.1 + 0.75 * i / max(n - 1, 1)) for i in range(n)]
    labels = [f"Re {b.re_min:.0f}-{b.re_max:.0f} ({b.n_trajectories} traj.)" for b in spectra.bins]

    fig, (ax_e, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6))

    for b, colour, label in zip(spectra.bins, colours, labels, strict=True):
        positive = b.spectrum > 0
        ax_e.plot(spectra.k[positive], b.spectrum[positive], color=colour, lw=1.8, label=label)

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("wavenumber $k$")
    ax_e.set_ylabel("energy $E(k)$")
    ax_e.set_title("Energy spectrum (trajectory-averaged)")
    ax_e.legend(fontsize=8)
    ax_e.grid(True, which="both", alpha=0.3)

    reference = spectra.bins[0].spectrum
    for b, colour, label in zip(spectra.bins, colours, labels, strict=True):
        ax_r.plot(spectra.k, b.spectrum / reference, color=colour, lw=1.8, label=label)

    ax_r.axhline(1.0, color="0.4", ls="--", lw=1)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("wavenumber $k$")
    ax_r.set_ylabel(r"$E(k)\,/\,E_{\mathrm{lowest\ Re}}(k)$")
    ax_r.set_title("Relative to the lowest-Re bin")
    ax_r.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Energy spectrum of the Kolmogorov dataset, binned by Reynolds number")
    p.add_argument("--data-path", type=Path, required=True, help="HDF5 file produced by dg-kolmogorov")
    p.add_argument("--max-traj-per-re", type=int, default=6, help="Trajectories sampled per Re on the ladder")
    p.add_argument("--n-frames", type=int, default=3, help="Evenly spaced frames sampled per trajectory")
    p.add_argument("--n-bins", type=int, default=4, help="Number of equal-count Reynolds bins")
    p.add_argument(
        "--k-max",
        type=int,
        default=85,
        help="Highest wavenumber plotted; defaults to the solver's 2/3-dealiasing limit (2/3 * 256/2)",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/analysis_re/re_spectrum.png"))
    args = p.parse_args(argv)

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spectra = load_re_spectra(args.data_path, args.max_traj_per_re, args.n_frames, args.n_bins, args.k_max)
    plot_re_spectra(spectra, out_path)

    lo, hi = spectra.bins[0], spectra.bins[-1]
    print(f"Sampled {sum(b.n_trajectories for b in spectra.bins)} trajectories across {len(spectra.bins)} Re bins")
    print(f"High-Re ({hi.re_min:.0f}-{hi.re_max:.0f}) / low-Re ({lo.re_min:.0f}-{lo.re_max:.0f}) energy ratio:")

    for k in (10, 20, 40, 80):
        hits = np.flatnonzero(spectra.k == k)
        if hits.size:
            j = int(hits[0])
            print(f"  k={k:3d}: {hi.spectrum[j] / lo.spectrum[j]:5.2f}x")

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
