"""Temporal autocorrelation of Kolmogorov velocity fields.

How fast does the flow decorrelate in time? Saving every frame of a fine
simulation yields training samples that are near-duplicates: the model sees the
same state many times and the effective dataset is far smaller than the frame
count suggests. This module measures the autocorrelation C(tau) of the velocity
fluctuations and reports the e-folding time tau_e as an iid diagnostic of that
temporal redundancy.

tau_e is a diagnostic only: it is NOT a recommended save spacing. The
autoregressive model's save spacing is its prediction step and must be chosen
for learnability (a small Δt where C is still high), independently of tau_e.

The central figure for the thesis's iid section is C(tau) with a mean +/- std
band over trajectories, the 1/e threshold, and tau_e marked.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class AutocorrResult:
    """Trajectory-averaged temporal autocorrelation curve.

    `lags` is physical time (seconds), so `c_mean[i]` is the correlation at lag
    `lags[i]`. `c_std` is the spread across trajectories at each lag (the band).
    """

    lags: np.ndarray
    c_mean: np.ndarray
    c_std: np.ndarray


def _cross_correlation_over_time(fluctuations: np.ndarray) -> np.ndarray:
    """Unnormalized autocorrelation summed over the feature axis, per time lag.

    `fluctuations` is `(T, D)`. Returns `(T,)` where entry `tau` is
    `sum over t and features of f[t] * f[t + tau]`. Computed with a zero-padded
    FFT (linear, not circular, correlation) and by summing the per-feature power
    spectra before one inverse transform, which avoids materializing a `(T, D)`
    correlation array.
    """
    t = fluctuations.shape[0]
    n = 2 * t  # zero-pad so wrap-around does not contaminate the linear correlation

    spectrum = np.fft.rfft(fluctuations, n=n, axis=0)
    power = (spectrum.real**2 + spectrum.imag**2).sum(axis=1)  # sum over features
    return np.fft.irfft(power, n=n)[:t]


def temporal_autocorrelation(trajectories: np.ndarray, output_dt: float) -> AutocorrResult:
    """Temporal autocorrelation of velocity fluctuations, averaged over trajectories.

    `trajectories` is `(n_traj, T, C, H, W)`; `output_dt` is the physical time
    between consecutive frames. Per trajectory the time-mean field is subtracted
    so only fluctuations remain, then
    `C(tau) = mean_t <f'(t), f'(t+tau)> / mean_t <f'(t), f'(t)>` with the inner
    product taken over channels and space, giving `C(0) = 1`. `c_mean`/`c_std`
    are the mean and standard deviation of `C(tau)` across trajectories.

    Raises:
        ValueError: if `T < 2` (a single frame has no temporal structure).
    """
    traj = np.asarray(trajectories, dtype=np.float64)
    if traj.ndim != 5:
        msg = f"expected (n_traj, T, C, H, W), got shape {traj.shape}"
        raise ValueError(msg)

    n_traj, t, c, h, w = traj.shape
    if t < 2:
        msg = f"need at least 2 time steps to measure autocorrelation, got T={t}"
        raise ValueError(msg)

    per_traj = np.empty((n_traj, t))
    counts = t - np.arange(t)  # number of (t, t+tau) pairs contributing to each lag
    for i in range(n_traj):
        field = traj[i].reshape(t, c * h * w)

        # Subtract the time-mean field: Kolmogorov forcing at mode 4 sustains a
        # mean flow that would otherwise pin C(tau) high forever. We want the
        # decorrelation of the fluctuations about that mean, not of the mean.
        fluctuations = field - field.mean(axis=0, keepdims=True)

        cross = _cross_correlation_over_time(fluctuations)
        numerator = cross / counts  # mean_t <f'(t), f'(t+tau)>
        denominator = cross[0] / t  # mean_t <f'(t), f'(t)>
        per_traj[i] = numerator / denominator

    return AutocorrResult(
        lags=np.arange(t) * output_dt,
        c_mean=per_traj.mean(axis=0),
        c_std=per_traj.std(axis=0),
    )


def decorrelation_time(result: AutocorrResult, threshold: float = 1.0 / np.e) -> float | None:
    """Physical time lag at which C(tau) first drops to `threshold` (default 1/e).

    Returns `result.lags` at the first lag whose mean autocorrelation is
    `<= threshold`, or `None` if C never crosses `threshold` within the measured
    window.

    This is a diagnostic of within-trajectory temporal redundancy for the iid
    discussion, NOT a recommended save spacing. The AR save spacing should be
    chosen for learnability (a small Δt where C is still high), independently of
    this decorrelation time.
    """
    below = np.flatnonzero(result.c_mean <= threshold)
    if below.size == 0:
        return None

    return float(result.lags[below[0]])


def correlation_at(result: AutocorrResult, delta_t: float) -> float:
    """How correlated two frames `delta_t` apart are: C(tau) at lag `delta_t`.

    Linearly interpolates the mean autocorrelation curve at `delta_t`.
    """
    return float(np.interp(delta_t, result.lags, result.c_mean))


@dataclass(frozen=True, slots=True)
class TrajectoryData:
    """Velocity trajectories and the physical time between saved frames.

    `trajectories` is `(n_traj, T, 2, H, W)`; `output_dt` is the saved-frame
    interval read from the file's HDF5 attrs.
    """

    trajectories: np.ndarray
    output_dt: float


def load_trajectories(data_path: Path, max_traj: int) -> TrajectoryData:
    """Load up to `max_traj` velocity trajectories plus the saved-frame interval.

    Reads only the first `max_traj` trajectories directly from the HDF5 (a chunked
    hyperslab read), so it never pulls the whole dataset into RAM — the full file
    can be tens of GB. Any `max_traj` trajectories are representative for the
    decorrelation diagnostic, so the train/val split is irrelevant here.

    Reads the lag time-unit from the file's `output_dt` HDF5 attr (= dt x keep_every,
    the physical time between saved frames).

    Raises:
        ValueError: if `output_dt` is not stored.
    """
    with h5py.File(data_path, "r") as f:
        if "output_dt" not in f.attrs:
            msg = f"{data_path} has no 'output_dt' attr"
            raise ValueError(msg)

        output_dt = float(f.attrs["output_dt"])
        n = min(max_traj, f["velocity"].shape[0])
        trajectories = f["velocity"][:n]  # (n, T, 2, H, W); reads only n chunks

    return TrajectoryData(trajectories=trajectories, output_dt=output_dt)


def plot_autocorrelation(
    result: AutocorrResult,
    threshold: float,
    tau_e: float | None,
    out_path: Path,
) -> None:
    """Write the C(tau) figure: mean +/- std band, threshold line, tau_e marked."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.lags, result.c_mean, color="tab:blue", lw=2, label="C(tau)")
    ax.fill_between(
        result.lags,
        result.c_mean - result.c_std,
        result.c_mean + result.c_std,
        color="tab:blue",
        alpha=0.2,
        label="+/- 1 std over trajectories",
    )
    ax.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold = {threshold:.3f}")

    if tau_e is not None:
        ax.axvline(tau_e, color="tab:red", ls=":", lw=1.5, label=f"tau_e (1/e decorrelation) = {tau_e:.3g}")

    ax.set_xlabel("time lag tau (physical time)")
    ax.set_ylabel("temporal autocorrelation C(tau)")
    ax.set_title("Temporal autocorrelation of Kolmogorov velocity fluctuations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Temporal autocorrelation study for Kolmogorov flow")
    p.add_argument("--data-path", type=Path, required=True, help="HDF5 file produced by dg-kolmogorov")
    p.add_argument("--max-traj", type=int, default=4, help="Cap on how many train trajectories to average over")
    p.add_argument(
        "--delta-t",
        type=float,
        default=None,
        help="Candidate saved-frame spacing (physical time) to diagnose C at",
    )
    p.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/analysis_autocorr"))
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_trajectories(args.data_path, args.max_traj)
    print(f"Loaded {data.trajectories.shape[0]} trajectories from {args.data_path} (output_dt={data.output_dt:.4g})")

    result = temporal_autocorrelation(data.trajectories, data.output_dt)
    threshold = 1.0 / np.e
    tau_e = decorrelation_time(result, threshold)

    out_path = args.output_dir / "autocorrelation.png"
    plot_autocorrelation(result, threshold, tau_e, out_path)

    if tau_e is None:
        print("\nDecorrelation time (C->1/e): no 1/e crossing within the measured window")
    else:
        print(f"\nDecorrelation time (C->1/e): tau_e = {tau_e:.3g} s")

    print(
        "  (iid diagnostic of temporal redundancy, NOT the save spacing; "
        "choose the AR Δt as a small lag where C is still high)"
    )

    if args.delta_t is not None:
        c_dt = correlation_at(result, args.delta_t)
        if tau_e is None:
            print(f"at Δt={args.delta_t:.3g}s: C={c_dt:.2f}")
        else:
            print(
                f"at Δt={args.delta_t:.3g}s: C={c_dt:.2f}; "
                f"~{tau_e / args.delta_t:.1f} saved frames per decorrelation time"
            )

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
