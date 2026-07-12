import argparse
import os
from pathlib import Path

# Must precede the jax import. JAX otherwise preallocates 75% of VRAM into its
# arena, while this generator's working set is ~200 MiB. On a GPU that also drives
# a desktop that leaves too little outside the arena for cuFFT to build its plans,
# and plan creation fails with CUFFT_INTERNAL_ERROR partway through a long run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import exponax as ex
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from tqdm import tqdm

from hiera_2d.analysis.spectra import vorticity


def save_trajectory_gif(vort_frames: np.ndarray, out_path: Path, fps: int = 10) -> None:
    """Animate a (T, H, W) vorticity field over all frames.

    Human stationarity check: eyeball whether the last frame is still developed
    turbulence rather than a decayed/laminar state.
    """
    vlim = float(np.abs(vort_frames).max()) or 1.0

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(vort_frames[0], cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="lower", animated=True)
    ax.set_xticks([])
    ax.set_yticks([])
    txt = fig.suptitle("frame 1")
    fig.tight_layout()

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out_path), dpi=90):
        for t in range(vort_frames.shape[0]):
            im.set_data(vort_frames[t])
            txt.set_text(f"frame {t + 1}")
            writer.grab_frame()

    plt.close(fig)


def save_stationarity_panel(vort_frames: np.ndarray, out_path: Path) -> None:
    """First / middle / last vorticity frames side by side as a static PNG.

    Single-frame trajectories reuse the one frame in all three positions.
    """
    t = vort_frames.shape[0]
    idx = [0, t // 2, t - 1]
    labels = ["first", "middle", "last"]

    # Shared symmetric scale across the three panels, so the colours are comparable
    # frame-to-frame (the whole point of the stationarity check).
    vlim = float(np.abs(vort_frames).max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, i, label in zip(axes, idx, labels, strict=True):
        im = ax.imshow(vort_frames[i], cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="lower")
        ax.set_title(f"{label} (frame {i + 1})")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label(r"vorticity $\omega$ [1/s]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate a diverse Kolmogorov flow dataset (varied Re + burn-in)")
    p.add_argument("--grid-size", type=int, required=True, help="spatial resolution; domain is grid_size x grid_size")
    p.add_argument("--re-min", type=float, required=True, help="minimum Reynolds number of the sampled family")
    p.add_argument("--re-max", type=float, required=True, help="maximum Reynolds number of the sampled family")
    p.add_argument(
        "--n-re-values",
        type=int,
        required=True,
        help="number of DISTINCT Re values sampled evenly across [re-min, re-max] (each compiled once)",
    )
    p.add_argument("--dt", type=float, required=True, help="solver timestep (numerical integration step)")
    p.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="number of independent trajectories to simulate (pool size)",
    )
    p.add_argument("--collect", type=int, required=True, help="frames saved per trajectory")
    p.add_argument(
        "--keep-every",
        type=int,
        required=True,
        help="solver steps between saved frames (saved spacing = dt x keep-every)",
    )
    p.add_argument(
        "--burn-min",
        type=float,
        required=True,
        help="minimum physical burn-in time discarded before collecting (per trajectory)",
    )
    p.add_argument(
        "--burn-max",
        type=float,
        required=True,
        help="maximum physical burn-in time discarded before collecting (per trajectory)",
    )
    p.add_argument("--output", "-o", type=Path, required=True, help="output HDF5 path")
    args = p.parse_args(argv)

    if args.keep_every < 1:
        msg = f"--keep-every must be >= 1, got {args.keep_every}"
        raise ValueError(msg)

    if args.collect < 1:
        msg = f"--collect must be >= 1, got {args.collect}"
        raise ValueError(msg)

    if args.grid_size <= 0:
        msg = f"--grid-size must be > 0, got {args.grid_size}"
        raise ValueError(msg)

    if args.num_seeds <= 0:
        msg = f"--num-seeds must be > 0, got {args.num_seeds}"
        raise ValueError(msg)

    if args.n_re_values < 1:
        msg = f"--n-re-values must be >= 1, got {args.n_re_values}"
        raise ValueError(msg)

    if args.re_min > args.re_max:
        msg = f"--re-min must be <= --re-max, got {args.re_min} > {args.re_max}"
        raise ValueError(msg)

    if args.burn_min > args.burn_max:
        msg = f"--burn-min must be <= --burn-max, got {args.burn_min} > {args.burn_max}"
        raise ValueError(msg)

    return args


def plan_trajectory_params(
    num_seeds: int,
    re_min: float,
    re_max: float,
    n_re_values: int,
    burn_min: float,
    burn_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically assign a Reynolds number and burn-in time to each trajectory.

    Re values form an evenly spaced grid of ``n_re_values`` points over
    ``[re_min, re_max]`` (collapsing to ``re_min`` when the grid is degenerate);
    trajectory ``i`` gets ``re_grid[i % n_re_values]`` so any contiguous or
    strided index subset stays Re-balanced. Burn-in is drawn continuously and
    uniformly from ``[burn_min, burn_max]``.

    Returns two float64 arrays of length ``num_seeds``, aligned to trajectory index.
    """
    rng = np.random.default_rng(0)
    re_grid = np.linspace(re_min, re_max, n_re_values, dtype=np.float64)
    re_per_traj = re_grid[np.arange(num_seeds) % n_re_values]
    burn_per_traj = rng.uniform(burn_min, burn_max, size=num_seeds)

    return re_per_traj, burn_per_traj


def burn_iters_for(burn_time: float, output_dt: float) -> int:
    """Number of ``output_dt``-sized advance steps that discard ``burn_time`` of physical time."""
    return round(burn_time / output_dt)


MAX_HALVINGS = 2
"""Times a diverged trajectory is retried at half the timestep before giving up."""


def refinement_ladder(dt: float, keep_every: int, max_halvings: int) -> tuple[tuple[float, int], ...]:
    """Successive ``(dt, keep_every)`` attempts, each halving dt and doubling keep_every.

    Their product — the saved-frame spacing, which is the step the AR model learns
    to predict across — is invariant along the ladder. So a trajectory retried at
    a finer timestep is integrated more accurately but saved at exactly the same
    physical spacing as every other trajectory in the dataset.

    Divergence is a solver failure, not a physical one: it selects for the
    trajectories with the sharpest small-scale structure. Refining recovers those
    trajectories rather than discarding them, which would censor the high-wavenumber
    tail of the dataset's energy spectrum.
    """
    return tuple((dt / 2**i, keep_every * 2**i) for i in range(max_halvings + 1))


def make_stepper(N, viscosity, dt):
    return ex.stepper.KolmogorovFlowVorticity(
        num_spatial_dims=2,
        domain_extent=float(2 * jnp.pi),
        num_points=N,
        dt=dt,
        diffusivity=viscosity,
        drag=-0.1,
        injection_mode=4,
        injection_scale=1.0,
    )


def make_rollout(N, re_value, dt, keep_every, collect):
    """JIT'd (single-saved-frame advance, full-trajectory collect) pair for one Re and timestep."""
    stepper = make_stepper(N, 1.0 / re_value, dt)
    advance = jax.jit(ex.repeat(stepper, keep_every))  # keep_every solver steps per saved frame
    return advance, jax.jit(ex.rollout(advance, collect))


def make_vort_to_vel(N):
    kx = jnp.fft.fftfreq(N, d=1.0 / N)
    ky = jnp.fft.rfftfreq(N, d=1.0 / N)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K2 = (KX**2 + KY**2).at[0, 0].set(1.0)

    def vort_to_vel(w):
        """(1,N,N) vorticity → (2,N,N) velocity"""
        w_hat = jnp.fft.rfftn(w[0])
        psi = -w_hat / K2
        return jnp.stack(
            [
                jnp.fft.irfftn(1j * KY * psi, s=(N, N)),
                jnp.fft.irfftn(-1j * KX * psi, s=(N, N)),
            ]
        )

    return vort_to_vel


def simulate_trajectory(seed_idx, burn_time, output_dt, N, ic_gen, advance, collect_rollout, vort_to_vel):
    """Burn in from seed `seed_idx`'s random IC, then collect the saved frames as velocity.

    Returns `(collect, 2, N, N)` float32. The caller checks finiteness: a diverged
    run yields non-finite values rather than raising.
    """
    w = ic_gen(num_points=N, key=jax.random.PRNGKey(seed_idx))

    for _ in range(burn_iters_for(burn_time, output_dt)):
        w = advance(w)

    w_traj = collect_rollout(w)  # (collect, 1, N, N), one JIT'd scan
    return np.array(jax.vmap(vort_to_vel)(w_traj), dtype=np.float32)


def main():
    args = parse_args()
    N = args.grid_size
    filename = args.output

    vort_to_vel = jax.jit(make_vort_to_vel(N))
    ic_gen = ex.ic.RandomTruncatedFourierSeries(num_spatial_dims=2, cutoff=5)

    output_dt = args.dt * args.keep_every  # physical time between saved frames

    re_per_traj, burn_per_traj = plan_trajectory_params(
        args.num_seeds,
        args.re_min,
        args.re_max,
        args.n_re_values,
        args.burn_min,
        args.burn_max,
    )

    # Solver timestep each trajectory actually integrated with: `args.dt` unless it
    # diverged and was refined. Saved as provenance; `output_dt` is the same for all.
    dt_per_traj = np.full(args.num_seeds, args.dt, dtype=np.float64)

    # Streaming stat accumulators (float64). The dataset is far too large to load
    # for stats — a 625-seed run is ~33 GB — so we fold each trajectory in as it
    # is generated rather than reading it back with dset[:].
    n_elem = 0
    u_sum = v_sum = 0.0
    u_sqsum = v_sqsum = 0.0
    speed_sum = 0.0
    u_min = v_min = speed_min = np.inf
    u_max = v_max = speed_max = -np.inf

    # ── Generate & save ──────────────────────────────────────────
    # Ctrl+C aborts and deletes the incomplete file (no partial dataset).
    try:
        with h5py.File(filename, "w") as f:
            dset = f.create_dataset(
                "velocity",
                shape=(args.num_seeds, args.collect, 2, N, N),
                dtype=np.float32,
                chunks=(1, args.collect, 2, N, N),
                compression="gzip",
                compression_opts=4,
            )

            # Compile the stepper once per distinct Re (~n_re_values compiles
            # total), then reuse it for every trajectory sharing that Re. Burn-in
            # varies per trajectory but is applied by repeatedly calling the same
            # JIT'd `advance` (no recompile per trajectory).
            ladder = refinement_ladder(args.dt, args.keep_every, MAX_HALVINGS)

            with tqdm(total=args.num_seeds, desc="Simulating") as pbar:
                for re_value in np.unique(re_per_traj):
                    base_rollout = make_rollout(N, float(re_value), args.dt, args.keep_every, args.collect)

                    for seed_idx in np.nonzero(re_per_traj == re_value)[0]:
                        seed_idx = int(seed_idx)
                        burn_time = float(burn_per_traj[seed_idx])

                        # `dt` is marginally stable near the CFL limit, so whether a
                        # trajectory diverges depends on its own vorticity extrema.
                        # Refine that trajectory alone until it integrates cleanly;
                        # output_dt is invariant along the ladder, so its saved frames
                        # stay aligned with every other trajectory's.
                        for attempt, (dt, keep_every) in enumerate(ladder):
                            advance, collect_rollout = (
                                base_rollout
                                if attempt == 0
                                else make_rollout(N, float(re_value), dt, keep_every, args.collect)
                            )
                            vel = simulate_trajectory(
                                seed_idx, burn_time, output_dt, N, ic_gen, advance, collect_rollout, vort_to_vel
                            )

                            if np.isfinite(vel).all():
                                break

                            if attempt == len(ladder) - 1:
                                msg = (
                                    f"Simulation diverged (non-finite values) at seed {seed_idx}, "
                                    f"Re={float(re_value):.1f}, even after {MAX_HALVINGS} halvings "
                                    f"down to dt={dt:g}. Lower --dt and retry."
                                )
                                raise FloatingPointError(msg)

                            tqdm.write(
                                f"[retry] seed {seed_idx} (Re={float(re_value):.1f}) diverged at dt={dt:g}; "
                                f"retrying at dt={dt / 2:g}, keep_every={keep_every * 2}"
                            )

                        dt_per_traj[seed_idx] = dt
                        dset[seed_idx] = vel

                        # Fold this trajectory into the running stats (never reload it).
                        u_ch = vel[:, 0].astype(np.float64)
                        v_ch = vel[:, 1].astype(np.float64)
                        speed = np.sqrt(u_ch**2 + v_ch**2)
                        n_elem += u_ch.size
                        u_sum += u_ch.sum()
                        v_sum += v_ch.sum()
                        u_sqsum += np.square(u_ch).sum()
                        v_sqsum += np.square(v_ch).sum()
                        speed_sum += speed.sum()
                        u_min = min(u_min, float(u_ch.min()))
                        u_max = max(u_max, float(u_ch.max()))
                        v_min = min(v_min, float(v_ch.min()))
                        v_max = max(v_max, float(v_ch.max()))
                        speed_min = min(speed_min, float(speed.min()))
                        speed_max = max(speed_max, float(speed.max()))
                        pbar.update(1)

            # ── Global normalization stats (from streaming accumulators) ──
            # Global (split-independent) stats: computing them over the full pool
            # rather than a train split is a weak, acceptable leak.
            u_mean = u_sum / n_elem
            v_mean = v_sum / n_elem
            u_std = np.sqrt(max(u_sqsum / n_elem - u_mean**2, 0.0))
            v_std = np.sqrt(max(v_sqsum / n_elem - v_mean**2, 0.0))

            # Per-trajectory provenance (aligned to trajectory index): the Re, burn-in
            # and solver timestep each trajectory was actually generated with. Viscosity
            # is derivable as 1/re, so it is not stored.
            f.create_dataset("re", data=re_per_traj.astype(np.float64))
            f.create_dataset("burn_time", data=burn_per_traj.astype(np.float64))
            f.create_dataset("solver_dt", data=dt_per_traj)

            for k, val in {
                "grid_size": N,
                "dt": args.dt,
                "keep_every": args.keep_every,
                "output_dt": output_dt,
                "collect": args.collect,
                "re_min": args.re_min,
                "re_max": args.re_max,
                "burn_min": args.burn_min,
                "burn_max": args.burn_max,
                "n_re_values": args.n_re_values,
                "u_mean": u_mean,
                "u_std": u_std,
                "v_mean": v_mean,
                "v_std": v_std,
            }.items():
                f.attrs[k] = float(val)
    except KeyboardInterrupt:
        filename.unlink(missing_ok=True)
        print("\nInterrupted — deleted incomplete dataset.")
        raise SystemExit(130) from None
    except FloatingPointError as e:
        filename.unlink(missing_ok=True)
        print(f"\n{e}")
        raise SystemExit(1) from None
    except BaseException:
        # Catch-all so no failure mode leaves a truncated dataset on disk: a
        # JAX/CUDA runtime error mid-run would otherwise pass through untouched.
        # Re-raised unchanged — the traceback is the diagnostic.
        filename.unlink(missing_ok=True)
        print("\nFailed — deleted incomplete dataset.")
        raise

    print(f"Saved {filename} ({Path(filename).stat().st_size / 1e6:.1f} MB)")

    refined = np.flatnonzero(dt_per_traj < args.dt)
    if refined.size:
        print(f"\n{refined.size}/{args.num_seeds} trajectories were refined below --dt after diverging:")
        for seed_idx in refined:
            print(f"  seed {int(seed_idx):4d}: dt={dt_per_traj[seed_idx]:g}  (Re={re_per_traj[seed_idx]:.1f})")

    # ── Summary stats (from the streaming accumulators; no full reload) ──
    print("\n── Summary statistics ──")
    with h5py.File(filename, "r") as f:
        for k, val in f.attrs.items():
            print(f"  {k:12s}: {val:.6g}")

        print(f"\n  Speed  — min: {speed_min:.4f}  max: {speed_max:.4f}  mean: {speed_sum / n_elem:.4f}")
        print(f"  u-vel  — min: {u_min:.4f}  max: {u_max:.4f}")
        print(f"  v-vel  — min: {v_min:.4f}  max: {v_max:.4f}")

        # ── Stationarity visualizations ───────────────────────────
        # Per sampled seed: a GIF over all saved frames + a first/mid/last
        # vorticity panel, so a human can confirm the last saved frame is still
        # developed turbulence. Read only the sampled trajectories, never the
        # whole dataset. GIF writing is I/O-heavy, so cap at 4 seeds.
        out_dir = Path(filename).with_suffix("") / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(0)
        n_samples = min(4, args.num_seeds)
        sample_ids = rng.choice(args.num_seeds, n_samples, replace=False)

        for sid in sample_ids:
            vel_sample = f["velocity"][sid]  # one trajectory (collect, 2, N, N)
            vort_frames = vorticity(vel_sample)  # → (collect, N, N)
            save_trajectory_gif(vort_frames, out_dir / f"sample_{sid:03d}.gif")
            save_stationarity_panel(vort_frames, out_dir / f"sample_{sid:03d}_firstmidlast.png")

        print(f"\n  Saved stationarity viz for {n_samples} samples to {out_dir}/")


if __name__ == "__main__":
    main()
