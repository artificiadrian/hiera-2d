import argparse
from pathlib import Path

import exponax as ex
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate Kolmogorov flow dataset")
    p.add_argument("--grid-size", "-N", type=int, default=256)
    p.add_argument("--re", type=float, default=3000, help="Reynolds number (viscosity = 1/Re)")
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--num-seeds", "-S", type=int, default=400)
    p.add_argument("--collect", type=int, default=100, help="Number of frames to collect per seed")
    p.add_argument("--output-dt", type=float, default=0.01, help="Physical time between output frames")
    p.add_argument("--burn-time", type=float, default=5.0, help="Physical burn-in time (seconds)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--output", "-o", type=str, default=None)
    args = p.parse_args()

    # ── Derived quantities ────────────────────────────────────────
    args.viscosity = 1.0 / args.re
    args.subsample = round(args.output_dt / args.dt)
    args.burn_in = round(args.burn_time / args.dt)

    return args


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


def main():
    args = parse_args()
    N = args.grid_size
    filename = args.output or f"kolmogorov2d_{N}_s{args.num_seeds}.h5"

    stepper = make_stepper(N, args.viscosity, args.dt)
    vort_to_vel = jax.jit(make_vort_to_vel(N))
    ic_gen = ex.ic.RandomTruncatedFourierSeries(num_spatial_dims=2, cutoff=5)

    burn_step = jax.jit(ex.repeat(stepper, args.burn_in))
    advance_step = ex.repeat(stepper, args.subsample)
    collect_rollout = jax.jit(ex.rollout(advance_step, args.collect))

    # ── Split indices ────────────────────────────────────────────
    idx = np.random.RandomState(42).permutation(args.num_seeds)
    nt = int(args.num_seeds * args.train_frac)
    splits = {"train": idx[:nt], "val": idx[nt:]}

    # Reverse mapping: seed_idx -> (split_name, position)
    seed_to_split = {}
    for name, si in splits.items():
        for pos, seed_idx in enumerate(si):
            seed_to_split[int(seed_idx)] = (name, pos)

    # ── Generate & save incrementally ────────────────────────────
    with h5py.File(filename, "w") as f:
        dsets = {}
        for name, si in splits.items():
            dsets[name] = f.create_dataset(
                f"{name}/velocity",
                shape=(len(si), args.collect, 2, N, N),
                dtype=np.float32,
                chunks=(1, args.collect, 2, N, N),
                compression="gzip",
                compression_opts=4,
            )

        for seed_idx in tqdm(range(args.num_seeds), desc="Simulating"):
            key = jax.random.PRNGKey(seed_idx)
            w = ic_gen(num_points=N, key=key)

            w = burn_step(w)

            # collect all frames in one JIT'd scan
            w_traj = collect_rollout(w)  # (collect, 1, N, N)
            vel = np.array(jax.vmap(vort_to_vel)(w_traj), dtype=np.float32)  # (collect, 2, N, N)
            split_name, pos = seed_to_split[seed_idx]
            dsets[split_name][pos] = vel

        # ── Training stats from disk ─────────────────────────────
        tv = dsets["train"][:]
        for k, v in {
            "re": args.re,
            "viscosity": args.viscosity,
            "grid_size": N,
            "dt": args.dt,
            "output_dt": args.output_dt,
            "burn_time": args.burn_time,
            "collect": args.collect,
            "u_mean": tv[:, :, 0].mean(),
            "u_std": tv[:, :, 0].std(),
            "v_mean": tv[:, :, 1].mean(),
            "v_std": tv[:, :, 1].std(),
        }.items():
            f.attrs[k] = float(v)

    for name, si in splits.items():
        print(f"  {name:5s}: {len(si)} seeds")

    print(f"Saved {filename} ({Path(filename).stat().st_size / 1e6:.1f} MB)")

    # ── Summary stats ─────────────────────────────────────────────
    with h5py.File(filename, "r") as f:
        print("\n── Summary statistics ──")
        for k, v in f.attrs.items():
            print(f"  {k:12s}: {v:.6g}")

        train_vel = f["train/velocity"][:]  # (n_train, collect, 2, N, N)
        speed = np.sqrt(train_vel[:, :, 0] ** 2 + train_vel[:, :, 1] ** 2)
        print(f"\n  Speed  — min: {speed.min():.4f}  max: {speed.max():.4f}  mean: {speed.mean():.4f}")
        print(f"  u-vel  — min: {train_vel[:,:,0].min():.4f}  max: {train_vel[:,:,0].max():.4f}")
        print(f"  v-vel  — min: {train_vel[:,:,1].min():.4f}  max: {train_vel[:,:,1].max():.4f}")

        # ── Sample images ─────────────────────────────────────────
        out_dir = Path(filename).with_suffix("") / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pick 4 random training samples, show first frame
        rng = np.random.RandomState(0)
        n_samples = min(4, train_vel.shape[0])
        sample_ids = rng.choice(train_vel.shape[0], n_samples, replace=False)

        for sid in sample_ids:
            u = train_vel[sid, 0, 0]  # first frame, u component
            v = train_vel[sid, 0, 1]  # first frame, v component
            spd = np.sqrt(u ** 2 + v ** 2)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, field, title in zip(axes, [u, v, spd], ["u", "v", "speed"]):
                im = ax.imshow(field, cmap="RdBu_r" if title != "speed" else "inferno", origin="lower")
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"Sample {sid}, frame 0", fontsize=13)
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{sid:03d}.png", dpi=150)
            plt.close(fig)

        print(f"\n  Saved {n_samples} sample images to {out_dir}/")


if __name__ == "__main__":
    main()
