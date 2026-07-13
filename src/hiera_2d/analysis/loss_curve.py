"""Training/validation loss curve for a run, read from its TensorBoard log.

A log-scaled loss axis is the reflex choice, but it is the wrong one here: the
MAE loss is not power-law-like, so log axes compress the part a reader cares
about (does it converge, and to what?) into a featureless downward line. This
module plots a linear axis instead, and recovers the one thing linear axes lose
-- the slow converged tail -- with an inset zoom rather than by distorting the
whole figure.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class LossHistory:
    """Per-epoch loss curves for one run.

    `epochs` is 0-based. `warmup_epochs` is the number of LR-warmup epochs,
    recovered from the logged learning rate (the LR rises to its peak during
    warmup, so the peak marks the last warmup epoch); it is `None` if the run
    did not log an `lr` scalar.
    """

    epochs: np.ndarray
    train: np.ndarray
    val: np.ndarray
    warmup_epochs: int | None


def load_loss_history(run_dir: Path) -> LossHistory:
    """Read the `loss/train`, `loss/val` and `lr` scalars from a run's event file.

    `run_dir` is a training run directory (the one holding `events.out.tfevents.*`),
    e.g. `outputs/kg_scaling/N250/mae_e120`.

    Raises:
        ValueError: if the run logged no `loss/train` or `loss/val` scalars.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]

    missing = {"loss/train", "loss/val"} - set(tags)
    if missing:
        msg = f"{run_dir} has no {sorted(missing)} scalars (found: {sorted(tags)})"
        raise ValueError(msg)

    train = np.array([e.value for e in accumulator.Scalars("loss/train")])
    val = np.array([e.value for e in accumulator.Scalars("loss/val")])

    warmup_epochs = None
    if "lr" in tags:
        lr = np.array([e.value for e in accumulator.Scalars("lr")])
        # The LR climbs through warmup and the peak is the FIRST epoch at full LR, i.e.
        # the first epoch after warmup -- so the peak's index IS the warmup length, and
        # adding one would count that post-warmup epoch as part of it.
        warmup_epochs = int(np.argmax(lr))

    return LossHistory(
        epochs=np.arange(len(val)),
        train=train,
        val=val,
        warmup_epochs=warmup_epochs,
    )


def tail_start(val: np.ndarray, factor: float = 2.0) -> int:
    """First epoch from which the validation loss is within `factor` x its final value.

    Defines the "converged tail" that the inset zooms into. Data-driven rather
    than a hand-picked epoch so the figure stays correct if the run length or
    schedule changes.
    """
    within = np.flatnonzero(val <= factor * val[-1])
    return int(within[0]) if within.size else 0


def plot_loss_curve(history: LossHistory, out_path: Path, title: str = "") -> None:
    """Write the loss figure: linear axes, warmup shaded, converged tail inset.

    Deliberately spare on text: the figure lives next to a caption in the write-up, so
    the title is empty by default and the numbers (best loss, epoch) belong in the
    caption rather than annotated onto the axes.
    """
    import matplotlib.pyplot as plt

    best = int(np.argmin(history.val))
    start = tail_start(history.val)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(history.epochs, history.train, color="tab:blue", lw=1.8, label="training")
    ax.plot(history.epochs, history.val, color="tab:orange", lw=1.8, label="validation")

    if history.warmup_epochs:
        ax.axvspan(0, history.warmup_epochs, color="gray", alpha=0.12, label="LR warmup")

    # The loss is MSE against per-patch-normalized targets, so it is dimensionless and
    # 1.0 is the trivial baseline: predicting each patch's own mean. Saying so on the
    # axis is what makes 0.033 legible as "3% of the per-patch variance is left".
    ax.set_xlim(0, history.epochs[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE\n(per-patch normalized; 1.0 = predicting the patch mean)")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    # Sits below the legend (upper right) and above the converged tail it magnifies.
    axins = ax.inset_axes((0.45, 0.28, 0.52, 0.50))
    axins.plot(history.epochs[start:], history.train[start:], color="tab:blue", lw=1.5)
    axins.plot(history.epochs[start:], history.val[start:], color="tab:orange", lw=1.5)
    axins.plot(best, history.val[best], "o", color="tab:red", ms=5, zorder=5)
    axins.margins(y=0.12)
    axins.set_xlim(start, history.epochs[-1])
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="gray")

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Plot the loss curve of a training run")
    p.add_argument("run_dir", type=Path, help="Run directory holding the TensorBoard event file")
    p.add_argument("--title", default="", help="Axes title; empty (the default) for a figure that carries a caption")
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/analysis_loss/loss_curve.png"))
    args = p.parse_args(argv)

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = load_loss_history(args.run_dir)
    plot_loss_curve(history, out_path, args.title)

    best = int(np.argmin(history.val))
    print(f"{len(history.epochs)} epochs; best val loss {history.val[best]:.4f} at epoch {best}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
