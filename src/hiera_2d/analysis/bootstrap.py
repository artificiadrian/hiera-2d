from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Percentile-bootstrap point estimate with a two-sided CI.

    On a paired-difference sample, `significant` reports whether the interval
    excludes 0 (i.e. the two models differ at the chosen alpha).
    """

    mean: float
    lo: float
    hi: float

    @property
    def significant(self):
        return self.lo > 0 or self.hi < 0


def bootstrap_ci(samples: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    """Percentile bootstrap of the mean over the first axis of `samples`."""
    n = samples.shape[0]
    boot_means = samples[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(float(samples.mean()), float(lo), float(hi))
