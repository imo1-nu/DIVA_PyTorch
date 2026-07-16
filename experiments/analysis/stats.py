"""Statistical helpers for experiment results."""

import numpy as np
from scipy import stats


def mean_sem(data: np.ndarray):
    """Return (mean, SEM) of 1-D array."""
    return data.mean(), stats.sem(data)


def bootstrap_ci(data: np.ndarray, n_boot: int = 2000,
                 ci: float = 0.95, rng=None) -> tuple:
    """Return (lower, upper) bootstrap confidence interval of the mean."""
    if rng is None:
        rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_means, 100 * (1 - (1 - ci) / 2))
    return lo, hi


def repetitions_per_100(repairs: np.ndarray) -> float:
    """Convert repair counts per trial to repetitions per 100 syllables."""
    return float(repairs.mean() * 100)
