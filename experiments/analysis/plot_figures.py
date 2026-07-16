"""Plotting utilities for Civier et al. (2010) reproduction figures."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_f2_error_timeseries(err_hz_dict: dict, out_path: str,
                             sample_rate_ms: int = 5) -> None:
    """
    Fig. 2 equivalent: F2 error time-series for two conditions.

    Parameters
    ----------
    err_hz_dict : {label: np.ndarray shape (T, 4)}
    """
    fig, axes = plt.subplots(1, len(err_hz_dict), figsize=(5 * len(err_hz_dict), 4),
                              sharey=True)
    if len(err_hz_dict) == 1:
        axes = [axes]
    for ax, (label, err_hz) in zip(axes, err_hz_dict.items()):
        t = np.arange(err_hz.shape[0]) * sample_rate_ms
        ax.plot(t, np.abs(err_hz[:, 2]), label='F2 error')
        ax.set_title(label)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('|F2 error| (Hz)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_repetition_counts(conditions: dict, out_path: str) -> None:
    """
    Fig. 3/4 equivalent: bar chart of repetitions per 100 syllables.

    Parameters
    ----------
    conditions : {label: float}  repetitions/100 syllables per condition
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = list(conditions.keys())
    values = [conditions[k] for k in labels]
    ax.bar(labels, values, color='steelblue')
    ax.set_ylabel('Repetitions per 100 syllables')
    ax.set_title('Sound/syllable repetitions by condition')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_repetitions_vs_noise(noise_levels: np.ndarray,
                               rep_counts: np.ndarray,
                               out_path: str) -> None:
    """
    Fig. 5 equivalent: repetitions per 100 syllables vs masking noise level.

    Parameters
    ----------
    noise_levels : shape (N,)
    rep_counts : shape (N,)  mean repetitions per 100 syllables
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(noise_levels, rep_counts, 'o-', color='darkorange')
    ax.set_xlabel('Masking noise level (I_masking)')
    ax.set_ylabel('Repetitions per 100 syllables')
    ax.set_title('Effect of masking noise on repetitions')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
