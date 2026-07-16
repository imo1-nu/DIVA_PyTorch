"""
Formant error analysis utilities.

The DIVA model outputs auditory errors in normalised units (0–1 per dimension).
Scale factors from vocaltract_args.AuditoryArgs.Scale convert to Hz:
  F0=100, F1=500, F2=1500, F3=3000  [Hz]
"""

import numpy as np

AUD_SCALE_HZ = np.array([100.0, 500.0, 1500.0, 3000.0])


def aud_err_to_hz(aud_err_log: list) -> np.ndarray:
    """
    Convert a list of normalised auditory error vectors to Hz.

    Parameters
    ----------
    aud_err_log : list of array (4,)
        Normalised auditory error at each timestep.

    Returns
    -------
    err_hz : np.ndarray, shape (T, 4)
        Signed error in Hz for F0, F1, F2, F3.
    """
    arr = np.array(aud_err_log)  # (T, 4)
    return arr * AUD_SCALE_HZ


def summarise_errors(aud_err_log: list) -> dict:
    """
    Compute summary statistics (mean, max absolute) per formant in Hz.

    Returns dict with keys 'mean_abs_hz' and 'max_abs_hz', each array (4,).
    """
    err_hz = aud_err_to_hz(aud_err_log)
    abs_err = np.abs(err_hz)
    return {
        'mean_abs_hz': abs_err.mean(axis=0),
        'max_abs_hz':  abs_err.max(axis=0),
    }


def peak_f2_error_hz(aud_err_log: list) -> float:
    """Return peak absolute F2 error across all timesteps (Hz)."""
    err_hz = aud_err_to_hz(aud_err_log)
    return float(np.abs(err_hz[:, 2]).max())
