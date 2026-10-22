"""Independent MATLAB-formula reference from DIVA-FORWARD-01.
Derived from stutter_tts_jp research/32-diva-forward-audit,
scripts/investigate_diva_forward.py. Does not import the observation module.
MATLAB/Octave execution is not claimed.
"""
import numpy as np
from scipy import signal

def matlab_xy2ab(outline):
    """diva_synth.m:236–318, MATLAB indices/rounding and missing bins explicit."""
    rotated = np.asarray(outline).reshape(-1) * np.exp(-1j * np.pi / 12)
    x, y = rotated.real, rotated.imag
    a, b = np.zeros(len(x)), np.zeros(len(x))
    below, left, bend = y < -100, x < 45, (y >= -100) & (x >= 45)
    a[below], b[below] = y[below] + 100, x[below] - 45
    a[left], b[left] = np.pi * 30 + 45 - x[left], y[left] + 100
    z = x[bend] - 45 + 1j * (y[bend] + 100)
    a[bend], b[bend] = 60 * np.angle(z), np.abs(z)
    amin, amax, d = -60, 220, .075

    def accumulate(indices, bins, upper, minimum):
        position = np.clip(bins(a[indices] - amin), 1, upper).astype(int) - 1
        out = np.full(upper, np.nan)
        for i, value in zip(position, b[indices]):
            out[i] = np.fmin(out[i], value) if minimum else np.fmax(out[i], value)
        return out

    outer, inner, inner_wall = np.arange(29, 164), np.arange(163, 302), np.arange(173, 257)
    outer_wall = accumulate(outer, lambda v: np.ceil(27 * v / amax), 27, True)
    inner_wall_values = accumulate(inner_wall, lambda v: np.ceil(27 * v / amax), 27, False)
    groups = (outer_wall[6:24] - inner_wall_values[6:24]).reshape((3, 6), order="F")
    mind = np.fmin.reduce(groups, axis=0)
    sc = d * np.r_[mind[:4], np.fmin.reduce(mind[4:]), np.min(b[29:45]) - np.max(b[256:302])]
    # All selected round arguments are positive; floor(v+.5) is MATLAB round.
    ab1 = accumulate(outer, lambda v: np.floor(v + .5), amax, True)
    ab2 = accumulate(inner, lambda v: np.floor(v + .5), amax, False)
    for _ in range(2):
        ab1[1:-1] = np.fmin(np.fmin(ab1[1:-1], ab1[:-2]), ab1[2:])
        ab2[1:-1] = np.fmax(np.fmax(ab2[1:-1], ab2[:-2]), ab2[2:])
    selected = (ab1 > 0) & (ab2 > 0)
    opening = d * (ab1[selected] - ab2[selected])
    half = .5 * (1 - np.cos(2 * np.pi * np.arange(1, 27) / 52))
    hann = np.r_[half, half[:-1][::-1]]
    hann /= hann.sum()
    alpha = np.convolve(np.pad(np.ones(220), 25, mode="edge"), hann, mode="valid")
    beta = np.convolve(np.pad(np.r_[np.full(60, .25), np.full(160, 1.25)], 25, mode="edge"), hann, mode="valid")
    positive = np.flatnonzero(opening > 0)
    start = positive[0] if len(positive) else len(opening)
    area = (np.minimum(0, opening) + alpha[selected] * np.maximum(0, opening) ** beta[selected])[start:]
    return {"area": area, "somatic": -np.tanh(sc), "a": a, "b": b,
            "alpha": alpha, "beta": beta, "selected_bins": np.flatnonzero(selected)[start:], "d": d}

def matlab_transfer(area, frequency, native_first_round=False):
    """Open-tract diva_synth.m:329–383, double precision unless specified."""
    r = np.r_[.9, np.diff(area) / np.maximum(np.finfo(float).eps, area[1:] + area[:-1])]
    lip = .9 * np.exp(-(frequency / 4000) ** 2)
    phase = np.exp(2j * np.pi * frequency * .075 / 34326)
    h1, h2 = np.ones(len(frequency)), np.zeros(len(frequency))
    for section, coefficient in enumerate(r[:-1]):
        u, v = h1 - coefficient * h2, h2 - coefficient * h1
        if section == 0 and native_first_round:
            u, v = u.astype(np.float32), v.astype(np.float32)
        h1, h2 = phase * u, phase.conj() * v
    return (1 + lip) * np.prod(1 + r) / (h1 - lip * h2)

def first_peaks(frequency, transfer):
    db = 20 * np.log10(np.maximum(1e-30, np.abs(transfer)))
    index, _ = signal.find_peaks(db, prominence=1., distance=int(np.ceil(50 / (frequency[1] - frequency[0]))))
    all_peaks = frequency[index[(frequency[index] >= 50) & (frequency[index] <= 4950)]]
    return all_peaks[:3].tolist() + [None] * max(0, 3 - len(all_peaks)), all_peaks.tolist()

def evaluate(raw, asset, scale):
    bounded = np.tanh(np.asarray(raw) * np.asarray(scale).reshape(13))
    outline = asset['Average'] + asset['Base'][:, :10] @ (asset['Scale'][:10] * bounded[:10])
    area = matlab_xy2ab(outline)['area']
    if not len(area) or not np.isfinite(area).all():
        return np.full(3, np.nan)
    area = np.maximum(0, area)
    area[(area > 0) & (area < .025)] = .025
    if np.any(area == 0):
        return np.full(3, np.nan)
    frequency = np.arange(32769) * 11025 / 65536
    transfer = matlab_transfer(area, frequency)
    peaks, _ = first_peaks(frequency, transfer)
    return np.asarray([np.nan if p is None else p for p in peaks])
