"""Static transfer maxima, not waveform formants or pole frequencies.

Only the observation boundary uses this module. Model Jacobians remain fmfit.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.signal import find_peaks
import torch


@dataclass
class PeakMeasurement:
    hz: np.ndarray
    valid: np.ndarray
    reasons: tuple
    peak_count: int
    elapsed_s: float = 0.0


@dataclass
class AuditoryObservation:
    values: torch.Tensor
    valid: torch.Tensor
    reasons: tuple
    step: int
    fmfit: torch.Tensor
    peaks: PeakMeasurement | None


def invalid_peaks(reason):
    return PeakMeasurement(np.full(3, np.nan), np.zeros(3, dtype=bool),
                           (reason,) * 3, 0)


def acoustic_area(area):
    """Exactly preserve complete closures in the sound-path area treatment."""
    area = np.maximum(0., np.asarray(area, dtype=float).reshape(-1)).copy()
    area[(area > 0) & (area < .025)] = .025
    return area


def measure_area(area, length):
    from diva_vocaltract import a2h

    if area is None:
        return invalid_peaks('empty_area')
    area = np.asarray(area, dtype=float).reshape(-1)
    if not len(area):
        return invalid_peaks('empty_area')
    if not np.isfinite(area).all() or length is None or not np.isfinite(length) or length <= 0:
        return invalid_peaks('nonfinite_or_invalid_geometry')
    area = acoustic_area(area)
    if np.any(area == 0):
        return invalid_peaks('closure')
    h, frequency, _ = a2h(torch.tensor(area).reshape(-1, 1), float(length),
                          65536, 11025, 0, float(area.min()))
    h = np.asarray(h).reshape(-1)[:32769]
    frequency = np.asarray(frequency).reshape(-1)[:32769]
    if not np.isfinite(h).all():
        return invalid_peaks('nonfinite_transfer')
    db = 20 * np.log10(np.maximum(1e-30, np.abs(h)))
    index, _ = find_peaks(db, prominence=1., distance=int(np.ceil(50 / (11025 / 65536))))
    peaks = frequency[index[(frequency[index] >= 50) & (frequency[index] <= 4950)]]
    count = len(peaks)
    result = invalid_peaks('insufficient_peaks')
    result.hz[:min(3, count)] = peaks[:3]
    result.valid[:min(3, count)] = True
    result.reasons = tuple('ok' if v else 'insufficient_peaks' for v in result.valid)
    result.peak_count = count
    return result


class TransferPeakObserver:
    """Own geometry/cache instance: never touch the waveform synthesizer state."""
    def __init__(self, geometry):
        self.geometry = geometry

    def measure(self, raw, input_scale):
        start = time.perf_counter()
        raw = torch.as_tensor(raw, dtype=torch.float64).reshape(13, 1)
        if not torch.isfinite(raw).all():
            result = invalid_peaks('nonfinite_raw')
        else:
            bounded = torch.tanh(raw * torch.as_tensor(input_scale, dtype=torch.float64).reshape(13, 1))
            if not torch.isfinite(bounded).all():
                result = invalid_peaks('nonfinite_scaled_input')
            else:
                # nargout=4 is for geometry only. Its somatic return has an
                # unset pressure component and must never reach sensory maps.
                try:
                    _, _, _, area, length = self.geometry.diva_synth_sample(bounded, 4)
                    result = measure_area(area, length)
                except (ValueError, IndexError, FloatingPointError) as error:
                    result = invalid_peaks(f'geometry_error:{type(error).__name__}:{error}')
        result.elapsed_s = time.perf_counter() - start
        return result
