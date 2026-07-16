import numpy as np


class ExcessiveErrorDetector:
    """
    Civier et al. (2010) monitoring subsystem, Section 2.2.

    P(R_error(t)) = epsilon * f_masking(f_size(dAu(t)), I_masking)
    f_masking(x, I)  = max(0, x - xi * I)
    f_size(dAu)      = sum_k |dAu_k| * Scale_k   [Hz]

    Parameters
    ----------
    epsilon : float
        Repair sensitivity (default 2e-3 per Hz).
    xi : float
        Masking factor (default 2.0).
    """

    AUD_SCALE_HZ = np.array([100.0, 500.0, 1500.0, 3000.0])

    def __init__(self, epsilon: float = 2e-5, xi: float = 2.0):
        self.epsilon = epsilon
        self.xi = xi

    def f_size(self, aud_err_normalized: np.ndarray) -> float:
        """Total auditory error magnitude in Hz."""
        err = np.asarray(aud_err_normalized).flatten()
        return float(np.sum(np.abs(err) * self.AUD_SCALE_HZ))

    def f_masking(self, x: float, I_masking: float) -> float:
        return max(0.0, x - self.xi * I_masking)

    def repair_probability(self, aud_err_normalized: np.ndarray,
                           I_masking: float = 0.0) -> float:
        """Probability of triggering self-repair at current timestep."""
        size = self.f_size(aud_err_normalized)
        return min(1.0, self.epsilon * self.f_masking(size, I_masking))
