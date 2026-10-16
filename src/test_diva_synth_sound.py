import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

import diva_vocaltract as dvt


class TestSynthSound(unittest.TestCase):
    def synthesize(self, f0_control=0.0, frames=80, release=False, noise=False):
        tract = dvt.VocalTract()
        tract.vt = dict(pressure=0, f0=120, closed=0, closure_time=0,
                        closure_position=0, opening_time=0)
        art = torch.zeros((13, frames), dtype=torch.float64)
        art[10] = f0_control
        art[11] = 0.9
        art[12] = 1.0
        if release:
            art[0, 10:20] = -1

        def geometry(sample, nargout):
            # Exercise the sound generator on an open tube, optionally closing it.
            area = torch.tensor([[1.0], [1.2], [0.8], [1.0], [1.1]], dtype=torch.float64)
            if sample[0, 0] < 0:
                area[2] = 0
            return None, None, None, area, 0.1

        synth = dvt.Synth()
        periods = []
        filter_tube = dvt.a2h

        def filter_period(area, length, n, fs, closure, minimum):
            self.assertEqual(n, math.ceil(fs / synth.f0))
            periods.append(n)
            return filter_tube(area, length, n, fs, closure, minimum)

        def zero_noise(*args, size=None, **kwargs):
            return 0.0 if size is None else np.zeros(size)

        normal = np.random.normal if noise else zero_noise
        with patch.object(tract, 'diva_synth_sample', side_effect=geometry), \
                patch.object(dvt, 'Synth', return_value=synth), \
                patch.object(dvt, 'a2h', side_effect=filter_period), \
                patch.object(dvt.np.random, 'normal', side_effect=normal), \
                patch.object(dvt.np.random, 'uniform', return_value=0.5), \
                patch.object(dvt.diva_utils, 'read_file_parameter',
                             side_effect=AssertionError('Sound synthesis must not load debug noise')):
            waveform = tract.diva_synth_sound(art)

        self.assertEqual(waveform.shape, (math.ceil(11025 * frames * 0.005), 1))
        self.assertTrue(torch.isfinite(waveform).all())
        self.assertGreater(waveform.abs().max().item(), 0)
        return waveform, periods

    def test_pitch_control_changes_period_and_waveform(self):
        waveforms = []
        for control in (-1.0, 0.0, 1.0):
            with self.subTest(f0_control=control):
                waveform, periods = self.synthesize(control)
                expected_period = math.ceil(11025 / (100 + 20 * control))
                self.assertEqual(periods[-1], expected_period)
                self.assertGreater(len(set(periods)), 1)
                waveforms.append(waveform)
        self.assertFalse(torch.equal(waveforms[0], waveforms[-1]))

    def test_release_with_variable_periods(self):
        _, periods = self.synthesize(release=True)
        self.assertGreater(len(set(periods)), 1)

    def test_short_output_is_trimmed_to_requested_duration(self):
        self.synthesize(frames=1)

    def test_random_noise_handles_variable_periods_and_can_be_seeded(self):
        random_state = np.random.get_state()
        try:
            np.random.seed(123)
            first, _ = self.synthesize(release=True, noise=True)
            np.random.seed(123)
            repeated, _ = self.synthesize(release=True, noise=True)
            np.random.seed(456)
            different, _ = self.synthesize(release=True, noise=True)
        finally:
            np.random.set_state(random_state)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))


if __name__ == '__main__':
    unittest.main()
