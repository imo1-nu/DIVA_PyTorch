"""Reproducible observation, inverse separation, and mask integration tests."""
import copy
from contextlib import chdir
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from diva_vocaltract import VocalTract
from diva_transfer_peaks import (acoustic_area, measure_area, TransferPeakObserver,
                                 AuditoryObservation, PeakMeasurement)
from diva_asm import AuditoryStateMap
from diva_aem import AuditoryErrorMap
from diva_weightsexplicit import ExplicitWeightBlock
from diva_nullspacegain import NullSpaceGain
from transfer_peak_reference import evaluate


class TransferPeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.context = chdir(ROOT / 'src')
        cls.context.__enter__()
        cls.states = json.loads((ROOT / 'test/fixtures/transfer_peak_states.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_forty_fixed_independent_references(self):
        vt = VocalTract('transfer_peaks')
        for state in self.states:
            with self.subTest(state=state['state']):
                obs, som = vt.observe(torch.tensor(state['raw']).reshape(13, 1))
                self.assertTrue(obs.valid.all())
                np.testing.assert_allclose(obs.peaks.hz, state['reference_hz'], atol=1, rtol=0)
                np.testing.assert_array_equal(obs.peaks.hz, state['native_hz'])
                model, fitted_som = vt.vocal_tract_compute(torch.tensor(state['raw']).reshape(13, 1), True)
                torch.testing.assert_close(obs.fmfit, model, rtol=0, atol=0)
                torch.testing.assert_close(obs.values[0], model[0], rtol=0, atol=0)
                torch.testing.assert_close(som, fitted_som, rtol=0, atol=0)

    def test_scale_and_tanh_once_with_independent_geometry(self):
        vt = VocalTract('transfer_peaks')
        scale = np.linspace(.6, 1.4, 13)
        vt.params.Input.Scale = scale.reshape(13, 1)
        raw = np.array(self.states[0]['raw'])
        observation, _ = vt.observe(torch.tensor(raw).reshape(13, 1))
        asset = loadmat('diva_synth.mat', simplify_cells=True)['vt']
        np.testing.assert_allclose(observation.peaks.hz, evaluate(raw, asset, scale), atol=1, rtol=0)

    def test_inverse_and_posture_use_fmfit_only(self):
        raw = torch.tensor(self.states[0]['raw'], dtype=torch.float64).reshape(13, 1)
        old, new = VocalTract(), VocalTract('transfer_peaks')
        obs, _ = new.observe(raw)
        self.assertGreater(abs(float(obs.values[3] - obs.fmfit[3])), .1)
        with patch.object(new, 'observe', side_effect=AssertionError('model called observation')):
            for cls, parameters in ((ExplicitWeightBlock, ('Auditory', .05, .05)),
                                    (ExplicitWeightBlock, ('Somatosensory', .05, .05)),
                                    (NullSpaceGain, ('Auditory&Somatosensory', .1, .05, .05))):
                blocks = [cls(*parameters, v) for v in (old, new)]
                for b in blocks:
                    if cls is ExplicitWeightBlock:
                        b.InputPorts[0] = torch.full((4 if parameters[0] == 'Auditory' else 8, 1), .1, dtype=torch.float64)
                        b.InputPorts[1] = raw
                    else:
                        b.InputPorts[0] = raw
                    b.output()
                torch.testing.assert_close(blocks[0].OutputPorts[0], blocks[1].OutputPorts[0], rtol=0, atol=0)

    def test_actual_observation_enters_error_map(self):
        vt = VocalTract('transfer_peaks')
        vt.InputPorts[0] = torch.tensor(self.states[0]['raw'], dtype=torch.float64).reshape(13, 1)
        vt.output()
        obs = vt.last_observation
        state = AuditoryStateMap(masked=True)
        state.setinputdims(4)
        state.input(obs.values, obs)
        for _ in range(11):
            state.output()
        self.assertEqual(state.observation.step, obs.step)
        error = AuditoryErrorMap()
        error.InputPorts[0] = torch.zeros((101, 1), dtype=torch.float64)
        error.InputPorts[1] = state.OutputPorts[0]
        error.validity = state.validity
        # Target weights with no activity yield their existing bias range.
        error.output()
        expected = obs.values - torch.clamp(obs.values,
            error.weight_min.bias.reshape(-1, 1), error.weight_max.bias.reshape(-1, 1))
        torch.testing.assert_close(error.OutputPorts[0], expected)
        self.assertFalse(torch.equal(obs.values, obs.fmfit))

    def test_closure_empty_nonfinite_and_partial_peaks(self):
        np.testing.assert_array_equal(acoustic_area([-1, 0, .001, .025, 2]), [0, 0, .025, .025, 2])
        for area, reason in (([], 'empty_area'), ([1, 0, 1], 'closure'),
                             ([1, -1, 1], 'closure'), ([1, np.nan], 'nonfinite_or_invalid_geometry')):
            result = measure_area(area, .075)
            self.assertFalse(result.valid.any())
            self.assertEqual(result.reasons, (reason,) * 3)
        f = np.arange(32769) * 11025 / 65536
        h = np.ones(32769)
        h[600] = 10
        with patch('diva_vocaltract.a2h', return_value=(h, f, None)):
            result = measure_area([1, 1], .075)
        np.testing.assert_array_equal(result.valid, [True, False, False])
        self.assertEqual(result.peak_count, 1)
        self.assertEqual(result.reasons, ('ok', 'insufficient_peaks', 'insufficient_peaks'))
        vt = VocalTract('transfer_peaks')
        vt.peak_observer = type('Failed', (), {'measure': lambda *args: result})()
        obs, som = vt.observe(torch.zeros((13, 1), dtype=torch.float64))
        self.assertTrue(torch.isfinite(obs.values).all())
        np.testing.assert_array_equal(obs.valid[:, 0], [True, True, False, False])
        self.assertEqual(float(obs.values[3]), 0)
        observer = TransferPeakObserver(VocalTract())
        self.assertEqual(observer.measure(torch.full((13, 1), float('nan')), np.ones(13)).reasons[0], 'nonfinite_raw')

    def test_invalid_error_excluded_and_mask_delay_reset(self):
        state = AuditoryStateMap(masked=True)
        state.setinputdims(4)
        valid = torch.tensor([True, True, False, True]).reshape(4, 1)
        values = torch.tensor([1., 2., float('nan'), 4.], dtype=torch.float64).reshape(4, 1)
        obs = AuditoryObservation(values, valid, ('ok', 'ok', 'missing', 'ok'), 37, values, None)
        for step in range(11):
            state.output()
            if step < 10:
                self.assertIsNone(state.observation)
                self.assertFalse(state.validity[1:].any())
            else:
                self.assertEqual(state.observation.step, 37)
                torch.testing.assert_close(state.validity, valid)
            state.input(values, obs)
        error = AuditoryErrorMap()
        error.InputPorts[0] = torch.zeros((101, 1), dtype=torch.float64)
        error.InputPorts[1] = values
        error.validity = valid
        error.output()
        self.assertTrue(torch.isfinite(error.OutputPorts[0]).all())
        self.assertEqual(float(error.OutputPorts[0][2]), 0)
        state.reset()
        state.output()
        self.assertIsNone(state.observation)
        self.assertFalse(state.validity[1:].any())
        vt = VocalTract('transfer_peaks')
        vt.InputPorts[0] = torch.zeros((13, 1), dtype=torch.float64)
        vt.output(); vt.reset(); vt.output()
        self.assertEqual(vt.last_observation.step, 0)

    def test_rng_and_synthesizer_state_unchanged(self):
        vt = VocalTract('transfer_peaks')
        vt.diva_vt()
        state = copy.deepcopy(vt.vt)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        vt.observe(torch.tensor(self.states[0]['raw'], dtype=torch.float64).reshape(13, 1))
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(vt.vt[key], value)
            else:
                self.assertEqual(vt.vt[key], value)
        self.assertIsNone(vt.ab_alpha)
        self.assertIsNone(vt.ab_beta)
        self.assertEqual(random.getstate(), python_rng)
        for a, b in zip(np.random.get_state(), numpy_rng):
            np.testing.assert_array_equal(a, b)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_default_is_fmfit_and_bad_mode_rejected(self):
        vt = VocalTract()
        x = torch.zeros((13, 1), dtype=torch.float64)
        y, z = vt.vocal_tract_compute(x, True)
        obs, som = vt.observe(x)
        torch.testing.assert_close(obs.values, y, rtol=0, atol=0)
        torch.testing.assert_close(som, z, rtol=0, atol=0)
        self.assertIsNone(vt.peak_observer)
        with self.assertRaises(ValueError):
            VocalTract('bad-mode')


if __name__ == '__main__':
    unittest.main()
