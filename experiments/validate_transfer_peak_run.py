"""Recheck saved experiment signals and main regression without model calls."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path) as f:
        return {k: f[k].copy() for k in f.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for trial in range(1, 5):
        old = load(args.root / 'main-regression' / f'learning_{trial:02d}.npz')
        new = load(args.root / 'fmfit' / f'learning_{trial:02d}.npz')
        checks = {}
        for name in ('raw', 'returned', 'somatic', 'ff', 'fb', 'auditory_error',
                     'somatic_error', 'weight', 'learning_signal', 'eligibility', 'perceived'):
            np.testing.assert_array_equal(old[name], new[name])
            checks[name] = float(np.max(np.abs(old[name] - new[name])))
        rows.append({'trial': trial, 'main_regression_max_abs': checks})
    delay_checks = []
    for mode in ('fmfit', 'transfer_peaks'):
        for path in sorted((args.root / mode).glob('*.npz')):
            data = load(path)
            n = len(data['raw'])
            assert n == 411
            expected_steps = np.r_[np.full(10, -1), np.arange(n - 10)]
            np.testing.assert_array_equal(data['delayed_observation_step'], expected_steps)
            np.testing.assert_array_equal(data['perceived'][10:], data['returned'][:-10])
            np.testing.assert_array_equal(data['delayed_valid'][10:], data['valid'][:-10])
            if mode == 'transfer_peaks':
                assert not data['delayed_valid'][:10, 1:].any()
                assert not data['auditory_error'][~data['delayed_valid']].any()
                assert not data['delayed_auditory_error'][1:][~data['delayed_valid'][:-1]].any()
                np.testing.assert_array_equal(data['returned'][:, 0], data['fmfit'][:, 0])
                scale = np.array([500., 1500., 3000.])
                valid = data['h_valid']
                np.testing.assert_allclose(data['returned'][:, 1:][valid] * np.broadcast_to(scale, valid.shape)[valid],
                                           data['h_peaks'][valid], rtol=1e-14, atol=1e-10)
            before = np.zeros((401, 13))
            if path.name.startswith('learning_'):
                trial = int(path.stem.split('_')[-1])
                if trial > 1:
                    before = load(path.parent / f'learning_{trial - 1:02d}.npz')['weight']
                # Existing MATLAB-order reshape: eligibility=[auditory, somatic]
                # and learn signal=[auditory 13, somatic 13].
                u = data['eligibility'].reshape(n, 2, 401).transpose(0, 2, 1)
                v = data['learning_signal'].reshape(n, 2, 13)
                update = .5 * np.einsum('nki,nij->kj', u, v)
                np.testing.assert_allclose(before + update, data['weight'], atol=1e-12, rtol=1e-12)
            delay_checks.append({'file': str(path.relative_to(args.root)), 'frames': n,
                                 'observation_mask_delay_exact': True, 'masked_errors_zero': True})
    args.output.write_text(json.dumps({'main_regression': rows, 'signal_checks': delay_checks}, indent=2) + '\n')
    print(f'4 main trials exact; {len(delay_checks)} trajectories and learning updates checked')


if __name__ == '__main__':
    main()
