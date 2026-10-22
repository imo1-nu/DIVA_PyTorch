"""Local check against the 40 saved raw states, areas and independent H arrays."""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import find_peaks
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from diva_vocaltract import VocalTract


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = [args.reference / 'results.json', *sorted((args.reference / 'states').glob('*.npz'))]
    before = {str(p): sha(p) for p in paths}
    rows = json.loads(paths[0].read_text())
    checks = []
    with chdir(ROOT / 'src'):
        vt = VocalTract('transfer_peaks')
        for row in rows:
            path = args.reference / 'states' / (row['state'] + '.npz')
            with np.load(path) as f:
                frequency, h = f['frequency_hz'], f['reference_transfer']
                index, _ = find_peaks(20 * np.log10(np.maximum(1e-30, np.abs(h))),
                                      prominence=1., distance=int(np.ceil(50 / np.diff(frequency)[0])))
                reference = frequency[index[(frequency[index] >= 50) & (frequency[index] <= 4950)]][:3]
                observation, _ = vt.observe(torch.tensor(row['raw'], dtype=torch.float64).reshape(13, 1))
                np.testing.assert_allclose(observation.peaks.hz, reference, atol=1, rtol=0)
                checks.append({'state': row['state'], 'max_abs_hz': float(np.max(np.abs(observation.peaks.hz - reference)))})
    assert before == {str(p): sha(p) for p in paths}
    args.output.write_text(json.dumps({'all_40_pass': len(checks) == 40, 'checks': checks,
                                      'input_unchanged': True, 'input_sha256': before}, indent=2) + '\n')
    print(f'{len(checks)} saved independent-H checks pass; max difference {max(c["max_abs_hz"] for c in checks)} Hz')


if __name__ == '__main__':
    main()
