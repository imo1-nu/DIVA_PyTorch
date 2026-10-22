"""Post-hoc, matched-target analysis of frozen happy gain1,1 trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import loadmat
from transfer_peak_reference import evaluate

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.array([500., 1500., 3000.])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def mean(values):
    good = np.isfinite(values)
    return float(values[good].mean()) if good.any() else None


def stats(value, lo, hi, selected):
    valid = np.isfinite(value)
    distance = np.abs(value - np.clip(value, lo, hi))
    v, d = valid[selected], distance[selected]
    return {'mean_outside_hz': [mean(d[:, i]) for i in range(3)],
            'overall_mean_hz_valid_only': mean(d), 'component_valid_rate': v.mean(axis=0).tolist(),
            'joint_valid_rate': float(v.all(axis=1).mean()),
            'strict_inside_rate': ((d == 0) & v).mean(axis=0).tolist(),
            'strict_joint_inside_rate': float(((d == 0) & v).all(axis=1).mean()),
            'tolerance_joint_inside_rate': float(((d <= .1) & v).all(axis=1).mean())}, distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve();out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    paths = sorted(root.glob('*/frozen_*.npz'))
    assert len(paths) == 8
    targets = root / 'inputs/diva_000001.mat';assets = root / 'inputs/diva_synth.mat'
    before = {str(p): sha(p) for p in [*paths, targets, assets]}
    target = loadmat(targets, simplify_cells=True)['timeseries']
    asset = loadmat(assets, simplify_cells=True)['vt']
    rows, arrays = [], {}
    start = time.monotonic()
    selection = None
    for path in paths:
        with np.load(path) as f:data = {k:f[k].copy() for k in f.files}
        activation = data['ff_activation']
        selected = np.any(activation != 0, axis=1)
        assert int(selected.sum()) == 99
        np.testing.assert_array_equal(activation[selected].sum(axis=1), np.ones(99))
        if selection is None:selection = selected
        else:np.testing.assert_array_equal(selection, selected)
        lo = activation @ target['Aud_min'][:, 1:]
        hi = activation @ target['Aud_max'][:, 1:]
        peaks = np.array([evaluate(raw, asset, np.ones(13)) for raw in data['raw']])
        fitted = data['fmfit'][:, 1:] * SCALE
        h, distance = stats(peaks, lo, hi, selected)
        m, _ = stats(fitted, lo, hi, selected)
        som = data['somatic'] - np.clip(data['somatic'], activation @ target['Som_min'], activation @ target['Som_max'])
        mode = path.parent.name;trial = int(path.stem.split('_')[1]);generation = path.stem.split('_',2)[2]
        difference = np.abs(peaks - data['h_peaks'])
        row = {'mode':mode, 'after_trial':trial, 'generation':generation, 'selected_frames':99,
               'H':h, 'M':m,
               'H_minus_M_mae_hz': [mean(np.abs(peaks[selected,i] - fitted[selected,i])) for i in range(3)],
               'somatic_sse':float(np.mean(np.sum(som[selected]**2,axis=1))),
               'native_reference_max_abs_hz':mean([]) if not np.isfinite(difference).any() else float(np.nanmax(difference)),
               'native_reference_validity_disagreements':int(np.sum(np.isfinite(peaks) != data['h_valid']))}
        rows.append(row)
        arrays[(mode,trial,generation)] = (peaks, distance)
        np.savez_compressed(out / (mode+'_'+path.name), independent_peaks=peaks,
                            selected=selected, lower_hz=lo, upper_hz=hi, outside_hz=distance)
        print(mode, trial, generation, h, flush=True)
    comparisons = []
    for trial in (4,20):
        for generation in ('ff_fb','ff_only'):
            pa,da = arrays[('fmfit',trial,generation)];pb,db = arrays[('transfer_peaks',trial,generation)]
            common = selection & np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
            a = mean(da[common]);b = mean(db[common])
            ra,rb = [next(r for r in rows if r['mode']==mode and r['after_trial']==trial and r['generation']==generation) for mode in ('fmfit','transfer_peaks')]
            comparisons.append({'after_trial':trial,'generation':generation,
                                'own_valid_mean_hz_fmfit':ra['H']['overall_mean_hz_valid_only'],
                                'own_valid_mean_hz_transfer':rb['H']['overall_mean_hz_valid_only'],
                                'joint_valid_rate_fmfit':ra['H']['joint_valid_rate'],
                                'joint_valid_rate_transfer':rb['H']['joint_valid_rate'],
                                'common_valid_frames':int(common.sum()),'common_mean_hz_fmfit':a,'common_mean_hz_transfer':b,
                                'common_change_hz':None if a is None or b is None else b-a,
                                'common_reduction_percent':None if a is None or b is None or a==0 else 100*(a-b)/a})
    assert before == {str(p):sha(p) for p in [*paths,targets,assets]}
    dump(out/'evaluations.json',rows);dump(out/'comparisons.json',comparisons)
    dump(out/'provenance.json', {'analysis_kind':'post-hoc, no learning or regeneration',
         'selection':'all nonzero saved motor FF SSM activations; 99 frames, program index1..99',
         'source_files':before,'inputs_unchanged':True,'script_sha256':sha(Path(__file__)),
         'reference_sha256':sha(ROOT/'experiments/transfer_peak_reference.py'),
         'python':sys.version,'numpy':np.__version__,'seconds':time.monotonic()-start})
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
