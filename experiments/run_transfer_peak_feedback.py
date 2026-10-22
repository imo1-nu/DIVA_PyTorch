"""Fixed /u/ experiment. All writable runtime assets live below --output.

Run with the repository's existing Torch/NumPy/SciPy environment. No retraining
of fmfit and no waveform analysis. Each worker is a separate process.
"""
import argparse
import copy
from contextlib import chdir
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.io import loadmat
import torch

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.array([100., 500., 1500., 3000.])


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.mat', '.txt', '.csv', '.npz', '.json')
            and '__pycache__' not in p.parts}


def vec(t):
    return np.asarray(t.detach().cpu()).reshape(-1).copy()


def build(source, target_dir, runtime, mode):
    runtime.mkdir(parents=True)
    for p in (source / 'src').iterdir():
        if p.suffix in ('.mat', '.txt', '.csv'):
            shutil.copyfile(p, runtime / p.name)
    with np.load(target_dir / 'target.npz') as f:
        target = {k: f[k].copy() for k in f.files}
    np.testing.assert_array_equal(target['Art'], np.zeros((401, 13)))
    sys.path.insert(0, str(source / 'src'))
    from main import DIVA
    from diva_vocaltract import VocalTract
    from diva_motorcortex import MotorCortex
    from diva_cerebellum import Cerebellum
    with chdir(runtime):
        # Legacy source is used by the regression worker only.
        diva = DIVA() if mode == 'legacy' else DIVA(observation_mode=mode)
        diva.production_info = __import__('diva_utils').txt2struct(str(target_dir / 'diva_000001.txt'))
        diva.production_art = copy.deepcopy(target)
        for name in ('ssm', 'cerebellum', 'motor_cortex', 'vocal_tract', 'aud_state_map',
                     'som_state_map', 'aud_err_map', 'som_err_map'):
            setattr(diva, name, None)
        diva.PrepareSimulation()
        diva.vocal_tract = VocalTract() if mode == 'legacy' else VocalTract(mode)
        diva.motor_cortex = MotorCortex(diva.vocal_tract)
        diva.motor_cortex.feedforward_gain = .85
        diva.motor_cortex.feedback_gain = .30
        diva.cerebellum = Cerebellum(401)
        # Explicit compatibility padding for the main-source regression worker.
        if diva.cerebellum.dataBuffer[0].shape[0] < 401:
            diva.cerebellum.dataBuffer[0] = torch.zeros((401, 1), dtype=torch.float64)
    return diva, target


class Recorder:
    def __init__(self, diva, learn, diagnostic=True):
        self.diva, self.learn, self.diagnostic = diva, learn, diagnostic
        self.rows = {}
        m, vt = diva.motor_cortex, diva.vocal_tract
        lmc_output, motor_output, vt_output = m.LMC.output, m.output, vt.output
        if diagnostic:
            from diva_transfer_peaks import TransferPeakObserver
            from diva_vocaltract import VocalTract
            self.observer = TransferPeakObserver(VocalTract())

        def lmc():
            self.record(learning_signal=m.LMC.InputPorts[2], eligibility=m.LMC.InputPorts[1],
                        ff_activation=m.LMC.InputPorts[0])
            if learn:
                lmc_output()
            else:
                m.LMC.OutputPorts[0] = m.LMC.InputPorts[0].reshape(1, -1) @ m.LMC.weight

        def motor():
            motor_output()
            self.record(raw=m.OutputPorts[0], ff=.85 * m.LMC.OutputPorts[0],
                        fb=m.feedback_gain * m.fb_gain_fixed * m.vel2pos.OutputPorts[0],
                        auditory_error=diva.aud_err_map.OutputPorts[0],
                        somatic_error=diva.som_err_map.OutputPorts[0],
                        delayed_auditory_error=m.AIM.InputPorts[0],
                        perceived=diva.aud_err_map.InputPorts[1])
            asm = diva.aud_state_map
            self.record(delayed_valid=getattr(asm, 'validity', torch.ones((4, 1), dtype=torch.bool)))
            obs = getattr(asm, 'observation', None)
            self.rows.setdefault('delayed_observation_step', []).append(-1 if obs is None else obs.step)
            self.rows.setdefault('delayed_reasons', []).append(('initial_buffer',) * 4 if obs is None else obs.reasons)

        def vocal():
            vt_output()
            observation = getattr(vt, 'last_observation', None)
            self.record(returned=vt.OutputPorts[0], somatic=vt.OutputPorts[1])
            if observation is not None:
                self.record(fmfit=observation.fmfit, valid=observation.valid)
            else:
                self.record(fmfit=vt.OutputPorts[0], valid=torch.ones((4, 1), dtype=torch.bool))
            if diagnostic:
                peaks = observation.peaks if observation is not None else None
                if peaks is None:
                    peaks = self.observer.measure(vt.InputPorts[0], vt.params.Input.Scale)
                for key, value in (('h_peaks', peaks.hz), ('h_valid', peaks.valid),
                                   ('reasons', peaks.reasons), ('peak_count', peaks.peak_count),
                                   ('observation_seconds', peaks.elapsed_s)):
                    self.rows.setdefault(key, []).append(value)

        m.LMC.output, m.output, vt.output = lmc, motor, vocal

    def record(self, **values):
        for k, v in values.items():
            self.rows.setdefault(k, []).append(vec(v))

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}


def run_sim(diva, recorder, runtime):
    recorder.rows = {}
    start = time.monotonic()
    with chdir(runtime), torch.no_grad():
        diva.StartSim(False, None, False)
    data = recorder.arrays()
    data['observation_step'] = np.arange(len(data['raw']))
    data['time_s'] = data['observation_step'] * .005
    data['weight'] = vec(diva.motor_cortex.LMC.weight).reshape(401, 13)
    data['seconds'] = np.array(time.monotonic() - start)
    for name in ('raw', 'ff', 'fb', 'returned', 'fmfit', 'somatic', 'weight', 'auditory_error'):
        if not np.isfinite(data[name]).all():
            raise ValueError(f'nonfinite {name}')
    return data


def metrics(data, target):
    from transfer_peak_reference import evaluate
    asset = loadmat(ROOT / 'src/diva_synth.mat', simplify_cells=True)['vt']
    peaks = np.array([evaluate(raw, asset, np.ones(13)) for raw in data['raw']])
    data['independent_peaks'] = peaks
    valid = np.isfinite(peaks)
    distance = np.abs(peaks - np.clip(peaks, target['Aud_min'][0, 1:], target['Aud_max'][0, 1:]))
    inside = valid & (distance <= .1)
    central = (data['time_s'] >= .5) & (data['time_s'] < 1.5)
    s = data['somatic'][central]
    se = s - np.clip(s, target['Som_min'][0], target['Som_max'][0])
    d, v, within = distance[central], valid[central], inside[central]
    mean = [float(np.mean(d[:, i][v[:, i]])) if v[:, i].any() else None for i in range(3)]
    # All denominators include all scheduled frames, including missing peaks.
    return {'mean_outside_hz': mean, 'overall_mean_outside_hz_valid_only': float(d[v].mean()) if v.any() else None,
            'valid_rate': v.mean(axis=0).tolist(), 'joint_valid_rate': float(v.all(axis=1).mean()),
            'within_rate': within.mean(axis=0).tolist(), 'joint_within_rate': float(within.all(axis=1).mean()),
            'somatic_sse': float(np.mean(np.sum(se**2, axis=1))),
            'all_frames_valid_rate': valid.mean(axis=0).tolist(), 'central_frames': int(central.sum()),
            'observation_seconds_mean': float(np.mean(data['observation_seconds'])),
            'observation_seconds_max': float(np.max(data['observation_seconds'])),
            'native_reference_max_hz': float(np.nanmax(np.abs(peaks - data['h_peaks']))) if np.isfinite(peaks - data['h_peaks']).any() else None,
            'native_reference_validity_disagreements': int(np.sum(valid != data['h_valid']))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('fmfit', 'transfer_peaks', 'legacy'), required=True)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--regression-only', action='store_true')
    args = parser.parse_args()
    source, target_dir, out = args.source.resolve(), args.target.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(target_dir, out / 'inputs')
    runtime = out / 'runtime'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    before = inventory(source / 'src')
    provenance = {'source': str(source), 'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                  'source_before': before, 'inputs': inventory(target_dir), 'config': vars(args) | {'source': str(source), 'target': str(target_dir), 'output': str(out)},
                  'seed': 0, 'feedforward_gain': .85, 'feedback_gain': .30, 'fixed_fb_gain': .5, 'learning_EPS': .5,
                  'auditory_delay_steps': 10, 'dt_s': .005, 'grid': 65536, 'fs': 11025,
                  'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
                  'experiment_sha': sha(Path(__file__)), 'reference_sha': sha(ROOT / 'experiments/transfer_peak_reference.py')}
    dump(out / 'provenance.json', provenance)
    diva, target = build(source, out / 'inputs', runtime, args.mode)
    clean = copy.deepcopy(diva)
    recorder = Recorder(diva, True, not args.regression_only)
    weights, learning = {}, []
    for trial in range(1, args.trials + 1):
        data = run_sim(diva, recorder, runtime)
        np.savez_compressed(out / f'learning_{trial:02d}.npz', **data)
        weights[trial] = data['weight'].copy()
        learning.append({'trial': trial, 'seconds': float(data['seconds']),
                         'weight_norm': float(np.linalg.norm(data['weight'])),
                         'measurement_failure_rate': float(1 - data['h_valid'].mean()) if 'h_valid' in data else None})
        dump(out / 'learning.json', learning)
        print(f'{args.mode} learning {trial}/{args.trials}: {data["seconds"]:.2f}s', flush=True)
    evaluations = []
    if not args.regression_only:
        for trial in (4, 20):
            if trial not in weights:
                continue
            for name, gain in (('ff_only', 0.), ('ff_fb', .30)):
                current = copy.deepcopy(clean)
                weight = torch.tensor(weights[trial])
                current.motor_cortex.LMC.get_set_weights(3, weight.clone())
                current.motor_cortex.feedback_gain = gain
                observed = Recorder(current, False)
                data = run_sim(current, observed, runtime)
                np.testing.assert_array_equal(data['weight'], weight.numpy())
                result = metrics(data, target)
                np.savez_compressed(out / f'frozen_{trial:02d}_{name}.npz', **data)
                evaluations.append({'after_trial': trial, 'generation': name, 'weight_unchanged': True,
                                    'seconds': float(data['seconds']), **result})
                dump(out / 'evaluations.json', evaluations)
                print(f'{args.mode} frozen {trial} {name}: {result}', flush=True)
    after = inventory(source / 'src')
    assert before == after, 'source/assets changed'
    dump(out / 'integrity.json', {'source_assets_unchanged': True, 'inputs_unchanged': inventory(target_dir) == provenance['inputs']})
    dump(out / 'completed.json', {'trials': args.trials, 'evaluations': len(evaluations)})


if __name__ == '__main__':
    main()
