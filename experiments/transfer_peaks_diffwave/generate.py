"""Generate the registered transfer-peak DIVA dataset without touching baseline data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/diffwave'))
from conditions import sha256


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(task):
    directory, sample, mode, seed = task
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = Path(directory)
    source = directory/'sources/transfer_peaks/src'
    sys.path[:0] = [str(source), str(ROOT/'experiments')]
    from main import DIVA
    from diva_setup import setup_target
    from diva_target_gen import Target_Gen
    from diva_vocaltract import VocalTract
    from diva_motorcortex import MotorCortex
    from run_transfer_peak_feedback import Recorder
    name = sample['name']
    condition = 'transfer_peaks' if mode == 'transfer_peaks' else 'fmfit_check'
    out = directory/'conditions'/condition/'pairs'/name
    out.parent.mkdir(parents=True, exist_ok=True)
    logdir = directory/'conditions'/condition/'trajectories'/Path(name).stem
    logdir.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((directory/'sources/transfer_peaks/provenance.json').read_text())
    code = {n:sha256(p) for n,p in (
        ('generate.py', Path(__file__)), ('recorder', ROOT/'experiments/run_transfer_peak_feedback.py'),
        ('setup_target', ROOT/'experiments/diva_setup.py'))}
    metadata = out.with_suffix('.json')
    if metadata.exists() and out.exists():
        saved = json.loads(metadata.read_text())
        if (saved['code_sha256']==code and saved['global_seed']==seed
            and saved['source_fingerprint']==provenance['source_fingerprint']
            and saved['human_sha256']==sample['wav_sha256']
            and saved['sha256']==sha256(out)
            and all((logdir/f'trial_{i:02}.npz').exists() for i in range(1,6))):
            return saved
        raise ValueError(f'Refusing stale generated data: {out}')
    if sha256(directory/sample['human_path']) != sample['wav_sha256']:
        raise ValueError('Human input hash mismatch')
    started = time.perf_counter()
    random_seed = seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    original_cwd = Path.cwd()
    runs = []
    with tempfile.TemporaryDirectory(prefix='hfb-diva-') as temporary:
        temp = Path(temporary)
        for pattern in ('*.mat','*.txt','*.csv'):
            for p in source.glob(pattern):
                shutil.copy2(p, temp/p.name)
        csv = temp/'diva.csv'
        csv.write_text('\n'.join(line for line in csv.read_text().splitlines()
            if line.strip() and (not line.split(',')[0].isdigit() or int(line.split(',')[0]) <= 9))+'\n')
        for p in temp.glob('diva_*.txt'):
            if int(p.stem.removeprefix('diva_'))>9:
                p.unlink()
        os.chdir(temp)
        try:
            with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
                extractor = Target_Gen()
                if not extractor.extract(SimpleNamespace(path=str(directory/sample['human_path']), name=name)):
                    raise ValueError('Published target extraction failed')
                diva = DIVA(observation_mode=mode)
                setup_target(diva, name)
                # setup_target eagerly creates a default fmfit tract. Replace
                # it BEFORE simulation so H observations actually enter FB.
                diva.vocal_tract = VocalTract(observation_mode=mode)
                diva.motor_cortex = MotorCortex(diva.vocal_tract)
                assert diva.motor_cortex.feedforward_gain == diva.motor_cortex.feedback_gain == 1
                np.testing.assert_array_equal(diva.production_art['Art'], 0.)
                np.savez_compressed(logdir/'target.npz', **diva.production_art)
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                recorder = Recorder(diva, True, diagnostic=True)
                for trial in range(1,6):
                    recorder.rows = {}
                    before = diva.motor_cortex.LMC.weight.detach().cpu().numpy().copy()
                    begin = time.perf_counter()
                    diva.StartSim(False, None, False)
                    arrays = recorder.arrays()
                    arrays.update(weight_before=before,
                        weight_after=diva.motor_cortex.LMC.weight.detach().cpu().numpy().copy(),
                        observation_step=np.arange(len(recorder.rows['raw'])))
                    arrays['time_s'] = arrays['observation_step']*.005
                    for key in ('raw','ff','fb','returned','fmfit','somatic','auditory_error','somatic_error','learning_signal'):
                        if not np.isfinite(arrays[key]).all():
                            raise ValueError(f'Nonfinite matrix input: {key}')
                    np.testing.assert_allclose(arrays['raw'], arrays['ff']+arrays['fb'], atol=1e-12, rtol=0)
                    if mode == 'transfer_peaks':
                        active = arrays['valid'][:,1:].astype(bool)
                        np.testing.assert_allclose((arrays['returned'][:,1:]*[500,1500,3000])[active],
                                                   arrays['h_peaks'][active], rtol=0, atol=1e-9)
                        if np.any(arrays['auditory_error'][~arrays['delayed_valid'].astype(bool)] != 0):
                            raise ValueError('Invalid auditory error reached controller')
                    np.savez_compressed(logdir/f'trial_{trial:02}.npz', **arrays)
                    validity = arrays['h_valid']
                    active_frames = np.any(arrays['ff_activation'] != 0, axis=1)
                    runs.append(dict(trial=trial, seconds=time.perf_counter()-begin,
                        frames=len(validity), active_frames=int(active_frames.sum()),
                        joint_valid_rate=float(validity.all(axis=1).mean()),
                        active_joint_valid_rate=float(validity[active_frames].all(axis=1).mean()),
                        reasons=dict(Counter(str(x) for x in arrays['reasons'].ravel())),
                        weight_update_norm=float(np.linalg.norm(arrays['weight_after']-before))))
                articulatory = torch.from_numpy(diva.production_log['VT_4'])
                waveform, *_ = diva.vocal_tract.diva_synth(articulatory, 'sound')
                waveform = waveform.detach().cpu().numpy().ravel()
        finally:
            os.chdir(original_cwd)
    if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) <= 1e-8:
        raise ValueError('Invalid or silent DIVA waveform')
    sf.write(out, waveform, 11025, subtype='FLOAT')
    row = dict(name=name, split=sample['split'], condition=condition, observation_mode=mode,
        seed=random_seed, global_seed=seed, target_ms=2000, learning_runs=4, saved_iteration=5,
        saved_iteration_online_learning=True, feedforward_gain=1., feedback_gain=1.,
        source_fingerprint=provenance['source_fingerprint'], code_sha256=code,
        human_sha256=sample['wav_sha256'], sha256=sha256(out),
        pcm_float32_sha256=hashlib.sha256(waveform.astype('<f4').tobytes()).hexdigest(),
        duration=len(waveform)/11025, sample_rate=11025, peak=float(np.max(np.abs(waveform))),
        seconds=time.perf_counter()-started, trajectories=str(logdir.relative_to(directory)), runs=runs)
    dump(metadata, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', choices=['fmfit','transfer_peaks'], default='transfer_peaks')
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    samples = json.loads((directory/'manifest.json').read_text())['samples'][:args.limit]
    condition = 'transfer_peaks' if args.mode=='transfer_peaks' else 'fmfit_check'
    rows, failures = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(prepare,(str(directory),s,args.mode,args.seed)):s for s in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(name=sample['name'], error=repr(error), traceback=traceback.format_exc()))
                print(f'FAILED {sample["name"]}: {error}', flush=True)
            if (len(rows)+len(failures))%10==0 or len(rows)+len(failures)==len(samples):
                result = dict(condition=condition, seed=args.seed, workers=args.workers,
                    seconds=time.perf_counter()-started, samples=sorted(rows,key=lambda r:r['name']), failures=failures,
                    provenance=json.loads((directory/'sources/transfer_peaks/provenance.json').read_text()))
                dest = directory/'conditions'/condition
                dest.mkdir(parents=True, exist_ok=True)
                dump(dest/'pairs-manifest.json', result)
                print(f'{condition}: {len(rows)} valid/{len(samples)}, {len(failures)} failed; {result["seconds"]:.1f}s', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} failures; no silent exclusion')


if __name__ == '__main__':
    main()
