"""Generate isolated human/TorchDIVA pairs using the author's target extractor."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import warnings

from conditions import CONDITIONS, materialize, sha256

ROOT = Path(__file__).resolve().parents[2]
TARGET_MS = 2000  # Published diva_target_gen.py sets length=2000.


def prepare_pair(task):
    data_directory, sample, seed, condition, provenance = task
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    source_directory = Path(data_directory)/'sources'/condition/'src'
    sys.path[:0] = [str(source_directory), str(ROOT/'experiments')]
    from main import DIVA
    from diva_setup import setup_target
    from diva_target_gen import Target_Gen

    directory = Path(data_directory)
    output = directory / 'conditions' / condition / 'pairs' / sample['name']
    output.parent.mkdir(parents=True, exist_ok=True)
    generator_fingerprint = sha256(__file__)
    metadata_path = output.with_suffix('.json')
    if output.exists() and metadata_path.exists():
        result = json.loads(metadata_path.read_text())
        if (result['human_sha256'] == sample['wav_sha256'] and
                result.get('source_fingerprint') == provenance['source_fingerprint'] and
                result.get('generator_fingerprint') == generator_fingerprint and
                result.get('global_seed') == seed and
                result['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()):
            return result
        # Regenerate only this generated cache when code/settings changed.
    start = time.perf_counter()
    random_seed = seed + int(hashlib.sha256(sample['name'].encode()).hexdigest()[:8], 16)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='diffwave-diva-') as temporary:
        temp = Path(temporary)
        for pattern in ('*.mat', '*.txt', '*.csv'):
            for source in source_directory.glob(pattern):
                shutil.copy2(source, temp/source.name)
        # Target_Gen numbers new files from 10. Keep its original built-in CSV
        # rather than the unrelated Civier custom targets with IDs 508..510.
        csv_path = temp/'diva.csv'
        csv_lines = [line for line in csv_path.read_text().splitlines()
                     if line.strip() and (not line.split(',')[0].isdigit()
                                         or int(line.split(',')[0]) <= 9)]
        csv_path.write_text('\n'.join(csv_lines) + '\n')
        for path in temp.glob('diva_*.txt'):
            if int(path.stem.removeprefix('diva_')) > 9:
                path.unlink()
        os.chdir(temp)
        try:
            with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
                extractor = Target_Gen()
                source = SimpleNamespace(path=str(directory/sample['human_path']), name=sample['name'])
                if not extractor.extract(source):
                    raise ValueError(f'Published target extraction failed: {sample["name"]}')
                diva = DIVA()
                setup_target(diva, sample['name'])
                # DIVA.__init__ resets both generators to zero. Apply the
                # shared, per-file seed after construction in both conditions.
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                for _ in range(5):
                    diva.StartSim(False, None, False)
                articulatory = torch.from_numpy(diva.production_log['VT_4'])
                waveform, *_ = diva.vocal_tract.diva_synth(articulatory, 'sound')
                waveform = waveform.detach().cpu().numpy().ravel()
        finally:
            os.chdir(original_cwd)
    if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) <= 1e-8:
        raise ValueError(f'Invalid or silent DIVA waveform: {sample["name"]}')
    # Save float WAV: no peak matching, clipping, or lossy PCM quantization.
    sf.write(output, waveform, 11025, subtype='FLOAT')
    result = dict(name=sample['name'], split=sample['split'], seed=random_seed,
                  global_seed=seed, condition=condition,
                  source_fingerprint=provenance['source_fingerprint'],
                  generator_fingerprint=generator_fingerprint,
                  target_ms=TARGET_MS, learning_runs=4, saved_iteration=5,
                  duration=len(waveform)/11025, sample_rate=11025,
                  peak=float(np.max(np.abs(waveform))),
                  seconds=time.perf_counter()-start,
                  human_sha256=sample['wav_sha256'],
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    metadata_path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def generate(data_directory, workers=8, seed=20261023, limit=None, conditions=None):
    directory = Path(data_directory).resolve()
    conditions = list(conditions or CONDITIONS)
    provenances = {condition: materialize(directory, condition) for condition in conditions}
    manifest = json.loads((directory/'manifest.json').read_text())
    samples = manifest['samples'][:limit]
    start = time.perf_counter()
    rows, failures = [], []
    # Spawn and a fresh process per pair avoid global cwd/model state sharing.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(prepare_pair, (str(directory), row, seed, condition, provenances[condition])): (condition, row)
                   for row in samples for condition in conditions}
        total = len(futures)
        for future in as_completed(futures):
            condition, row = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(name=row['name'], condition=condition, error=repr(error), traceback=traceback.format_exc()))
                print(f'FAILED {condition}/{row["name"]}: {error}', flush=True)
            done = len(rows)+len(failures)
            if done % 10 == 0 or done == total:
                print(f'Pairs {done}/{total}: {len(rows)} valid, '
                      f'{len(failures)} failed, elapsed {time.perf_counter()-start:.1f}s', flush=True)
                (directory/'pairs-progress.json').write_text(json.dumps(
                    dict(completed=len(rows), failed=failures, total=total,
                         elapsed_seconds=time.perf_counter()-start), indent=2))
    result = dict(seed=seed, workers=workers, target_ms=TARGET_MS,
                  seconds=time.perf_counter()-start, samples=rows, failures=failures)
    for condition in conditions:
        condition_directory = directory/'conditions'/condition
        condition_directory.mkdir(parents=True, exist_ok=True)
        condition_result = dict(result, condition=condition, provenance=provenances[condition],
            samples=[row for row in rows if row['condition']==condition],
            failures=[row for row in failures if row['condition']==condition])
        (condition_directory/'pairs-manifest.json').write_text(json.dumps(condition_result, indent=2)+'\n')
    (directory/'paired-generation.json').write_text(json.dumps(result, indent=2)+'\n')
    if failures:
        raise RuntimeError(f'{len(failures)} pairs failed; failures recorded, no silent exclusions.')
    return result


def load_pair(directory, sample, condition='corrected'):
    """22.05 kHz, identical duration, independently centered sustained vowels.

    The author fixes DIVA targets at 2 s although SVD clips have variable
    durations. Center-crop both to their common duration; no time warping,
    pitch alteration, reference-guided alignment, or loudness matching.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    directory = Path(directory)
    signals = []
    for path in (directory/sample['human_path'], directory/'conditions'/condition/'pairs'/sample['name']):
        waveform, rate = sf.read(path, dtype='float32')
        if waveform.ndim != 1:
            raise ValueError('Mono audio required')
        divisor = math.gcd(22050, rate)
        waveform = resample_poly(waveform, 22050//divisor, rate//divisor)
        signals.append(waveform)
    length = min(len(waveform) for waveform in signals)
    if length < 1024:
        raise ValueError('Pair too short for a mel spectrogram')
    return tuple(np.ascontiguousarray(waveform[(len(waveform)-length)//2:
                    (len(waveform)-length)//2+length]) for waveform in signals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()
    generate(args.data_dir, args.workers, args.seed, args.limit, args.conditions)
