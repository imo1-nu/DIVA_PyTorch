"""Generate native happy with the exact corrected source used for DiffWave.

No DiffWave inference and no HTML. Preserve the author's WAV unmodified.
"""
import argparse
from contextlib import chdir
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.io import wavfile, loadmat
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/tmp/torchdiva-diffwave/sources/corrected/src'))
    parser.add_argument('--training-config', type=Path, default=ROOT.parent / 'torchdiva-diffwave/experiments/diffwave/results/corrected/config.json')
    parser.add_argument('--prior-waveform', type=Path, default=ROOT.parent / 'torchdiva-diffwave/experiments/diffwave/artifacts/happy_corrected/happy_corrected_diva_11025_float.wav')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    start = time.monotonic()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    provenance_path = source.parent / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    config = json.loads(args.training_config.read_text())
    expected = provenance['source_files_sha256']
    before = {name: sha(source / name) for name in expected}
    assert before == expected
    fingerprint = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    assert fingerprint == provenance['source_fingerprint'] == config['source_fingerprint']
    staged = out / 'source/src'
    for name in expected:
        dest = staged / name;dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    shutil.copyfile(provenance_path, out / 'source/provenance.json')
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    shutil.copyfile(ROOT / 'docs/research/DIFFWAVE_CORRECTED_HAPPY_PROTOCOL.md', out / 'protocol.md')
    runtime = out / 'runtime';runtime.mkdir()
    for path in staged.iterdir():
        if path.suffix in ('.mat', '.txt', '.csv'):
            shutil.copyfile(path, runtime / path.name)
    sys.path.insert(0, str(staged))
    from main import DIVA
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with chdir(runtime), torch.no_grad():
        model = DIVA()
        assert model.production() == 'happy'
        initial = model.production_art['Art'].copy()
        assert np.count_nonzero(initial) == 1188
        np.testing.assert_array_equal(loadmat('diva_weights_SSM2FF.mat')['W'], initial)
        np.random.seed(args.seed);torch.manual_seed(args.seed)
        model.StartSim(False, None, False)
        raw = model.production_log['VT_0'].copy()
        assert raw.shape == (13, 111)
        assert model.motor_cortex.feedforward_gain == model.motor_cortex.feedback_gain == 1
        np.testing.assert_array_equal(np.asarray(model.vocal_tract.params.Input.Scale), np.ones((13, 1)))
        sound = model.vocal_tract.diva_synth(torch.from_numpy(raw), 'sound')[0].numpy().ravel()
        after = model.motor_cortex.LMC.weight.numpy().copy()
    assert np.isfinite(sound).all() and 1e-4 < np.max(np.abs(sound)) <= 1
    np.savez_compressed(out / 'state.npz', raw=raw, initial_ff=initial, final_ff=after, waveform=sound)
    sf.write(out / 'happy_corrected_native_float.wav', sound, 11025, subtype='FLOAT')
    current = out / 'happy_corrected_native.wav'
    corrected_pcm = (sound * 32767).astype('<i2')
    wavfile.write(current, 11025, corrected_pcm)
    author = out / '1661288163_TORCHDIVA_OUT.wav'
    original = ROOT / 'test/figure_data/1661288163_TORCHDIVA_OUT.wav'
    assert sha(original) == '20cbbe4ebcaca58bb29cfb23178208c48b588ffa75f7c367b4256fdd2b1ed73d'
    shutil.copyfile(original, author)
    fs, author_pcm = wavfile.read(author)
    assert fs == 11025 and author_pcm.dtype == np.int16 and author_pcm.ndim == 1
    silence = np.zeros(round(.5 * fs), dtype=np.int16)
    joined = np.r_[author_pcm, silence, corrected_pcm]
    pair = out / 'happy_author_then_corrected.wav'
    wavfile.write(pair, fs, joined)
    _, decoded = wavfile.read(pair)
    np.testing.assert_array_equal(decoded[:len(author_pcm)], author_pcm)
    np.testing.assert_array_equal(decoded[-len(corrected_pcm):], corrected_pcm)
    assert sha(author) == sha(original)
    prior, prior_fs = sf.read(args.prior_waveform, dtype='float32')
    assert prior_fs == 11025
    np.testing.assert_array_equal(sound.astype('float32'), prior)
    assert before == {name: sha(source / name) for name in expected}
    assert before == {name: sha(staged / name) for name in expected}
    result = {'revision': provenance['revision'], 'source_fingerprint': fingerprint, 'source_files_match': len(expected),
              'training_config': str(args.training_config.resolve()), 'training_config_sha256': sha(args.training_config),
              'source_provenance': provenance, 'script_sha256': sha(Path(__file__)),
              'initialization': 'Stored trained Art; no reset or extra learning iterations',
              'initial_ff_nonzero': int(np.count_nonzero(initial)), 'initial_ff_norm': float(np.linalg.norm(initial)),
              'simulation_runs': 1, 'standard_online_ff_updates_enabled': True,
              'ff_weight_change_norm': float(np.linalg.norm(after - initial)),
              'ff_gain': 1, 'fb_gain': 1, 'fixed_fb_gain': .5, 'seed': args.seed,
              'diffwave': False, 'transfer_peaks': False, 'loudness_normalization': False,
              'resampling': False, 'trimming': False, 'time_stretch': False,
              'sample_rate': fs, 'author_samples': len(author_pcm), 'corrected_samples': len(corrected_pcm),
              'author_duration_s': len(author_pcm)/fs, 'corrected_duration_s': len(corrected_pcm)/fs,
              'silence_samples': len(silence), 'order': ['author', 'silence', 'corrected'],
              'author_sha256': sha(author), 'corrected_sha256': sha(current), 'comparison_sha256': sha(pair),
              'prior_native_float_wav': str(args.prior_waveform.resolve()), 'prior_native_float_sha256': sha(args.prior_waveform),
              'prior_native_float_samples_exact': True, 'source_assets_unchanged': True,
              'author_original_bytes_unchanged': True, 'comparison_pcm_exact': True,
              'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
              'seconds': time.monotonic() - start,
              'limitation': 'Original archived author WAV generation history is unknown; not a claim of identical controller history.'}
    dump(out / 'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_provenance'}, indent=2))


if __name__ == '__main__':
    main()
