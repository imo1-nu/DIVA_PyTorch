"""Native happy audio: matched zero FF, frozen checkpoints, both observers."""
import argparse
import copy
from contextlib import chdir
import html
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.io import loadmat, wavfile
import torch

from run_transfer_peak_feedback import Recorder, dump, sha, inventory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from main import DIVA
from diva_vocaltract import VocalTract
from diva_motorcortex import MotorCortex
from diva_cerebellum import Cerebellum


def seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def build(runtime, mode, ff_gain, fb_gain):
    runtime.mkdir(parents=True)
    for p in (ROOT / 'src').iterdir():
        if p.suffix in ('.mat', '.txt', '.csv'):
            shutil.copyfile(p, runtime / p.name)
    with chdir(runtime):
        seed()
        diva = DIVA(observation_mode=mode)
        assert diva.production() == 'happy'
        count = len(diva.production_art['Art'])
        assert count == 101
        diva.vocal_tract = VocalTract(mode)
        diva.motor_cortex = MotorCortex(diva.vocal_tract)
        diva.motor_cortex.feedforward_gain = ff_gain
        diva.motor_cortex.feedback_gain = fb_gain
        diva.motor_cortex.LMC.get_set_weights(3, torch.zeros((count, 13), dtype=torch.float64))
        diva.cerebellum = Cerebellum(count)
    return diva


def generate(diva, recorder, runtime):
    recorder.rows = {}
    start = time.monotonic()
    with chdir(runtime), torch.no_grad():
        diva.StartSim(False, None, False)
    data = recorder.arrays()
    data['weight'] = diva.motor_cortex.LMC.weight.numpy().copy()
    data['observation_step'] = np.arange(len(data['raw']))
    data['time_s'] = data['observation_step'] * .005
    data['seconds'] = np.array(time.monotonic() - start)
    for k in ('weight', 'raw', 'fmfit', 'returned', 'ff', 'fb', 'auditory_error', 'somatic_error'):
        assert np.isfinite(data[k]).all(), k
    np.testing.assert_allclose(data['ff'] + data['fb'], data['raw'], rtol=1e-13, atol=1e-13)
    n = len(data['raw'])
    assert n == 111
    np.testing.assert_array_equal(data['perceived'][10:], data['returned'][:-10])
    np.testing.assert_array_equal(data['delayed_valid'][10:], data['valid'][:-10])
    if diva.observation_mode == 'transfer_peaks':
        assert not data['auditory_error'][~data['delayed_valid']].any()
        assert not data['delayed_auditory_error'][1:][~data['delayed_valid'][:-1]].any()
    return data


def waveform(data, runtime, destination):
    seed()
    with chdir(runtime), torch.no_grad():
        # Fresh sound state; no calls to observation. Scale and tanh once.
        vt = VocalTract()
        raw = torch.tensor(data['raw'].T, dtype=torch.float64)
        scale = torch.tensor(vt.params.Input.Scale, dtype=torch.float64)
        sound = vt.diva_synth(raw * scale, 'sound')[0].flatten().numpy()
    assert np.isfinite(sound).all()
    assert float(np.max(np.abs(sound))) > 1e-4
    assert float(np.max(np.abs(sound))) <= 1.
    pcm = (sound * (2**15 - 1)).astype('<i2')
    wavfile.write(destination, 11025, pcm)
    return {'file': destination.name, 'sample_rate': 11025, 'samples': len(pcm),
            'duration_s': len(pcm) / 11025, 'peak': float(np.max(np.abs(sound))),
            'rms': float(np.sqrt(np.mean(sound**2))), 'sha256': sha(destination)}


def worker(out, mode, ff_gain, fb_gain):
    folder = out / mode
    folder.mkdir()
    runtime = folder / 'runtime'
    before = inventory(ROOT / 'src')
    model = build(runtime, mode, ff_gain, fb_gain)
    clean = copy.deepcopy(model)
    observer = Recorder(model, True)
    learning, weights = [], {}
    for trial in range(1, 21):
        data = generate(model, observer, runtime)
        np.savez_compressed(folder / f'learning_{trial:02d}.npz', **data)
        learning.append({'trial': trial, 'seconds': float(data['seconds']),
                         'h_component_valid_rate': data['h_valid'].mean(axis=0).tolist(),
                         'weight_norm': float(np.linalg.norm(data['weight']))})
        dump(folder / 'learning.json', learning)
        if trial in (4, 20):
            weights[trial] = data['weight'].copy()
        print(f'{mode} happy learning {trial}/20', flush=True)
    cases = []
    for trial in (4, 20):
        for generation, gain in (('ff_only', 0.), ('ff_fb', fb_gain)):
            current = copy.deepcopy(clean)
            current.motor_cortex.LMC.get_set_weights(3, torch.tensor(weights[trial]))
            current.motor_cortex.feedback_gain = gain
            recorded = Recorder(current, False)
            data = generate(current, recorded, runtime)
            np.testing.assert_array_equal(data['weight'], weights[trial])
            np.savez_compressed(folder / f'frozen_{trial:02d}_{generation}.npz', **data)
            tag = f'ff{ff_gain:g}_fb{fb_gain:g}'.replace('.', 'p')
            name = f'happy_{tag}_{mode}_{trial:02d}_{generation}.wav'
            sound = waveform(data, runtime, out / 'audio' / name)
            cases.append({'observation_mode': mode, 'after_trial': trial, 'generation': generation,
                          'h_component_valid_rate': data['h_valid'].mean(axis=0).tolist(),
                          'h_joint_valid_rate': float(data['h_valid'].all(axis=1).mean()),
                          'learning_ff_gain': ff_gain, 'learning_fb_gain': fb_gain,
                          'generation_ff_gain': ff_gain, 'generation_fb_gain': gain,
                          'waveform_synthesizer': 'DIVA native, corrected main c01cee5',
                          'diffwave': False, 'weight_unchanged': True, **sound})
            dump(folder / 'cases.json', cases)
            print(f'{mode} happy audio {name}: {sound["duration_s"]:.3f}s', flush=True)
    assert before == inventory(ROOT / 'src')
    dump(folder / 'integrity.json', {'source_assets_unchanged': True, 'signal_delay_checks': True})


def page(out):
    cases = sum([json.loads((out / mode / 'cases.json').read_text()) for mode in ('fmfit', 'transfer_peaks')], [])
    config = json.loads((out / 'provenance.json').read_text())
    ff_gain, fb_gain = config['ff_gain'], config['fb_gain']
    tag = f'ff{ff_gain:g}_fb{fb_gain:g}'.replace('.', 'p')
    entries = []
    for trial in (4, 20):
        for generation in ('ff_only', 'ff_fb'):
            a, b = [next(c for c in cases if c['observation_mode'] == mode and c['after_trial'] == trial and c['generation'] == generation)
                    for mode in ('fmfit', 'transfer_peaks')]
            fs, x = wavfile.read(out / 'audio' / a['file'])
            fs2, y = wavfile.read(out / 'audio' / b['file'])
            assert fs == fs2 == 11025
            pair = np.r_[x, np.zeros(round(.5 * fs), dtype=x.dtype), y]
            name = f'happy_{tag}_AB_{trial:02d}_{generation}.wav'
            wavfile.write(out / 'audio' / name, fs, pair)
            entries.append({'trial': trial, 'generation': generation, 'a': a, 'b': b, 'pair': name})
    rows = []
    for e in entries:
        label = 'FFのみ' if e['generation'] == 'ff_only' else 'FF + FB'
        cells = []
        for c in (e['a'], e['b']):
            name = html.escape(c['file'])
            cells.append(f'<td><audio controls preload="none" src="audio/{name}"></audio>'
                         f'<p><a download href="audio/{name}">WAV</a> · H全3成分有効 {c["h_joint_valid_rate"]:.1%}</p></td>')
        pair = html.escape(e['pair'])
        rows.append(f'<tr><th>{e["trial"]}回後<br>{label}</th>{"".join(cells)}'
                    f'<td><audio controls preload="none" src="audio/{pair}"></audio>'
                    f'<p><a download href="audio/{pair}">A→B連続再生WAV</a></p></td></tr>')
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>happy 聴き比べ</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#172a3a;background:#f7f9fb}h1{font-size:28px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:16px;border:1px solid #d8e1e8;text-align:left}thead th{background:#eaf0f5}audio{width:260px;max-width:100%}td p{margin:6px 0;font-size:13px}a{color:#1265a0}.wrap{overflow-x:auto}p{max-width:1000px}</style>
<h1>happy：従来観測とHピーク観測（FF=__FF_GAIN__ / FB=__FB_GAIN__）</h1>
<p>既存happy目標（500 ms）。両方式ともゼロFF・seed 0・FFゲイン__FF_GAIN__・FBゲイン__FB_GAIN__で学習し、4回／20回後に重みを固定して生成しました。列の同じ行を比べてください。FF+FBの生成ゲインは学習時と同じで、FFのみの参考生成ではFBを0にします（内部固定FB係数0.5）。</p>
<p>連続再生は <strong>A: fmfit → 0.5秒無音 → B: transfer_peaks</strong> の順です。音声はDIVA内蔵の音源・声道合成器から直接生成しています。DiffWaveや生成後の音声補正は使っていません。11025 Hz／16-bit PCMで、個別の音量正規化や時間伸縮はしていません。Hの欠測は記録しており、下の有効率は各111時点の静的H計算についての値です。</p>
<div class="wrap"><table><thead><tr><th>固定評価</th><th>A: fmfit（従来）</th><th>B: transfer_peaks</th><th>A → B</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>ここで返す値は静的Hのピークです。波形フォルマント・極周波数・知覚品質の改善を示すものではありません。<a href="cases.json">全WAVの設定・有効率・SHA</a>。</p></html>'''
    content = content.replace('__FF_GAIN__', f'{ff_gain:g}').replace('__FB_GAIN__', f'{fb_gain:g}')
    (out / 'index.html').write_text(content)
    dump(out / 'cases.json', cases)
    dump(out / 'pairs.json', entries)
    for e in entries:
        for name in (e['a']['file'], e['b']['file'], e['pair']):
            fs, pcm = wavfile.read(out / 'audio' / name)
            assert fs == 11025 and pcm.dtype == np.int16 and np.any(pcm)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ff-gain', type=float, default=1.)
    parser.add_argument('--fb-gain', type=float, default=1.)
    parser.add_argument('--worker', choices=('fmfit', 'transfer_peaks'))
    args = parser.parse_args()
    out = args.output.resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.worker:
        worker(out, args.worker, args.ff_gain, args.fb_gain)
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / 'audio').mkdir()
    (out / 'inputs').mkdir()
    for name in ('diva_000001.mat', 'diva_000001.txt', 'diva_synth.mat'):
        shutil.copyfile(ROOT / 'src' / name, out / 'inputs' / name)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    shutil.copyfile(ROOT / 'docs/research/HAPPY_TRANSFER_PEAK_PROTOCOL.md', out / 'protocol.md')
    before = inventory(ROOT / 'src')
    dump(out / 'provenance.json', {'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                                  'script_sha256': sha(Path(__file__)), 'sources': before, 'inputs': inventory(out / 'inputs'),
                                  'initial_ff': 'all zeros; original saved Art not used as initial FF',
                                  'seed': 0, 'ff_gain': args.ff_gain, 'fb_gain': args.fb_gain, 'fixed_fb_gain': .5, 'learning_EPS': .5,
                                  'target_steps': 101, 'simulation_frames': 111, 'learning_runs': 20,
                                  'frozen_checkpoints': [4, 20], 'sample_rate': 11025,
                                  'waveform_synthesizer': 'DIVA native, corrected main c01cee5', 'diffwave': False, 'python': sys.version,
                                  'torch': torch.__version__, 'numpy': np.__version__})
    jobs = []
    for mode in ('fmfit', 'transfer_peaks'):
        log = (out / (mode + '.log')).open('w')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--worker', mode, '--ff-gain', str(args.ff_gain), '--fb-gain', str(args.fb_gain)], stdout=log, stderr=subprocess.STDOUT)
        jobs.append((mode, process, log))
    for mode, process, log in jobs:
        result = process.wait()
        log.close()
        if result:
            raise RuntimeError(f'{mode} worker failed; see {out / (mode + ".log")}')
    page(out)
    assert before == inventory(ROOT / 'src')
    dump(out / 'artifact_sha256.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    print(out / 'index.html', flush=True)


if __name__ == '__main__':
    main()
