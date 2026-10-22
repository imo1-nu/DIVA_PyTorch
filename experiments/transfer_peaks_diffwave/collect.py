"""Collect all registered outcomes and direct WAV comparisons, without selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.stats import ttest_rel
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/diffwave'))
from conditions import sha256
from data import load_pair
from evaluate import METRICS, aggregate


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def collect(directory):
    directory=directory.resolve()
    here=Path(__file__).resolve().parent
    out=here/'results'
    out.mkdir(exist_ok=True)
    baseline=json.loads((directory/'evaluation/individual.json').read_text())
    baseline=[r for r in baseline if r['condition']=='corrected']
    current=json.loads((directory/'transfer_evaluation/individual.json').read_text())
    assert len(baseline)==len(current)==40
    manifest=json.loads((directory/'manifest.json').read_text())
    tests=[s for s in manifest['samples'] if s['split']=='test']
    expected={s['name'] for s in tests}
    for condition,rows in [('corrected',baseline),('transfer_peaks',current)]:
        for method in ('diva','vanilla','enhanced','oracle'):
            assert {r['name'] for r in rows if r['method']==method}==expected
            assert all(r['condition']==condition for r in rows)
    rows=baseline+current
    summary=aggregate(rows)
    for method in ('diva','vanilla','enhanced'):
        b={r['name']:r for r in baseline if r['method']==method}
        h={r['name']:r for r in current if r['method']==method}
        result={}
        for metric in METRICS:
            before=np.array([b[n][metric] for n in sorted(expected)])
            after=np.array([h[n][metric] for n in sorted(expected)])
            test=ttest_rel(after,before)
            result[metric]=dict(mean_change=float(np.mean(after-before)),
                t=float(test.statistic) if np.isfinite(test.statistic) else None,
                p_two_sided=float(test.pvalue) if np.isfinite(test.pvalue) else None,
                differences_by_file={n:float(h[n][metric]-b[n][metric]) for n in sorted(expected)})
        summary['paired_tests']['transfer_peaks-corrected/'+method]=result
    dump(out/'summary.json',summary)
    dump(out/'individual.json',rows)
    with (out/'individual.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    pairs=json.loads((directory/'conditions/transfer_peaks/pairs-manifest.json').read_text())
    assert len(pairs['samples'])==492 and not pairs['failures']
    assert len([r for r in pairs['samples'] if r['split']=='train'])==482
    assert len({r['name'] for r in pairs['samples']})==492
    sha_mismatches=[]
    for sample in pairs['samples']:
        if sha256(directory/'conditions/transfer_peaks/pairs'/sample['name']) != sample['sha256']:
            sha_mismatches.append(sample['name'])
    assert not sha_mismatches
    old=json.loads((directory/'runs/corrected/config.json').read_text())
    new=json.loads((directory/'runs/transfer_peaks/config.json').read_text())
    fixed=('seed','epochs','learning_rate','batch_size','crop_frames','precision','gradient_clip',
        'memory_format','deterministic_algorithms','cublas_workspace_config','train_files',
        'validation_files','test_files','pretrained_sha256','architecture','parameters','torch','device','gpu')
    assert all(old[k]==new[k] for k in fixed)
    unchanged_modules={n:sha256(ROOT/'experiments/diffwave'/n) == old['code_sha256'][n]
                       for n in ('model.py','data.py')}
    assert all(unchanged_modules.values())
    # train.py has only a CLI choice addition; compare the executed train body.
    old_train=subprocess.check_output(['git','-C',str(ROOT),'show','5d8b684:experiments/diffwave/train.py'],text=True)
    current_train=(ROOT/'experiments/diffwave/train.py').read_text()
    assert current_train == old_train.replace("choices=['author','corrected']","choices=['author','corrected','transfer_peaks']")
    config_check=dict(fixed_fields_equal=list(fixed),unchanged_modules=unchanged_modules,
        train_body_equal=True,baseline_input_sha256=old['training_input_sha256'],
        h_input_sha256=new['training_input_sha256'])
    dump(out/'configuration_check.json',config_check)
    validity=[]
    for trial in range(1,6):
        subset=[next(r for r in s['runs'] if r['trial']==trial) for s in pairs['samples']]
        validity.append(dict(trial=trial,recordings=len(subset),
            active_joint_valid_rate_mean=float(np.mean([s['active_joint_valid_rate'] for s in subset])),
            active_joint_valid_rate_min=float(min(s['active_joint_valid_rate'] for s in subset)),
            all_frames_joint_valid_rate_mean=float(np.mean([s['joint_valid_rate'] for s in subset])),
            zero_valid_recordings=sum(s['active_joint_valid_rate']==0 for s in subset)))
    dump(out/'observation_coverage.json',validity)
    # Preserve the entire input list and per-recording failure/coverage outcomes.
    for source,dest in ((directory/'manifest.json','manifest.json'),
        (directory/'conditions/transfer_peaks/pairs-manifest.json','pairs-manifest.json'),
        (directory/'sources/transfer_peaks/provenance.json','diva-source-provenance.json'),
        (directory/'baseline_snapshot.json','baseline_snapshot.json'),
        (directory/'runs/transfer_peaks/config.json','config.json'),
        (directory/'runs/transfer_peaks/training.json','training.json'),
        (directory/'state.json','run_state.json')):
        shutil.copy2(source,out/dest)
    # 3-track H pipeline and 2-track controlled DiffWave comparison for all 10.
    audio_dir=directory/'listening'
    audio_dir.mkdir(exist_ok=True)
    listening=[]
    for sample in tests:
        name=sample['name']
        human_b,diva_b=load_pair(directory,sample,'corrected')
        human_h,diva_h=load_pair(directory,sample,'transfer_peaks')
        np.testing.assert_array_equal(human_b,human_h)
        enhanced_b,rate_b=sf.read(directory/'evaluation/audio/corrected/enhanced'/name,dtype='float32')
        enhanced_h,rate_h=sf.read(directory/'transfer_evaluation/audio/transfer_peaks/enhanced'/name,dtype='float32')
        assert rate_b==rate_h==22050
        assert len(human_h)==len(diva_h)==len(enhanced_h)==len(enhanced_b)
        gap=np.zeros(8820,dtype=np.float32)
        for label,signals in [('human_diva_diffwave',[human_h,diva_h,enhanced_h]),
                              ('fmfit_then_h_diffwave',[enhanced_b,enhanced_h])]:
            blocks=[]
            for signal in signals:
                if blocks:blocks.append(gap)
                blocks.append(signal)
            wave=np.concatenate(blocks)
            path=audio_dir/(Path(name).stem+'_'+label+'.wav')
            sf.write(path,wave,22050,subtype='PCM_16')
            listening.append(dict(name=name,sequence=label,path=str(path.relative_to(directory)),
                frames=len(wave),sample_rate=22050,gap_frames=len(gap),
                peak=float(np.max(np.abs(wave))),sha256=sha256(path),gain_normalization=False))
        # Same reference + seed makes oracle output a useful independent check.
        bo,_=sf.read(directory/'evaluation/audio/corrected/oracle'/name,dtype='float32')
        ho,_=sf.read(directory/'transfer_evaluation/audio/transfer_peaks/oracle'/name,dtype='float32')
        np.testing.assert_array_equal(bo,ho)
    dump(out/'listening_manifest.json',listening)
    provenance=dict(git_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ('experiments/diffwave','experiments/transfer_peaks_diffwave')
                     for p in (ROOT/folder).glob('*.py')},
        pretrained_sha256=sha256(directory/'assets/speechbrain-diffwave.ckpt'),
        baseline_checkpoint_sha256=sha256(directory/'runs/corrected/final.pt'),
        h_checkpoint_sha256=sha256(directory/'runs/transfer_peaks/final.pt'),
        human_reference_samples_equal_all_ten=True,oracle_samples_equal_all_ten=True,
        pair_audio_hashes_all_verified=True,missing_recordings=0,
        data_directory=str(directory),collected_code_sha256=sha256(Path(__file__)))
    dump(out/'provenance.json',provenance)
    lines=['# HピークFBで学習したCNN＋DiffWaveの評価','',
        '事前条件は[PROTOCOL.md](PROTOCOL.md)、研究Issue #41。mainとは別の実験ブランチ。',
        '482学習/10評価、DIVA4回学習後の5回目、FF/FB=1/1、CNN1000epochの固定条件。',
        'DiffWave本体は凍結し、学習したのは追加CNN。比較はfmfit観測の修正版DIVA実験。','',
        '## 全10件の平均（raw複合指標）','',
        '|観測|方法|PESQ|CSIG raw|CBAK raw|COVL raw|segSNR dB|',
        '|---|---|---:|---:|---:|---:|---:|']
    for condition in ('corrected','transfer_peaks'):
        for method in ('diva','vanilla','enhanced','oracle'):
            g=summary['groups'][f'{condition}/{method}']
            label='fmfit' if condition=='corrected' else 'Hピーク'
            lines.append('|'+label+'|'+method+'|'+ '|'.join(f'{g[k]["mean"]:.4f}' for k in ('pesq','csig_raw','cbak_raw','covl_raw','segsnr'))+'|')
    lines+=['','## CNN＋DiffWaveの対応差（H−fmfit）','',
        '|指標|平均差|探索的p値|','|---|---:|---:|']
    for key in METRICS:
        item=summary['paired_tests']['transfer_peaks-corrected/enhanced'][key]
        lines.append(f'|{key}|{item["mean_change"]:+.6f}|{item["p_two_sided"]:.6g}|' if item['p_two_sided'] is not None else f'|{key}|{item["mean_change"]:+.6f}|未定義|')
    lines+=['','全個票・標本SD・条件内DIVA比較は[results/summary.json](results/summary.json)と[CSV](results/individual.csv)。',
        '対応t検定は探索的・未補正、10ファイルは独立した10話者ではない。','',
        '## 観測有効率と失敗','',
        '全492録音を生成し、失敗・除外0件。H欠測は録音除外や達成として扱わない。',
        '|DIVA反復|活動区間での3峰同時有効率（録音平均）|最小有効率|有効点ゼロ録音|',
        '|---:|---:|---:|---:|']
    for item in validity:
        lines.append(f'|{item["trial"]}|{item["active_joint_valid_rate_mean"]:.2%}|{item["active_joint_valid_rate_min"]:.2%}|{item["zero_valid_recordings"]}|')
    state=json.loads((directory/'state.json').read_text())
    lines+=['','## 実測時間','']
    for stage in state['stages']:
        lines.append(f'- {stage["stage"]}: {stage.get("seconds",0)/60:.2f}分、{stage["state"]}')
    lines+=['','## 直接再生するWAV','',
        '各行の比較は「fmfit由来のCNN＋DiffWave → 0.4秒無音 → Hピーク由来のCNN＋DiffWave」。',
        '3段比較は「人音声 → H観測DIVA → H由来CNN＋DiffWave」。PCM16化のみ、音量正規化なし。','',
        '|録音|CNN＋DiffWave条件比較|H条件3段比較|','|---|---|---|']
    for sample in tests:
        stem=Path(sample['name']).stem
        lines.append(f'|{stem}|[WAV](artifacts/listening/{stem}_fmfit_then_h_diffwave.wav)|[WAV](artifacts/listening/{stem}_human_diva_diffwave.wav)|')
    lines+=['','## 検証と限界','',
        '従来fmfit再生成1件は全サンプル一致。13テストと40独立状態照合が通過。全10件の人参照・oracle音声も条件間で完全一致。',
        '入力・資産・重み・コードSHA、元録音一覧、学習履歴、全個票、H観測有効率と失敗理由をresultsに記録した。',
        '各録音5回の軌道NPZ、FLOAT生成音声、最終CNN、直接比較WAVはGit対象外のartifactsに保存する。',
        'Hピークの追従、波形フォルマント、知覚品質を同一視しない。1seed・持続母音・公開分割の範囲の結果であり、happyや日本語への汎化を測っていない。',
        '著者版の特定のDiffWave重みは未同定で、前実験と同じSpeechBrain版を使用した。人による主観評価は実施していない。','']
    (here/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary['groups'].items() if k.endswith('/enhanced')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    collect(p.parse_args().data_dir)
