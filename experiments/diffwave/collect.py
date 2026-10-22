"""Keep compact evidence and a Japanese result report in the repository."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(directory):
    directory=Path(directory).resolve()
    output=HERE/'results';output.mkdir(exist_ok=True)
    generation=json.loads((directory/'paired-generation.json').read_text())
    if len(generation['samples'])!=984 or generation['failures']:
        raise ValueError('Complete generation with no excluded file is required')
    summary=json.loads((directory/'evaluation/summary.json').read_text())
    rows=json.loads((directory/'evaluation/individual.json').read_text())
    if len(rows)!=80:
        raise ValueError('Expected 2 conditions x 4 methods x 10 files')
    for name in ('manifest.json','paired-generation.json'):
        shutil.copy2(directory/name,output/name)
    for condition in ('author','corrected'):
        target=output/condition;target.mkdir(exist_ok=True)
        for name in ('config.json','training.json'):
            shutil.copy2(directory/'runs'/condition/name,target/name)
        shutil.copy2(directory/'sources'/condition/'provenance.json',target/'diva-source-provenance.json')
        training=json.loads((target/'training.json').read_text())
        if training['last_epoch']!=1000 or training['steps']!=31000:
            raise ValueError('Primary run must use 1000 epochs x 31 steps')
    for name in ('summary.json','individual.json','individual.csv'):
        shutil.copy2(directory/'evaluation'/name,output/name)
    snapshots=output/'reference-configs';snapshots.mkdir(exist_ok=True)
    sources={}
    for name in ('hyperparams.yaml','speechbrain-train.yaml','model-card.md'):
        path=directory/'assets'/name
        if path.exists():
            shutil.copy2(path,snapshots/name)
            sources[name]=digest(path)
    for name in ('vendor/model.py','vendor/params.py','assets.py','conditions.py','data.py','model.py','train.py','evaluate.py','run_study.py','collect.py','../diva_setup.py'):
        sources[name]=digest(HERE/name)
    checkpoints={c:digest(directory/'runs'/c/'final.pt') for c in ('author','corrected')}
    # Keep results beyond /tmp cleanup/reboot, without adding large files to Git.
    artifacts=HERE/'artifacts'
    artifacts.mkdir(exist_ok=True)
    for name in ('assets','raw_nsp','human','conditions','sources','runs','evaluation','logs'):
        if (directory/name).resolve() != (artifacts/name).resolve():
            shutil.copytree(directory/name,artifacts/name,dirs_exist_ok=True)
    for name in ('manifest.json','paired-generation.json'):
        if (directory/name).resolve() != (artifacts/name).resolve():
            shutil.copy2(directory/name,artifacts/name)
    for condition,expected in checkpoints.items():
        if digest(artifacts/'runs'/condition/'final.pt') != expected:
            raise ValueError('Persistent checkpoint copy checksum mismatch')
    provenance=dict(collected_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        sources_sha256=sources,final_checkpoints_sha256=checkpoints,
        temporary_run_directory=str(directory),persistent_artifact_directory=str(artifacts),
        evaluation_audio_directory=str(artifacts/'evaluation/audio'))
    (output/'source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for c in ('author','corrected'):
        log=directory/'logs'/('train-'+c+'.log')
        if log.exists():shutil.copy2(log,output/c/'training.log')
    environment=json.loads((output/'environment.json').read_text())
    lines=['# TorchDIVA + DiffWave 実験結果','',
        f'集計: {provenance["collected_utc"]}。ブランチ `experiment/torchdiva-diffwave`。',
        '[実行条件](README.md)と[実行記録](RUN_LOG.md)を併せて参照。',
        '著者の公開DIVA実装を基準にした条件と修正版を比較した。DiffWaveは代替のSpeechBrain重みであり、原著との完全同一条件ではない。','',
        '## 完了した処理','',
        '- 公開SVDリスト492件、著者版・修正版の計984ペアを生成。失敗・除外0件。',
        '- 各条件で全482学習ファイルを1000エポック（31,000更新）、凍結DiffWaveのCNNだけ学習。',
        '- 両条件について評価10件 × 4手法、全80個票を保存。評価によるモデル選択なし。','',
        '## 実測時間','',
        '| 処理 | 所要時間 |','| --- | ---: |',
        f'| DIVA生成、984ペア、workers12 | {generation["seconds"]/60:.2f}分 |']
    for c in ('author','corrected'):
        t=json.loads((output/c/'training.json').read_text())
        lines.append(f'| CNN学習 `{c}`、1000epoch | {t["seconds"]/60:.2f}分 |')
    for c in ('author','corrected'):
        r=[r for r in rows if r['condition']==c and r['method']=='enhanced']
        lines.append(f'| 学習済みモデルの50段推論 `{c}`、10件平均 | {sum(x["synthesis_seconds"] for x in r)/len(r):.3f}秒/件 |')
    lines.extend(['',
        f'環境: {environment["hardware"]["cpu"]}、{environment["hardware"]["gpu"]}、RAM {environment["hardware"]["ram"]}。',
        '学習時間は学習ループの実測。mel前処理・環境構築・データ取得・事前検査は別。詳細な秒数はJSONに保存。','',
        '## 評価10件の平均','',
        'PESQ、CSIG、CBAK、COVL、segSNRはいずれも高い値を良い方向とする。',
        '原著に負の複合指標があるため、ここでは切り詰め前を示す。標準の1–5制限版とSDもJSONに保存。','',
        '| DIVA条件 | 方法 | PESQ | CSIG raw | CBAK raw | COVL raw | segSNR dB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for c in ('author','corrected'):
        for method in ('diva','vanilla','enhanced','oracle'):
            g=summary['groups'][c+'/'+method]
            values=[g[k]['mean'] for k in ('pesq','csig_raw','cbak_raw','covl_raw','segsnr')]
            lines.append('| '+c+' | '+method+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
    lines.extend(['',
        '`diva`: 合成音声のみ。`vanilla`: 追加学習なしの標準DiffWave。',
        '`enhanced`: DIVA音声を入力に学習したCNNへ置換。`oracle`: 人音声melを標準DiffWaveへ入力する対照。','',
        '## 学習したCNNによる変化','',
        '以下は各DIVA条件の `enhanced - diva`。対応のある両側t検定、補正なしの探索的p値。',
        'n=10ファイルであり、独立した10話者の評価ではない。','',
        '| 条件 | 指標 | 平均差 | p値 |','| --- | --- | ---: | ---: |'])
    for c in ('author','corrected'):
        for key in ('pesq','csig_raw','cbak_raw','covl_raw','segsnr'):
            t=summary['paired_tests'][c+'/enhanced-diva'][key]
            p='未定義' if t['p_two_sided'] is None else f'{t["p_two_sided"]:.6g}'
            lines.append(f'| {c} | {key} | {t["mean_change"]:+.4f} | {p} |')
    lines.extend(['','## 解釈の制限','',
        '両条件はF0だけの比較ではなく、周期・雑音・補間・返却長の修正をまとめて比較している。',
        'CNNは参照図の解釈による16層、DiffWave重みはSpeechBrain版。著者独自の学習コード・重みと同一とは確認できない。',
        '公開リストのファイル分割を保持したため先頭ID 1と2が学習・評価で共有される。',
        '原著の個票とSummaryは一致しないため、都合のよい集合を選んで再現成功と判定しない。',
        '時間合わせは中央切り出しのみで、位相・遅延・音量を評価用の人音声に合わせて最適化していない。',
        '各条件は1学習seed、各評価は1推論seed。seed間のばらつきや話者汎化は評価していない。','',
        '## 証拠と再実行','',
        '- [個票CSV](results/individual.csv)、[集計とt検定](results/summary.json)',
        '- [使用データ・SHA](results/manifest.json)、[生成ペア・SHA](results/paired-generation.json)',
        '- [著者版の設定・学習履歴](results/author/training.json)、[修正版の設定・学習履歴](results/corrected/training.json)',
        '- [環境](results/environment.json)、[ソースと最終重みのSHA](results/source_provenance.json)',
        '- 元音声・生成音声・最終重みは `artifacts/` にも保存（Git対象外）。`/tmp` の消去後もこの作業ツリーに残る。',
        '- 最終CNNは `artifacts/runs/author/final.pt` と `artifacts/runs/corrected/final.pt`、試聴音声は `artifacts/evaluation/audio/`。',
        '', '取得手順、依存関係、コマンドは [README.md](README.md) に記載。'])
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    collect(parser.parse_args().data_dir)
