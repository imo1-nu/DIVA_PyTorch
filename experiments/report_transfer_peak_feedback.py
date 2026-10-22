"""Report all prespecified frozen evaluations, preserving failure denominators."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evaluations, learning, provenance = [], {}, {}
    for mode in ('fmfit', 'transfer_peaks'):
        folder = args.root / mode
        assert json.loads((folder / 'completed.json').read_text()) == {'trials': 20, 'evaluations': 4}
        learning[mode] = json.loads((folder / 'learning.json').read_text())
        provenance[mode] = json.loads((folder / 'provenance.json').read_text())
        target = np.load(folder / 'inputs/target.npz')
        for row in json.loads((folder / 'evaluations.json').read_text()):
            path = folder / f'frozen_{row["after_trial"]:02d}_{row["generation"]}.npz'
            data = np.load(path)
            central = (data['time_s'] >= .5) & (data['time_s'] < 1.5)
            peaks = data['independent_peaks'][central]
            valid = np.isfinite(peaks)
            errors = np.maximum(0, np.maximum(target['Aud_min'][0, 1:] - peaks, peaks - target['Aud_max'][0, 1:]))
            means = [float(errors[:, i][valid[:, i]].mean()) if valid[:, i].any() else None for i in range(3)]
            np.testing.assert_allclose(means, row['mean_outside_hz'], atol=1e-12, rtol=1e-12)
            # Add strict band inclusion as well as the saved <=.1Hz numerical tolerance.
            strict = valid & (errors == 0)
            evaluations.append({'mode': mode, **row, 'strict_within_rate': strict.mean(axis=0).tolist(),
                                'strict_joint_within_rate': float(strict.all(axis=1).mean())})
    contrasts = []
    for trial in (4, 20):
        for generation in ('ff_only', 'ff_fb'):
            a, b = [next(r for r in evaluations if r['mode'] == mode and r['after_trial'] == trial and r['generation'] == generation)
                    for mode in ('fmfit', 'transfer_peaks')]
            contrasts.append({'after_trial': trial, 'generation': generation,
                              'change_mean_outside_hz': (np.array(b['mean_outside_hz']) - a['mean_outside_hz']).tolist(),
                              'change_overall_mean_hz': b['overall_mean_outside_hz_valid_only'] - a['overall_mean_outside_hz_valid_only'],
                              'change_joint_valid_rate': b['joint_valid_rate'] - a['joint_valid_rate'],
                              'change_somatic_sse': b['somatic_sse'] - a['somatic_sse']})
    dump(args.output / 'evaluations.json', evaluations)
    dump(args.output / 'contrasts.json', contrasts)
    dump(args.output / 'learning.json', learning)
    dump(args.output / 'provenance.json', provenance)
    with (args.output / 'evaluations.csv').open('w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['mode', 'after_trial', 'generation', 'outside1_hz', 'outside2_hz', 'outside3_hz',
                         'mean_hz', 'valid_rate', 'joint_inside_strict', 'somatic_sse'])
        for r in evaluations:
            writer.writerow([r['mode'], r['after_trial'], r['generation'], *r['mean_outside_hz'],
                             r['overall_mean_outside_hz_valid_only'], r['joint_valid_rate'], r['strict_joint_within_rate'], r['somatic_sse']])
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, contrast in zip(axes.flat, contrasts):
        trial, generation = contrast['after_trial'], contrast['generation']
        label = {'ff_only': 'FF only', 'ff_fb': 'FF + FB'}[generation]
        for offset, mode, color in ((-.18, 'fmfit', '#6d819c'), (.18, 'transfer_peaks', '#dc8b28')):
            r = next(r for r in evaluations if r['mode'] == mode and r['after_trial'] == trial and r['generation'] == generation)
            bars = ax.bar(np.arange(3) + offset, r['mean_outside_hz'], .36, label=mode, color=color)
            ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)
        ax.set(title=f'After {trial} trials, {label}', xticks=np.arange(3), xticklabels=['Peak 1', 'Peak 2', 'Peak 3'],
               ylabel='Mean outside-band distance (Hz)')
        ax.margins(y=.22)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Independent static-H evaluation; frozen weights; /u/, seed 0')
    fig.savefig(args.output / 'outside_distances.png', dpi=170)
    plt.close(fig)
    lines = ['# TRANSFER-PEAK-01 固定比較結果', '',
             '2026-10-23。[実装前Issue](https://github.com/imo1-nu/stutter_tts_jp/issues/34)、'
             '[事前条件](../../TRANSFER_PEAK_PROTOCOL.md)、[使用方法](../../TRANSFER_PEAK_USAGE.md)。', '',
             '## 仮説', '',
             '同じ調音状態の静的Hピークで聴覚誤差を計算すると、従来fmfit観測よりHピークの目標帯外距離が減る可能性。'
             '既存逆写像との不一致、体性感覚との競合、姿勢補正、ピーク順位切替も競合説明として残す。', '',
             '## 実装', '',
             '実状態観測のF1–F3のみを変更。既定fmfit、F0、体性感覚8成分、目標、逆写像・姿勢補正、音源・波形合成式は保持。'
             '別インスタンスで幾何を評価し、値・マスク・時刻・理由を同じ10ステップ遅延へ渡す。'
             '欠測は0の輸送値とfalseマスクで表し、聴覚誤差を0にしてから従来の補正・学習へ渡す。'
             '401点初期バッファの既知失敗は再現し、PR #1相当の互換修正を独立コミット00268e7へ分離した。', '',
             '## 方法', '',
             '同じ既存/u/401点目標・ゼロFF・seed0、FFゲイン.85、FBゲイン.30（内部.5）、学習EPS=.5で20回。'
             '4回・20回後の重みを保存し、毎回同一初期内部状態から学習停止のFF-only / FF+FB生成を行った。'
             '固定評価8件すべてで重み完全不変を確認。全411フレームを保存し、主評価は.5≤t<1.5秒の200点。', '',
             '主評価Hは観測関数を呼ばず、raw→独立NumPy輪郭→xy2ab式→面積処理→H式で再計算。'
             'MATLAB式の転記でありMATLAB実行ではない。同じ定義のH最大値の参照で、人体の独立実測ではない。'
             'fs11025 Hz、65536点、prominence1 dB、間隔50 Hz、50–4950 Hz、周波数順先頭3本を固定。'
             '目標はF1=306.56–431.71、F2=1155.02–1771.98、F3=2301.46–2732.63 Hz（丸め表示）。', '',
             '## 結果：固定重み生成', '',
             '帯内率・有効率の分母は欠測を含む全予定観測。表は厳密な帯内率で、保存コードの数値許容差.1 Hzによる率もJSONに併記。'
             '距離は有効点平均なので、有効率低下を伴う場合は単独で改善判定に使わない。体性SSEは8成分の帯外距離二乗和の時間平均。', '',
             '|観測|学習回数|生成|ピーク1/2/3 平均帯外Hz|3成分平均Hz|有効率|全3成分同時帯内率|体性SSE|',
             '|---|---:|---|---|---:|---:|---:|---:|']
    for r in evaluations:
        values = ' / '.join(f'{v:.3f}' for v in r['mean_outside_hz'])
        lines.append(f'|{r["mode"]}|{r["after_trial"]}|{r["generation"]}|{values}|{r["overall_mean_outside_hz_valid_only"]:.3f}|{r["joint_valid_rate"]:.1%}|{r["strict_joint_within_rate"]:.1%}|{r["somatic_sse"]:.6f}|')
    lines += ['', '![独立Hピークの固定評価](outside_distances.png)', '',
              '### 新モード − fmfit（負は距離減少）', '',
              '|学習回数|生成|ピーク1/2/3 ΔHz|全成分平均 ΔHz|体性SSE Δ|', '|---:|---|---|---:|---:|']
    for r in contrasts:
        values = ' / '.join(f'{v:+.3f}' for v in r['change_mean_outside_hz'])
        lines.append(f'|{r["after_trial"]}|{r["generation"]}|{values}|{r["change_overall_mean_hz"]:+.3f}|{r["change_somatic_sse"]:+.6f}|')
    lines += ['', '## 測定不能・処理時間', '']
    for mode in ('fmfit', 'transfer_peaks'):
        rates = [r['measurement_failure_rate'] for r in learning[mode]]
        rows = [r for r in evaluations if r['mode'] == mode]
        lines.append(f'- {mode}: 学習20回の平均測定不能率 {np.mean(rates):.2%}、固定評価有効率 '
                     f'{min(r["joint_valid_rate"] for r in rows):.1%}–{max(r["joint_valid_rate"] for r in rows):.1%}。'
                     f'学習1試行平均 {np.mean([r["seconds"] for r in learning[mode]]):.2f}秒。'
                     f'H観測1回の固定評価平均 {np.mean([r["observation_seconds_mean"] for r in rows])*1000:.2f} ms、'
                     f'最大 {max(r["observation_seconds_max"] for r in rows)*1000:.2f} ms。')
    lines += ['', 'fmfit条件にも保存用の診断H計算を追加したため、上の試行時間を通常fmfit単体の速度とは解釈しない。'
              '静的Hの計算負荷は5 ms刻みの実時間動作を保証しない。周波数格子長を観測窓や追加遅延と解釈しない。', '',
              '## 検証', '',
              '- 固定テスト40状態と保存NPZの独立H再抽出40状態：最大差0 Hz（1 Hz条件を満たす）。',
              '- main2920c35参照の4回：感覚・運動・FF/FB指令・誤差・FF更新が全要素完全一致。参照にはCPU置換と明示的初回ゼロパディングのみ。',
              '- 11テスト成功。新観測→非ゼロの聴覚誤差→fmfit逆写像、NullSpaceGainのfmfit維持、Scale/tanh一回、F0/体性感覚、閉鎖・空・非有限H・不足・リセット・マスク遅延、乱数/合成状態を検証。',
              '- 全保存系列で値・マスクの10ステップ対応と無効誤差0、EPS=.5の全FF更新の再構成を検算。source/assets/inputの前後SHA不変。',
              '- 既存xy2abのcenter02ではAll-NaN axis警告が残る。40状態の採用面積・H・ピークは有限で、警告を成功の代わりに扱っていない。', '',
              '## 解釈と限界', '',
              '以下の差は固定した観測経路の効果であり、制御へ返した同じ値を再表示しただけの判定ではない。'
              '独立式による固定評価8系列の全raw状態の再評価と、別に生成したFF-only結果を含めて比較した。'
              '成分間の悪化、体性誤差、有効率を併記し、学習中の信号を固定評価へ混在させない。'
              '単一/u/・seed0・既存目標・既存inverseに限定する。競合説明の個別アブレーションは行っておらず、原因を一つに確定しない。'
              '波形推定・極周波数・知覚品質を評価していないため、波形一致、自然性、日本語母音の達成へ拡張しない。', '',
              '## 保存と再現', '',
              '[全8条件JSON](evaluations.json)、[CSV](evaluations.csv)、[条件差](contrasts.json)、'
              '[オンライン学習ログ](learning.json)、[コード・入力・資産・環境来歴](provenance.json)、'
              '[main回帰と全系列検算](validation.json)、[40状態独立H照合](reference_check.json)、[テスト](tests.txt)。', '',
              '全NPZ・raw・重み・マスク/理由/時刻・FF/FB・エラー・入力コピー・ソーススナップショットは専用worktreeの'
              '`outputs/transfer_peak01/`へ保存（大きな元データはGit追跡外）。'
              '[実行スクリプト](../../../../experiments/run_transfer_peak_feedback.py)、'
              '[独立参照](../../../../experiments/transfer_peak_reference.py)、'
              '[保存系列検算](../../../../experiments/validate_transfer_peak_run.py)、'
              '[集計/図](../../../../experiments/report_transfer_peak_feedback.py)。', '',
              '比較条件を結果に合わせて変更していない。追加した厳密帯内率は保存済みの.1 Hz許容差率を残して併記し、'
              '全8条件へ一律適用した集計上の明示である。']
    (args.output / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    artifacts = {str(p.relative_to(args.root)): sha(p) for p in sorted(args.root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    dump(args.output / 'artifact_sha256.json', artifacts)
    print(json.dumps(contrasts, indent=2))


if __name__ == '__main__':
    main()
