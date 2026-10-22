# DiffWave学習時の修正版DIVA：happyと著者公開WAV

[連続比較WAV](happy_author_then_corrected.wav)：著者の1661288163_TORCHDIVA_OUT.wav → 約0.5秒無音 → 修正版DIVA。

[修正版DIVA単体](happy_corrected_native.wav)、[著者公開WAVの無加工コピー](1661288163_TORCHDIVA_OUT.wav)。HTMLページは作成していない。

## 条件

- DiffWave学習のcorrected条件と同じrevision2920c35503711a9faa4780e3a34ed6d74adcd65b、同じCPU等の実行用アダプタ。77ファイルのSHAとsource fingerprintを正式学習configへ照合して一致。
- 組み込みhappyの保存済み学習済みArt（101×13、非ゼロ1188要素）をそのままロード。ゼロFF化、追加の反復学習、新しい目標抽出は行わない。
- FF=1、FB=1、既存内部FB係数.5。seed20261023は当時のhappy生成記録と同じ。通常のStartSimを1回実行し、VT_0（13×111）をnative合成器でWAV化。その1回の通常オンラインFF更新は維持し、FF変化ノルムは.283878。
- DIVA内蔵の声門音源・声道合成を使用する。DiffWave、Hピーク観測、生成後の音声補正を使用しない。
- 両音声とも11025Hz/PCM16。音量正規化・切り出し・時間伸縮・リサンプルなし。著者6175サンプル（.560091秒）、修正版6119サンプル（.555011秒）の全長を保持。

## 検証と来歴

今回の修正版波形をfloat32へ変換すると、既存DiffWave実験の`happy_corrected_diva_11025_float.wav`と全サンプル完全一致。DiffWave推論を実行せずに、その入力となったnative波形を再現した。

著者WAVのSHA256は`20cbbe4ebcaca58bb29cfb23178208c48b588ffa75f7c367b4256fdd2b1ed73d`で、保存ファイルとバイト完全一致。連結WAVの前後部分はそれぞれのPCMと完全一致し、source/assetsの前後SHAも不変。正式学習のsource fingerprintは`f9cdee5d7a9027dd9152f621af39592928b4b19b154837684a0f317a8a0b8276`。

著者公開WAVの厳密な初期FF・学習履歴・seedは不明であり、すべての制御条件を一致させた著者音声の完全再現とは主張しない。聴取用に形式と比較順をそろえた。先行のゼロFFから4/20回学習したhappy・Hピーク比較の結果は保持する。

[全条件・検証・SHA](manifest.json)、[実行前条件](../DIFFWAVE_CORRECTED_HAPPY_PROTOCOL.md)、[生成コード](../../../experiments/generate_diffwave_corrected_happy_author.py)。全source・runtime・運動指令・初期/最終FF・FLOAT波形・ログは専用worktreeの`outputs/diffwave_corrected_happy_author/`へ保存。

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/diva-mpl-cache \
  /tmp/torchdiva-diffwave/venv/bin/python experiments/generate_diffwave_corrected_happy_author.py \
  --output /tmp/diffwave-corrected-happy-author-repeat
```
