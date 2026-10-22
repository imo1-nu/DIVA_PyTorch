# happyの聴き比べ

[ユーザー指定のゲインFF=1 / FB=1版](../happy_comparison_gain11/index.html)を追加した。以下は旧ゲインFF=.85 / FB=.30の保存結果。

[比較ページを開く](index.html)。ローカルのHTMLで同じ行の2方式を比較できる。

既存happy目標（500 ms、101点）を変更せず、両モードをゼロFFからseed0・FF=.85/FB=.30（内部.5）・EPS=.5で20回学習した。保存済みArtの学習値を初期FFには使わず、メモリ上でゼロへそろえた。4回／20回後に重みを固定し、同一初期内部状態からFF-only/FF+FBを別生成した。音声は毎回freshなnative合成器とseed0で生成した。

個別WAVは各0.555秒、11025 Hz／16-bit PCM。時間伸縮・切り出し・個別の音量正規化をしていない。A→Bはfmfit、約0.5秒無音、transfer_peaksの順。

|学習|生成|fmfit|transfer_peaks|連続比較|
|---:|---|---|---|---|
|4回|ff_only|[WAV](audio/happy_fmfit_04_ff_only.wav)|[WAV](audio/happy_transfer_peaks_04_ff_only.wav)|[A→B](audio/happy_AB_04_ff_only.wav)|
|4回|ff_fb|[WAV](audio/happy_fmfit_04_ff_fb.wav)|[WAV](audio/happy_transfer_peaks_04_ff_fb.wav)|[A→B](audio/happy_AB_04_ff_fb.wav)|
|20回|ff_only|[WAV](audio/happy_fmfit_20_ff_only.wav)|[WAV](audio/happy_transfer_peaks_20_ff_only.wav)|[A→B](audio/happy_AB_20_ff_only.wav)|
|20回|ff_fb|[WAV](audio/happy_fmfit_20_ff_fb.wav)|[WAV](audio/happy_transfer_peaks_20_ff_fb.wav)|[A→B](audio/happy_AB_20_ff_fb.wav)|

## 記録と検証

8固定評価ともFF重み不変。freshな合成器で20回後FF+FBの新方式を再生成しWAVバイト完全一致も確認した。全WAVの非無音・有限性・sample rate/PCM形式を検証し、連結4本が個別PCM＋無音＋個別PCMと完全一致すること、HTMLの12再生参照が実在することを確認した。観測値・マスクの10ステップ遅延と無効誤差0は実行時に検査した。source/assetの前後SHAは両モードで不変。

|観測|学習|生成|静的Hの全3ピーク有効率|
|---|---:|---|---:|
|fmfit|4|ff_only|95.50%|
|fmfit|4|ff_fb|99.10%|
|fmfit|20|ff_only|81.98%|
|fmfit|20|ff_fb|81.98%|
|transfer_peaks|4|ff_only|84.68%|
|transfer_peaks|4|ff_fb|84.68%|
|transfer_peaks|20|ff_only|81.98%|
|transfer_peaks|20|ff_fb|81.98%|

happyの閉鎖時にはHピークが無効となる。欠測をfmfitへ戻したり、達成と見なしたりしていない。[理由別ログとWAV検証](validation.json)、[WAV情報・SHA](cases.json)、[来歴](provenance.json)。初期バッファのF1–F3無効は別途保持する。

この追加生成は聴取用であり、音の違いを改善とは判定しない。/u/の固定比較結果・条件は変更していない。波形フォルマント、極周波数、自然性・知覚品質の達成へ一般化しない。

全オンライン・固定軌道、raw/fmfit/H/返却値、時刻・マスク・理由、誤差、FF/FB、重み、入力コピーは専用worktreeの`outputs/happy_transfer_peak01/`に保存。聴取ページと12 WAVはこのディレクトリにもコピーしてGit追跡する。

[追加条件](../HAPPY_TRANSFER_PEAK_PROTOCOL.md)、[実行スクリプト](../../../experiments/generate_happy_transfer_comparison.py)、[研究Issue](https://github.com/imo1-nu/stutter_tts_jp/issues/34)、[Draft PR #2](https://github.com/imo1-nu/DIVA_PyTorch/pull/2)。

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/diva-mpl-cache \
  /tmp/stutter_diva_audit_venv/bin/python experiments/generate_happy_transfer_comparison.py \
  --ff-gain 0.85 --fb-gain 0.30 --output /tmp/happy-transfer-peak-repeat
```
