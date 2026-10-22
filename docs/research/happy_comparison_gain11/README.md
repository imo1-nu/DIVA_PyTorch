# happy：FF=1 / FB=1 の比較

[再生ページ](index.html)。A→Bはfmfit → 約0.5秒無音 → transfer_peaksの順。

ユーザー指定により、学習とFF+FB固定生成のゲインを両方式ともFF=1、FB=1に変更した。内部固定FB係数0.5、ゼロFF、seed0、EPS=.5、既存happy目標（500ms・101点）、逆写像・姿勢補正・遅延・合成コードは維持する。4回・20回後の重みを固定して生成した。参考のFF-onlyはFF=1、FB=0である。

**音声は修正済みDIVA内蔵の音源・声道モデルから直接生成している。DiffWave、ニューラル自然化、生成後の音声補正は使っていない。**「修正」はこの実験以前にmainへ入ったc01cee5のF0参照・声門周期・乱数・補間等のコード修正を指す。波形は11025Hz・PCM16で各0.555秒。個別音量正規化・切り出し・時間伸縮は行わない。

|学習|生成ゲイン|fmfit|transfer_peaks|連続比較|
|---:|---|---|---|---|
|4回|FF=1 / FB=1|[WAV](audio/happy_ff1_fb1_fmfit_04_ff_fb.wav)|[WAV](audio/happy_ff1_fb1_transfer_peaks_04_ff_fb.wav)|[A→B](audio/happy_ff1_fb1_AB_04_ff_fb.wav)|
|4回|FF=1 / FB=0（参考）|[WAV](audio/happy_ff1_fb1_fmfit_04_ff_only.wav)|[WAV](audio/happy_ff1_fb1_transfer_peaks_04_ff_only.wav)|[A→B](audio/happy_ff1_fb1_AB_04_ff_only.wav)|
|20回|FF=1 / FB=1|[WAV](audio/happy_ff1_fb1_fmfit_20_ff_fb.wav)|[WAV](audio/happy_ff1_fb1_transfer_peaks_20_ff_fb.wav)|[A→B](audio/happy_ff1_fb1_AB_20_ff_fb.wav)|
|20回|FF=1 / FB=0（参考）|[WAV](audio/happy_ff1_fb1_fmfit_20_ff_only.wav)|[WAV](audio/happy_ff1_fb1_transfer_peaks_20_ff_only.wav)|[A→B](audio/happy_ff1_fb1_AB_20_ff_only.wav)|

## 有効性と検証

|方式|学習|生成|H全3成分有効率|
|---|---:|---|---:|
|fmfit|4|ff_only|87.39%|
|fmfit|4|ff_fb|90.09%|
|fmfit|20|ff_only|81.98%|
|fmfit|20|ff_fb|81.98%|
|transfer_peaks|4|ff_only|85.59%|
|transfer_peaks|4|ff_fb|85.59%|
|transfer_peaks|20|ff_only|81.98%|
|transfer_peaks|20|ff_fb|81.98%|

8固定評価で重み不変、全48系列でFF+FB=rawを確認し、40回の学習更新をEPS=.5で独立再構成した。観測値とマスクの10ステップ対応、無効誤差0、12再生参照の実在・PCM16/11025Hz/非無音、4連結PCMの完全一致、source/assets前後SHA不変を確認した。閉鎖等の欠測理由は[検証JSON](validation.json)に保持する。

[旧ゲインFF=.85/FB=.30の音声](../happy_comparison/index.html)と[/u/固定比較](../results/transfer_peak01/RESULT.md)の条件・結果は保持した。この聴取用生成から自然性・波形フォルマント・知覚品質の改善を判定しない。

[WAV情報](cases.json)、[来歴](provenance.json)、[変更前後の条件](../HAPPY_TRANSFER_PEAK_PROTOCOL.md)。全軌道・重み・入力・ログとZIPは専用worktreeの`outputs/happy_transfer_peak_gain11/`へ保存。

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/diva-mpl-cache \
  /tmp/stutter_diva_audit_venv/bin/python experiments/generate_happy_transfer_comparison.py \
  --ff-gain 1 --fb-gain 1 --output /tmp/happy-transfer-peak-gain11-repeat
```
