# TorchDIVA + DiffWave: 著者版と修正版の比較

実験開始日: 2026-10-23。作業ブランチ: `experiment/torchdiva-diffwave`。
[実装監査](IMPLEMENTATION_AUDIT.md)を受け、利用者の指定で以下の2条件を比較する。
原著と同一条件の完全再現ではなく、**公開版を基準にした追試と修正版の比較**である。

## 固定した条件

| 項目 | 条件 |
| --- | --- |
| 著者版 `author` | `9170932e9d77a6bcfb2c00e05df362660d11b3e0` |
| 修正版 `corrected` | `2920c35503711a9faa4780e3a34ed6d74adcd65b` (`main`) |
| 比較対象の差 | F0の参照成分、固定/可変周期、固定/生成雑音、周期間補間、返す音声長。詳細は監査参照 |
| データ | 著者公開リストの482学習 + 10評価。両条件で同一ファイル |
| データ取得 | SVD healthy.zip内の該当NSPをHTTP Rangeで取得し、CRC検証、PCMを変えずWAV化 |
| DIVA教師ターゲット | 著者のParselmouth抽出処理。2秒固定、4回学習し5回目の運動指令から音声生成 |
| 初期化 | 各ファイル・各条件で新規プロセスと一時作業領域。元ブランチの作業用重みを上書きしない |
| seed | 20261023 + SHA256(ファイル名)の先頭32bit。DIVA構築後に再設定。両条件で同じ値 |
| 学習済みDiffWave | SpeechBrain `tts-diffwave-ljspeech`, revision `3a0cbf1`。全254テンソル厳密読込、凍結 |
| 重みSHA256 | `b434ac56f45486ec899e4ef50fb1bab09443bbe4ccc844acae35486e89541fc2` |
| mel | 22050 Hz, FFT1024, hop256, 80 Slaney mel, 0–8000 Hz, magnitude、自然対数圧縮 ln(max(mel,1e-5)) |
| 時間合わせ | 各音声を22050 Hzへ再標本化し、短い方の長さへそれぞれ中央切り出し。音量合わせ・時間伸縮なし |
| CNN | 8 encoder + 8 decoderの16畳み込み、473842パラメータ、時間方向256倍 |
| 学習目的 | DIVA melからCNNが出す特徴を、人音声melから凍結済み標準アップサンプラが出す特徴へL1で合わせる |
| 学習長 | 1000 epochs。各epochは全482件を一度ずつ、shuffle。同じseed・順序、batch16、最終batch2 |
| 最適化 | Adam lr0.001、gradient norm上限10、BF16 autocast、channels_last。seed20261023 |
| 切り出し | 62 mel framesのランダム区間。短い音声は端値でpadし、有効区間だけL1計算 |
| 評価モデル | epoch1000の最終モデル。評価10件によるモデル選択・早期終了なし |
| 推論 | DiffWaveの通常50段、seed固定。評価用音声は学習時のランダム切り出しを使わない |

### 共通の実行対応と未確定事項

`conditions.py` が両コミットの `src` を別領域に展開し、次の最小対応を**両条件に同じように**適用する。

1. 初期ゼロバッファを `max(101,num_steps)` にし、著者の2秒ターゲットに対応。
2. ターゲット最小/最大値がfloatのときの未初期化変数を初期化。
3. NumPyを含む声道形状計算をCPUで実行するため、固定CUDA転送をCPUへ置換。

F0、励振雑音、周期・補間の計算式には追加変更しない。
展開後の各ファイルと全体のSHA256、および適用した置換内容を保存する。
元の `src` と元の作業ブランチの未コミット変更は実験入力にしない。

著者のDiffWave学習コードとチェックポイントは公開ブランチで確認できなかった。
LMNTの既知のチェックポイントURLは403を返したため、SpeechBrainの別の学習済み重みを採用した。
Zhang et al. の本文は15層・encoder8層だが図の文字通りの列は16層になる。
本実験はencoder出力を `1,4,8,16,64,64,64,64`、decoderを
`CT64,C16,CT16,C8,CT8,C4,C1,CT1` と解釈する。
著者独自のCNNを再現できたとは主張しない。この解釈を両条件で固定する。
1000epochは著者の補助Excelの値を採用。batch、crop、BF16、padding等は追加の実装条件。

公開リストの先頭ID 1と2は学習/評価の双方にあり、話者完全分離を主張しない。
公式CSVでファイル名の先頭=話者ID、5番目=録音IDを全件照合した。
52話者・56録音セッションで、分割には公開ファイル一覧をそのまま使う。
監査で指摘した438/44分割と早期終了の草稿は正式実験に採用しなかった。
修正前の2件の動作確認キャッシュも正式条件とは別領域である。

## 実行方法

Pythonの依存版・環境は `results/environment.json`、実測結果は `RESULTS.md` に残す。
以下はリポジトリルートからのコマンド。`PYTHON` は必要な依存関係を入れたPythonに変更する。
GPUを使う学習・推論はRTX5090対応のPyTorch/CUDA環境が必要。

```sh
export PYTHON=/tmp/torchdiva-diffwave/venv/bin/python
export DATA=/tmp/torchdiva-diffwave
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$DATA/mpl"
export PYTHONDONTWRITEBYTECODE=1
$PYTHON experiments/diffwave/assets.py --data-dir "$DATA" --workers 4
$PYTHON experiments/diffwave/data.py --data-dir "$DATA" --workers 18
$PYTHON experiments/diffwave/train.py --data-dir "$DATA" --condition author --output "$DATA/runs/author"
$PYTHON experiments/diffwave/train.py --data-dir "$DATA" --condition corrected --output "$DATA/runs/corrected"
$PYTHON experiments/diffwave/evaluate.py --data-dir "$DATA" --output "$DATA/evaluation"
$PYTHON experiments/diffwave/collect.py --data-dir "$DATA"
```

中断した学習は同じコマンドに `--resume` を追加する。
生成から集計までまとめて実行する場合は
`$PYTHON experiments/diffwave/run_study.py --data-dir "$DATA"` を使う。
最終保存epochからモデル・Adam・乱数状態を復元する。設定や学習コードの変更時は再開を拒否する。
データ生成は入力音声・ソース・生成コード・seed・出力音声のSHAが一致したペアだけ再利用する。

データ、生成音声、チェックポイントは `$DATA` に保存し、大容量ファイルはGitに含めない。
Gitには取得/生成手順、入力と出力のSHA、設定、学習曲線、個票・集計結果、実験記録を保存する。
集計時にはデータ・音声・重みを `experiments/diffwave/artifacts/` にもコピーする
（Git対象外）。`/tmp` の消去後にも、この作業ツリーで学習済み重みを利用できる。

## データの入手性と出典

SVDは登録不要の[Zenodo公開データ](https://doi.org/10.5281/zenodo.16874898)でCC BY 4.0。
全healthy.zipは約6.02GBだが、今回必要な492ファイルだけ取得可能。
旧ファイル名から現行ZIP内のパスへの対応とNSP変換が必要で、コードなしでは多少の作業を要する。
本コードは対応付け、取得、破損検知、WAV化を自動化する。
データはPützer and Barry, Saarbruecken Voice Databaseに帰属する。

- [Kinahan et al., TorchDIVA, PLOS ONE 2023](https://doi.org/10.1371/journal.pone.0281306)
- [Zhang et al., Interspeech 2021](https://arxiv.org/abs/2104.11347)
- [DiffWave, ICLR 2021](https://arxiv.org/abs/2009.09761)
- [SpeechBrainモデル](https://huggingface.co/speechbrain/tts-diffwave-ljspeech)
- [LMNT DiffWaveソース](https://github.com/lmnt-com/diffwave)

## 学習前の検証で修正した事項

SpeechBrainには [0,1] 正規化の `DiffWave.diffwave_mel_spectrogram` もあるが、
この公開モデルのカードと公式学習YAMLが指定するのは
`HifiGAN.mel_spectrogram(compression=True)`、すなわち自然対数melだった。
初回の対照60件は前者で測ってしまったため、正式結果から除外して
`$DATA/diagnostics/wrong-mel/` へ隔離し、正しい前処理で全件再測定した。
この段階ではCNN学習は未開始で、テスト結果に合わせたパラメータ調整は行っていない。
DIVAのターゲット抽出・音声生成はこの修正の影響を受けない。
一次ソースは [公開モデルカード](https://huggingface.co/speechbrain/tts-diffwave-ljspeech/blob/3a0cbf1/README.md) と
[公式学習設定](https://github.com/speechbrain/speechbrain/blob/develop/recipes/LJSpeech/TTS/vocoder/diffwave/hparams/train.yaml)。
ソース内容とSHAは実験記録に保存する。

### 依存環境

実際に使った版は [requirements.txt](requirements.txt) と
[environment.json](results/environment.json) を参照。PyTorchは `2.13.0+cu130`。
このPCでは既存のPyTorch環境を専用venvの `.pth` から参照し、
Parselmouth、PESQ、pystoi、pysepmを専用venvへ追加した。
新規環境ではGPUに対応したPyTorchを先に用意し、そのPythonで次を実行する。

```sh
python -m pip install -r experiments/diffwave/requirements.txt
python -m pip install --no-deps https://github.com/schmiph2/pysepm/archive/7ef88aff2c56201a2d0470aaeb58e77e47a914d2.zip
python -m pytest -q experiments/diffwave/test_pipeline.py
```

PESQがソースビルドになる場合はCコンパイラと対応Pythonの開発ヘッダが必要。
TorchDIVAがimportするsounddeviceにはPortAudioの共有ライブラリも必要だが、音声再生は行わない。
このPCではシステムライブラリは既存のものを使い、PESQ用ヘッダだけ `/tmp` に展開した。
SRMRは今回の指標に含めず、pysepmの未使用のオプション依存は導入していない。

生成FLOAT WAVはPEAKチャンクに作成時刻を含むため、再実行でWAV全体のSHAが変わる場合がある。
そのSHAは個々の保存物の同定用で、数値的再現性の判定には音声サンプルのSHAを使う。
`results/audio_pcm_sha256.json` にfloat32サンプル列のSHAを保存する。
別作業領域での再生成では両DIVA条件のサンプルが完全一致した。

### 録音IDの対応誤りと修正

最初の取得コードはファイル名の先頭をZIPのディレクトリに使っていたが、先頭は話者ID、
5番目が録音IDだった。例えば `12_n_w_26_2200_a_h.wav` の正しい入力は
`2200/vowels/2200-a_h.nsp`。`12/vowels/12-a_h.nsp` ではない。
公式CSVの AufnahmeID / SprecherID / 性別 / 健康区分 / 年齢が全492件で一致することを
確認し、取得時に照合を必須にした。

誤った対応の生成は810ペア完了時点で停止し、関連する音声・マニフェスト・評価を
`$DATA/diagnostics/wrong-recording-id/` へ隔離した。正式CNN学習は未開始だった。
正しい元データを取得し直し、DIVA音声も両条件で再生成する。
誤ったファイル名の下に保存されていたNSPは、ZIPの実際のメンバー名とSHAを
照合できた442件だけ、正しいファイル名の取得キャッシュへ再配置した。
生成済みDIVA音声や対照指標はこのキャッシュ移行の対象にしていない。

公式の根拠: https://stimmdb.coli.uni-saarland.de/data/voice_data.csv 。
この誤りを防ぐ回帰テストも追加した。過去の実装監査にある暫定IDの説明は、
この公式メタデータ照合結果で置き換える。

### 音質評価の定義

評価は16 kHzへ再標本化した同じ長さの信号で行い、PESQはwidebandモード。
pysepmの固定コミットからLLR、WSS、segSNRを計算し、次の複合指標を保存する。

```text
CSIG = 3.093 - 1.029*LLR + 0.603*PESQ - 0.009*WSS
CBAK = 1.634 + 0.478*PESQ - 0.007*WSS + 0.063*segSNR
COVL = 1.594 + 0.805*PESQ - 0.512*LLR - 0.007*WSS
```

原著に負のCSIG等があるため、無制限のraw値と、標準の1–5制限版を両方示す。
segSNRのフレーム値はpysepmに従い -10〜35 dBへ制限。
平均、標本SD（ddof=1）、同一10ファイルに対する対応のある両側t検定を保存する。
検定は探索的・多重比較補正なし。個々のp値だけで一般的な改善を断定しない。
`enhanced - diva` に加え、同じ方法の `corrected - author` も直接比較する。
学習・推論は各1seedで、評価10ファイルは2話者からの音声である。
