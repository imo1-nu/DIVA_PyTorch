# 静的Hピーク聴覚FBとCNN＋DiffWave

[実行前条件](PROTOCOL.md)、[研究Issue #41](https://github.com/imo1-nu/stutter_tts_jp/issues/41)。

DIVAの観測F1–F3だけをtransfer_peaksにした音声492件を作り、その482件からCNNを新規学習し、同じ10件で評価する。DiffWave本体は既存のSpeechBrain重みを凍結する。比較対象は前実験の修正版DIVA・fmfit観測。既存実験の未コミット変更は読み込まない。実験ブランチはmainから作り、コミット済みのDIVA観測実装とDiffWaveパイプラインをmergeしている。

`prepare.py`は人音声・資産・corrected条件・最終CNN・元評価結果をSHA照合して別領域へコピーし、H観測実装b89a625のsrcを展開する。元実験と同じfloat目標の初期化対応のみ追加する。`generate.py`は各録音を新規プロセス・一時作業領域で実行する。setup_targetは既定fmfitのVocalTractを先に作るため、シミュレーション前に明示的にHモードのVocalTractとMotorCortexへ差し替える。逆写像と姿勢補正はそのVocalTractのfmfitモデル経路を使う。

各録音の5回分のraw・FF/FB・fmfit/H/返却値・有効マスク・時刻・失敗理由・誤差信号・学習前後重みと目標を圧縮NPZに保存する。5回目も従来通りオンライン重み更新があり、固定重み評価ではない。欠測成分の誤差ゼロとraw=FF+FBを毎系列で照合する。学習対象は追加CNNのみ、条件名以外は元train.py/evaluate.pyの処理を維持する。

## 再実行

作業ツリーのルートから、既存依存環境を使用する例。出力はGit対象外の`artifacts/`。GPU学習・推論はGPUアクセス可能な環境で実行する。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/torchdiva-diffwave/venv/bin/python experiments/transfer_peaks_diffwave/prepare.py \
  --baseline ../torchdiva-diffwave/experiments/diffwave/artifacts \
  --output experiments/transfer_peaks_diffwave/artifacts
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/torchdiva-diffwave/venv/bin/python experiments/transfer_peaks_diffwave/generate.py \
  --data-dir experiments/transfer_peaks_diffwave/artifacts --mode fmfit --limit 1 --workers 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/torchdiva-diffwave/venv/bin/python experiments/transfer_peaks_diffwave/generate.py \
  --data-dir experiments/transfer_peaks_diffwave/artifacts --workers 18
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/torchdiva-diffwave/venv/bin/python experiments/diffwave/train.py \
  --data-dir experiments/transfer_peaks_diffwave/artifacts --condition transfer_peaks \
  --output experiments/transfer_peaks_diffwave/artifacts/runs/transfer_peaks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/torchdiva-diffwave/venv/bin/python experiments/diffwave/evaluate.py \
  --data-dir experiments/transfer_peaks_diffwave/artifacts --conditions transfer_peaks \
  --output experiments/transfer_peaks_diffwave/artifacts/transfer_evaluation
```

既存出力を上書きしないため、再生成時は新しい出力先を使う。途中までの生成はソース・コード・入力・音声SHAの一致するペアのみ再利用する。CNN学習の再開は`--resume`。音声ページは作成しない。
