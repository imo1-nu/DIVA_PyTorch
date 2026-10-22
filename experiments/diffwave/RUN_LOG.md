# 実験実行記録

日時は日本時間。正式結果の集計は [RESULTS.md](RESULTS.md)、設定は [README.md](README.md)。

## 2026-10-23: 実験前の確認と準備

- 著者版 `9170932` と統合版 `2920c35` の差を監査し、監査報告を `4f9b9cd` に保存。
- 利用者の指定により、著者版基準 `author` と修正版 `corrected` を両方実行する。
- 元の `area-synthesis-comparison` の作業は別worktreeに残し、取り込まない。
- SVDの公開ファイル一覧492件を取得。NSPのCRCを全件照合し、WAVが元のPCMと全件一致することを確認。
  NSP合計61,780,386 bytes、WAV合計61,772,514 bytes。healthy.zip全体の取得は不要だった。
- LMNT重みURLは403。代替としてSpeechBrain `3a0cbf1` の重みを取得、SHA256確認、254テンソル厳密読込。
- 専用venvにParselmouth、PESQ、pysepm等を追加。既存のPyTorch環境は変更しなかった。
  PESQビルドにはUbuntuのPython3.12ヘッダdebを `/tmp` へ展開して使用（システムへのインストールなし）。
- CPUで2条件各1ファイルの動作確認: 2/2成功、約28秒。元ソース3ファイルの草稿変更は戻し、
  必要な実行対応は `conditions.py` で条件別ソースコピーへ同じように適用する方式に変更。
- 正式生成をworkers12で開始。コンストラクタ後のseed再設定、コード/入力/出力SHAによるキャッシュ検証を有効化。
- GPU速度試測: batch16で約0.055秒/step、channels_lastで約0.042秒/step（定常時）。
  これらは性能の事前試測で、正式学習時間とは分けて扱う。
- 初回対照評価で使った [0,1] melは、この公開重みの公式設定と異なることが分かった。
  正しい自然対数melへ修正。誤った60個票は `diagnostics/wrong-mel/` へ隔離し、正式結果から除外。
  この時点ではCNN学習未開始。データ生成に影響なし。
- 自然対数melによる対照評価60件が完走。人音声入力のoracleも測り、ボコーダの動作を確認。
  正式集計時には全80条件を同じ評価コード・決定的実行設定で改めて測定する。
- 小規模CPUテスト4件成功: キャッシュ化したノイズ予測の元forwardとの一致、CNN形状と凍結重み、
  paddingのloss除外、固定seedの推論再現性。
- 隔離した16件・2epochのGPU動作確認: L1 0.540694 → 0.431381、保存成功。
  正式モデルはこの重みを引き継がず、初期化から全482件で学習する。
- 中断への備えとして `study-state.json`、条件別 `training.json`、再開可能な `last.pt`、実行ログを保存する。
  `run_study.py --wait-for-pairs` は進行中の生成完了後に2学習、評価、集計を順に実行する。

## 再開・状態確認

```sh
cat /tmp/torchdiva-diffwave/study-state.json
cat /tmp/torchdiva-diffwave/pairs-progress.json
tail -n 5 /tmp/torchdiva-diffwave/logs/train-author.log
tail -n 5 /tmp/torchdiva-diffwave/logs/train-corrected.log
```

既に実行中なら同じ処理を重ねて起動しない。中断した学習はREADMEの `--resume` を使う。
`/tmp` を失った場合は、取得・生成からやり直す。小容量のマニフェストと実験結果はGitに残す。

- 再開検証: 16件・3epochにおいて、中断再開と連続実行の全112保存テンソルが完全一致。
  証拠は `results/resume_check.json`。元の未コミット17ファイルもSHA256全件一致。
