# TorchDIVA + DiffWave 実験結果

集計: 2026-10-22T18:09:06.364594+00:00。ブランチ `experiment/torchdiva-diffwave`。
[実行条件](README.md)と[実行記録](RUN_LOG.md)を併せて参照。
著者の公開DIVA実装を基準にした条件と修正版を比較した。DiffWaveは代替のSpeechBrain重みであり、原著との完全同一条件ではない。

## 今回の所見

- `author`: DIVA単体との平均比較で、上昇はCSIG raw, CBAK raw, COVL raw, segSNR、低下はPESQ。
- `corrected`: DIVA単体との平均比較で、上昇はPESQ, CSIG raw, CBAK raw, COVL raw, segSNR、低下はなし。
- CNN追加後と標準DiffWaveのPESQ平均は、author: 1.1405 / 1.1870、corrected: 1.1464 / 1.2344。
これらは1seed・評価10ファイルの結果であり、全指標での改善や話者全般への有効性を一括して主張しない。
raw複合指標の上昇と1–5制限のスコアを区別する。人による聴取評価は行っていない。

## 完了した処理

- 公開SVDリスト492件、著者版・修正版の計984ペアを生成。失敗・除外0件。
- 各条件で全482学習ファイルを1000エポック（31,000更新）、凍結DiffWaveのCNNだけ学習。
- 両条件について評価10件 × 4手法、全80個票を保存。評価によるモデル選択なし。

## 実測時間

| 処理 | 所要時間 |
| --- | ---: |
| DIVA生成、984ペア、workers18 | 30.96分 |
| CNN学習 `author`、1000epoch | 22.51分 |
| CNN学習 `corrected`、1000epoch | 22.49分 |
| 学習済みモデルの50段推論 `author`、10件平均 | 0.287秒/件 |
| 学習済みモデルの50段推論 `corrected`、10件平均 | 0.282秒/件 |

環境: Intel Core Ultra 7 265K (20 CPUs)、NVIDIA RTX 5090 32GB、RAM 62 GiB。
学習時間は学習ループの実測。mel前処理・環境構築・データ取得・事前検査・誤ったID対応による破棄試行は別。詳細な秒数はJSONに保存。

## 評価10件の平均

PESQ、CSIG、CBAK、COVL、segSNRはいずれも高い値を良い方向とする。
原著に負の複合指標があるため、ここでは切り詰め前を示す。標準の1–5制限版とSDもJSONに保存。

| DIVA条件 | 方法 | PESQ | CSIG raw | CBAK raw | COVL raw | segSNR dB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| author | diva | 1.2500 | -7.2301 | -0.9441 | -3.9670 | -3.8936 |
| author | vanilla | 1.1870 | -1.7789 | -0.7988 | -1.2397 | -3.1667 |
| author | enhanced | 1.1405 | -0.2822 | -0.0704 | -0.2873 | -1.4389 |
| author | oracle | 2.9149 | 4.1751 | 2.7629 | 3.5402 | -1.3739 |
| corrected | diva | 1.0884 | -6.5411 | -0.3803 | -3.4776 | -3.7685 |
| corrected | vanilla | 1.2344 | -1.1894 | -0.3591 | -0.7686 | -3.3241 |
| corrected | enhanced | 1.1464 | -0.2265 | -0.0707 | -0.2428 | -2.0952 |
| corrected | oracle | 2.9149 | 4.1751 | 2.7629 | 3.5402 | -1.3739 |

## 1–5へ制限した複合指標の平均

raw値に加え、標準範囲への制限後も示す。標本SDは集計JSONに保存。

| DIVA条件 | 方法 | CSIG | CBAK | COVL |
| --- | --- | ---: | ---: | ---: |
| author | diva | 1.0000 | 1.0000 | 1.0000 |
| author | vanilla | 1.0000 | 1.0000 | 1.0000 |
| author | enhanced | 1.0141 | 1.0032 | 1.0000 |
| author | oracle | 4.1751 | 2.7629 | 3.5402 |
| corrected | diva | 1.0000 | 1.0000 | 1.0000 |
| corrected | vanilla | 1.0000 | 1.0000 | 1.0000 |
| corrected | enhanced | 1.0856 | 1.0642 | 1.0397 |
| corrected | oracle | 4.1751 | 2.7629 | 3.5402 |

`diva`: 合成音声のみ。`vanilla`: 追加学習なしの標準DiffWave。
`enhanced`: DIVA音声を入力に学習したCNNへ置換。`oracle`: 人音声melを標準DiffWaveへ入力する対照。

## 著者公開Excelとの比較

著者の `figs_1000.xlsx` は個票とSummaryで値が異なる。両集合を別々に載せ、どちらか一方に合わせて成功を判定しない。

| 著者資料 | PESQ | CSIG | CBAK | COVL | segSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| TorchDIVA 個票平均 | 1.0607 | -5.5438 | -0.3725 | -2.9747 | -4.3476 |
| DiffWave epoch1000 個票平均 | 1.0636 | -1.2856 | -0.7928 | -1.0342 | -3.1202 |
| DiffWave Summary保存値 | 1.6223 | 2.8414 | 1.5670 | 2.1110 | -5.5737 |

## 学習したCNNによる変化

以下は各DIVA条件の `enhanced - diva`。対応のある両側t検定、補正なしの探索的p値。
n=10ファイルであり、独立した10話者の評価ではない。

| 条件 | 指標 | 平均差 | p値 |
| --- | --- | ---: | ---: |
| author | pesq | -0.1095 | 0.482749 |
| author | csig_raw | +6.9479 | 8.76116e-08 |
| author | cbak_raw | +0.8737 | 0.000796724 |
| author | covl_raw | +3.6797 | 5.51861e-07 |
| author | segsnr | +2.4547 | 0.00296988 |
| corrected | pesq | +0.0581 | 0.0625516 |
| corrected | csig_raw | +6.3146 | 2.09605e-06 |
| corrected | cbak_raw | +0.3096 | 0.304843 |
| corrected | covl_raw | +3.2348 | 1.87496e-05 |
| corrected | segsnr | +1.6733 | 0.0127135 |

## 修正版と著者版の直接比較

同じファイルに対する `corrected - author`。修正一式と、それに由来するCNN学習の違いを含む。

| 方法 | 指標 | 平均差 | p値 |
| --- | --- | ---: | ---: |
| diva | pesq | -0.1616 | 0.270496 |
| diva | csig_raw | +0.6890 | 0.00708659 |
| diva | cbak_raw | +0.5638 | 0.0039411 |
| diva | covl_raw | +0.4893 | 0.00944788 |
| diva | segsnr | +0.1251 | 0.00278575 |
| vanilla | pesq | +0.0474 | 0.683711 |
| vanilla | csig_raw | +0.5895 | 0.116353 |
| vanilla | cbak_raw | +0.4397 | 0.117587 |
| vanilla | covl_raw | +0.4711 | 0.0989038 |
| vanilla | segsnr | -0.1574 | 0.0866046 |
| enhanced | pesq | +0.0059 | 0.893304 |
| enhanced | csig_raw | +0.0557 | 0.750457 |
| enhanced | cbak_raw | -0.0003 | 0.998057 |
| enhanced | covl_raw | +0.0445 | 0.752128 |
| enhanced | segsnr | -0.6563 | 0.246298 |

## 解釈の制限

両条件はF0だけの比較ではなく、周期・雑音・補間・返却長の修正をまとめて比較している。
CNNは参照図の解釈による16層、DiffWave重みはSpeechBrain版。著者独自の学習コード・重みと同一とは確認できない。
公開リストのファイル分割を保持したため先頭ID 1と2が学習・評価で共有される。
原著の個票とSummaryは一致しないため、都合のよい集合を選んで再現成功と判定しない。
時間合わせは中央切り出しのみで、位相・遅延・音量を評価用の人音声に合わせて最適化していない。
各条件は1学習seed、各評価は1推論seed。seed間のばらつきや話者汎化は評価していない。

## 証拠と再実行

- [個票CSV](results/individual.csv)、[集計とt検定](results/summary.json)
- [使用データ・SHA](results/manifest.json)、[生成ペア・SHA](results/paired-generation.json)
- [著者版の設定・学習履歴](results/author/training.json)、[修正版の設定・学習履歴](results/corrected/training.json)
- [環境](results/environment.json)、[ソースと最終重みのSHA](results/source_provenance.json)
- [学習曲線](results/training_curves.png)、[評価音声のmel比較](results/test_spectrograms.png)（SVGも保存）
- 元音声・生成音声・最終重みは `artifacts/` にも保存（Git対象外）。`/tmp` の消去後もこの作業ツリーに残る。
- 最終CNNは `artifacts/runs/author/final.pt` と `artifacts/runs/corrected/final.pt`、試聴音声は `artifacts/evaluation/audio/`。

取得手順、依存関係、コマンドは [README.md](README.md) に記載。
