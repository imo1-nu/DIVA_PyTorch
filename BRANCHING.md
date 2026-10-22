# ブランチ運用

このリポジトリでは、著者のオリジナルと独自の実装を分けて管理します。

| 名前 | 用途 | 追跡先 |
| --- | --- | --- |
| `upstream-original` | 著者公開版の参照用。独自変更は加えない | `upstream/main` |
| `main` | 自分用の統合ブランチ。暫定名 `mine` に相当 | `origin/main` |
| `feature/<内容>` | 機能追加 | `main` から作成 |
| `experiment/<内容>` | 再現実験 | `main` から作成 |
| `fix/<内容>` | 修正 | `main` から作成 |
| `chore/<内容>` | 運用や設定の変更 | `main` から作成 |
| `area-synthesis-comparison` | 整理前から継続している音声合成比較の作業 | 既存のコミット・未コミット変更を維持 |
| `archive/previous-f0-fix` | 旧ローカルブランチ `origin` の履歴保存 | 通常の開発には使わない |

リモート `origin` は自分のフォーク
(`https://github.com/imo1-nu/DIVA_PyTorch`) 、`upstream` は著者の公開リポジトリ
(`https://github.com/skinahan/DIVA_PyTorch`) です。
ブランチ名 `main` は著者版ではなく、自分用の統合先を指します。

## 作業を始める

作業中の変更は現在の作業ブランチで保存してから切り替えます。
新しい実装・実験ブランチは、必ず `main` から作成します。

```sh
git switch main
git pull --ff-only origin main
git switch -c experiment/torchdiva-diffwave
```

上の実験ブランチ名は作成例です。
未コミットの別作業がある場合は、それを保存するか、別の worktree を使います。

## 必要な変更を main に統合する

作業ブランチで変更をコミットし、内容に応じた動作確認を行ってから統合します。
実験途中の変更や生成物を一律に `main` へ取り込む必要はありません。

```sh
git switch main
git pull --ff-only origin main
git merge --no-ff experiment/torchdiva-diffwave
git push origin main
```

GitHub のプルリクエストを利用する場合も、統合先は自分の `origin/main` にします。

## 著者版を更新する

`upstream-original` は著者の履歴だけを保ちます。

```sh
git fetch upstream
git switch upstream-original
git merge --ff-only upstream/main
git push origin upstream-original
```

著者の更新を自分の実装へ取り込むときも、作業ブランチで確認します。

```sh
git switch main
git switch -c chore/update-upstream
git merge upstream-original
```

確認できた変更だけを `main` に統合します。
著者版を `main` に強制的に上書きしたり、`upstream-original` に独自実装を混ぜたりしません。
