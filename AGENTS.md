# 開発ブランチのルール

- ブランチ運用は [BRANCHING.md](BRANCHING.md) に従う。
- 自分用の統合先は `main`。新しい実装・実験は `main` から作業ブランチを作って行い、必要な変更を確認後に `main` へ統合する。
- 著者版 `upstream-original` は `upstream/main` と揃え、独自変更を加えない。
- 別ブランチの未コミット変更を新しい作業へ持ち込まない。既存の変更を保存するか、別の worktree を使う。
