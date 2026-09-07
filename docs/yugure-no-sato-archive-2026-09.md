# 夕暮れの里 — 経緯アーカイブ 2026-09

> 台帳（`yugure-no-sato-handoff.md`）から**原文のまま**移した記録。
> 要約・圧縮はしない（原文保存が目的）。新しいものが上。

---

## 2026-09-06 — セットアップ（フェーズ開始前）

**やったこと**

- `~/dev/yugure-no-sato` を作成し `git init`（既定ブランチ main）
- GitHub 公開リポジトリ `kanda-houtokukai/yugure-no-sato` を作成（MIT）
- `.gitignore` / `LICENSE` / `README.md` を配置
- dev-workflow の `claude-md-template.md` から `CLAUDE.md` を作成し、案件固有の絶対規則
  （素材ゼロ／外部ライブラリ禁止／区切りで push＋SHA報告／台帳二層運用／仕様外は停止して報告）を記載
- `docs/decisions.md`（決定事項 13 件の正本）と本台帳を初期化

**やっていないこと（意図的）**

- アプリのソースコード・`package.json` の作成 → 次の指示「フェーズ0: 自己修正ループの構築」で扱う
- 依存パッケージのインストール → 同上
- GitHub Pages の設定 → フェーズ1以降で扱う

**★教訓**

- Chrome Canary が無い環境だと分かった時点で、「安定版で WebGPU が動くか」は
  **推測せずフェーズ0の最初に機械確認する**項目として台帳に繰り越した。
  ツール側の制約で判定できないものを「たぶん動く」で通さない。
