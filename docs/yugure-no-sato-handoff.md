# 夕暮れの里（yugure-no-sato）— 台帳

> **引き継ぎの入口はこのファイル1本。** 新チャットは「この台帳を読んで再開して」で始める。
> 台帳は二層運用（現在地サマリ ＋ 経緯アーカイブ）。維持規則は末尾。

---

# 現在地サマリ

## 今どこか

**フェーズ0（自己修正ループの構築）の停止ポイント① まで完了。承認待ち。**（2026-09-06）

Vite + TypeScript の最小プロジェクトを作り、**WebGPU が実際に取れることを機械確認した**
（結果は下記「WebGPU 環境（確定・2026-09-06 実測）」）。
自己修正ループの本体（スクショ自動化・ゴールデンビュー・自己点検レポート・`npm run verify`）は未着手。
田園・神社・人物・地形変形はもちろん未着手。GitHub Pages も未設定。

## 次の一手

**停止ポイント①の承認後、フェーズ0 本体（A〜D）に着手する。**

- A: Playwright でスクショ自動化（検証ツールなので絶対規則2の例外として許可済み）
- B: ゴールデンビュー3つ（正面・俯瞰・地面すれすれ）。検証用の仮表示は市松模様の地面＋空のグラデーションのみ
- C: 自己点検レポート（平均輝度と分散・極端輝度の比率・GPU エラー件数・フレーム時間）→ `.screenshots/report-<日時>.json`
- D: `npm run verify` の一括実行（dev 起動 → 3視点スクショ → レポート → 停止）

## 決定事項（正本は別ファイル）

📌 **正本: [`decisions.md`](./decisions.md)** — 決定内容と**その理由**はすべてこちらにある。
以下は検索用の索引（1行要約）。判断に使うときは必ず正本を読むこと。

- **[DECISION] 1** 案件種別 = 個人開発の実験。参照元は「Opus 5 で9時間・約400万トークンの雪原シミュレータ」事例。同じ土俵で挑む
- **[DECISION] 2** 題材 = 昔の田舎の日本の集落を散策。**夏の夕暮れで固定**
- **[DECISION] 3** 街の構成 = 半径数百m。田んぼ／あぜ道／小川／石垣／木造家屋数軒／神社（鳥居・石段・杉並木・境内）／遠景の山並みと夕焼け空
- **[DECISION] 4** 触れて応える地形（**中核の見どころ**）= 足跡が残り戻る／水面の波紋と濁り／踏んだ草が倒れて戻る／風で稲が一斉に揺れる
- **[DECISION] 5** 素材ゼロ = テクスチャ・メッシュ・画像に加え**音も持ち込まない**。建物・鳥居はパラメトリック生成
- **[DECISION] 6** 視点 = 三人称・後方カメラ。人物はシルエット調（麦わら帽子・着物の裾など2〜3点で時代性）
- **[DECISION] 7** 人物の動作 = 手続き型。初期フェーズは**4動作のみ**（歩く・走る／立ち止まる／水に入る／見回す）。所作は後半フェーズ
- **[DECISION] 8** 操作 = まず PC（WASD＋マウス）。iPad タッチは後半フェーズ
- **[DECISION] 9** 技術 = 素の WebGPU（WGSL）＋ TypeScript ＋ Vite。描画ライブラリ不使用
- **[DECISION] 10** 自己修正ループを**描画コードより先に**作る
- **[DECISION] 11** 公開 = GitHub 公開リポジトリ・MIT・GitHub Pages。設計側は raw で裏取り
- **[DECISION] 12** モデル運用 = 準備・調査は Opus、長時間の自律実装は Fable 5.1（費用は要確認）
- **[DECISION] 13** 応用先は決めない。まず散策そのものの心地よさを完成させる

## 生きている注意事項

- ⚠️ **素材ファイルを1つでも入れたら実験として失格。** テクスチャ・3Dモデル・画像・音源すべて。
  判定は `git ls-files` にそれらの拡張子が現れないこと。
- ⚠️ **描画ライブラリ禁止。** `dependencies` は原則空。`devDependencies` は vite / typescript / 型定義と、
  自己修正ループに必要な最小限のみ。増やしたくなったら**停止して報告**。
- ⚠️ **仕様に無い選択を迫られたら、実装せず報告して停止する。** 勝手に決めない。
- ⚠️ **区切りごとに push して SHA を報告する。** 設計側は raw で裏取りするため、
  push 済みでないものは存在しないものとして扱われる。
- ⚠️ **public を維持する。** private 化すると設計側の裏取り手段が消える。
- ✅ **解消済み（2026-09-06）**: Chrome 152 安定版で WebGPU が取れることを機械確認した。
  **Canary は不要**（下記「WebGPU 環境」）。
- ⚠️ **`npm install` はグローバル設定の deny で止まる**（`~/.claude/settings.json` の `Bash(npm install*)`）。
  これは意図的なガード。Code が勝手に外さず、**神田さんに自分のターミナルで実行してもらう運用**とする
  （2026-09-06 に本人が選択）。依存を足す必要が出たら、コマンドを提示して待つ。
- ⚠️ **`@types/node` を入れない。** devDependencies を vite / typescript / @webgpu/types の3点に保つため、
  `vite.config.ts` は `tsconfig.json` の型検査対象から外してある（`include: ["src"]`）。
  Vite は設定ファイルを実行時にトランスパイルするだけなので実害はないが、
  **`vite.config.ts` の型は機械保証の外**である点は把握しておくこと。

## ブロッカー

なし。

## ファイルの地図

| ファイル | 役割 | 正本か |
|---|---|---|
| `docs/yugure-no-sato-handoff.md` | **台帳（このファイル）**。引き継ぎの入口 | ○ |
| `docs/decisions.md` | **決定事項リスト**。決定内容＋理由 | ○（台帳の索引は要約にすぎない） |
| `CLAUDE.md` | 案件の絶対規則・台帳の維持規則。Code が起動時に自動で読む | ○ |
| `README.md` | 案件名と一行説明のみ | ○ |
| `LICENSE` | MIT | ○ |
| `.gitignore` | node_modules / dist ほか | ○ |
| `docs/yugure-no-sato-archive-YYYY-MM.md` | 経緯アーカイブ（まだ存在しない。台帳が規定を超えたら作る） | — |

## WebGPU 環境（確定・2026-09-06 実測）

`node tools/probe.mjs` による実測。**Chrome の起動条件4通りすべてで WebGPU が取れた。**

| 起動条件 | 結果 |
|---|---|
| headed（フラグなし） | ✅ |
| headed + `--enable-unsafe-webgpu` | ✅ |
| **`--headless=new`（フラグなし）** | ✅ |
| `--headless=new` + `--enable-unsafe-webgpu` + `--use-angle=metal` | ✅ |

- **特別な起動フラグは不要。** `--enable-unsafe-webgpu` も `--use-angle=metal` も要らない。
- **ヘッドレスでも WebGPU が動く。** パートAで想定していた「ヘッドレスでは動かない」懸念は外れた。
- アダプタ: `vendor: apple` / `architecture: metal-3` / subgroup サイズ 32 固定
- `preferredCanvasFormat: bgra8unorm`
- 主要な上限値: `maxBufferSize` **4,294,967,292**（約4GiB）／`maxStorageBufferBindingSize` 同値／
  `maxComputeWorkgroupSizeX,Y` 1024・`Z` 64／`maxComputeInvocationsPerWorkgroup` 1024／
  `maxComputeWorkgroupsPerDimension` 65,535／`maxComputeWorkgroupStorageSize` 32,768／
  `maxTextureDimension2D` 16,384／`maxStorageBuffersPerShaderStage` 10／`maxColorAttachments` 8
- 使える主な機能: `shader-f16`・`subgroups`（size-control 付き）・`timestamp-query`・`float32-filterable`・
  `float32-blendable`・`dual-source-blending`・`depth-clip-control`・`bgra8unorm-storage`・
  `primitive-index`・`clip-distances`
- 全文は `.screenshots/probe-attempts.json`（gitignore 済み）。再取得は `node tools/probe.mjs`。

## 環境（2026-09-06 実測）

- Node.js **v24.16.0** / npm 11.13.0
- gh CLI: ログイン済み `kanda-houtokukai`（scopes: gist, read:org, repo, workflow）
- ブラウザ: **Google Chrome 152.0.7977.76 安定版のみ**。Chrome Canary・Chromium・Edge は未インストール
- git 2.50.1
- 導入済み devDependencies: vite **8.2.2** / typescript **7.0.2** / @webgpu/types **0.1.72**
  （`dependencies` は空。描画・数学ライブラリは無し＝絶対規則2）

---

# 経緯アーカイブ（版/フェーズの記録）

> 新しいものを上に足す。**11件目を足す前に**、最も古い1件を
> `docs/yugure-no-sato-archive-YYYY-MM.md` へ**原文のまま**移す（要約禁止）。

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

---

# 台帳の維持規則

- 台帳は**毎セッションが起動時に読む**。肥大はコンテキストを毎回浪費する。
- **現在地サマリは常に 1〜2画面以内**に保つ。それ以外は経緯アーカイブへ。
- **版/フェーズの記録は直近 10 件まで。** 11 件目を足す前に、最も古い1件を
  `docs/yugure-no-sato-archive-YYYY-MM.md` へ**原文のまま**移してから追記する。
- **台帳が 3 万文字を超えたら**、まず記録の件数を数える:
  1. 規定数（10件）以内 → 規則は守られている。重い回が続いただけ。何もしない
  2. 規定数超過 → **超過分だけ**をアーカイブへ移す
- **チェックの引き金は「版/フェーズを追記するとき」**。追記の直前に必ず数える。誰かが気づくのを待たない。
- ⚠️ **要約・圧縮による削減は禁止**（アーカイブは原文保存が目的）。
- アーカイブは**月別**に分ける（`yugure-no-sato-archive-YYYY-MM.md`）。
- 移す前に、その中の**いまも効いている制約**（確定値・禁則・落とし穴）を
  「生きている注意事項」へ昇格させる。迷ったら残す側に倒す。
- 決定には `[DECISION]`、教訓には `★` を付け、後から検索できるようにする。
