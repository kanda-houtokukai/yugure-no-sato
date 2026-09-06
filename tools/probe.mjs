// この Mac の Chrome で WebGPU が実際に取れるかを機械確認する（フェーズ0 停止ポイント①で使用）。
// 依存パッケージなし（Node 標準のみ）。Chrome を起動条件を変えながら開き、
// ページが送ってきた診断 JSON を読む。どの起動条件で通ったかも記録する。
//
// 実測結果（2026-09-06）: 4通りすべて成功。特別な起動フラグは不要でヘッドレスでも動く。
// 環境が変わったとき（Chrome 更新・別マシン）に再実行して前提を取り直すためのもの。

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ROOT, sleep, startVite, waitForServer } from './dev-server.mjs';

const OUT_DIR = join(ROOT, '.screenshots');
const LATEST = join(OUT_DIR, 'probe-latest.json');
const PORT = 5199;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const PAGE_URL = `${ORIGIN}/?mode=probe`;

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

/** 起動条件の候補。上から順に試し、最初に通ったものを「動く組み合わせ」として記録する。 */
const CANDIDATES = [
  { name: 'headed (フラグなし)', flags: [] },
  { name: 'headed + --enable-unsafe-webgpu', flags: ['--enable-unsafe-webgpu'] },
  { name: 'headless=new', flags: ['--headless=new'] },
  {
    name: 'headless=new + --enable-unsafe-webgpu + --use-angle=metal',
    flags: ['--headless=new', '--enable-unsafe-webgpu', '--use-angle=metal'],
  },
];

async function tryCandidate(candidate) {
  rmSync(LATEST, { force: true });
  const profile = mkdtempSync(join(tmpdir(), 'yugure-chrome-'));
  const args = [
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-timer-throttling',
    '--window-size=900,700',
    ...candidate.flags,
    PAGE_URL,
  ];
  const chrome = spawn(CHROME, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  const chromeLog = [];
  chrome.stdout.on('data', (d) => chromeLog.push(String(d)));
  chrome.stderr.on('data', (d) => chromeLog.push(String(d)));

  let report = null;
  const deadline = Date.now() + 25000;
  while (Date.now() < deadline) {
    if (existsSync(LATEST)) {
      try {
        report = JSON.parse(readFileSync(LATEST, 'utf8'));
        break;
      } catch {
        /* 書き込み途中。次のループで読み直す */
      }
    }
    await sleep(250);
  }

  chrome.kill('SIGTERM');
  await sleep(500);
  if (!chrome.killed) chrome.kill('SIGKILL');
  rmSync(profile, { recursive: true, force: true });

  return {
    candidate: candidate.name,
    flags: candidate.flags,
    ok: report?.ok === true,
    stage: report?.stage ?? null,
    error: report?.error ?? (report ? null : 'ページから診断が届かなかった（タイムアウト）'),
    report,
    chromeStderrTail: chromeLog.join('').split('\n').slice(-8).join('\n').trim() || null,
  };
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true });
  const vite = startVite(PORT, { quiet: false });
  const cleanup = () => vite.kill('SIGTERM');
  process.on('exit', cleanup);
  process.on('SIGINT', () => {
    cleanup();
    process.exit(130);
  });

  if (!(await waitForServer(ORIGIN))) {
    console.error('dev サーバが起動しなかった');
    cleanup();
    process.exit(2);
  }

  const attempts = [];
  let winner = null;
  for (const candidate of CANDIDATES) {
    process.stdout.write(`\n--- 試行: ${candidate.name} ---\n`);
    const result = await tryCandidate(candidate);
    attempts.push(result);
    process.stdout.write(
      `  結果: ${result.ok ? 'OK' : 'NG'}  stage=${result.stage ?? '-'}  ${result.error ?? ''}\n`,
    );
    if (result.ok && !winner) winner = result;
  }

  writeFileSync(
    join(OUT_DIR, 'probe-attempts.json'),
    JSON.stringify({ generatedAt: new Date().toISOString(), attempts }, null, 2),
  );

  console.log('\n================ まとめ ================');
  for (const a of attempts) console.log(`${a.ok ? '✅' : '❌'} ${a.candidate}`);

  if (winner) {
    const r = winner.report;
    console.log(`\n--- 動いた起動条件 ---\n  ${winner.candidate}`);
    console.log('\n--- アダプタ ---');
    console.log(JSON.stringify(r.adapter.info, null, 2));
    console.log(`  preferredCanvasFormat: ${r.preferredCanvasFormat}`);
    console.log('\n--- 主要な上限値 (adapter.limits) ---');
    for (const k of Object.keys(r.adapter.limits).sort()) {
      console.log(`  ${k}: ${r.adapter.limits[k]}`);
    }
  } else {
    console.log('\nどの起動条件でも WebGPU が取れなかった。');
  }

  cleanup();
  process.exit(winner ? 0 : 1);
}

void main();
