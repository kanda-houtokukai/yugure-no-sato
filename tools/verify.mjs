// フェーズ0 パートD: 一括実行。
// dev サーバ起動 → 3視点のスクショ → 自己点検レポート出力 → サーバ停止。
// 異常があれば終了コードを非0にし、何が失敗したかを標準出力に出す。
//
// Playwright 同梱の Chromium ではなく channel:'chrome' で実機の Chrome 安定版を起動する。
// 理由: tools/probe.mjs で WebGPU の動作を実証したのがその Chrome だから。

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { ROOT, startVite, stamp, waitForServer } from './dev-server.mjs';

const PORT = 5198;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const OUT_DIR = join(ROOT, '.screenshots');
const VIEWS = ['front', 'bird', 'ground'];
const PAGE_TIMEOUT_MS = 60000;

async function captureView(browser, view, at) {
  const context = await browser.newContext({
    viewport: { width: 1360, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();

  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(`pageerror: ${err.message}`));

  // コンソールの 404 は本文に URL が出ない。取り逃さないよう応答そのものを見る
  const httpErrors = [];
  page.on('response', (res) => {
    if (res.status() >= 400) httpErrors.push(`${res.status()} ${res.url()}`);
  });
  page.on('requestfailed', (req) => {
    httpErrors.push(`failed ${req.url()} (${req.failure()?.errorText ?? '理由不明'})`);
  });

  const url = `${ORIGIN}/?view=${view}`;
  let report = null;
  let failure = null;
  let screenshot = null;

  try {
    await page.goto(url, { waitUntil: 'load', timeout: PAGE_TIMEOUT_MS });
    // 「最初のフレームが描き終わった」より先には撮らない。sceneDone は計測完了まで含む
    await page.waitForFunction(() => window.__yugure?.firstFrameDone === true, null, {
      timeout: PAGE_TIMEOUT_MS,
    });
    await page.waitForFunction(() => window.__yugure?.sceneDone === true, null, {
      timeout: PAGE_TIMEOUT_MS,
    });
    report = await page.evaluate(() => window.__yugure.report);

    screenshot = join(OUT_DIR, `${at}-${view}.png`);
    await page.locator('#view').screenshot({ path: screenshot });
  } catch (e) {
    failure = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  } finally {
    await context.close();
  }

  return {
    view,
    url,
    screenshot,
    harnessError: failure,
    consoleErrors,
    httpErrors,
    ok:
      failure === null &&
      consoleErrors.length === 0 &&
      httpErrors.length === 0 &&
      report?.ok === true,
    report,
  };
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true });
  const at = stamp();

  const vite = startVite(PORT);
  let browser = null;
  const shutdown = async () => {
    if (browser) await browser.close().catch(() => {});
    vite.kill('SIGTERM');
  };
  process.on('SIGINT', () => {
    void shutdown().then(() => process.exit(130));
  });

  if (!(await waitForServer(ORIGIN))) {
    console.error('dev サーバが起動しなかった');
    await shutdown();
    process.exit(2);
  }

  browser = await chromium.launch({ channel: 'chrome', headless: true });

  const results = [];
  for (const view of VIEWS) {
    process.stdout.write(`--- 視点 ${view} ---\n`);
    const result = await captureView(browser, view, at);
    results.push(result);
    process.stdout.write(`  ${result.ok ? 'OK' : 'NG'}  ${result.screenshot ?? '(スクショなし)'}\n`);
  }

  const reportPath = join(OUT_DIR, `report-${at}.json`);
  const ok = results.every((r) => r.ok);
  writeFileSync(
    reportPath,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        origin: ORIGIN,
        browser: 'Chrome 安定版 (playwright channel:chrome, headless)',
        ok,
        views: results,
      },
      null,
      2,
    ),
  );

  console.log('\n================ verify まとめ ================');
  for (const r of results) {
    console.log(`\n${r.ok ? '✅' : '❌'} ${r.view}`);
    if (r.harnessError) console.log(`   ハーネス失敗: ${r.harnessError}`);
    for (const e of r.consoleErrors) console.log(`   コンソールエラー: ${e}`);
    for (const e of r.httpErrors) console.log(`   HTTP エラー: ${e}`);
    for (const c of r.report?.checks ?? []) {
      console.log(`   ${c.ok ? '○' : '×'} ${c.name} — ${c.detail}`);
    }
    if (r.report?.error) console.log(`   ページ内エラー: ${r.report.error}`);
  }
  console.log(`\nスクショ: ${OUT_DIR}`);
  console.log(`レポート: ${reportPath}`);
  console.log(ok ? '\n結果: 全視点 OK' : '\n結果: 異常あり（上の × を参照）');

  await shutdown();
  process.exit(ok ? 0 : 1);
}

void main();
