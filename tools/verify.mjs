// フェーズ0 パートD: 一括実行。
// dev サーバ起動 → 3視点のスクショ → 自己点検レポート出力 → サーバ停止。
// 異常があれば終了コードを非0にし、何が失敗したかを標準出力に出す。
//
// Playwright 同梱の Chromium ではなく channel:'chrome' で実機の Chrome 安定版を起動する。
// 理由: tools/probe.mjs で WebGPU の動作を実証したのがその Chrome だから。

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { ROOT, startVite, stamp, waitForServer } from './dev-server.mjs';

const PORT = 5198;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const OUT_DIR = join(ROOT, '.screenshots');
const VIEWS = ['front', 'bird', 'ground', 'water', 'walk'];
const PAGE_TIMEOUT_MS = 60000;

/** 太陽の方位をずらして撮り直す幅 [度]。映り込みがこれに応じて動くことを確かめる */
const SUN_SHIFT_DEG = 8;

async function captureView(browser, view, at, extraQuery = '', suffix = '') {
  const context = await browser.newContext({
    viewport: { width: 1360, height: 900 },
    deviceScaleFactor: 2  /* 実機は Retina（DPR 2）。スクショも物理解像度 2560×1440 で撮る */,
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

  // mode=scene: 既定は実操作モードなので、検証では明示的に計測モードを指定する
  const url = `${ORIGIN}/?mode=scene&view=${view}${extraQuery}`;
  let report = null;
  let failure = null;
  let screenshot = null;
  let sha256 = null;

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

    screenshot = join(OUT_DIR, `${at}-${view}${suffix}.png`);
    await page.locator('#view').screenshot({ path: screenshot });
    sha256 = createHash('sha256').update(readFileSync(screenshot)).digest('hex');
  } catch (e) {
    failure = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  } finally {
    await context.close();
  }

  return {
    view,
    url,
    screenshot,
    sha256,
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

    // 映り込みの検証: 太陽をずらして撮り直し、明るい点が予測どおり動くか
    if (result.report?.glint) {
      const base = result.report.glint;
      const shifted = await captureView(browser, view, at, `&sunAz=${275 + SUN_SHIFT_DEG}`, `-sunshift`);
      const g = shifted.report?.glint;
      let check = { name: `太陽を ${SUN_SHIFT_DEG}° 動かすと映り込みも動く`, ok: false, detail: '撮り直しに失敗' };
      if (g && base.predictedPixel && g.predictedPixel) {
        const predDx = g.predictedPixel.x - base.predictedPixel.x;
        const obsDx = g.observedPixel.x - base.observedPixel.x;
        const tol = Math.max(12, Math.abs(predDx) * 0.25);
        check = {
          name: `太陽を ${SUN_SHIFT_DEG}° 動かすと映り込みも動く`,
          ok: Math.abs(obsDx - predDx) <= tol && g.angleDeg < 2.0,
          detail: `予測 Δx ${predDx.toFixed(0)}px / 観測 Δx ${obsDx.toFixed(0)}px（許容 ±${tol.toFixed(0)}px）／ ずらした後のずれ ${g.angleDeg.toFixed(2)}°`,
        };
      }
      result.sunShift = { screenshot: shifted.screenshot, sha256: shifted.sha256, check, glint: g ?? null };
      result.report.checks.push(check);
      if (!check.ok) result.ok = false;
      process.stdout.write(`  映り込み検証 ${check.ok ? 'OK' : 'NG'}  ${check.detail}\n`);
    }
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
    if (r.sha256) console.log(`   sha256: ${r.sha256}`);
    if (r.report?.error) console.log(`   ページ内エラー: ${r.report.error}`);
  }
  console.log(`\nスクショ: ${OUT_DIR}`);
  console.log(`レポート: ${reportPath}`);
  console.log(ok ? '\n結果: 全視点 OK' : '\n結果: 異常あり（上の × を参照）');

  await shutdown();
  process.exit(ok ? 0 : 1);
}

void main();
