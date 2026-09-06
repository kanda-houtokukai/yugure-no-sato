// 1視点だけ撮って数値を出す。verify より軽く、試行錯誤のための道具。
//   node tools/shot.mjs 'view=front&dbg=1' [出力名]
// 出力: .screenshots/shot-<名前>.png と主要な計測値。

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { ROOT, startVite, waitForServer } from './dev-server.mjs';

const PORT = 5193;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const OUT_DIR = join(ROOT, '.screenshots');
const query = process.argv[2] ?? 'view=front';
const name = process.argv[3] ?? query.replace(/[^A-Za-z0-9]+/g, '_').slice(0, 60);

mkdirSync(OUT_DIR, { recursive: true });
const vite = startVite(PORT);
if (!(await waitForServer(ORIGIN))) { console.error('vite 起動せず'); vite.kill(); process.exit(2); }
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1360, height: 900 }, deviceScaleFactor: 2  /* 実機は Retina（DPR 2）。スクショも物理解像度 2560×1440 で撮る */ });
const page = await context.newPage();
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(`${ORIGIN}/?${query}`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__yugure?.sceneDone === true, null, { timeout: 90000 });
const report = await page.evaluate(() => window.__yugure.report);
const path = join(OUT_DIR, `shot-${name}.png`);
await page.locator('#view').screenshot({ path });
await browser.close();
vite.kill('SIGTERM');

const s = report.scene;
if (report.error) console.log('ページ内エラー:', report.error);
for (const e of errors) console.log('コンソールエラー:', e);
for (const c of report.checks) if (!c.ok) console.log(`× ${c.name} — ${c.detail}`);
if (s) {
  console.log(`GPU ${s.gpu.avgMs?.toFixed(2)} ms (max ${s.gpu.maxMs?.toFixed(2)}) | CPU ${s.cpu.avgMs?.toFixed(2)} ms | mean ${s.image.meanLuminance.toFixed(3)} sd ${s.image.stdDevLuminance.toFixed(3)} | black ${(s.image.nearBlackRatio*100).toFixed(1)}% white ${(s.image.nearWhiteRatio*100).toFixed(1)}%`);
  if (s.shaderMessages?.length) console.log('シェーダ:', JSON.stringify(s.shaderMessages));
  if (s.gpuErrors?.length) console.log('GPUエラー:', JSON.stringify(s.gpuErrors));
  if (s.scene?.resolved) console.log('resolved:', JSON.stringify(s.scene.resolved));
  if (s.scene?.extra) console.log('extra:', JSON.stringify(s.scene.extra));
}
console.log(`→ ${path}`);
process.exit(0);
