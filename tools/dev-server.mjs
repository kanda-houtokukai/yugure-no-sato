// vite dev サーバの起動と待機。probe.mjs / verify.mjs の共通部品。

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const ROOT = resolve(import.meta.dirname, '..');

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function stamp(date = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return (
    `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}` +
    `-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`
  );
}

export function startVite(port, { quiet = true } = {}) {
  const bin = join(ROOT, 'node_modules', '.bin', 'vite');
  if (!existsSync(bin)) {
    throw new Error(`vite が見つからない: ${bin}\nnpm install を先に実行すること。`);
  }
  const child = spawn(bin, ['--port', String(port), '--strictPort'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (!quiet) child.stdout.on('data', (d) => process.stdout.write(`[vite] ${d}`));
  else child.stdout.on('data', () => {});
  child.stderr.on('data', (d) => process.stderr.write(`[vite:err] ${d}`));
  return child;
}

export async function waitForServer(origin, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${origin}/`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return true;
    } catch {
      /* まだ起動していない */
    }
    await sleep(250);
  }
  return false;
}
