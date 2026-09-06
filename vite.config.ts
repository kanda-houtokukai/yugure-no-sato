import { defineConfig, type Plugin } from 'vite';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const OUT_DIR = resolve(process.cwd(), '.screenshots');

/** 日時を YYYYMMDD-HHmmss（ローカル時刻）で。ファイル名に使う */
function stamp(): string {
  const d = new Date();
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  return (
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
}

/**
 * ブラウザ側から POST /__sink/<name> で送られた JSON を .screenshots/ に落とす。
 * 履歴用 <name>-<日時>.json と、ハーネスが待ち受ける <name>-latest.json の2本を書く。
 * dev サーバ限定（本番ビルドには入らない）。
 */
function sinkPlugin(): Plugin {
  return {
    name: 'yugure-sink',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__sink', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('{"ok":false,"error":"POST only"}');
          return;
        }
        const name = decodeURIComponent((req.url ?? '/').replace(/^\//, '')) || 'unnamed';
        if (!/^[A-Za-z0-9_-]+$/.test(name)) {
          res.statusCode = 400;
          res.end('{"ok":false,"error":"bad name"}');
          return;
        }
        const chunks: Buffer[] = [];
        req.on('data', (c: Buffer) => chunks.push(c));
        req.on('end', () => {
          const body = Buffer.concat(chunks).toString('utf8');
          mkdirSync(OUT_DIR, { recursive: true });
          writeFileSync(resolve(OUT_DIR, `${name}-${stamp()}.json`), body);
          writeFileSync(resolve(OUT_DIR, `${name}-latest.json`), body);
          // 本文の無い 204 を返すと、書き込みが成功していても Chrome が要求を
          // net::ERR_ABORTED として扱い、verify が偽の異常を報告する。必ず本文を返す
          res.statusCode = 200;
          res.setHeader('content-type', 'application/json');
          res.end('{"ok":true}');
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [sinkPlugin()],
  server: { host: '127.0.0.1' },
});
