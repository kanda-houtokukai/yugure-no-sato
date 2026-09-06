import { probeWebGPU } from './gpu/probe';
import { postToSink } from './sink';
import { runScene } from './harness/runner';
import { WorldScene } from './scene/world';
import { viewFromUrl } from './views';
import { evaluateChecks, type Check } from './selfcheck';

export interface ViewReport {
  view: string;
  ok: boolean;
  checks: Check[];
  error: string | null;
  scene: unknown;
}

declare global {
  interface Window {
    /** 検証ハーネスが待ち受ける旗と結果。ここ以外の経路でページの状態を判定しない */
    __yugure: {
      mode: 'probe' | 'scene';
      probeDone: boolean;
      probe: unknown;
      /** 最初のフレームが GPU 側で描き終わった。スクショはこれ以降でないと真っ黒を撮る */
      firstFrameDone: boolean;
      /** 計測とレポート生成まで終わった */
      sceneDone: boolean;
      report: ViewReport | null;
    };
  }
}

const params = new URLSearchParams(location.search);
const mode = params.get('mode') === 'probe' ? 'probe' : 'scene';

window.__yugure = { mode, probeDone: false, probe: null, firstFrameDone: false, sceneDone: false, report: null };

function status(text: string): void {
  const el = document.getElementById('out');
  if (el) el.textContent = text;
}

async function runProbe(): Promise<void> {
  status('WebGPU を確認中…');
  const report = await probeWebGPU();
  if (report.ok) console.log('[probe] WebGPU OK', report.adapter?.info);
  else console.error('[probe] WebGPU NG at', report.stage, report.error);
  status(JSON.stringify(report, null, 2));
  window.__yugure.probe = report;
  window.__yugure.probeDone = true;
  await postToSink('probe', report);
}

async function runView(): Promise<void> {
  const view = viewFromUrl(location.search);
  status(`視点 ${view.name} を描画中…`);
  const canvas = document.getElementById('view');
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('#view キャンバスが無い');

  let report: ViewReport;
  try {
    const result = await runScene(canvas, new WorldScene(view), () => {
      window.__yugure.firstFrameDone = true;
    });
    const checks = evaluateChecks({
      image: result.image,
      gpuErrors: result.gpuErrors,
      shaderErrors: result.shaderErrors,
      framesRendered: result.framesRendered,
      framesExpected: result.framesExpected,
      timedOut: result.timedOut,
      gpu: result.gpu,
      cpu: result.cpu,
    });
    // 生の画素はレポートに載せない（大きすぎる）
    const { pixels: _pixels, ...rest } = result;
    report = { view: view.name, ok: checks.every((c) => c.ok), checks, error: null, scene: rest };
  } catch (e) {
    report = {
      view: view.name,
      ok: false,
      checks: [],
      error: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
      scene: null,
    };
    console.error('[scene] 失敗', e);
  }

  status(
    `${report.view}: ${report.ok ? 'OK' : 'NG'}\n` +
      (report.error ?? report.checks.map((c) => `${c.ok ? '○' : '×'} ${c.name} — ${c.detail}`).join('\n')),
  );
  window.__yugure.report = report;
  // sink への送出を終えてから旗を立てる（先に立てると送出中の POST が中断され偽の異常になる）
  await postToSink(`scene-${view.name}`, report);
  window.__yugure.sceneDone = true;
}

void (mode === 'probe' ? runProbe() : runView());
