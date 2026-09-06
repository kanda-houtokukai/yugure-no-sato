import { probeWebGPU } from './gpu/probe';
import { postToSink } from './sink';
import { runScene } from './harness/runner';
import { runPlay } from './harness/play';
import { WorldScene } from './scene/world';
import { viewFromUrl } from './views';
import { brightestRegion, evaluateChecks, type Check } from './selfcheck';
import { dot } from './math/vec';

export interface GlintReport {
  predictedPixel: { x: number; y: number } | null;
  observedPixel: { x: number; y: number };
  angleDeg: number;
  horizonY: number;
  maxLuminance: number;
  count: number;
}

export interface ViewReport {
  view: string;
  ok: boolean;
  checks: Check[];
  error: string | null;
  /** 太陽の映り込み位置（水面の視点のみ） */
  glint: GlintReport | null;
  scene: unknown;
}

declare global {
  interface Window {
    /** 検証ハーネスが待ち受ける旗と結果。ここ以外の経路でページの状態を判定しない */
    __yugure: {
      mode: 'probe' | 'scene' | 'play';
      /** 実操作モードの計測値（play.ts） */
      play?: unknown;
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
const modeRaw = params.get('mode');
const mode = modeRaw === 'probe' ? 'probe' : modeRaw === 'play' ? 'play' : 'scene';

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
    const scene = new WorldScene(view);
    // 検証用の観測点: probe=x,z;x,z（世界座標）または e+dx,dz（視点相対）
    const probeRaw = params.get('probe');
    if (probeRaw) {
      const pts: [number, number][] = [];
      for (const item of probeRaw.split(';')) {
        const rel = item.startsWith('e');
        const [a, b] = item.replace(/^e\+?/, '').split(',').map(Number);
        if (Number.isFinite(a) && Number.isFinite(b)) pts.push(rel ? [view.eye.x + a, view.eye.z + b] : [a, b]);
      }
      scene.setProbePoints(pts);
    }
    const result = await runScene(canvas, scene, () => {
      window.__yugure.firstFrameDone = true;
    }, view.frames ?? 120);
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
    // 太陽の映り込み位置の機械検証: 平らな水面なら太陽の鏡像方向に最も明るい点が出るはず
    let glint: GlintReport | null = null;
    if (view.glintCheck) {
      const pred = scene.predictGlint();
      const region = brightestRegion(result.pixels, result.canvas.width, result.canvas.height, result.bytesPerRow, result.canvas.format, pred.horizonY + 12 * result.canvas.dpr);
      const observedDir = scene.pixelToDir(region.centroid.x, region.centroid.y);
      const angleDeg = (Math.acos(Math.max(-1, Math.min(1, dot(observedDir, pred.reflectedDir)))) * 180) / Math.PI;
      glint = {
        predictedPixel: pred.pixel,
        observedPixel: region.centroid,
        angleDeg,
        horizonY: pred.horizonY,
        maxLuminance: region.maxLuminance,
        count: region.count,
      };
      checks.push({
        name: '太陽の映り込み位置（鏡像方向と最も明るい点）',
        ok: angleDeg < 2.0 && region.count > 0,
        detail: `予測 (${pred.pixel?.x.toFixed(0)}, ${pred.pixel?.y.toFixed(0)}) / 観測 (${region.centroid.x.toFixed(0)}, ${region.centroid.y.toFixed(0)}) / ずれ ${angleDeg.toFixed(2)}°（< 2° を要求）`,
      });
    }
    // 生の画素はレポートに載せない（大きすぎる）
    const { pixels: _pixels, ...rest } = result;
    report = { view: view.name, ok: checks.every((c) => c.ok), checks, error: null, glint, scene: rest };
  } catch (e) {
    report = {
      view: view.name,
      ok: false,
      checks: [],
      error: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
      glint: null,
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

async function runLive(): Promise<void> {
  const view = viewFromUrl(location.search);
  const canvas = document.getElementById('view');
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('#view キャンバスが無い');
  status('起動中…');
  await runPlay(canvas, new WorldScene(view), status);
}

void (mode === 'probe' ? runProbe() : mode === 'play' ? runLive() : runView());
