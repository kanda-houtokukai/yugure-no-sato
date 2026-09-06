import { probeWebGPU } from './gpu/probe';
import { postToSink } from './sink';

declare global {
  interface Window {
    /** 検証ハーネスが「ページ側の作業が終わったか」を判定するための旗 */
    __yugure: {
      probeDone: boolean;
      probe: unknown;
    };
  }
}

window.__yugure = { probeDone: false, probe: null };

function show(text: string): void {
  const el = document.getElementById('out');
  if (el) el.textContent = text;
}

async function main(): Promise<void> {
  show('WebGPU を確認中…');
  const report = await probeWebGPU();

  // 人が見る用（ブラウザのコンソール）
  if (report.ok) {
    console.log('[probe] WebGPU OK');
    console.log('[probe] adapter.info', report.adapter?.info);
    console.log('[probe] adapter.features', report.adapter?.features);
    console.log('[probe] device.limits', report.device?.limits);
    console.log('[probe] preferredCanvasFormat', report.preferredCanvasFormat);
  } else {
    console.error('[probe] WebGPU NG at', report.stage, report.error);
  }

  // 機械が見る用（画面・旗・dev サーバへの送出）
  show(JSON.stringify(report, null, 2));
  window.__yugure.probe = report;
  window.__yugure.probeDone = true;
  await postToSink('probe', report);
}

void main();
