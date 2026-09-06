// 描画の枠組み。フレームを回し、画素を読み戻し、GPU/CPU のフレーム時間と GPU エラーを集める。
// 「描く中身」はシーン側（SceneRenderer）に委ね、ここは計測だけを担う。

import { acquireDevice, popErrorScopes, pushErrorScopes, type DeviceBundle, type GpuErrorRecord } from '../gpu/device';
import { analyzePixels, detectQuantumNs, summarizeMs, type FrameTimeStats, type ImageStats } from '../selfcheck';

/** キャンバスの CSS サイズ。実描画解像度はこれに devicePixelRatio（上限 2）を掛ける */
export const CSS_WIDTH = 1280;
export const CSS_HEIGHT = 720;
export const MAX_DPR = 2;

/** 現在の実描画解像度。init 時に決まる（Retina なら 2560×1440） */
export const resolution = { width: CSS_WIDTH, height: CSS_HEIGHT, dpr: 1 };

/** copyTextureToBuffer の bytesPerRow は 256 の倍数。1280*4 と 2560*4 はいずれも条件を満たす */
function bytesPerRowFor(width: number): number {
  return Math.ceil((width * 4) / 256) * 256;
}
const FRAME_COUNT = 120;
const MEASURE_TAIL = 60;
const LOOP_TIMEOUT_MS = 30000;

export interface ShaderMessage {
  module: string;
  type: string;
  message: string;
  lineNum: number;
  linePos: number;
}

/** シーンがフレームを描くときに受け取るもの */
export interface FrameContext {
  encoder: GPUCommandEncoder;
  /** 表示先（bgra8unorm のキャンバス） */
  target: GPUTextureView;
  frameIndex: number;
  /** 最初のパスに付ける timestampWrites（無ければ undefined） */
  tsBegin: GPURenderPassTimestampWrites | undefined;
  /** 最後のパスに付ける timestampWrites */
  tsEnd: GPURenderPassTimestampWrites | undefined;
}

export interface SceneRenderer {
  /** パイプラインと資源を作る。compute で CPU が要る値を読み戻すのもここ */
  init(bundle: DeviceBundle, canvasFormat: GPUTextureFormat): Promise<void>;
  render(ctx: FrameContext): void;
  /** 描画ループの後に GPU から読み戻すものがあれば */
  afterRun?(): Promise<void>;
  /** レポートに載せるシーン固有の情報 */
  describe(): unknown;
  shaderMessages(): ShaderMessage[];
}

export interface RunResult {
  canvas: { width: number; height: number; dpr: number; format: GPUTextureFormat };
  adapter: Record<string, string>;
  image: ImageStats;
  /** 読み戻した生の画素（解析の追加に使う。レポートには載せない） */
  pixels: Uint8Array;
  bytesPerRow: number;
  gpuErrors: GpuErrorRecord[];
  shaderMessages: ShaderMessage[];
  shaderErrors: number;
  framesRendered: number;
  framesExpected: number;
  timedOut: boolean;
  gpu: FrameTimeStats;
  cpu: FrameTimeStats;
  scene: unknown;
}

export async function runScene(
  canvas: HTMLCanvasElement,
  scene: SceneRenderer,
  onFirstFrame: () => void,
): Promise<RunResult> {
  // Retina では CSS 1px = 物理 2px。描画をこれに合わせないと半分の解像度で描いて引き伸ばすことになる
  // （フェーズ2で実測: 実ディスプレイ DPR 2、キャンバス 1280×720 が物理 2560×1440 を覆っていた）
  const override = Number(new URLSearchParams(location.search).get('dpr'));
  const dpr = Number.isFinite(override) && override > 0
    ? Math.min(MAX_DPR, override)   // 診断用の上書き（?dpr=1）
    : Math.min(MAX_DPR, Math.max(1, window.devicePixelRatio || 1));
  resolution.width = Math.round(CSS_WIDTH * dpr);
  resolution.height = Math.round(CSS_HEIGHT * dpr);
  resolution.dpr = dpr;
  canvas.width = resolution.width;
  canvas.height = resolution.height;
  const WIDTH = resolution.width;
  const HEIGHT = resolution.height;
  const BYTES_PER_ROW = bytesPerRowFor(WIDTH);

  const bundle = await acquireDevice();
  const { device, adapter, format, hasTimestampQuery, errors } = bundle;
  pushErrorScopes(device);

  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('canvas.getContext("webgpu") が null');
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  await scene.init(bundle, format);

  const pixelBuffer = device.createBuffer({
    size: BYTES_PER_ROW * HEIGHT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const queryCount = FRAME_COUNT * 2;
  const tsBytes = queryCount * 8;
  const querySet = hasTimestampQuery ? device.createQuerySet({ type: 'timestamp', count: queryCount }) : null;
  const tsResolve = querySet
    ? device.createBuffer({ size: tsBytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null;
  const tsReadback = querySet
    ? device.createBuffer({ size: tsBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    : null;

  function renderFrame(index: number, isLast: boolean): void {
    const encoder = device.createCommandEncoder();
    const texture = context!.getCurrentTexture();
    scene.render({
      encoder,
      target: texture.createView(),
      frameIndex: index,
      tsBegin: querySet ? { querySet, beginningOfPassWriteIndex: index * 2 } : undefined,
      tsEnd: querySet ? { querySet, endOfPassWriteIndex: index * 2 + 1 } : undefined,
    });
    if (isLast) {
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: pixelBuffer, bytesPerRow: BYTES_PER_ROW, rowsPerImage: HEIGHT },
        { width: WIDTH, height: HEIGHT, depthOrArrayLayers: 1 },
      );
    }
    device.queue.submit([encoder.finish()]);
  }

  const cpuFrameMs: number[] = [];
  let framesRendered = 0;
  let timedOut = false;

  await new Promise<void>((resolve) => {
    let lastTs = -1;
    const watchdog = setTimeout(() => {
      timedOut = true;
      resolve();
    }, LOOP_TIMEOUT_MS);
    const tick = (now: number): void => {
      if (lastTs >= 0) cpuFrameMs.push(now - lastTs);
      lastTs = now;
      renderFrame(framesRendered, framesRendered === FRAME_COUNT - 1);
      framesRendered++;
      if (framesRendered === 1) void device.queue.onSubmittedWorkDone().then(onFirstFrame);
      if (framesRendered >= FRAME_COUNT) {
        clearTimeout(watchdog);
        resolve();
        return;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  let gpu: FrameTimeStats = {
    available: false,
    reason: hasTimestampQuery ? '未計測' : 'timestamp-query 非対応',
    frames: 0, avgMs: null, maxMs: null, minMs: null, quantumNs: null,
  };
  if (querySet && tsResolve && tsReadback && !timedOut) {
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, queryCount, tsResolve, 0);
    encoder.copyBufferToBuffer(tsResolve, 0, tsReadback, 0, tsBytes);
    device.queue.submit([encoder.finish()]);
    await tsReadback.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(tsReadback.getMappedRange().slice(0));
    tsReadback.unmap();
    const deltas: bigint[] = [];
    for (let i = Math.max(0, framesRendered - MEASURE_TAIL); i < framesRendered; i++) {
      const a = stamps[i * 2];
      const b = stamps[i * 2 + 1];
      if ((a === 0n && b === 0n) || b <= a) continue;
      deltas.push(b - a);
    }
    const ms = deltas.map((d) => Number(d) / 1e6);
    gpu = {
      available: ms.length > 0,
      reason: ms.length > 0 ? null : 'タイムスタンプが書かれなかった',
      frames: ms.length,
      ...summarizeMs(ms),
      quantumNs: detectQuantumNs(deltas),
    };
  }

  await pixelBuffer.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(pixelBuffer.getMappedRange().slice(0));
  pixelBuffer.unmap();
  const image = analyzePixels(pixels, WIDTH, HEIGHT, BYTES_PER_ROW, format);

  const cpuTail = cpuFrameMs.slice(-MEASURE_TAIL);
  const cpu: FrameTimeStats = {
    available: cpuTail.length > 0,
    reason: cpuTail.length > 0 ? null : 'フレームが進まなかった',
    frames: cpuTail.length,
    ...summarizeMs(cpuTail),
    quantumNs: null,
  };

  if (scene.afterRun) await scene.afterRun();
  await popErrorScopes(device, errors);

  const adapterInfo: Record<string, string> = {};
  const rawInfo = (adapter as unknown as { info?: Record<string, unknown> }).info;
  if (rawInfo) {
    for (const key of ['vendor', 'architecture', 'device', 'description']) {
      const value = rawInfo[key];
      if (typeof value === 'string' && value !== '') adapterInfo[key] = value;
    }
  }

  const shaderMessages = scene.shaderMessages();
  return {
    canvas: { width: WIDTH, height: HEIGHT, dpr: resolution.dpr, format },
    adapter: adapterInfo,
    image,
    pixels,
    bytesPerRow: BYTES_PER_ROW,
    gpuErrors: errors,
    shaderMessages,
    shaderErrors: shaderMessages.filter((m) => m.type === 'error').length,
    framesRendered,
    framesExpected: FRAME_COUNT,
    timedOut,
    gpu,
    cpu,
    scene: scene.describe(),
  };
}
