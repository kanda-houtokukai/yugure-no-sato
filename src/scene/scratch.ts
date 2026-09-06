// 検証用の仮の景色を描き、同時に自己点検の材料（画素・GPUエラー・フレーム時間）を集める。
// フェーズ1でこのファイルの「描く中身」は捨てるが、計測の枠組みはそのまま使う。

import shaderSource from './scratch.wgsl?raw';
import { acquireDevice, popErrorScopes, pushErrorScopes, type GpuErrorRecord } from '../gpu/device';
import {
  analyzePixels,
  detectQuantumNs,
  summarizeMs,
  type FrameTimeStats,
  type ImageStats,
} from '../selfcheck';
import type { View } from '../views';

/** 計測に使うフレーム数。末尾 MEASURE_TAIL 件を採用する（立ち上がりの揺れを外す） */
const FRAME_COUNT = 120;
const MEASURE_TAIL = 60;
const WIDTH = 1280;
const HEIGHT = 720;
/** copyTextureToBuffer の bytesPerRow は 256 の倍数でなければならない。1280*4 = 5120 は条件を満たす */
const BYTES_PER_ROW = WIDTH * 4;
const LOOP_TIMEOUT_MS = 20000;

export interface ShaderMessage {
  type: string;
  message: string;
  lineNum: number;
  linePos: number;
}

export interface SceneResult {
  view: View;
  canvas: { width: number; height: number; format: GPUTextureFormat };
  adapter: Record<string, string>;
  image: ImageStats;
  gpuErrors: GpuErrorRecord[];
  shaderMessages: ShaderMessage[];
  shaderErrors: number;
  framesRendered: number;
  framesExpected: number;
  timedOut: boolean;
  gpu: FrameTimeStats;
  cpu: FrameTimeStats;
}

type Vec3 = [number, number, number];

const sub = (a: readonly number[], b: readonly number[]): Vec3 => [
  a[0] - b[0],
  a[1] - b[1],
  a[2] - b[2],
];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, -1];
}

/** eye / target から画面の基底（前・右・上）を作る */
function cameraBasis(view: View): { forward: Vec3; right: Vec3; up: Vec3 } {
  const forward = normalize(sub(view.target, view.eye));
  // 視線がほぼ真上・真下だと worldUp との外積が潰れるので基準を差し替える
  const worldUp: Vec3 = Math.abs(forward[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize(cross(forward, worldUp));
  const up = normalize(cross(right, forward));
  return { forward, right, up };
}

export async function runScratchScene(
  canvas: HTMLCanvasElement,
  view: View,
  onFirstFrame: () => void,
): Promise<SceneResult> {
  canvas.width = WIDTH;
  canvas.height = HEIGHT;

  const bundle = await acquireDevice();
  const { device, adapter, format, hasTimestampQuery, errors } = bundle;

  pushErrorScopes(device);

  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('canvas.getContext("webgpu") が null');

  context.configure({
    device,
    format,
    alphaMode: 'opaque',
    // 画素を読み戻して解析するので COPY_SRC が要る
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const module = device.createShaderModule({ code: shaderSource, label: 'scratch' });
  const info = await module.getCompilationInfo();
  const shaderMessages: ShaderMessage[] = info.messages.map((m) => ({
    type: m.type,
    message: m.message,
    lineNum: m.lineNum,
    linePos: m.linePos,
  }));
  const shaderErrors = shaderMessages.filter((m) => m.type === 'error').length;

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  // eye / forward / right / up / params の 5 × vec4f = 80 バイト
  const uniformData = new Float32Array(20);
  const { forward, right, up } = cameraBasis(view);
  const tanHalfFov = Math.tan((view.fovDeg * Math.PI) / 360);
  uniformData.set([view.eye[0], view.eye[1], view.eye[2], 0], 0);
  uniformData.set([forward[0], forward[1], forward[2], 0], 4);
  uniformData.set([right[0], right[1], right[2], 0], 8);
  uniformData.set([up[0], up[1], up[2], 0], 12);
  uniformData.set([tanHalfFov, WIDTH / HEIGHT, view.time, 0], 16);

  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const pixelBuffer = device.createBuffer({
    size: BYTES_PER_ROW * HEIGHT,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const queryCount = FRAME_COUNT * 2;
  const tsBytes = queryCount * 8;
  const querySet = hasTimestampQuery
    ? device.createQuerySet({ type: 'timestamp', count: queryCount })
    : null;
  const tsResolve = querySet
    ? device.createBuffer({
        size: tsBytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null;
  const tsReadback = querySet
    ? device.createBuffer({ size: tsBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    : null;

  function renderFrame(index: number, isLast: boolean): void {
    const encoder = device.createCommandEncoder();
    const texture = context!.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: texture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      ...(querySet
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: index * 2,
              endOfPassWriteIndex: index * 2 + 1,
            },
          }
        : {}),
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    if (isLast) {
      // 提示される前の、いま描いたテクスチャをそのまま読み戻す
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
    // requestAnimationFrame が一度も来ない環境で黙って固まらないよう、外側にも番人を置く
    const watchdog = setTimeout(() => {
      timedOut = true;
      resolve();
    }, LOOP_TIMEOUT_MS);

    const tick = (now: number): void => {
      if (lastTs >= 0) cpuFrameMs.push(now - lastTs);
      lastTs = now;

      renderFrame(framesRendered, framesRendered === FRAME_COUNT - 1);
      framesRendered++;

      if (framesRendered === 1) {
        // 「最初のフレームが描き終わった」= GPU 側の完了。ここで旗を立てる
        void device.queue.onSubmittedWorkDone().then(onFirstFrame);
      }
      if (framesRendered >= FRAME_COUNT) {
        clearTimeout(watchdog);
        resolve();
        return;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  // --- GPU 側フレーム時間 ---
  let gpu: FrameTimeStats = {
    available: false,
    reason: hasTimestampQuery ? '未計測' : 'timestamp-query 非対応',
    frames: 0,
    avgMs: null,
    maxMs: null,
    minMs: null,
    quantumNs: null,
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
    const start = Math.max(0, framesRendered - MEASURE_TAIL);
    for (let i = start; i < framesRendered; i++) {
      const a = stamps[i * 2];
      const b = stamps[i * 2 + 1];
      if (a === 0n && b === 0n) continue; // 書かれなかった枠
      if (b <= a) continue;
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

  // --- 画素の読み戻しと解析 ---
  await pixelBuffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(pixelBuffer.getMappedRange().slice(0));
  pixelBuffer.unmap();
  const image = analyzePixels(bytes, WIDTH, HEIGHT, BYTES_PER_ROW, format);

  const cpuTail = cpuFrameMs.slice(-MEASURE_TAIL);
  const cpu: FrameTimeStats = {
    available: cpuTail.length > 0,
    reason: cpuTail.length > 0 ? null : 'フレームが進まなかった',
    frames: cpuTail.length,
    ...summarizeMs(cpuTail),
    quantumNs: null,
  };

  await popErrorScopes(device, errors);

  const adapterInfo: Record<string, string> = {};
  const rawInfo = (adapter as unknown as { info?: Record<string, unknown> }).info;
  if (rawInfo) {
    for (const key of ['vendor', 'architecture', 'device', 'description']) {
      const value = rawInfo[key];
      if (typeof value === 'string' && value !== '') adapterInfo[key] = value;
    }
  }

  return {
    view,
    canvas: { width: WIDTH, height: HEIGHT, format },
    adapter: adapterInfo,
    image,
    gpuErrors: errors,
    shaderMessages,
    shaderErrors,
    framesRendered,
    framesExpected: FRAME_COUNT,
    timedOut,
    gpu,
    cpu,
  };
}
