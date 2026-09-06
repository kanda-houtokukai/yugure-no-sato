// 実操作モード。WASD で歩き、マウス（クリックでポインタロック）または矢印キーで向きを回す。
// 計測はしない代わりに、直近 120 フレームの CPU/GPU 時間を HUD と window.__yugure.play に出す
// （画面付き Chrome での実測に使う）。

import { acquireDevice, type DeviceBundle } from '../gpu/device';
import { CSS_HEIGHT, CSS_WIDTH, MAX_DPR, resolution, type SceneRenderer } from './runner';
import type { WalkerInput } from '../scene/walker';

export interface PlayStats {
  frames: number;
  cpuMs: number;    // 直近 120 フレームの平均（rAF 間隔）
  cpuMaxMs: number;
  gpuMs: number | null;
  gpuMaxMs: number | null;
  pos: { x: number; y: number; z: number; yawDeg: number };
}

export interface PlayableScene extends SceneRenderer {
  live: boolean;
  liveInput: WalkerInput;
  walker: { x: number; y: number; z: number; yawDeg: number };
}

export async function runPlay(canvas: HTMLCanvasElement, scene: PlayableScene, status: (t: string) => void): Promise<void> {
  const dpr = Math.min(MAX_DPR, Math.max(1, window.devicePixelRatio || 1));
  resolution.width = Math.round(CSS_WIDTH * dpr);
  resolution.height = Math.round(CSS_HEIGHT * dpr);
  resolution.dpr = dpr;
  canvas.width = resolution.width;
  canvas.height = resolution.height;

  const bundle: DeviceBundle = await acquireDevice();
  const { device, format, hasTimestampQuery } = bundle;
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('canvas.getContext("webgpu") が null');
  context.configure({ device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  await scene.init(bundle, format);
  scene.live = true;

  // ---- 入力 ----
  const keys = new Set<string>();
  let mouseDx = 0, mouseDy = 0;
  window.addEventListener('keydown', (e) => { keys.add(e.code); if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault(); });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  canvas.addEventListener('click', () => { void canvas.requestPointerLock(); });
  window.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas) { mouseDx += e.movementX; mouseDy += e.movementY; }
  });

  // ---- 計測（直近 120 フレーム） ----
  const TS = 120;
  const querySet = hasTimestampQuery ? device.createQuerySet({ type: 'timestamp', count: TS * 2 }) : null;
  const tsResolve = querySet ? device.createBuffer({ size: TS * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const tsRead = querySet ? device.createBuffer({ size: TS * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
  let tsBusy = false;
  const cpuHist: number[] = [];
  const stats: PlayStats = { frames: 0, cpuMs: 0, cpuMaxMs: 0, gpuMs: null, gpuMaxMs: null, pos: { x: 0, y: 0, z: 0, yawDeg: 0 } };
  (window as unknown as { __yugure: { play?: PlayStats } }).__yugure.play = stats;

  let last = -1;
  let frame = 0;
  const tick = (now: number): void => {
    if (last >= 0) { cpuHist.push(now - last); if (cpuHist.length > TS) cpuHist.shift(); }
    last = now;

    const yawKeys = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
    const pitchKeys = (keys.has('ArrowDown') ? 1 : 0) - (keys.has('ArrowUp') ? 1 : 0);
    scene.liveInput = {
      forward: (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0),
      strafe: (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0),
      yawDelta: mouseDx * 0.15 + yawKeys * 1.5,
      pitchDelta: mouseDy * 0.1 + pitchKeys * 1.0,
      run: keys.has('ShiftLeft') || keys.has('ShiftRight'),
    };
    mouseDx = 0; mouseDy = 0;

    const encoder = device.createCommandEncoder();
    const ti = frame % TS;
    scene.render({
      encoder,
      target: context.getCurrentTexture().createView(),
      frameIndex: frame,
      tsBegin: querySet ? { querySet, beginningOfPassWriteIndex: ti * 2 } : undefined,
      tsEnd: querySet ? { querySet, endOfPassWriteIndex: ti * 2 + 1 } : undefined,
    });
    // 120 フレームごとに GPU 時間を非同期で読む（待たない）
    if (querySet && tsResolve && tsRead && ti === TS - 1 && !tsBusy) {
      encoder.resolveQuerySet(querySet, 0, TS * 2, tsResolve, 0);
      encoder.copyBufferToBuffer(tsResolve, 0, tsRead, 0, TS * 16);
      tsBusy = true;
      device.queue.submit([encoder.finish()]);
      void tsRead.mapAsync(GPUMapMode.READ).then(() => {
        const st = new BigUint64Array(tsRead.getMappedRange().slice(0));
        tsRead.unmap();
        let sum = 0, max = 0, n = 0;
        for (let i = 0; i < TS; i++) {
          const a = st[i * 2], b = st[i * 2 + 1];
          if (b > a) { const ms = Number(b - a) / 1e6; sum += ms; max = Math.max(max, ms); n++; }
        }
        stats.gpuMs = n > 0 ? sum / n : null;
        stats.gpuMaxMs = n > 0 ? max : null;
        tsBusy = false;
      });
    } else {
      device.queue.submit([encoder.finish()]);
    }

    frame++;
    stats.frames = frame;
    if (cpuHist.length > 0) {
      stats.cpuMs = cpuHist.reduce((a, b) => a + b, 0) / cpuHist.length;
      stats.cpuMaxMs = Math.max(...cpuHist);
    }
    stats.pos = { x: scene.walker.x, y: scene.walker.y, z: scene.walker.z, yawDeg: scene.walker.yawDeg };
    if (frame % 15 === 0) {
      status(
        `実操作モード  WASD 移動 / Shift 走る / クリックでマウス視点（矢印キーでも回せる）\n` +
        `CPU ${stats.cpuMs.toFixed(2)} ms (max ${stats.cpuMaxMs.toFixed(1)})  GPU ${stats.gpuMs?.toFixed(2) ?? '-'} ms (max ${stats.gpuMaxMs?.toFixed(1) ?? '-'})  ` +
        `${resolution.width}×${resolution.height} DPR ${resolution.dpr}\n` +
        `位置 (${stats.pos.x.toFixed(1)}, ${stats.pos.y.toFixed(2)}, ${stats.pos.z.toFixed(1)}) 向き ${stats.pos.yawDeg.toFixed(0)}°  frame ${frame}`,
      );
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
