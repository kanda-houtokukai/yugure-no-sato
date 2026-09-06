// 実操作モード（既定）。ゲームに馴染みのない人が開いても操作が分かることを優先する。
//
// 視線の回し方は 3 通りを用意し、どれか 1 つでも通れば操作できるようにする:
//   ① ドラッグ（押したまま動かす）— どの環境でも通る。これを主にする
//   ② ポインタロック — 成功した環境でだけ有効。失敗しても黙ってドラッグへ落ちる
//      （Playwright 経由の Chrome では WrongDocumentError で失敗するのを実測）
//   ③ 矢印キー — マウスを使いたくない場合
// 数値の HUD は既定で非表示（H で切り替え）。案内は操作を始めたら消える。

import { acquireDevice, type DeviceBundle } from '../gpu/device';
import { MAX_DPR, resolution, type SceneRenderer } from './runner';
import type { WalkerInput } from '../scene/walker';

export interface PlayStats {
  frames: number;
  cpuMs: number;
  cpuMaxMs: number;
  gpuMs: number | null;
  gpuMaxMs: number | null;
  pos: { x: number; y: number; z: number; yawDeg: number };
  /** 検証用: 入力がどの経路で届いているか */
  input: { keys: string[]; look: 'none' | 'drag' | 'pointerlock' | 'keys'; pointerLock: boolean; pointerLockError: string | null };
  guideVisible: boolean;
}

export interface PlayableScene extends SceneRenderer {
  live: boolean;
  liveInput: WalkerInput;
  walker: { x: number; y: number; z: number; yawDeg: number };
}

/** 画面いっぱいに、起動時の縦横比を保って収める */
function fitCanvas(canvas: HTMLCanvasElement, aspect: number): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const fitW = Math.min(w, h * aspect);
  canvas.style.width = `${Math.round(fitW)}px`;
  canvas.style.height = `${Math.round(fitW / aspect)}px`;
}

export async function runPlay(canvas: HTMLCanvasElement, scene: PlayableScene): Promise<void> {
  document.body.classList.add('playing');
  const guide = document.getElementById('guide') as HTMLElement | null;
  const hint = document.getElementById('hint') as HTMLElement | null;
  const hud = document.getElementById('hud') as HTMLElement | null;
  if (guide) guide.hidden = false;

  // 描画解像度は起動時のウィンドウ（× DPR、上限 2）で固定する。
  // 途中でウィンドウが変わっても CSS で収めるだけ（描画先の作り直しは要らない）
  const dprOverride = Number(new URLSearchParams(location.search).get('dpr'));
  const dpr = Number.isFinite(dprOverride) && dprOverride > 0
    ? Math.min(MAX_DPR, dprOverride)   // 診断用の上書き（?dpr=1）
    : Math.min(MAX_DPR, Math.max(1, window.devicePixelRatio || 1));
  resolution.width = Math.round(Math.max(640, window.innerWidth) * dpr);
  resolution.height = Math.round(Math.max(360, window.innerHeight) * dpr);
  resolution.dpr = dpr;
  canvas.width = resolution.width;
  canvas.height = resolution.height;
  const aspect = resolution.width / resolution.height;
  fitCanvas(canvas, aspect);
  window.addEventListener('resize', () => fitCanvas(canvas, aspect));

  const bundle: DeviceBundle = await acquireDevice();
  const { device, format, hasTimestampQuery } = bundle;
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('canvas.getContext("webgpu") が null');
  context.configure({ device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  await scene.init(bundle, format);
  scene.live = true;

  // ---------------- 入力 ----------------
  const keys = new Set<string>();
  let lookDx = 0;
  let lookDy = 0;
  let lookSource: PlayStats['input']['look'] = 'none';
  let dragging = false;
  let pointerLockError: string | null = null;
  let started = false;
  let hudVisible = false;

  const dismissGuide = (): void => {
    if (started) return;
    started = true;
    if (guide) {
      guide.classList.add('hiding');
      window.setTimeout(() => { guide.hidden = true; }, 450);
    }
    if (hint) hint.hidden = false;
  };

  const updateHint = (): void => {
    if (!hint) return;
    const look = document.pointerLockElement === canvas ? 'マウスで視線（Esc で戻す）' : '押したまま動かして視線';
    hint.textContent = `W A S D 移動 ／ ${look} ／ Shift 走る ／ H 数値`;
  };
  updateHint();

  window.addEventListener('keydown', (e) => {
    if (e.repeat) { return; }
    keys.add(e.code);
    if (e.code === 'KeyH') {
      hudVisible = !hudVisible;
      if (hud) hud.hidden = !hudVisible;
    }
    if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
    if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) dismissGuide();
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => keys.clear());

  // ① ドラッグ（主）
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dismissGuide();
    if (document.pointerLockElement === canvas) return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.classList.add('dragging');
    canvas.setPointerCapture(e.pointerId);
  });
  const endDrag = (): void => { dragging = false; canvas.classList.remove('dragging'); };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('pointermove', (e) => {
    if (dragging && document.pointerLockElement !== canvas) {
      // movementX は合成イベントで 0 になる環境がある。前回位置との差分を主にする
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      lookDx += dx !== 0 || dy !== 0 ? dx : e.movementX;
      lookDy += dx !== 0 || dy !== 0 ? dy : e.movementY;
      if (dx !== 0 || dy !== 0 || e.movementX !== 0 || e.movementY !== 0) lookSource = 'drag';
    }
  });

  // ② ポインタロック（通る環境でだけ）。失敗しても案内は変えない＝ドラッグで操作できる
  canvas.addEventListener('click', () => {
    dismissGuide();
    if (document.pointerLockElement === canvas) return;
    try {
      const r = canvas.requestPointerLock() as unknown as Promise<void> | undefined;
      if (r && typeof r.catch === 'function') r.catch((err: unknown) => { pointerLockError = String(err); });
    } catch (err) {
      pointerLockError = String(err);
    }
  });
  document.addEventListener('pointerlockerror', () => { pointerLockError = 'pointerlockerror'; });
  document.addEventListener('pointerlockchange', updateHint);
  window.addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas) {
      lookDx += e.movementX;
      lookDy += e.movementY;
      lookSource = 'pointerlock';
    }
  });

  // ---------------- 計測（H で見せる） ----------------
  const TS = 120;
  const querySet = hasTimestampQuery ? device.createQuerySet({ type: 'timestamp', count: TS * 2 }) : null;
  const tsResolve = querySet ? device.createBuffer({ size: TS * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const tsRead = querySet ? device.createBuffer({ size: TS * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
  let tsBusy = false;
  const cpuHist: number[] = [];
  const stats: PlayStats = {
    frames: 0, cpuMs: 0, cpuMaxMs: 0, gpuMs: null, gpuMaxMs: null,
    pos: { x: 0, y: 0, z: 0, yawDeg: 0 },
    input: { keys: [], look: 'none', pointerLock: false, pointerLockError: null },
    guideVisible: true,
  };
  (window as unknown as { __yugure: { play?: PlayStats; scene?: PlayableScene } }).__yugure.play = stats;
  // 計測・診断から scene.describe() を呼べるようにする
  (window as unknown as { __yugure: { scene?: PlayableScene } }).__yugure.scene = scene;

  let last = -1;
  let frame = 0;
  const tick = (now: number): void => {
    if (last >= 0) { cpuHist.push(now - last); if (cpuHist.length > TS) cpuHist.shift(); }
    last = now;

    const yawKeys = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
    const pitchKeys = (keys.has('ArrowDown') ? 1 : 0) - (keys.has('ArrowUp') ? 1 : 0);
    if (yawKeys !== 0 || pitchKeys !== 0) lookSource = 'keys';
    scene.liveInput = {
      forward: (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0),
      strafe: (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0),
      yawDelta: lookDx * 0.14 + yawKeys * 1.6,
      pitchDelta: lookDy * 0.10 + pitchKeys * 1.1,
      run: keys.has('ShiftLeft') || keys.has('ShiftRight'),
    };
    lookDx = 0;
    lookDy = 0;

    const encoder = device.createCommandEncoder();
    const ti = frame % TS;
    scene.render({
      encoder,
      target: context.getCurrentTexture().createView(),
      frameIndex: frame,
      tsBegin: querySet ? { querySet, beginningOfPassWriteIndex: ti * 2 } : undefined,
      tsEnd: querySet ? { querySet, endOfPassWriteIndex: ti * 2 + 1 } : undefined,
    });
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
          const a = st[i * 2];
          const b = st[i * 2 + 1];
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
    stats.input = {
      keys: [...keys],
      look: lookSource,
      pointerLock: document.pointerLockElement === canvas,
      pointerLockError,
    };
    stats.guideVisible = !!guide && !guide.hidden;
    if (hudVisible && hud && frame % 10 === 0) {
      hud.textContent =
        `${stats.cpuMs.toFixed(2)} ms / フレーム（${(1000 / Math.max(stats.cpuMs, 0.001)).toFixed(0)} fps）  GPU ${stats.gpuMs?.toFixed(2) ?? '-'} ms\n` +
        `${resolution.width}×${resolution.height}  DPR ${resolution.dpr}\n` +
        `位置 (${stats.pos.x.toFixed(1)}, ${stats.pos.y.toFixed(2)}, ${stats.pos.z.toFixed(1)})  向き ${((stats.pos.yawDeg % 360) + 360) % 360 | 0}°`;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
