// ゴールデンビュー（決め打ちの視点）。毎回まったく同じ画角で撮るための固定値。
// 視点の高さは「地面からの高さ」で持ち、地面の高さは起動時に GPU（world.wgsl）へ問い合わせる。

export interface View {
  readonly name: string;
  /** 世界の xz と、その地点の地面からの高さ [m] */
  readonly eye: { x: number; z: number; above: number };
  /**
   * 視点を畦や道の線の上へ吸着する（線の位置は正本の WGSL にしかないので GPU に問い合わせる）。
   * family 0 = 縦線 B（x を合わせる）、1 = 横線 A（z を合わせる）。offset は線からの横ずれ [m]
   */
  readonly snap?: { family: 0 | 1; index: number; offset: number };
  /** 太陽の映り込み位置の機械検証を行う視点か */
  readonly glintCheck?: boolean;
  /** 歩き手の筋書き名（walker.ts の scriptInput）。指定すると三人称カメラで歩く */
  readonly script?: string;
  /** 描画フレーム数（既定 120）。筋書きの長さに合わせる */
  readonly frames?: number;
  /** 三人称カメラの距離 [m]（既定 3.0）。人物を遠景で見せるときに伸ばす */
  readonly camDist?: number;
  /** 方位（北 = 0°, 東 = 90°）と仰角（度） */
  readonly yawDeg: number;
  readonly pitchDeg: number;
  readonly fovDeg: number;
  readonly time: number;
  readonly sunAzimuthDeg: number;
  readonly sunElevationDeg: number;
  readonly exposure: number;
  /** 0 = 通常、1 = 法線、2 = 材質、3 = 高さ */
  readonly debug: number;
}

/**
 * 夏の夕暮れ。太陽は低空（[DECISION] 2）。方位は谷の出口（西）に合わせる。
 * 盆地を囲む丘（70m）や山（400m）は仰角 3.5° の太陽を隠してしまうので、
 * 川が西へ抜ける切れ目を低くし、そこへ夕日を沈める（world.wgsl の outlet）。
 */
const SUN_AZ = 275;
const SUN_EL = 3.5;

const base = { time: 0, sunAzimuthDeg: SUN_AZ, sunElevationDeg: SUN_EL, exposure: 0.45, debug: 0 };

export const VIEWS: Readonly<Record<string, View>> = {
  // 集落側（南）の主道に立ち、田を越えて神社の丘（北）を望む
  front: { name: 'front', eye: { x: 0, z: -135, above: 1.6 }, snap: { family: 0, index: 0, offset: 0 }, yawDeg: -8, pitchDeg: 2, fovDeg: 55, ...base },
  // 谷の南の上空から盆地全体を見下ろす
  bird: { name: 'bird', eye: { x: -80, z: -330, above: 75 }, yawDeg: 14, pitchDeg: -20, fovDeg: 50, ...base },
  // 主道の上、地面すれすれから北へ
  ground: { name: 'ground', eye: { x: 0.9, z: -70, above: 0.12 }, snap: { family: 0, index: 0, offset: 0.3 }, yawDeg: 0, pitchDeg: 1, fovDeg: 65, ...base },
  // 畦（縦線 -1）の上にしゃがみ、夕日の方向へ田の水面を見る（水面の見せ場）
  // 歩き手の筋書き（道 → 田 → 畦 → 振り返る）。跡が田・道・草地に残る画
  walk: { name: 'walk', eye: { x: 0, z: -119, above: 0 }, snap: { family: 0, index: 0, offset: 0 }, script: 'walk1', frames: 420, yawDeg: 0, pitchDeg: 0, fovDeg: 55, ...base },
  // 風が田を渡る様子を見る専用視点（verify の 4 視点には含めない）: 主道の上 2.5m から西の田を逆光で見渡す
  wind: { name: 'wind', eye: { x: 0, z: -40, above: 2.5 }, snap: { family: 0, index: 0, offset: 0 }, yawDeg: 290, pitchDeg: -6, fovDeg: 60, ...base },
  water: { name: 'water', eye: { x: -26, z: 22, above: 0.9 }, snap: { family: 0, index: -1, offset: 0 }, glintCheck: true, yawDeg: 272, pitchDeg: -3, fovDeg: 55, ...base },
};

export const VIEW_NAMES = Object.keys(VIEWS);

function num(q: URLSearchParams, key: string, fallback: number): number {
  const v = Number(q.get(key));
  return q.has(key) && Number.isFinite(v) ? v : fallback;
}

/**
 * URL から視点を決める。`?view=front` で名前指定。
 * `eye=x,z,above` `yaw` `pitch` `fov` `t` `sunAz` `sunEl` `exposure` `dbg` `frames` `script` `camDist` で個別に上書きできる。
 */
export function viewFromUrl(search: string): View {
  const q = new URLSearchParams(search);
  const b = VIEWS[q.get('view') ?? 'front'] ?? VIEWS.front;
  let eye = b.eye;
  const eyeRaw = q.get('eye');
  if (eyeRaw) {
    const parts = eyeRaw.split(',').map(Number);
    if (parts.length === 3 && parts.every(Number.isFinite)) eye = { x: parts[0], z: parts[1], above: parts[2] };
  }
  return {
    name: q.get('view') ?? b.name,
    eye,
    ...(b.snap && !eyeRaw ? { snap: b.snap } : {}),
    ...(b.glintCheck ? { glintCheck: true } : {}),
    ...(q.get('script') ?? b.script ? { script: q.get('script') ?? b.script } : {}),
    frames: num(q, 'frames', b.frames ?? 120),
    ...(q.has('camDist') ? { camDist: num(q, 'camDist', 3) } : b.camDist !== undefined ? { camDist: b.camDist } : {}),
    yawDeg: num(q, 'yaw', b.yawDeg),
    pitchDeg: num(q, 'pitch', b.pitchDeg),
    fovDeg: num(q, 'fov', b.fovDeg),
    time: num(q, 't', b.time),
    sunAzimuthDeg: num(q, 'sunAz', b.sunAzimuthDeg),
    sunElevationDeg: num(q, 'sunEl', b.sunElevationDeg),
    exposure: num(q, 'exposure', b.exposure),
    debug: num(q, 'dbg', b.debug),
  };
}
