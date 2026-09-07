// 里の建物を数値から起こす（パラメトリック生成）。素材ファイルは使わない。
// 昭和期以前の農家: 切妻、深い軒、土台の上に立つ、真壁（柱が見える）、縁側、格子窓、瓦屋根。
//
// 材質の種別（シェーダの kind）:
//   0 木の柱・梁  1 板壁  2 漆喰壁  3 瓦  4 基礎石  5 縁側の床板  6 格子（窓）  7 石（石垣・石段）
export const MAT_POST = 0;
export const MAT_PLANK = 1;
export const MAT_PLASTER = 2;
export const MAT_TILE = 3;
export const MAT_FOUNDATION = 4;
export const MAT_DECK = 5;
export const MAT_LATTICE = 6;
export const MAT_STONE = 7;
export const MAT_THATCH = 8;        // 茅葺き
export const MAT_GRASS_RIDGE = 9;   // 芝棟（棟に生える草）

import { MeshBuilder, rng, type Built, type V3 } from './mesh';

export interface FarmhouseParams {
  /** 間口 [m]（棟が走る向き = X） */
  width: number;
  /** 奥行き [m]（軒が下る向き = Z） */
  depth: number;
  /** 軒の高さ（壁の上端）[m] */
  eaveHeight: number;
  /** 棟の高さ [m] */
  ridgeHeight: number;
  /** 軒の出 [m]。日本建築の要。浅いと洋風に見える */
  eaveOut: number;
  /** けらば（妻側の出）[m] */
  gableOut: number;
  /** 床の高さ（土台の上）[m] */
  floorHeight: number;
  /** 縁側の奥行き [m]。0 なら縁側なし */
  veranda: number;
  /** 下屋（差し掛けの庇）を付ける側: 0 なし, 1 背面 */
  lean: number;
  /** 建物の種類。壁の作りが変わる */
  style: 'house' | 'barn' | 'storehouse';
  /** 屋根の葺き方 */
  roof: 'tile' | 'thatch';
  /** 茅葺きの厚み [m]（roof='thatch' のとき） */
  thatchThickness?: number;
}

export const FARMHOUSE_DEFAULT: FarmhouseParams = {
  width: 10.8,        // 6 間
  depth: 7.2,         // 4 間
  eaveHeight: 3.05,
  ridgeHeight: 6.15,   // 勾配 約 35°。緩いと屋根が横に広がって平べったく見える
  eaveOut: 1.15,       // 深すぎると壁が全部隠れる（1.5 では腰板しか見えなかった）
  gableOut: 0.65,
  floorHeight: 0.52,
  veranda: 1.2,
  lean: 1,
  style: 'house',
  roof: 'tile',
};

/**
 * 茅葺きの屋根。瓦と作りが別物なので分けた。
 * 要点: 分厚い（40〜60cm）、屋根面が緩やかにふくらむ、軒先に厚みの断面が見える、棟に押さえ（芝棟）。
 * 勾配は瓦より急（水を切るため 45° 前後）。
 */
function buildThatchRoof(
  m: MeshBuilder, r: () => number, p: FarmhouseParams,
  rw: number, re: number, eh: number, rh: number, hw: number, wallTop: number, D: number,
): void {
  const th = p.thatchThickness ?? 0.5;
  const N = 10;                     // 流れ方向の分割
  const M = 8;                      // 棟に沿う分割（けらばの反りに使う）
  /** 流れ方向 t（0 = 棟、1 = 軒先）での断面。ふくらみを持たせる */
  const prof = (t: number): { z: number; y: number } => {
    const bulge = 0.34 * Math.sin(Math.PI * Math.min(1, t * 1.06));
    return { z: re * t, y: rh - (rh - eh) * t + bulge };
  };
  // 軒先とけらばの輪郭を不揃いにする。遠目に「毛羽立ち」を伝えるのは明るさのむらでなく形の乱れ
  const wob = (x: number, k: number): number => 0.05 * Math.sin(x * k) + 0.035 * Math.sin(x * k * 2.7 + 1.3);
  for (const sg of [1, -1]) {
    for (let i = 0; i < N; i++) {
      const t0 = i / N, t1 = (i + 1) / N;
      const a0 = prof(t0), a1 = prof(t1);
      for (let j = 0; j < M; j++) {
        const x0 = -rw + (2 * rw * j) / M;
        const x1 = -rw + (2 * rw * (j + 1)) / M;
        // 上面（茅）
        const uv: [number, number][] = [[x0, t0 * re], [x1, t0 * re], [x1, t1 * re], [x0, t1 * re]];
        const e0 = i === N - 1 ? wob(x0, 1.7) : 0;
        const e1 = i === N - 1 ? wob(x1, 1.7) : 0;
        const P0: V3 = [x0, a0.y, sg * a0.z];
        const P1: V3 = [x1, a0.y, sg * a0.z];
        const P2: V3 = [x1, a1.y + e1 * 0.5, sg * (a1.z + e1)];
        const P3: V3 = [x0, a1.y + e0 * 0.5, sg * (a1.z + e0)];
        if (sg > 0) m.quad(P3, P2, P1, P0, MAT_THATCH, 0, uv);
        else m.quad(P0, P1, P2, P3, MAT_THATCH, 0, uv);
        // 下面（暗い野地）
        const d = th;
        if (sg > 0) m.quad([x0, P0[1] - d, P0[2]], [x1, P1[1] - d, P1[2]], [x1, P2[1] - d, P2[2]], [x0, P3[1] - d, P3[2]], MAT_POST, 0, undefined, [0, -1, 0]);
        else m.quad([x0, P3[1] - d, P3[2]], [x1, P2[1] - d, P2[2]], [x1, P1[1] - d, P1[2]], [x0, P0[1] - d, P0[2]], MAT_POST, 0, undefined, [0, -1, 0]);
      }
      // けらば（妻側）の切り口。茅の断面が厚く見える
      for (const x of [rw, -rw]) {
        const nx = x > 0 ? 1 : -1;
        const A: V3 = [x, a0.y, sg * a0.z];
        const B: V3 = [x, a1.y, sg * a1.z];
        const A2: V3 = [x, a0.y - th, sg * a0.z];
        const B2: V3 = [x, a1.y - th, sg * a1.z];
        const uv: [number, number][] = [[a0.z, 0], [a1.z, 0], [a1.z, th], [a0.z, th]];
        if (nx * sg > 0) m.quad(A, B, B2, A2, MAT_THATCH, 0, uv, [nx, 0, 0]);
        else m.quad(B, A, A2, B2, MAT_THATCH, 0, uv, [nx, 0, 0]);
      }
    }
    // 軒先の断面（茅を切り揃えた厚い小口）。茅葺きの最大の見どころ
    const e = prof(1);
    for (let j = 0; j < M; j++) {
      const x0 = -rw + (2 * rw * j) / M;
      const x1 = -rw + (2 * rw * (j + 1)) / M;
      // 軒先を 1 枚ずつ前後・上下にずらす（切り揃えた茅の不揃いさ）
      const w0 = wob(x0, 1.7), w1 = wob(x1, 1.7);
      const A: V3 = [x0, e.y + w0 * 0.5, sg * (e.z + w0)];
      const B: V3 = [x1, e.y + w1 * 0.5, sg * (e.z + w1)];
      const A2: V3 = [x0, e.y - th + w0 * 0.8, sg * (e.z - 0.06 + w0)];
      const B2: V3 = [x1, e.y - th + w1 * 0.8, sg * (e.z - 0.06 + w1)];
      const uv: [number, number][] = [[x0, 0], [x1, 0], [x1, th], [x0, th]];
      if (sg > 0) m.quad(A2, B2, B, A, MAT_THATCH, 0, uv, [0, -0.3, sg]);
      else m.quad(A, B, B2, A2, MAT_THATCH, 0, uv, [0, -0.3, sg]);
    }
  }
  // 棟: 芝棟（土をかぶせた丸い棟）。竹の押さえを数本
  const top = prof(0);
  m.tube([-rw - 0.05, top.y + 0.30, 0], [rw + 0.05, top.y + 0.30, 0], 0.44, 0.44, 10, MAT_THATCH);
  m.tube([-rw - 0.05, top.y + 0.58, 0], [rw + 0.05, top.y + 0.58, 0], 0.22, 0.22, 8, MAT_GRASS_RIDGE);
  for (let i = 0; i <= 7; i++) {
    const x = -rw + (2 * rw * i) / 7;
    m.tube([x, top.y + 0.62, -0.5], [x, top.y + 0.62, 0.5], 0.035, 0.035, 5, MAT_POST);
  }
  // 妻壁（茅の下、小屋裏）
  for (const x of [hw, -hw]) {
    const nx = x > 0 ? 1 : -1;
    const px = x - nx * 0.06;
    m.box([px, wallTop + 0.42, 0], [0.16, 0.26, D * 0.92], MAT_POST);
    m.box([px, (wallTop + 0.42 + rh - 0.3) / 2, 0], [0.13, rh - 0.3 - wallTop - 0.42, 0.16], MAT_POST);
    m.box([px - nx * 0.02, wallTop + 1.35, 0], [0.06, 0.66, 1.6], MAT_LATTICE);
    m.box([px - nx * 0.01, wallTop + 0.25, 0], [0.06, 0.5, D * 0.9], MAT_PLANK);
  }
  void r;
}

/**
 * 農家 1 棟。原点は地面の中心、+Z が正面（縁側のある側）。
 * 棟は X 方向に走る（平入り）。
 */
export function buildFarmhouse(seed: number, p: FarmhouseParams): Built {
  const r = rng(seed);
  const m = new MeshBuilder();
  const W = p.width, D = p.depth;
  const hw = W / 2, hd = D / 2;
  const fl = p.floorHeight;
  const eh = p.eaveHeight;
  const rh = p.ridgeHeight;

  // ---- 基礎: 玉石と土台の帯 ----
  const postXs: number[] = [];
  const bays = Math.max(3, Math.round(W / 1.9));
  for (let i = 0; i <= bays; i++) postXs.push(-hw + (W * i) / bays);
  for (const x of postXs) {
    for (const z of [-hd, hd]) {
      m.box([x, fl * 0.28, z], [0.42 + 0.1 * r(), fl * 0.56, 0.42 + 0.1 * r()], MAT_FOUNDATION);
    }
  }
  // 土台（足元をぐるりと回る横木）
  m.box([0, fl - 0.09, hd], [W + 0.24, 0.18, 0.24], MAT_POST);
  m.box([0, fl - 0.09, -hd], [W + 0.24, 0.18, 0.24], MAT_POST);
  m.box([hw, fl - 0.09, 0], [0.24, 0.18, D], MAT_POST);
  m.box([-hw, fl - 0.09, 0], [0.24, 0.18, D], MAT_POST);
  // 床下は暗い隙間
  m.box([0, fl * 0.5, 0], [W - 0.5, fl * 0.9, D - 0.5], MAT_FOUNDATION);

  // ---- 壁と柱（真壁: 柱が外に見える） ----
  const wallTop = fl + (eh - fl);
  const plinth = fl + 0.95;   // 腰板の上端
  for (const z of [hd, -hd]) {
    const nz = z > 0 ? 1 : -1;
    for (let i = 0; i < bays; i++) {
      const x0 = postXs[i] + 0.07;
      const x1 = postXs[i + 1] - 0.07;
      const cx = (x0 + x1) / 2;
      const bw = x1 - x0;
      // 腰: 土蔵は黒い板（海鼠壁の代わり）、他は腰板
      m.box([cx, (fl + plinth) / 2, z - nz * (p.style === 'storehouse' ? 0.02 : 0.05)],
        [bw, plinth - fl, p.style === 'storehouse' ? 0.16 : 0.1], MAT_PLANK);
      // 上部: 種類で変わる
      const isFront = z > 0;
      const upH = wallTop - plinth;
      if (p.style === 'barn') {
        // 納屋: 全面が板壁。正面の中央に大きな引き戸
        const isDoor = isFront && i >= Math.floor(bays / 2) - 1 && i <= Math.floor(bays / 2);
        m.box([cx, plinth + upH * 0.5, z - nz * 0.05], [bw, upH, 0.1], isDoor ? MAT_POST : MAT_PLANK);
      } else if (p.style === 'storehouse') {
        // 土蔵: 分厚い漆喰。窓は小さく 1 つだけ
        const isWindow = isFront && i === Math.floor(bays / 2);
        m.box([cx, plinth + upH * 0.5, z - nz * 0.02], [bw, upH, 0.16], MAT_PLASTER);
        if (isWindow) m.box([cx, plinth + upH * 0.62, z - nz * 0.10], [Math.min(bw * 0.5, 0.9), upH * 0.42, 0.08], MAT_LATTICE);
      } else {
        const isWindow = isFront ? i >= 1 && i <= bays - 2 && i % 2 === 1 : i % 3 === 1;
        if (isWindow) {
          m.box([cx, plinth + upH * 0.5, z - nz * 0.06], [bw, upH, 0.06], MAT_LATTICE);
        } else {
          m.box([cx, plinth + upH * 0.5, z - nz * 0.05], [bw, upH, 0.1], MAT_PLASTER);
        }
      }
    }
  }
  // 妻側の壁
  for (const x of [hw, -hw]) {
    const nx = x > 0 ? 1 : -1;
    const upperMat = p.style === 'barn' ? MAT_PLANK : MAT_PLASTER;
    m.box([x - nx * 0.05, (fl + plinth) / 2, 0], [0.1, plinth - fl, D], MAT_PLANK);
    m.box([x - nx * 0.05, (plinth + wallTop) / 2, 0], [0.1, wallTop - plinth, D], upperMat);
    // 妻壁（軒より上の三角）
    const tri: V3[] = [
      [x - nx * 0.05, wallTop, -hd],
      [x - nx * 0.05, wallTop, hd],
      [x - nx * 0.05, rh - 0.12, 0],
    ];
    m.polygon(nx > 0 ? tri : [tri[1], tri[0], tri[2]], upperMat, 0, [nx, 0, 0]);
  }
  // 柱（外に見える）
  for (const x of postXs) {
    for (const z of [hd, -hd]) {
      m.box([x, (fl + wallTop) / 2, z], [0.14, wallTop - fl, 0.14], MAT_POST);
    }
  }
  for (const x of [hw, -hw]) {
    for (const z of [-hd * 0.5, 0, hd * 0.5]) {
      m.box([x, (fl + wallTop) / 2, z], [0.14, wallTop - fl, 0.14], MAT_POST);
    }
  }
  // 桁（軒を受ける横木）
  m.box([0, wallTop + 0.11, hd], [W + 0.3, 0.22, 0.2], MAT_POST);
  m.box([0, wallTop + 0.11, -hd], [W + 0.3, 0.22, 0.2], MAT_POST);

  // ---- 屋根 ----
  const rw = hw + p.gableOut;         // 屋根の半幅（けらば込み）
  const re = hd + p.eaveOut;          // 軒先の Z
  if (p.roof === 'thatch') {
    buildThatchRoof(m, r, p, rw, re, eh, rh, hw, wallTop, D);
  } else {
  const thick = 0.16;
  const slopes: 1[] | number[] = [1, -1];
  for (const s of slopes) {
    const ridge0: V3 = [-rw, rh, 0];
    const ridge1: V3 = [rw, rh, 0];
    const eave0: V3 = [-rw, eh, s * re];
    const eave1: V3 = [rw, eh, s * re];
    const run = Math.hypot(re, rh - eh);
    // 上面（瓦）。uv は m 単位: x = 棟に沿う向き, y = 流れ（軒→棟）
    const uv: [number, number][] = s > 0
      ? [[0, 0], [2 * rw, 0], [2 * rw, run], [0, run]]
      : [[0, 0], [2 * rw, 0], [2 * rw, run], [0, run]];
    if (s > 0) m.quad(eave0, eave1, ridge1, ridge0, MAT_TILE, 0, uv);
    else m.quad(ridge0, ridge1, eave1, eave0, MAT_TILE, 0, uv);
    // 下面（野地板）と軒先の小口
    const dn: V3 = [0, -thick, 0];
    const e0d: V3 = [eave0[0], eave0[1] - thick, eave0[2]];
    const e1d: V3 = [eave1[0], eave1[1] - thick, eave1[2]];
    const r0d: V3 = [ridge0[0], ridge0[1] - thick, ridge0[2]];
    const r1d: V3 = [ridge1[0], ridge1[1] - thick, ridge1[2]];
    if (s > 0) m.quad(r0d, r1d, e1d, e0d, MAT_POST, 0, undefined, [0, -1, 0]);
    else m.quad(e0d, e1d, r1d, r0d, MAT_POST, 0, undefined, [0, -1, 0]);
    // 軒先の厚み（軒瓦の小口。深い軒の見どころなので少し厚く）
    const lip = 0.1;
    const e0l: V3 = [eave0[0], eave0[1] - thick - lip, eave0[2]];
    const e1l: V3 = [eave1[0], eave1[1] - thick - lip, eave1[2]];
    if (s > 0) m.quad(e0l, e1l, eave1, eave0, MAT_TILE, 0, [[0, 0], [2 * rw, 0], [2 * rw, 0.3], [0, 0.3]], [0, 0, 1]);
    else m.quad(eave1, e1l, e0l, eave0, MAT_TILE, 0, [[0, 0], [2 * rw, 0], [2 * rw, 0.3], [0, 0.3]], [0, 0, -1]);
    void dn;
    // 垂木（軒の下に並ぶ）。深い軒の見どころ。
    // 軒先より内側で止める: 小口が軒先に出ると、西日を正面から受けて白い点の列になる（実測）
    const rafters = Math.round(W / 0.55);
    const rLen = p.eaveOut + 0.25;
    const rCenter = hd + p.eaveOut * 0.45;
    for (let i = 0; i <= rafters; i++) {
      const x = -rw + (2 * rw * i) / rafters;
      const yc = eh + (rh - eh) * (1 - rCenter / re) * 0.5 - thick - 0.05;
      m.box([x, yc, s * rCenter], [0.07, 0.09, rLen], MAT_POST);
    }
  }
  // 大棟: 熨斗瓦を数段積み、上に丸瓦を伏せる。両端に鬼瓦
  const ridgeLen = 2 * rw - 0.1;
  for (let i = 0; i < 3; i++) {
    const w = 0.52 - i * 0.07;
    m.box([0, rh + 0.055 + i * 0.10, 0], [ridgeLen, 0.10, w], MAT_TILE);
  }
  m.box([0, rh + 0.40, 0], [ridgeLen, 0.14, 0.30], MAT_TILE);
  for (const x of [rw - 0.06, -(rw - 0.06)]) {
    const nx = x > 0 ? 1 : -1;
    // 鬼瓦: 棟端で立ち上がる板
    m.box([x + nx * 0.05, rh + 0.34, 0], [0.12, 0.62, 0.42], MAT_TILE);
    m.box([x + nx * 0.05, rh + 0.68, 0], [0.14, 0.16, 0.26], MAT_TILE);
  }
  // 破風（妻の縁）。太い板を屋根の勾配に沿わせる
  for (const x of [rw, -rw]) {
    const nx = x > 0 ? 1 : -1;
    for (const sg of [1, -1]) {
      const a: V3 = [x + nx * 0.03, rh + 0.08, 0];
      const b: V3 = [x + nx * 0.03, eh, sg * re];
      const c: V3 = [x + nx * 0.03, eh - 0.34, sg * re];
      const d: V3 = [x + nx * 0.03, rh - 0.30, 0];
      m.quad(a, b, c, d, MAT_PLANK, 0, undefined, [nx, 0, 0]);
      // 破風の小口（厚み）
      const a2: V3 = [x - nx * 0.09, rh + 0.08, 0];
      const b2: V3 = [x - nx * 0.09, eh, sg * re];
      if (sg > 0) m.quad(a, b, b2, a2, MAT_PLANK, 0, undefined, [0, 1, 0]);
      else m.quad(b, a, a2, b2, MAT_PLANK, 0, undefined, [0, 1, 0]);
    }
  }
  // 妻面の作り込み: 梁の見え、小屋裏の開口、板張り
  for (const x of [hw, -hw]) {
    const nx = x > 0 ? 1 : -1;
    const px = x - nx * 0.11;
    // 桁の上に載る梁（妻壁を横切る）
    m.box([px, wallTop + 0.42, 0], [0.16, 0.26, D * 0.92], MAT_POST);
    m.box([px, wallTop + 1.05, 0], [0.14, 0.22, D * 0.62], MAT_POST);
    // 束（棟を受ける）
    m.box([px, (wallTop + 0.42 + rh) / 2, 0], [0.14, rh - wallTop - 0.42, 0.16], MAT_POST);
    // 小屋裏の開口（煙出しの格子）
    m.box([px - nx * 0.02, wallTop + 1.55, 0], [0.06, 0.62, 1.5], MAT_LATTICE);
    // 妻壁の下半分は板張り
    m.box([px - nx * 0.01, wallTop + 0.2, 0], [0.06, 0.4, D * 0.9], MAT_PLANK);
  }

  }

  // ---- 縁側（葺き方によらず共通） ----
  if (p.veranda > 0) {
    const v0 = hd;
    const v1 = hd + p.veranda;
    m.box([0, fl - 0.04, (v0 + v1) / 2], [W - 0.3, 0.08, p.veranda], MAT_DECK);
    // 束と束石（無いと縁側が浮いて見える）
    for (let i = 0; i <= 5; i++) {
      const x = -(W - 1.2) / 2 + ((W - 1.2) * i) / 5;
      m.box([x, 0.1, v1 - 0.18], [0.3 + 0.06 * r(), 0.2, 0.3 + 0.06 * r()], MAT_FOUNDATION);
      m.box([x, (fl - 0.08 + 0.2) / 2, v1 - 0.18], [0.1, fl - 0.28, 0.1], MAT_POST);
    }
    // 床下を塞ぐ暗い帯（縁の下の陰）
    m.box([0, (fl - 0.1) / 2, (v0 + v1) / 2], [W - 0.9, fl - 0.2, p.veranda - 0.5], MAT_FOUNDATION);
    // 縁の先の板
    m.box([0, fl - 0.02, v1], [W - 0.3, 0.06, 0.12], MAT_DECK);
  }

  // ---- 下屋（背面の差し掛け） ----
  if (p.lean > 0) {
    const lz0 = -hd;
    const lz1 = -hd - 2.2;
    const ly0 = fl + 2.25;
    const ly1 = fl + 1.75;
    m.quad([-hw, ly0, lz0], [hw, ly0, lz0], [hw, ly1, lz1], [-hw, ly1, lz1], MAT_TILE, 0,
      [[0, 0], [W, 0], [W, 2.3], [0, 2.3]]);
    m.quad([-hw, ly1 - 0.1, lz1], [hw, ly1 - 0.1, lz1], [hw, ly0 - 0.1, lz0], [-hw, ly0 - 0.1, lz0], MAT_POST, 0, undefined, [0, -1, 0]);
    for (const x of [-hw + 0.3, 0, hw - 0.3]) {
      m.box([x, (ly1 - 0.1) / 2, lz1 + 0.1], [0.12, ly1 - 0.1, 0.12], MAT_POST);
    }
    // 背面の板壁
    m.box([0, (fl + ly1 - 0.2) / 2, lz1 + 0.06], [W, ly1 - 0.2 - fl, 0.1], MAT_PLANK);
  }

  // ---- 玄関の踏み石 ----
  m.box([W * 0.18, 0.11, hd + p.veranda + 0.35], [1.1, 0.22, 0.8], MAT_STONE);
  void r;

  // 当たり判定: 壁の外周。軒（eaveOut）は含めないので軒下はくぐれる。
  // 縁側がある側（+Z）はその分だけ広げる（縁側に上がる高さは作らない＝通り抜けない塊にする）
  const cd = D + p.veranda;
  m.solid([0, (eh + 1.0) / 2, p.veranda / 2], [W, eh + 1.0, cd]);

  return m.built();
}


/** 納屋: 板壁だけの小屋。縁側も下屋もない */
export const BARN_DEFAULT: FarmhouseParams = {
  width: 7.6, depth: 5.4,
  eaveHeight: 3.3, ridgeHeight: 5.3,
  eaveOut: 0.85, gableOut: 0.5,
  floorHeight: 0.30, veranda: 0, lean: 0,
  style: 'barn', roof: 'tile',
};

/** 土蔵: 分厚い漆喰の壁、小さな窓、背が高い */
export const STOREHOUSE_DEFAULT: FarmhouseParams = {
  width: 5.6, depth: 4.6,
  eaveHeight: 4.3, ridgeHeight: 6.3,
  eaveOut: 0.75, gableOut: 0.45,
  floorHeight: 0.42, veranda: 0, lean: 0,
  style: 'storehouse', roof: 'tile',
};

// ============ 神社 ============

/**
 * 明神鳥居。笠木が反り、島木・貫・額束を持つ。柱は上でわずかに内へ倒す（転び）。
 * 原点は地面の中心、参道は Z 方向（+Z が手前）。
 */
export function buildTorii(seed: number, height: number, span: number): Built {
  const r = rng(seed);
  const m = new MeshBuilder();
  const pr = height * 0.052;              // 柱の半径
  const lean = height * 0.018;            // 転び（上で内へ）
  const nukiY = height * 0.62;            // 貫の高さ
  const shimaY = height * 0.93;           // 島木の下端
  const hs = span / 2;

  for (const sx of [1, -1]) {
    m.tube([sx * hs, 0, 0], [sx * (hs - lean), height, 0], pr * 1.12, pr * 0.94, 12, MAT_POST);
    // 亀腹（柱の根元の石）
    m.tube([sx * hs, 0, 0], [sx * hs, 0.26, 0], pr * 1.55, pr * 1.32, 12, MAT_STONE);
  }
  // 貫（柱を貫いて外へ出る）
  m.box([0, nukiY, 0], [span + pr * 3.4, height * 0.052, pr * 1.5], MAT_POST);
  // 島木（笠木の下の直線材）。柱より外へ張り出す
  const outw = span + height * 0.30;
  m.box([0, shimaY, 0], [outw, height * 0.055, pr * 2.3], MAT_POST);
  // 笠木（両端が反り上がる）。分割して曲げる
  const N = 14;
  const kh = height * 0.062;
  const rise = height * 0.075;
  const kasagi = (t: number): number => shimaY + height * 0.058 + rise * (t * t);   // t = -1..1
  for (let i = 0; i < N; i++) {
    const t0 = -1 + (2 * i) / N, t1 = -1 + (2 * (i + 1)) / N;
    const x0 = (t0 * outw) / 2 * 1.06, x1 = (t1 * outw) / 2 * 1.06;
    const y0 = kasagi(t0), y1 = kasagi(t1);
    const w = pr * 2.7;
    // 上面・前後面・下面の 4 枚で角材を曲げる
    m.quad([x0, y0 + kh, -w], [x1, y1 + kh, -w], [x1, y1 + kh, w], [x0, y0 + kh, w], MAT_POST, 0, undefined, [0, 1, 0]);
    m.quad([x0, y0, w], [x1, y1, w], [x1, y1 + kh, w], [x0, y0 + kh, w], MAT_POST, 0, undefined, [0, 0, 1]);
    m.quad([x1, y1, -w], [x0, y0, -w], [x0, y0 + kh, -w], [x1, y1 + kh, -w], MAT_POST, 0, undefined, [0, 0, -1]);
    m.quad([x1, y1, w], [x0, y0, w], [x0, y0, -w], [x1, y1, -w], MAT_POST, 0, undefined, [0, -1, 0]);
  }
  // 額束（貫と島木の間の短い柱）
  m.box([0, (nukiY + shimaY) / 2, 0], [height * 0.055, shimaY - nukiY, pr * 1.2], MAT_POST);
  void r;

  // 当たり判定: 柱 2 本だけ（間はくぐれる）
  for (const sx of [-1, 1]) m.solid([sx * span / 2, height / 2, 0], [0.62, height, 0.62]);

  return m.built();
}

/**
 * 小さな村の社殿。高床・板壁・切妻の妻入・階段と縁・千木と鰹木。
 * 原点は地面の中心、+Z が正面（階段のある側）。
 */
export function buildShrine(seed: number, scale: number): Built {
  const r = rng(seed);
  const m = new MeshBuilder();
  const W = 3.6 * scale, D = 3.2 * scale;
  const hw = W / 2, hd = D / 2;
  const fl = 1.15 * scale;        // 高床
  const eh = fl + 2.15 * scale;   // 軒
  const rh = fl + 3.5 * scale;    // 棟
  const eo = 0.75 * scale;        // 軒の出
  const go = 0.55 * scale;

  // 束石と床束
  for (const x of [-hw + 0.3, 0, hw - 0.3]) {
    for (const z of [-hd + 0.3, hd - 0.3]) {
      m.box([x, 0.14, z], [0.34, 0.28, 0.34], MAT_STONE);
      m.box([x, (fl + 0.28) / 2, z], [0.16, fl - 0.28, 0.16], MAT_POST);
    }
  }
  // 床と縁（周囲に回る）
  m.box([0, fl - 0.06, 0], [W + 0.7, 0.12, D + 0.7], MAT_DECK);
  // 高欄（縁の手すり）
  for (const sz of [1, -1]) {
    m.box([0, fl + 0.34, sz * (hd + 0.33)], [W + 0.7, 0.07, 0.07], MAT_POST);
    for (let i = 0; i <= 4; i++) {
      const x = -(W + 0.5) / 2 + ((W + 0.5) * i) / 4;
      m.box([x, fl + 0.17, sz * (hd + 0.33)], [0.06, 0.34, 0.06], MAT_POST);
    }
  }
  // 壁（板張り）と柱
  m.box([0, (fl + eh) / 2, 0], [W, eh - fl, D], MAT_PLANK);
  for (const sx of [1, -1]) for (const sz of [1, -1]) {
    m.box([sx * hw, (fl + eh) / 2, sz * hd], [0.15, eh - fl, 0.15], MAT_POST);
  }
  // 扉（正面）
  m.box([0, fl + (eh - fl) * 0.47, hd - 0.02], [W * 0.42, (eh - fl) * 0.78, 0.08], MAT_POST);
  // 屋根（切妻・妻入: 棟は Z 方向に走る）
  const rwz = hd + go;   // 棟の長さ方向の半分
  const rex = hw + eo;   // 軒先の X
  const th = 0.14;
  for (const s of [1, -1]) {
    const ridge0: V3 = [0, rh, -rwz];
    const ridge1: V3 = [0, rh, rwz];
    const eave0: V3 = [s * rex, eh, -rwz];
    const eave1: V3 = [s * rex, eh, rwz];
    const run = Math.hypot(rex, rh - eh);
    const uv: [number, number][] = [[0, 0], [2 * rwz, 0], [2 * rwz, run], [0, run]];
    if (s > 0) m.quad(ridge0, ridge1, eave1, eave0, MAT_PLANK, 0, uv);
    else m.quad(eave0, eave1, ridge1, ridge0, MAT_PLANK, 0, uv);
    const d: V3 = [0, -th, 0];
    const r0: V3 = [ridge0[0] + d[0], ridge0[1] + d[1], ridge0[2]];
    const r1: V3 = [ridge1[0], ridge1[1] - th, ridge1[2]];
    const e0: V3 = [eave0[0], eave0[1] - th, eave0[2]];
    const e1: V3 = [eave1[0], eave1[1] - th, eave1[2]];
    if (s > 0) m.quad(e0, e1, r1, r0, MAT_POST, 0, undefined, [0, -1, 0]);
    else m.quad(r0, r1, e1, e0, MAT_POST, 0, undefined, [0, -1, 0]);
    // 軒先の小口
    if (s > 0) m.quad(e0, e1, eave1, eave0, MAT_PLANK, 0, undefined, [s, -0.3, 0]);
    else m.quad(eave1, e1, e0, eave0, MAT_PLANK, 0, undefined, [s, -0.3, 0]);
  }
  // 鰹木（棟に横たわる丸太 3 本）
  for (let i = 0; i < 3; i++) {
    const z = -rwz * 0.55 + (rwz * 1.1 * i) / 2;
    m.tube([-0.42 * scale, rh + 0.14 * scale, z], [0.42 * scale, rh + 0.14 * scale, z], 0.115 * scale, 0.115 * scale, 8, MAT_POST);
  }
  // 千木（棟の両端で X 字に交差する板）
  for (const sz of [1, -1]) {
    for (const sx of [1, -1]) {
      const base: V3 = [sx * 0.16 * scale, rh - 0.35 * scale, sz * (rwz - 0.12 * scale)];
      const tip: V3 = [sx * 0.62 * scale, rh + 1.05 * scale, sz * (rwz + 0.20 * scale)];
      m.tube(base, tip, 0.075 * scale, 0.055 * scale, 4, MAT_POST);
    }
  }
  // 正面（+Z 側。扉と同じ側）の階段。
  // 建物に近い段ほど「高い」。逆にすると床へ上がるのに下る形になる（実機の指摘）
  const steps = 5;
  for (let i = 0; i < steps; i++) {
    const y = (fl * (steps - i)) / steps;   // i=0（建物際）が床の高さ、i=4（手前）が最下段
    const z = hd + 0.45 + i * 0.30;
    m.box([0, y - fl / steps / 2, z], [W * 0.55, fl / steps, 0.34], MAT_POST);
  }
  void r;

  // 当たり判定: 社殿の身舎（階段と縁は含めない）
  m.solid([0, 1.6 * scale, 0], [4.2 * scale, 3.2 * scale, 3.4 * scale]);

  return m.built();
}

/** 石灯籠。基礎・竿・中台・火袋・笠・宝珠 */
export function buildLantern(seed: number, h: number): Built {
  const r = rng(seed);
  const m = new MeshBuilder();
  const s = h / 2.2;
  m.tube([0, 0, 0], [0, 0.26 * s, 0], 0.34 * s, 0.30 * s, 8, MAT_STONE);          // 基礎
  m.tube([0, 0.26 * s, 0], [0, 1.10 * s, 0], 0.14 * s, 0.125 * s, 8, MAT_STONE);  // 竿
  m.tube([0, 1.10 * s, 0], [0, 1.30 * s, 0], 0.30 * s, 0.26 * s, 8, MAT_STONE);   // 中台
  // 火袋（六角の箱。開口があるので柱と面で作る）
  for (let i = 0; i < 6; i++) {
    const a0 = (i / 6) * Math.PI * 2, a1 = ((i + 1) / 6) * Math.PI * 2;
    const rr = 0.24 * s;
    const p0: V3 = [Math.cos(a0) * rr, 1.30 * s, Math.sin(a0) * rr];
    const p1: V3 = [Math.cos(a1) * rr, 1.30 * s, Math.sin(a1) * rr];
    if (i % 2 === 0) {
      m.quad(p0, p1, [p1[0], 1.72 * s, p1[2]], [p0[0], 1.72 * s, p0[2]], MAT_STONE);
    } else {
      m.tube(p0, [p0[0], 1.72 * s, p0[2]], 0.045 * s, 0.045 * s, 5, MAT_STONE);
    }
  }
  m.tube([0, 1.72 * s, 0], [0, 1.98 * s, 0], 0.46 * s, 0.20 * s, 8, MAT_STONE);   // 笠
  m.tube([0, 1.98 * s, 0], [0, 2.20 * s, 0], 0.10 * s, 0.02 * s, 6, MAT_STONE);   // 宝珠
  void r;

  // 当たり判定: 竿から火袋まで
  m.solid([0, h / 2, 0], [0.52, h, 0.52]);

  return m.built();
}

/**
 * 石段。境内の縁の斜面を上がる。段の一枚ずつに厚みと不揃いがある。
 * 原点は下端の中心、+Z へ上がる。
 */
export function buildStairs(seed: number, steps: number, width: number, rise: number, run: number): Built {
  const r = rng(seed);
  const m = new MeshBuilder();
  for (let i = 0; i < steps; i++) {
    const y = i * rise;
    const z = i * run;
    const w = width * (0.94 + 0.12 * r());
    // 段は下へ深く伸ばす。地形は S 字に曲がるので、薄いと中間で埋もれる（実測）
    const t = rise * (2.2 + 0.6 * r());
    const skew = (r() - 0.5) * 0.06;
    m.box([skew, y + rise * 0.5 - t / 2 + rise * 0.5, z + run / 2], [w, t, run * (1.0 + 0.14 * r())], MAT_STONE);
    // 両脇の縁石
    for (const sx of [1, -1]) {
      m.box([sx * (w / 2 + 0.16), y + rise * 0.5 - t * 0.25, z + run / 2], [0.32 + 0.1 * r(), t * 1.5, run * 0.95], MAT_STONE);
    }
  }
  return m.built();
}

/**
 * 野面積みの石垣。自然石を不揃いに積む。
 * 原点は中央下端、X 方向に長さ len、高さ h。規則的な格子に見えないよう
 * 段ごとに石の幅・高さ・出入りを変え、目地に隙間を残す。
 */
export function buildStoneWall(seed: number, len: number, h: number, depth = 0.55): Built {
  const r = rng(seed);
  const m = new MeshBuilder();
  let y = 0;
  let course = 0;
  while (y < h - 0.05) {
    // 段の高さは 0.22〜0.46m でばらす
    const ch = Math.min(0.22 + 0.24 * r(), h - y);
    // 段ごとに開始位置をずらす（縦目地を通さない）
    let x = -len / 2 - r() * 0.4;
    while (x < len / 2) {
      const cw = 0.30 + 0.55 * r() * (1 + 0.3 * Math.sin(course * 1.7));
      const w = Math.min(cw, len / 2 - x);
      if (w < 0.12) break;
      // 石ごとに高さ・奥行き・傾きを変える。目地に隙間を残すため 0.03〜0.06 縮める
      const gap = 0.03 + 0.03 * r();
      const sh = ch * (0.82 + 0.16 * r());
      const sd = depth * (0.8 + 0.35 * r());
      const tilt = (r() - 0.5) * 0.10;
      // 石は上に行くほどわずかに内へ（勾配）
      const batter = 0.10 * (y / Math.max(h, 0.001));
      m.box([x + w / 2, y + sh / 2 + tilt * 0.05, -batter], [w - gap, sh - gap * 0.7, sd], MAT_STONE);
      x += w;
    }
    y += ch;
    course++;
  }

  // 当たり判定: 石垣の全長。メッシュと同じ len / h / depth から出す
  m.solid([0, h / 2, 0], [len, h, depth]);

  return m.built();
}
