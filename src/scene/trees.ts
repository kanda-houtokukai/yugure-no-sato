// 木のテンプレートメッシュを数値から起こす（パラメトリック生成）。素材ファイルは使わない。
// 種類は 2 つ: 神社の杉（真っ直ぐ高い針葉樹）と、里の広葉樹（丸い樹形）。
// 頂点: 位置 3, 法線 3, uv 2, 種別 1（0 = 幹枝, 1 = 杉の葉, 2 = 広葉樹の葉）, 揺れ重み 1 = 10 floats

export const TREE_VERTEX_FLOATS = 10;

export interface CedarParams {
  height: number;        // 幹の高さ [m]
  baseRadius: number;    // 根元の半径 [m]
  crownStart: number;    // 樹冠が始まる高さの比（0..1）
  crownWidth: number;    // 樹冠の最大半径 [m]
  whorlSpacing: number;  // 枝輪の間隔 [m]
  branchesPerWhorl: number;
}

export interface BroadleafParams {
  trunkHeight: number;
  baseRadius: number;
  branchAngleDeg: number;   // 幹からの分岐角
  branchesLevel1: number;
  branchesLevel2: number;
  branchLength: number;     // 一段目の枝の長さ [m]
  leafCluster: number;      // 葉塊の大きさ [m]
}

class MeshBuilder {
  data: number[] = [];
  push(p: [number, number, number], n: [number, number, number], uv: [number, number], kind: number, sway: number): void {
    this.data.push(p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1], kind, sway);
  }
  /** 四角形（2 三角形）。両面で描くので向きは気にしない */
  quad(a: V3, b: V3, c: V3, d: V3, n: V3, kind: number, sway: [number, number, number, number]): void {
    this.push(a, n, [0, 0], kind, sway[0]);
    this.push(b, n, [1, 0], kind, sway[1]);
    this.push(c, n, [1, 1], kind, sway[2]);
    this.push(a, n, [0, 0], kind, sway[0]);
    this.push(c, n, [1, 1], kind, sway[2]);
    this.push(d, n, [0, 1], kind, sway[3]);
  }
  /** 先細りの角柱（幹・枝）。from → to、半径 r0 → r1、sides 面 */
  tube(from: V3, to: V3, r0: number, r1: number, sides: number, swayFrom: number, swayTo: number): void {
    const axis = norm(sub(to, from));
    const ref: V3 = Math.abs(axis[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = norm(cross(axis, ref));
    const v = cross(axis, u);
    for (let i = 0; i < sides; i++) {
      const a0 = (i / sides) * Math.PI * 2;
      const a1 = ((i + 1) / sides) * Math.PI * 2;
      const d0: V3 = add(scale(u, Math.cos(a0)), scale(v, Math.sin(a0)));
      const d1: V3 = add(scale(u, Math.cos(a1)), scale(v, Math.sin(a1)));
      const p0 = add(from, scale(d0, r0));
      const p1 = add(from, scale(d1, r0));
      const p2 = add(to, scale(d1, r1));
      const p3 = add(to, scale(d0, r1));
      const n = norm(add(d0, d1));
      this.push(p0, n, [i / sides, 0], 0, swayFrom);
      this.push(p1, n, [(i + 1) / sides, 0], 0, swayFrom);
      this.push(p2, n, [(i + 1) / sides, 1], 0, swayTo);
      this.push(p0, n, [i / sides, 0], 0, swayFrom);
      this.push(p2, n, [(i + 1) / sides, 1], 0, swayTo);
      this.push(p3, n, [i / sides, 1], 0, swayTo);
    }
  }
}

type V3 = [number, number, number];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** 決定的な乱数（種から）。ライブラリを使わないので自前の LCG */
function rng(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** 杉: 真っ直ぐな幹 ＋ 枝輪ごとに外へ張り出す葉の板（上へ行くほど短く＝円錐の樹冠） */
export function buildCedar(seed: number, p: CedarParams): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  // 幹（10 段で先細り）
  const segs = 10;
  for (let i = 0; i < segs; i++) {
    const h0 = (i / segs) * p.height;
    const h1 = ((i + 1) / segs) * p.height;
    const rr = (h: number) => p.baseRadius * Math.pow(1 - h / p.height, 0.8) + 0.02;
    m.tube([0, h0, 0], [0, h1, 0], rr(h0), rr(h1), 8, 0, 0);
  }
  // 枝輪
  const crownBase = p.crownStart * p.height;
  for (let h = crownBase; h < p.height * 0.97; h += p.whorlSpacing) {
    const f = (h - crownBase) / (p.height - crownBase);           // 樹冠の下端 0 → 先端 1
    const len = p.crownWidth * (1 - f * 0.9) * (0.85 + 0.3 * r());
    const n = p.branchesPerWhorl;
    const phase = r() * Math.PI * 2;
    for (let k = 0; k < n; k++) {
      const a = phase + (k / n) * Math.PI * 2 + (r() - 0.5) * 0.5;
      const dir: V3 = [Math.cos(a), -0.18 - 0.1 * r(), Math.sin(a)];   // やや下がり気味に張り出す
      const d = norm(dir);
      const base: V3 = [0, h, 0];
      const tip = add(base, scale(d, len));
      const side = norm(cross(d, [0, 1, 0]));
      const w = len * 0.42;
      const sway = 0.3 + 0.7 * f;
      // 水平寄りの板と、それに直交する板の 2 枚で厚みを出す
      m.quad(add(base, scale(side, -w * 0.3)), add(base, scale(side, w * 0.3)), add(tip, scale(side, w)), add(tip, scale(side, -w)), [0, 1, 0], 1, [sway * 0.5, sway * 0.5, sway, sway]);
      const up: V3 = [0, 1, 0];
      m.quad(add(base, scale(up, -w * 0.15)), add(base, scale(up, w * 0.15)), add(tip, scale(up, w * 0.7)), add(tip, scale(up, -w * 0.7)), side, 1, [sway * 0.5, sway * 0.5, sway, sway]);
    }
  }
  return new Float32Array(m.data);
}

/** 広葉樹: 幹 → 分岐（2 段）→ 枝先に丸い葉塊（交差する 3 枚の板） */
export function buildBroadleaf(seed: number, p: BroadleafParams): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  const trunkTop: V3 = [0, p.trunkHeight, 0];
  m.tube([0, 0, 0], trunkTop, p.baseRadius, p.baseRadius * 0.6, 8, 0, 0.05);

  const leafCluster = (c: V3, size: number, sway: number): void => {
    // 交差する 3 枚。塊ごとに向きを乱して、板が揃って平らに見えるのを防ぐ
    const s = size * (0.8 + 0.4 * r());
    const rot = r() * Math.PI;
    const tilt = (r() - 0.5) * 0.6;
    const ax = (a: number): V3 => norm([Math.cos(a), tilt * Math.sin(a * 2.0), Math.sin(a)]);
    const planes: [V3, V3, V3][] = [0, 1, 2].map((k) => {
      const u = ax(rot + (k * Math.PI) / 3);
      const n = norm(cross(u, [0, 1, 0]));
      const v = norm(cross(n, u));
      return [u, v, n] as [V3, V3, V3];
    });
    for (const [u, v, n] of planes) {
      const a = add(c, add(scale(u, -s), scale(v, -s)));
      const b = add(c, add(scale(u, s), scale(v, -s)));
      const cc = add(c, add(scale(u, s), scale(v, s)));
      const d = add(c, add(scale(u, -s), scale(v, s)));
      m.quad(a, b, cc, d, n, 2, [sway, sway, sway, sway]);
    }
  };

  const ang1 = (p.branchAngleDeg * Math.PI) / 180;
  const phase = r() * Math.PI * 2;
  for (let i = 0; i < p.branchesLevel1; i++) {
    const az = phase + (i / p.branchesLevel1) * Math.PI * 2 + (r() - 0.5) * 0.6;
    const d1: V3 = norm([Math.sin(ang1) * Math.cos(az), Math.cos(ang1), Math.sin(ang1) * Math.sin(az)]);
    const len1 = p.branchLength * (0.8 + 0.4 * r());
    const tip1 = add(trunkTop, scale(d1, len1));
    m.tube(trunkTop, tip1, p.baseRadius * 0.5, p.baseRadius * 0.28, 6, 0.05, 0.25);
    for (let j = 0; j < p.branchesLevel2; j++) {
      const az2 = az + (j - (p.branchesLevel2 - 1) / 2) * 0.9 + (r() - 0.5) * 0.5;
      const el2 = ang1 * 0.6 + (r() - 0.5) * 0.5;
      const d2: V3 = norm([Math.sin(el2) * Math.cos(az2), Math.cos(el2), Math.sin(el2) * Math.sin(az2)]);
      const len2 = len1 * 0.65;
      const tip2 = add(tip1, scale(d2, len2));
      m.tube(tip1, tip2, p.baseRadius * 0.26, 0.04, 5, 0.25, 0.6);
      leafCluster(tip2, p.leafCluster, 0.8);
      leafCluster(add(tip1, scale(d2, len2 * 0.45)), p.leafCluster * 0.8, 0.6);
    }
    leafCluster(tip1, p.leafCluster * 0.9, 0.5);
  }
  return new Float32Array(m.data);
}

export interface TreeVariant { name: string; species: 0 | 1; vertices: Float32Array }

/** 各種 2 変種のテンプレート */
export function buildTreeVariants(): TreeVariant[] {
  return [
    { name: 'cedar-a', species: 0, vertices: buildCedar(11, { height: 22, baseRadius: 0.45, crownStart: 0.22, crownWidth: 3.2, whorlSpacing: 0.75, branchesPerWhorl: 8 }) },
    { name: 'cedar-b', species: 0, vertices: buildCedar(23, { height: 18, baseRadius: 0.38, crownStart: 0.18, crownWidth: 2.9, whorlSpacing: 0.7, branchesPerWhorl: 7 }) },
    { name: 'broadleaf-a', species: 1, vertices: buildBroadleaf(31, { trunkHeight: 3.2, baseRadius: 0.22, branchAngleDeg: 42, branchesLevel1: 4, branchesLevel2: 3, branchLength: 2.6, leafCluster: 1.5 }) },
    { name: 'broadleaf-b', species: 1, vertices: buildBroadleaf(47, { trunkHeight: 2.6, baseRadius: 0.19, branchAngleDeg: 50, branchesLevel1: 3, branchesLevel2: 3, branchLength: 2.2, leafCluster: 1.35 }) },
  ];
}

/** 配置の指定（谷座標 (u,v) または 線への吸着）。世界座標への変換は正本への問い合わせで行う */
export interface TreePlacement {
  variant: number;
  u: number;
  v: number;
  scale: number;
  rotation: number;
  /** 縦線への吸着（参道の並木用）: index と横ずれ */
  snap?: { index: number; offset: number };
}

export function planTrees(): TreePlacement[] {
  const r = rng(7);
  const out: TreePlacement[] = [];
  // 参道の杉並木: 主道（縦線 0）の延長、v = 192〜256、両脇 ±3.6m、6.5m 間隔
  for (let v = 192; v <= 256; v += 6.5) {
    for (const side of [-1, 1]) {
      out.push({ variant: r() < 0.5 ? 0 : 1, u: 0, v: v + (r() - 0.5) * 0.8, scale: 0.9 + 0.3 * r(), rotation: r() * Math.PI * 2, snap: { index: 0, offset: side * 3.6 } });
    }
  }
  // 丘の上の杉の杜（参道の線から 6m 以上離す）
  for (let i = 0; i < 30; i++) {
    const a = r() * Math.PI * 2;
    const rad = Math.sqrt(r());
    const u = Math.cos(a) * 34 * rad;
    const v = 276 + Math.sin(a) * 22 * rad;
    if (Math.abs(u) < 6 && v < 262) continue;
    out.push({ variant: r() < 0.5 ? 0 : 1, u, v, scale: 0.8 + 0.4 * r(), rotation: r() * Math.PI * 2 });
  }
  // 里の広葉樹: 集落の敷地（南）
  for (let i = 0; i < 12; i++) {
    out.push({ variant: 2 + (r() < 0.5 ? 0 : 1), u: -85 + 170 * r(), v: -186 + 40 * r(), scale: 0.85 + 0.45 * r(), rotation: r() * Math.PI * 2 });
  }
  // 丘の麓（北東）
  for (let i = 0; i < 10; i++) {
    out.push({ variant: 2 + (r() < 0.5 ? 0 : 1), u: 70 + 130 * r(), v: 182 + 45 * r(), scale: 0.8 + 0.5 * r(), rotation: r() * Math.PI * 2 });
  }
  // 畦の脇に点在（横線 4 の副道の北側、横線 -3 の副道の南側）
  for (let i = 0; i < 6; i++) out.push({ variant: 2, u: -190 + 60 * i + 20 * r(), v: 63 + 2.5, scale: 0.7 + 0.4 * r(), rotation: r() * Math.PI * 2 });
  for (let i = 0; i < 5; i++) out.push({ variant: 3, u: -120 + 70 * i + 20 * r(), v: -60 - 2.8, scale: 0.7 + 0.4 * r(), rotation: r() * Math.PI * 2 });
  return out;
}
