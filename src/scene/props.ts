// 暮らしの跡。敷地の余白を埋める、背の低い作りもの。すべて数値から起こす（素材ファイルなし）。
// 建物と同じ頂点形式・同じインスタンス描画に乗せるので、材質の種別だけ足す。
//
// 追加した材質の種別（building.wgsl の kind）:
//   10 作物の葉  11 竹  12 藁・干し草  13 割った薪の木口  14 耕した土（畝）  15 農具の刃（鉄）  16 薪の樹皮

import { MAT_PLANK, MAT_POST, MAT_STONE } from './buildings';
import { MeshBuilder, rng, type V3 } from './mesh';

export const MAT_LEAF = 10;
export const MAT_BAMBOO = 11;
export const MAT_STRAW = 12;
export const MAT_SPLITWOOD = 13;
export const MAT_SOIL = 14;
export const MAT_METAL = 15;
export const MAT_LOG = 16;      // 薪の側面（樹皮）。柱の黒い木より明るい

/** 円板（筒の蓋）。薪の木口・桶の底に使う */
function disc(m: MeshBuilder, center: V3, normal: V3, r: number, sides: number, kind: number): void {
  // 法線が y 軸に近いかで基底を変える（tube と同じ作り）
  const up: V3 = Math.abs(normal[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  const ux = normal[1] * up[2] - normal[2] * up[1];
  const uy = normal[2] * up[0] - normal[0] * up[2];
  const uz = normal[0] * up[1] - normal[1] * up[0];
  const ul = Math.hypot(ux, uy, uz) || 1;
  const u: V3 = [ux / ul, uy / ul, uz / ul];
  const v: V3 = [
    normal[1] * u[2] - normal[2] * u[1],
    normal[2] * u[0] - normal[0] * u[2],
    normal[0] * u[1] - normal[1] * u[0],
  ];
  const pts: V3[] = [];
  for (let i = 0; i < sides; i++) {
    const a = (i / sides) * Math.PI * 2;
    const c = Math.cos(a) * r, s = Math.sin(a) * r;
    pts.push([center[0] + u[0] * c + v[0] * s, center[1] + u[1] * c + v[1] * s, center[2] + u[2] * c + v[2] * s]);
  }
  m.polygon(pts, kind, 0, normal);
}

/**
 * 角を崩した石。箱の 8 隅をばらして 6 面を張る。
 * 直方体のままだと「置いた箱」に見えるので、隅ごとに 3 軸へ散らす。
 */
function rock(m: MeshBuilder, center: V3, size: V3, seed: number, kind = MAT_STONE, jitter = 0.42): void {
  const r = rng(seed);
  const c: V3[] = [];
  for (let i = 0; i < 8; i++) {
    const sx = (i & 1) ? 1 : -1, sy = (i & 2) ? 1 : -1, sz = (i & 4) ? 1 : -1;
    c.push([
      center[0] + sx * size[0] * 0.5 * (1.0 - jitter / 2 + jitter * r()),
      center[1] + sy * size[1] * 0.5 * (1.0 - jitter / 2 + jitter * r()),
      center[2] + sz * size[2] * 0.5 * (1.0 - jitter / 2 + jitter * r()),
    ]);
  }
  const f = (a: number, b: number, d: number, e: number): void => m.quad(c[a], c[b], c[d], c[e], kind);
  f(0, 1, 3, 2);   // -Z
  f(5, 4, 6, 7);   // +Z
  f(4, 0, 2, 6);   // -X
  f(1, 5, 7, 3);   // +X
  f(2, 3, 7, 6);   // +Y
  f(4, 5, 1, 0);   // -Y
}

/**
 * 葉。中脈で二つに折った面。平らな一枚だと逆光で紙に見えるので折る。
 * dir = 葉の伸びる向き、up = 面の起き上がる向き。
 */
function leaf(m: MeshBuilder, base: V3, dir: V3, up: V3, len: number, wid: number, kind = MAT_LEAF): void {
  // 中脈に沿った 3 点（付け根・中ほど・先端）と、左右の張り出し
  const side: V3 = [
    dir[1] * up[2] - dir[2] * up[1],
    dir[2] * up[0] - dir[0] * up[2],
    dir[0] * up[1] - dir[1] * up[0],
  ];
  const sl = Math.hypot(side[0], side[1], side[2]) || 1;
  const s: V3 = [side[0] / sl, side[1] / sl, side[2] / sl];
  const at = (t: number, w: number, drop: number): V3 => [
    base[0] + dir[0] * len * t + s[0] * w + up[0] * drop,
    base[1] + dir[1] * len * t + s[1] * w + up[1] * drop,
    base[2] + dir[2] * len * t + s[2] * w + up[2] * drop,
  ];
  const mid = wid * 0.5;
  // 中脈は少し高く、縁は垂れる（＝折れ目が入り、両面で明るさが変わる）
  const b = at(0, 0, 0);
  const c1 = at(0.45, 0, 0.045 * len);
  const tip = at(1, 0, -0.02 * len);
  for (const k of [1, -1]) {
    const e1 = at(0.35, mid * k, -0.05 * len);
    const e2 = at(0.75, mid * 0.62 * k, -0.06 * len);
    m.polygon(k > 0 ? [b, e1, c1] : [b, c1, e1], kind);
    m.polygon(k > 0 ? [c1, e1, e2] : [c1, e2, e1], kind);
    m.polygon(k > 0 ? [c1, e2, tip] : [c1, tip, e2], kind);
  }
}

/** 1 株の作物。丈と葉の付き方だけ変えて、茄子・胡瓜・トマトを作り分ける */
function crop(m: MeshBuilder, x: number, z: number, kindOfCrop: number, r: () => number): void {
  const h = 0.46 + 0.32 * r() + kindOfCrop * 0.07;
  const lean = (r() - 0.5) * 0.16;
  m.tube([x, 0, z], [x + lean, h, z + lean * 0.6], 0.016, 0.010, 4, MAT_POST);
  const leaves = 9 + Math.floor(r() * 5);
  for (let i = 0; i < leaves; i++) {
    const t = 0.25 + 0.75 * (i / leaves);
    const a = r() * Math.PI * 2;
    const rise = 0.35 + 0.5 * r();
    const dir: V3 = [Math.cos(a), rise, Math.sin(a)];
    const dl = Math.hypot(dir[0], dir[1], dir[2]);
    const d: V3 = [dir[0] / dl, dir[1] / dl, dir[2] / dl];
    const base: V3 = [x + lean * t, h * t, z + lean * 0.6 * t];
    leaf(m, base, d, [0, 1, 0], 0.17 + 0.13 * r(), 0.115 + 0.075 * r());
  }
}

/**
 * 菜園。畝を立てて作物を並べ、支柱を立てて紐で結わえる。
 * rows 本の畝が Z 方向に並び、各畝は X 方向に len m 伸びる。
 */
export function buildVegetablePatch(seed: number, rows: number, len: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  const pitch = 0.75;
  const z0 = -((rows - 1) * pitch) / 2;
  for (let i = 0; i < rows; i++) {
    const z = z0 + i * pitch;
    const half = len / 2;
    // 畝: 台形の断面（底 0.58m・天 0.32m・高さ 0.17m）。手で立てたので稜が波打つ
    const hgt = 0.19 + 0.05 * r();
    const wobble = (x: number): number => 0.02 * Math.sin(x * 1.7 + i * 2.3) + 0.012 * Math.sin(x * 4.1 + i);
    const seg = Math.max(4, Math.round(len / 0.8));
    for (let k = 0; k < seg; k++) {
      const xa = -half + (len * k) / seg;
      const xb = -half + (len * (k + 1)) / seg;
      const ha = hgt + wobble(xa), hb = hgt + wobble(xb);
      // 両側の斜面と天端
      m.quad([xa, 0, z - 0.29], [xb, 0, z - 0.29], [xb, hb, z - 0.16], [xa, ha, z - 0.16], MAT_SOIL);
      m.quad([xa, ha, z - 0.16], [xb, hb, z - 0.16], [xb, hb, z + 0.16], [xa, ha, z + 0.16], MAT_SOIL);
      m.quad([xa, ha, z + 0.16], [xb, hb, z + 0.16], [xb, 0, z + 0.29], [xa, 0, z + 0.29], MAT_SOIL);
    }
    // 株。畝の上に等間隔＋ばらつき
    const n = Math.max(2, Math.round(len / 0.62));
    for (let k = 0; k < n; k++) {
      const x = -half + 0.3 + ((len - 0.6) * k) / Math.max(1, n - 1) + (r() - 0.5) * 0.12;
      crop(m, x, z + (r() - 0.5) * 0.1, i % 3, r);
    }
    // 支柱と紐（胡瓜・トマトの列だけ）
    if (i % 3 === 1) {
      const posts = Math.max(2, Math.round(len / 1.5));
      const tops: V3[] = [];
      for (let k = 0; k < posts; k++) {
        const x = -half + 0.25 + ((len - 0.5) * k) / Math.max(1, posts - 1);
        const th = 0.68 + 0.16 * r();
        const tilt = (r() - 0.5) * 0.05;
        m.tube([x, 0, z], [x + tilt, th, z + tilt], 0.020, 0.014, 4, MAT_BAMBOO);
        tops.push([x + tilt, th, z + tilt]);
      }
      // 結わえた紐: 支柱の上端と、その少し下を横に渡す
      for (let k = 0; k + 1 < tops.length; k++) {
        const a = tops[k], b = tops[k + 1];
        const sag = 0.035;
        const mid: V3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - sag, (a[2] + b[2]) / 2];
        m.tube(a, mid, 0.006, 0.006, 3, MAT_STRAW);
        m.tube(mid, b, 0.006, 0.006, 3, MAT_STRAW);
      }
    }
  }
  return m.toFloat32Array();
}

/**
 * 干し場。竹の脚を交叉させ、横竿を渡して藁束を掛ける。
 * 稲刈り前なので稲架ではなく、日常の物干し・藁干しの体。
 */
export function buildDryingRack(seed: number, len: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  const h = 1.32 + 0.12 * r();
  const half = len / 2;
  const spread = 0.42;
  for (const sx of [-1, 1]) {
    const x = sx * (half - 0.15);
    // 交叉した 2 本（X 字）。上で少し飛び出す
    m.tube([x, 0, -spread], [x, h + 0.16, spread], 0.030, 0.024, 5, MAT_BAMBOO);
    m.tube([x, 0, spread], [x, h + 0.16, -spread], 0.030, 0.024, 5, MAT_BAMBOO);
    // 結束（交点の縄）
    m.tube([x - 0.05, h * 0.94, -0.03], [x + 0.05, h * 0.94, 0.03], 0.032, 0.032, 5, MAT_STRAW);
  }
  // 横竿。わずかにたわむ
  const sag = 0.045;
  m.tube([-half - 0.2, h, 0], [0, h - sag, 0], 0.026, 0.026, 5, MAT_BAMBOO);
  m.tube([0, h - sag, 0], [half + 0.2, h, 0], 0.026, 0.026, 5, MAT_BAMBOO);
  // 掛けた藁束。竿をまたいで両側に垂れる
  const bundles = Math.max(3, Math.round(len / 0.55));
  for (let i = 0; i < bundles; i++) {
    const x = -half + 0.35 + ((len - 0.7) * i) / Math.max(1, bundles - 1) + (r() - 0.5) * 0.06;
    const y = h - sag * (1 - Math.abs(x) / half);
    const drop = 0.36 + 0.22 * r();
    const w = 0.10 + 0.05 * r();
    // 束は 1 本の筒だと板に見える。細い茎を数本ずつ扇に垂らす
    for (const sz of [-1, 1]) {
      const strands = 5;
      for (let k = 0; k < strands; k++) {
        const fx = ((k / (strands - 1)) - 0.5) * w * 1.7;
        const spreadZ = 0.06 + drop * (0.16 + 0.22 * r());
        const tip: V3 = [x + fx * 1.35 + (r() - 0.5) * 0.04, y - drop * (0.62 + 0.62 * r()), sz * spreadZ];
        m.tube([x + fx * 0.35, y - 0.01, sz * 0.03], tip, w * 0.30, w * 0.16, 4, MAT_STRAW);
      }
    }
  }
  return m.toFloat32Array();
}

/**
 * 薪の積み場。割った薪を木口を手前に向けて積む。
 * 端は崩れやすいので、段ごとに本数と高さをばらす。
 */
export function buildWoodpile(seed: number, wide: number, high: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  const logR = 0.062;
  const depth = 0.46;
  const rows = Math.max(2, Math.round(high / (logR * 2)));
  for (let row = 0; row < rows; row++) {
    const y = logR + row * logR * 1.92;
    // 上の段ほど短く（山なりに積む）
    // 上ほど短く、最上段は本数を欠いて崩す（壁のように整うと積み場に見えない）
    const shrink = 1.0 - 0.20 * (row / rows) * (row / rows) * 3.0;
    const w = Math.max(0.5, wide * Math.max(0.3, shrink));
    var n = Math.max(2, Math.floor(w / (logR * 2.05)));
    if (row === rows - 1) n = Math.max(1, n - 1 - Math.floor(r() * 2));
    const off = (r() - 0.5) * logR;
    for (let i = 0; i < n; i++) {
      const x = -w / 2 + logR + i * logR * 2.05 + off;
      const d = depth * (0.86 + 0.26 * r());
      const jz = (r() - 0.5) * 0.05;
      const jy = (r() - 0.5) * 0.012;
      const rr = logR * (0.82 + 0.30 * r());
      const sides = 6;
      m.tube([x, y + jy, -d / 2 + jz], [x, y + jy, d / 2 + jz], rr, rr, sides, MAT_LOG);
      // 木口（割った面）。どちらから見ても見えるよう両端に張る
      disc(m, [x, y + jy, d / 2 + jz], [0, 0, 1], rr, sides, MAT_SPLITWOOD);
      disc(m, [x, y + jy, -d / 2 + jz], [0, 0, -1], rr, sides, MAT_SPLITWOOD);
    }
  }
  return m.toFloat32Array();
}

/** 井戸。石の井筒に、四本柱と小さな切妻の覆い。釣瓶の桶を下げる */
export function buildWell(seed: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  const rad = 0.62;
  const curb = 0.52;
  // 井筒: 石を輪に積む。1 個ずつ独立した箱にすると必ず隙間が空くので、
  // 円環を扇形に割り、石ごとに半径を少しずらして目地の段差を作る（面は全部張るので穴は開かない）
  for (let ring = 0; ring < 2; ring++) {
    const n = 9;
    const y0 = (curb / 2) * ring;
    const y1 = y0 + curb / 2;
    for (let i = 0; i < n; i++) {
      const a0 = ((i + ring * 0.5) / n) * Math.PI * 2;
      const a1 = ((i + 1 + ring * 0.5) / n) * Math.PI * 2;
      const gap = 0.005;                       // 目地
      const push = (r() - 0.5) * 0.026;        // 石ごとの出入り
      const ro = rad + 0.13 + push, ri = rad - 0.13 + push * 0.4;
      const top = y1 - 0.012 * r();
      const pt = (a: number, rr: number, y: number): V3 => [Math.cos(a) * rr, y, Math.sin(a) * rr];
      const b0 = a0 + gap, b1 = a1 - gap;
      const oa = pt(b0, ro, y0), ob = pt(b1, ro, y0), oc = pt(b1, ro, top), od = pt(b0, ro, top);
      const ia = pt(b0, ri, y0), ib = pt(b1, ri, y0), ic = pt(b1, ri, top), id = pt(b0, ri, top);
      m.quad(oa, ob, oc, od, MAT_STONE);   // 外面
      m.quad(ib, ia, id, ic, MAT_STONE);   // 内面
      m.quad(od, oc, ic, id, MAT_STONE);   // 天端
      m.quad(ob, ib, ic, oc, MAT_STONE);   // 木口（隣との合わせ）
      m.quad(ia, oa, od, id, MAT_STONE);
    }
  }
  // 四本柱と桁
  const ph = 1.85;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    m.tube([sx * 0.66, curb, sz * 0.5], [sx * 0.62, ph, sz * 0.47], 0.045, 0.038, 5, MAT_POST);
  }
  for (const sz of [-1, 1]) m.tube([-0.66, ph, sz * 0.48], [0.66, ph, sz * 0.48], 0.036, 0.036, 4, MAT_POST);
  // 覆い（板葺きの小屋根）
  const ridge = ph + 0.42;
  for (const sz of [-1, 1]) {
    m.quad([-0.86, ridge, 0], [0.86, ridge, 0], [0.86, ph - 0.06, sz * 0.72], [-0.86, ph - 0.06, sz * 0.72], MAT_PLANK);
  }
  // 桁に渡した横木と、下がる縄・桶
  m.tube([-0.2, ph - 0.10, 0], [0.2, ph - 0.10, 0], 0.030, 0.030, 4, MAT_POST);
  m.tube([0.0, ph - 0.12, 0], [0.02, curb + 0.55, 0.02], 0.007, 0.007, 3, MAT_STRAW);
  // 桶
  const br = 0.15, bh = 0.22;
  m.tube([0.02, curb + 0.55, 0.02], [0.02, curb + 0.55 - bh, 0.02], br, br * 0.88, 8, MAT_PLANK);
  disc(m, [0.02, curb + 0.55 - bh, 0.02], [0, -1, 0], br * 0.88, 8, MAT_PLANK);
  return m.toFloat32Array();
}

/** 農具の立てかけ。鍬・熊手・箕を壁に立てかける（壁は別に建っている前提で、根元を少し前に出す） */
export function buildTools(seed: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  const mk = (x: number, len: number, head: 'hoe' | 'rake' | 'pole'): void => {
    const foot: V3 = [x, 0, 0.42 + 0.12 * r()];
    const top: V3 = [x + (r() - 0.5) * 0.1, len, 0.04];
    m.tube(foot, top, 0.021, 0.017, 4, MAT_POST);
    if (head === 'hoe') {
      // 柄の先に直角の刃
      m.box([foot[0] + 0.02, 0.10, foot[2] + 0.06], [0.19, 0.035, 0.15], MAT_METAL);
    } else if (head === 'rake') {
      const teeth = 5;
      m.tube([foot[0] - 0.16, 0.14, foot[2]], [foot[0] + 0.16, 0.14, foot[2]], 0.016, 0.016, 4, MAT_POST);
      for (let i = 0; i < teeth; i++) {
        const tx = foot[0] - 0.14 + (0.28 * i) / (teeth - 1);
        m.tube([tx, 0.14, foot[2]], [tx, 0.01, foot[2] + 0.05], 0.008, 0.005, 3, MAT_POST);
      }
    }
  };
  mk(-0.42, 1.42, 'hoe');
  mk(-0.06, 1.55, 'rake');
  mk(0.34, 1.30, 'pole');
  return m.toFloat32Array();
}

/** 桶と籠。数点まとめて 1 テンプレートにし、散らして置く */
export function buildVessels(seed: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  // 伏せた桶
  {
    const rad = 0.21, h = 0.26;
    m.tube([0, 0, 0], [0.01, h, 0.01], rad, rad * 0.94, 9, MAT_PLANK);
    disc(m, [0.01, h, 0.01], [0, 1, 0], rad * 0.94, 9, MAT_PLANK);
    // たが（竹の輪）
    for (const y of [h * 0.22, h * 0.8]) m.tube([0, y, 0], [0.004, y + 0.03, 0.004], rad * 1.03, rad * 1.03, 9, MAT_BAMBOO);
  }
  // 立てた籠（浅い）
  {
    const cx = 0.58 + 0.1 * r(), cz = -0.22;
    const rad = 0.26, h = 0.17;
    m.tube([cx, 0, cz], [cx, h, cz], rad * 0.78, rad, 9, MAT_BAMBOO);
    disc(m, [cx, 0.005, cz], [0, 1, 0], rad * 0.78, 9, MAT_BAMBOO);
  }
  // 立てかけた笊
  {
    const cx = -0.52, cz = 0.3;
    m.tube([cx, 0.02, cz], [cx - 0.10, 0.42, cz - 0.08], 0.30, 0.28, 10, MAT_BAMBOO);
  }
  return m.toFloat32Array();
}

/** 大きな石を数個。庭石・踏み石として敷地に散らす */
export function buildStones(seed: number): Float32Array {
  const r = rng(seed);
  const m = new MeshBuilder();
  rock(m, [0, 0.20, 0], [0.85, 0.48, 0.72], seed + 1);
  rock(m, [1.05, 0.11, 0.42], [0.52, 0.26, 0.46], seed + 2);
  rock(m, [-0.72, 0.09, -0.55], [0.44, 0.22, 0.40], seed + 3);
  if (r() > 0.4) rock(m, [0.35, 0.07, -0.95], [0.34, 0.17, 0.33], seed + 4);
  return m.toFloat32Array();
}
