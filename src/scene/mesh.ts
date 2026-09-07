// パラメトリック生成の土台。木・建物・石垣が共有する。素材ファイルは使わない。
// 頂点: 位置 3, 法線 3, uv 2, 種別 1, 補助 1 = 10 floats
// （木では補助 = 揺れ重み、建物では 0）

export const VERTEX_FLOATS = 10;

export type V3 = [number, number, number];

export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** 決定的な乱数（種から）。ライブラリを使わないので自前の LCG */
export function rng(seed: number): () => number {
  let s = (seed * 2654435761) >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export class MeshBuilder {
  data: number[] = [];

  push(p: V3, n: V3, uv: [number, number], kind: number, aux: number): void {
    this.data.push(p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1], kind, aux);
  }

  /** 四角形（2 三角形）。uv は m 単位で渡すと、シェーダ側で実寸の模様が作れる */
  quad(a: V3, b: V3, c: V3, d: V3, kind: number, aux = 0, uvA?: [number, number][], n?: V3): void {
    const nn = n ?? norm(cross(sub(b, a), sub(d, a)));
    const uv = uvA ?? [
      [0, 0],
      [Math.hypot(...(sub(b, a) as [number, number, number])), 0],
      [Math.hypot(...(sub(b, a) as [number, number, number])), Math.hypot(...(sub(d, a) as [number, number, number]))],
      [0, Math.hypot(...(sub(d, a) as [number, number, number]))],
    ];
    this.push(a, nn, uv[0], kind, aux);
    this.push(b, nn, uv[1], kind, aux);
    this.push(c, nn, uv[2], kind, aux);
    this.push(a, nn, uv[0], kind, aux);
    this.push(c, nn, uv[2], kind, aux);
    this.push(d, nn, uv[3], kind, aux);
  }

  /** 軸に沿った直方体。center は中心、size は各辺の長さ */
  box(center: V3, size: V3, kind: number, aux = 0): void {
    const [cx, cy, cz] = center;
    const [sx, sy, sz] = size;
    const hx = sx / 2, hy = sy / 2, hz = sz / 2;
    const p = (dx: number, dy: number, dz: number): V3 => [cx + dx * hx, cy + dy * hy, cz + dz * hz];
    // 面ごとに uv を実寸で
    this.quad(p(-1, -1, 1), p(1, -1, 1), p(1, 1, 1), p(-1, 1, 1), kind, aux, [[0, 0], [sx, 0], [sx, sy], [0, sy]], [0, 0, 1]);   // +Z
    this.quad(p(1, -1, -1), p(-1, -1, -1), p(-1, 1, -1), p(1, 1, -1), kind, aux, [[0, 0], [sx, 0], [sx, sy], [0, sy]], [0, 0, -1]); // -Z
    this.quad(p(1, -1, 1), p(1, -1, -1), p(1, 1, -1), p(1, 1, 1), kind, aux, [[0, 0], [sz, 0], [sz, sy], [0, sy]], [1, 0, 0]);    // +X
    this.quad(p(-1, -1, -1), p(-1, -1, 1), p(-1, 1, 1), p(-1, 1, -1), kind, aux, [[0, 0], [sz, 0], [sz, sy], [0, sy]], [-1, 0, 0]); // -X
    this.quad(p(-1, 1, 1), p(1, 1, 1), p(1, 1, -1), p(-1, 1, -1), kind, aux, [[0, 0], [sx, 0], [sx, sz], [0, sz]], [0, 1, 0]);    // +Y
    this.quad(p(-1, -1, -1), p(1, -1, -1), p(1, -1, 1), p(-1, -1, 1), kind, aux, [[0, 0], [sx, 0], [sx, sz], [0, sz]], [0, -1, 0]); // -Y
  }

  /** 先細りの角柱（幹・枝）。from → to、半径 r0 → r1、sides 面 */
  tube(from: V3, to: V3, r0: number, r1: number, sides: number, kind: number, auxFrom = 0, auxTo = 0): void {
    const axis = norm(sub(to, from));
    const ref: V3 = Math.abs(axis[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u = norm(cross(axis, ref));
    const v = cross(axis, u);
    for (let i = 0; i < sides; i++) {
      const a0 = (i / sides) * Math.PI * 2;
      const a1 = ((i + 1) / sides) * Math.PI * 2;
      const d0 = add(scale(u, Math.cos(a0)), scale(v, Math.sin(a0)));
      const d1 = add(scale(u, Math.cos(a1)), scale(v, Math.sin(a1)));
      const p0 = add(from, scale(d0, r0));
      const p1 = add(from, scale(d1, r0));
      const p2 = add(to, scale(d1, r1));
      const p3 = add(to, scale(d0, r1));
      const n = norm(add(d0, d1));
      this.push(p0, n, [i / sides, 0], kind, auxFrom);
      this.push(p1, n, [(i + 1) / sides, 0], kind, auxFrom);
      this.push(p2, n, [(i + 1) / sides, 1], kind, auxTo);
      this.push(p0, n, [i / sides, 0], kind, auxFrom);
      this.push(p2, n, [(i + 1) / sides, 1], kind, auxTo);
      this.push(p3, n, [i / sides, 1], kind, auxTo);
    }
  }

  /** 任意の凸多角形（面）。法線は自動 */
  polygon(pts: V3[], kind: number, aux = 0, n?: V3): void {
    if (pts.length < 3) return;
    const nn = n ?? norm(cross(sub(pts[1], pts[0]), sub(pts[2], pts[0])));
    for (let i = 1; i + 1 < pts.length; i++) {
      this.push(pts[0], nn, [0, 0], kind, aux);
      this.push(pts[i], nn, [1, 0], kind, aux);
      this.push(pts[i + 1], nn, [1, 1], kind, aux);
    }
  }

  toFloat32Array(): Float32Array {
    return new Float32Array(this.data);
  }
}
