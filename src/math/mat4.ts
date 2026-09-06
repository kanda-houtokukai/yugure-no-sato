// 4x4 行列（列優先 = WGSL の mat4x4f と同じ並び）。必要な操作だけ。

import { cross, normalize, sub, type Vec3 } from './vec';

export type Mat4 = Float32Array;

export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** a × b（列優先） */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = s;
    }
  }
  return out;
}

/**
 * 逆 Z・無限遠の透視投影。近いほど深度が 1 に、無限遠が 0 になる。
 * 深度バッファは depth32float、比較は greater。16km 先まで精度が保てる。
 */
export function perspectiveReversedInfinite(fovYRad: number, aspect: number, near: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = 0;
  m[11] = -1;
  m[14] = near;
  return m;
}

/** カメラ相対の視線変換（平行移動なし）。eye は原点にあるものとして forward の向きだけ使う */
export function viewRotation(forward: Vec3, up: Vec3): Mat4 {
  const f = normalize(forward);
  const r = normalize(cross(f, up));
  const u = cross(r, f);
  const m = new Float32Array(16);
  // 行 = 基底（世界 → 視点）。視点空間では -z が前
  m[0] = r[0]; m[4] = r[1]; m[8] = r[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[15] = 1;
  return m;
}

export function lookBasis(eye: Vec3, target: Vec3): { forward: Vec3; right: Vec3; up: Vec3 } {
  const forward = normalize(sub(target, eye));
  const worldUp: Vec3 = Math.abs(forward[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const right = normalize(cross(forward, worldUp));
  const up = cross(right, forward);
  return { forward, right, up };
}
