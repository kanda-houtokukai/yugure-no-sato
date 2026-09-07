// 変形の場の更新（毎フレーム、固定刻み dt = 1/60）。
// 1) 窓に新しく入ったテクセルは 0 に戻す 2) 素材ごとの時定数で戻す 3) 踏み跡（スタンプ）を書き込む 4) 波紋を 1 歩進める
// 素材は焼いた材質テクスチャ（種別）から取る。

struct DeformParams {
  prevOrigin: vec2f,   // 前フレームの窓の原点（世界 xz）
  curOrigin: vec2f,    // 今の窓の原点
  texel: f32,
  size: f32,
  dt: f32,
  stampCount: f32,
};

struct Stamp {
  a: vec4f,   // x,z = 位置, z = 半径, w = 種類（0 = 足跡, 1 = 通り跡（倒すだけ））
  b: vec4f,   // x,y = 進行方向（xz）, z = 沈みの深さ [m], w = 倒れ量 0..1
};

@group(3) @binding(0) var updMat: texture_2d_array<f32>;
@group(3) @binding(1) var updDeformIn: texture_2d<f32>;
@group(3) @binding(2) var updRippleIn: texture_2d<f32>;
@group(3) @binding(3) var updDeformOut: texture_storage_2d<rgba32float, write>;
@group(3) @binding(4) var updRippleOut: texture_storage_2d<rgba32float, write>;
@group(3) @binding(5) var<storage, read> stamps: array<Stamp, 32>;
@group(3) @binding(6) var<uniform> dp: DeformParams;

/** 格納位置 (i, j) が今の窓で担う世界テクセル番号 */
fn worldTexel(ij: vec2i) -> vec2i {
  let n = i32(dp.size);
  let o = vec2i(floor(dp.curOrigin / dp.texel));
  return o + ((ij - o) % n + n) % n;
}

fn materialKind(p: vec2f) -> u32 {
  let li = heightLevelFor(p);
  let lv = hmLevels.l[li];
  let uv = (p - lv.xy) / (lv.z * lv.w);
  let m = textureSampleLevel(updMat, hmSamp, uv, li, 0.0);
  return u32(round(m.a * 16.0));
}

/** 素材ごとの戻り方: x = 沈みの時定数 [s], y = 倒れの時定数 [s], z = 沈める最大深さ [m] */
fn materialResponse(kind: u32) -> vec3f {
  switch (kind) {
    case 1u: { return vec3f(40.0, 25.0, 0.12); }    // 田の泥: 深く沈み、ゆっくり戻る
    case 3u: { return vec3f(240.0, 6.0, 0.02); }    // あぜ道の土: 浅く沈み、ほぼ戻らない
    case 4u: { return vec3f(20.0, 8.0, 0.06); }     // 川床
    case 2u: { return vec3f(15.0, 12.0, 0.01); }    // 畦の草: 沈まず、草が倒れて早く起きる
    default: { return vec3f(15.0, 12.0, 0.01); }    // 草地・丘
  }
}

fn readIn(tex: texture_2d<f32>, ij: vec2i) -> vec4f {
  let n = i32(dp.size);
  return textureLoad(tex, ((ij % n) + n) % n, 0);
}

@compute @workgroup_size(8, 8)
fn deformUpdate(@builtin(global_invocation_id) id: vec3u) {
  let n = i32(dp.size);
  if (i32(id.x) >= n || i32(id.y) >= n) { return; }
  let ij = vec2i(id.xy);
  let wt = worldTexel(ij);
  let p = (vec2f(wt) + 0.5) * dp.texel;

  // 1) 前の窓の外にあった（＝今入ってきた）テクセルは 0 から
  let po = vec2i(floor(dp.prevOrigin / dp.texel));
  let wasInside = wt.x >= po.x && wt.y >= po.y && wt.x < po.x + n && wt.y < po.y + n;
  var d = select(vec4f(0.0), readIn(updDeformIn, ij), wasInside);
  var r = select(vec4f(0.0), readIn(updRippleIn, ij), wasInside);

  let kind = materialKind(p);
  let resp = materialResponse(kind);

  // 2) 時間で戻る（指数減衰）
  d.r *= exp(-dp.dt / resp.x);
  let bendMag = length(d.gb);
  if (bendMag > 0.0) { d = vec4f(d.r, d.gb * exp(-dp.dt / resp.y), d.a); }
  d.a *= exp(-dp.dt / 30.0);   // 濁りは沈殿する

  // 3) 踏み跡
  let count = i32(dp.stampCount);
  var rippleImpulse = 0.0;
  for (var k = 0; k < count; k++) {
    let s = stamps[k];
    let dist = distance(p, s.a.xy);
    if (dist >= s.a.z) { continue; }
    let w = smootherstep(s.a.z, s.a.z * 0.35, dist);
    if (s.a.w < 0.5) {
      d.r = max(d.r, min(s.b.z, resp.z) * w);
      if (kind == 1u) { d.a = min(1.0, d.a + 0.9 * w); rippleImpulse += w; }
    }
    let goal = s.b.xy * s.b.w;   // target は WGSL の予約語
    let nb = mix(d.gb, goal, w);
    let m = length(nb);
    d = vec4f(d.r, select(nb, nb / m, m > 1.0), d.a);
  }

  // 4) 波紋: 波動方程式を 1 歩。田の泥の上だけ伝わり、外では強く減衰
  let h = r.r;
  let hp = r.g;
  let lap = readIn(updRippleIn, ij + vec2i(1, 0)).r + readIn(updRippleIn, ij - vec2i(1, 0)).r
          + readIn(updRippleIn, ij + vec2i(0, 1)).r + readIn(updRippleIn, ij - vec2i(0, 1)).r - 4.0 * h;
  let c = 1.2;   // 浅い水の波の速さ [m/s]。安定条件 (c·dt/texel)² < 0.25 を満たす
  let k2 = (c * dp.dt / dp.texel) * (c * dp.dt / dp.texel);
  var hn = 2.0 * h - hp + k2 * lap;
  hn *= select(0.85, 0.993, kind == 1u);
  hn -= rippleImpulse * 0.006;
  hn = clamp(hn, -0.05, 0.05);

  textureStore(updDeformOut, ij, d);
  textureStore(updRippleOut, ij, vec4f(hn, h, 0.0, 0.0));
}
