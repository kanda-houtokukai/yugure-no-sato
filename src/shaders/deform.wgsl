// 変形の場。地形の数式（正本）は変えず、その上に加算される変位として持つ。
// カメラ周りの 1024² × 0.1m（102.4m 四方）をトーラス状（wrap）に持つ: 格納位置は世界テクセル番号の mod なので、
// カメラが動いてもコピーは要らない。窓に新しく入ったテクセルだけ 0 に戻す。窓の外に出た変形は失われる。
//
// deformTex: r = 沈み [m]（正 = 深い）, g,b = 倒れ方向（xz、長さ = 倒れ量 0..1）, a = 濁り 0..1
// rippleTex: r = 波紋の高さ [m], g = 前フレームの高さ（波動方程式用）
// 全消費者（地形の頂点・法線、水面、稲、草）は deformAt(p) の 1 タップで同じ場を見る。

@group(2) @binding(8) var deformTex: texture_2d<f32>;
@group(2) @binding(9) var rippleTex: texture_2d<f32>;
@group(2) @binding(10) var deformSamp: sampler;   // repeat・線形。wrap 格納と相性がよい

struct Deform {
  sink: f32,
  bend: vec2f,
  turbidity: f32,
  ripple: f32,
  inside: bool,
};

/** 点 p が現在の窓の中か */
fn deformInside(p: vec2f) -> bool {
  let o = frame.deform.xy;
  let extent = frame.deform.z * frame.deform.w;
  let q = p - o;
  let m = frame.deform.z;   // 縁 1 テクセルは補間で隣（反対側）を拾うので除く
  return q.x > m && q.y > m && q.x < extent - m && q.y < extent - m;
}

fn deformAt(p: vec2f) -> Deform {
  var d: Deform;
  d.sink = 0.0; d.bend = vec2f(0.0); d.turbidity = 0.0; d.ripple = 0.0; d.inside = false;
  if (!deformInside(p)) { return d; }
  // wrap 格納なので、サンプル座標は世界座標を窓の大きさで割った端数でよい（原点は引かない）
  let uv = p / (frame.deform.z * frame.deform.w);
  let t = textureSampleLevel(deformTex, deformSamp, uv, 0.0);
  let r = textureSampleLevel(rippleTex, deformSamp, uv, 0.0);
  d.sink = t.r;
  d.bend = t.gb;
  d.turbidity = t.a;
  d.ripple = r.r;
  d.inside = true;
  return d;
}

/** 沈みの勾配（法線の補正用）。テクセル間隔で差分 */
fn deformSinkGradient(p: vec2f) -> vec2f {
  if (!deformInside(p)) { return vec2f(0.0); }
  let e = frame.deform.z;
  let sx0 = deformAt(p - vec2f(e, 0.0)).sink;
  let sx1 = deformAt(p + vec2f(e, 0.0)).sink;
  let sz0 = deformAt(p - vec2f(0.0, e)).sink;
  let sz1 = deformAt(p + vec2f(0.0, e)).sink;
  return vec2f(sx1 - sx0, sz1 - sz0) / (2.0 * e);
}

/** 波紋の勾配（水面の法線用） */
fn rippleGradient(p: vec2f) -> vec2f {
  if (!deformInside(p)) { return vec2f(0.0); }
  let e = frame.deform.z;
  let n = frame.deform.z * frame.deform.w;
  let x0 = textureSampleLevel(rippleTex, deformSamp, (p - vec2f(e, 0.0)) / n, 0.0).r;
  let x1 = textureSampleLevel(rippleTex, deformSamp, (p + vec2f(e, 0.0)) / n, 0.0).r;
  let z0 = textureSampleLevel(rippleTex, deformSamp, (p - vec2f(0.0, e)) / n, 0.0).r;
  let z1 = textureSampleLevel(rippleTex, deformSamp, (p + vec2f(0.0, e)) / n, 0.0).r;
  return vec2f(x1 - x0, z1 - z0) / (2.0 * e);
}
