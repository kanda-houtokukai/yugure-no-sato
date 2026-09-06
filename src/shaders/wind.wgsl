// 風の場。稲・草・木・遠景の田の面が同じ場を参照する（株ごとにバラバラに揺らさない）。
// 面として吹き、田の上を波のように渡っていく: 大きなうねり（波長 25m）＋細かい波（6m）＋移動する突風のむら。
// t は frame.params.x（固定なら絵も固定 = 再現性）。

const WIND_DIR: vec2f = vec2f(0.9165, 0.4);   // 風下の向き（xz）。おおむね東北東へ
const WIND_SPEED: f32 = 5.5;                  // うねりが渡る速さ [m/s]

struct Wind { bend: vec2f, strength: f32 };

fn windField(p: vec2f, t: f32) -> Wind {
  let along = dot(p, WIND_DIR);
  let across = dot(p, vec2f(-WIND_DIR.y, WIND_DIR.x));
  // うねりの位相を低周波ノイズで乱し、まっすぐな縞にならないようにする
  let warp = 2.2 * gnoise(vec2f(along, across) / 45.0 + vec2f(t * 0.15, 0.0), 101u);
  let big = 0.5 + 0.5 * sin((along - t * WIND_SPEED) * (TAU / 25.0) + warp);
  let small = 0.5 + 0.5 * sin((along - t * WIND_SPEED * 0.8) * (TAU / 6.0) + across * 0.35);
  // 風下へ流れる突風のむら
  let gust = 0.5 + 0.5 * gnoise((p - WIND_DIR * t * WIND_SPEED) / 22.0, 102u);
  let strength = clamp(0.12 + 0.88 * gust * (0.65 * big + 0.35 * small), 0.0, 1.0);
  var w: Wind;
  w.bend = WIND_DIR * strength;
  w.strength = strength;
  return w;
}

// ---- 焼いた風の場（毎フレーム compute で 512² × 1.2m = 614m 四方をカメラ周りに焼く） ----
// 稲の頂点（数百万回）・遠景の田の面（数百万画素）から数式を直接呼ぶと重いので、1 タップにする。
// group(2) binding 6/7。origin/texel は frame.wind から。
@group(2) @binding(6) var windTex: texture_2d<f32>;
@group(2) @binding(7) var windSamp: sampler;

fn windAt(p: vec2f) -> Wind {
  let uv = (p - frame.wind.xy) / (frame.wind.z * frame.wind.w);
  let t = textureSampleLevel(windTex, windSamp, uv, 0.0);
  var w: Wind;
  w.bend = t.xy;
  w.strength = t.z;
  return w;
}
