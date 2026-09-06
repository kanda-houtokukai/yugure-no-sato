// 焼いた高さテクスチャの参照（描画側）。group(1) に束ねる。

struct HeightLevels {
  // 各段: xy = origin, z = texel, w = size
  l: array<vec4f, 3>,
};

@group(1) @binding(0) var<uniform> hmLevels: HeightLevels;
@group(1) @binding(1) var hmTex: texture_2d_array<f32>;
@group(1) @binding(2) var hmSamp: sampler;

/** 点 p を含む最も細かい段。境界の 2 テクセル内側までを有効とする */
fn heightLevelFor(p: vec2f) -> i32 {
  for (var i = 0; i < 3; i++) {
    let lv = hmLevels.l[i];
    let extent = lv.z * lv.w;
    let q = p - lv.xy;
    let margin = lv.z * 2.0;
    if (q.x > margin && q.y > margin && q.x < extent - margin && q.y < extent - margin) { return i; }
  }
  return 2;
}

fn sampleHeight(levelIndex: i32, p: vec2f) -> f32 {
  let lv = hmLevels.l[levelIndex];
  let uv = (p - lv.xy) / (lv.z * lv.w);
  return textureSampleLevel(hmTex, hmSamp, uv, levelIndex, 0.0).r;
}

/**
 * 双三次（B スプライン）補間。双線形は勾配が区分定数になり、近景で横縞として見える。
 * 4 回の双線形サンプルで 16 テクセル分の B スプラインを作る定番の手法。
 */
fn sampleHeightCubic(levelIndex: i32, p: vec2f) -> f32 {
  let lv = hmLevels.l[levelIndex];
  let texel = lv.z;
  let n = lv.w;
  let coord = (p - lv.xy) / texel - 0.5;
  let tc = floor(coord);
  let f = coord - tc;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let s0 = w0 + w1;
  let s1 = w2 + w3;
  let t0 = tc - 1.0 + w1 / s0;
  let t1 = tc + 1.0 + w3 / s1;
  let uv00 = (vec2f(t0.x, t0.y) + 0.5) / n;
  let uv10 = (vec2f(t1.x, t0.y) + 0.5) / n;
  let uv01 = (vec2f(t0.x, t1.y) + 0.5) / n;
  let uv11 = (vec2f(t1.x, t1.y) + 0.5) / n;
  let a = textureSampleLevel(hmTex, hmSamp, uv00, levelIndex, 0.0).r;
  let b = textureSampleLevel(hmTex, hmSamp, uv10, levelIndex, 0.0).r;
  let c = textureSampleLevel(hmTex, hmSamp, uv01, levelIndex, 0.0).r;
  let d = textureSampleLevel(hmTex, hmSamp, uv11, levelIndex, 0.0).r;
  return mix(mix(a, b, s1.x / (s0.x + s1.x)), mix(c, d, s1.x / (s0.x + s1.x)), s1.y / (s0.y + s1.y));
}

/** 法線。最も細かい段（近景）は双三次、それ以外は双線形。差分の刻みは画素の足元とテクセルの大きい方 */
fn sampleNormal(p: vec2f, pixelSize: f32) -> vec3f {
  let li = heightLevelFor(p);
  let d = max(hmLevels.l[li].z, pixelSize * 0.75);
  var hl: f32; var hr: f32; var hd: f32; var hu: f32;
  if (li == 0) {
    hl = sampleHeightCubic(0, p - vec2f(d, 0.0));
    hr = sampleHeightCubic(0, p + vec2f(d, 0.0));
    hd = sampleHeightCubic(0, p - vec2f(0.0, d));
    hu = sampleHeightCubic(0, p + vec2f(0.0, d));
  } else {
    hl = sampleHeight(li, p - vec2f(d, 0.0));
    hr = sampleHeight(li, p + vec2f(d, 0.0));
    hd = sampleHeight(li, p - vec2f(0.0, d));
    hu = sampleHeight(li, p + vec2f(0.0, d));
  }
  return normalize(vec3f(hl - hr, 2.0 * d, hd - hu));
}
