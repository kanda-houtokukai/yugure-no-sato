// 整数ハッシュに基づくノイズ。sin ハッシュは GPU ごとに結果が変わるので使わない。
// 同じ入力には常に同じ値が返る（決定性）ことが、ゴールデンビューの再現性の土台。

fn hashU32(x0: u32) -> u32 {
  // lowbias32
  var x = x0;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn hash2u(p: vec2i, seed: u32) -> u32 {
  let a = bitcast<u32>(p.x) * 0x9E3779B1u;
  let b = bitcast<u32>(p.y) * 0x85EBCA77u;
  return hashU32(a ^ hashU32(b ^ (seed * 0xC2B2AE3Du)));
}

/** [0,1) */
fn hash2f(p: vec2i, seed: u32) -> f32 {
  return f32(hash2u(p, seed) >> 8u) / 16777216.0;
}

fn hash1f(i: i32, seed: u32) -> f32 {
  return hash2f(vec2i(i, 0x51), seed);
}

/** 勾配ベクトル。cos/sin は GPU で高価なので、ハッシュの上下 16bit から直接作る（長さは 0〜√2 で揺れるが地形には十分） */
fn grad2(p: vec2i, seed: u32) -> vec2f {
  let h = hash2u(p, seed);
  return vec2f(f32(h & 0xffffu), f32(h >> 16u)) * (2.0 / 65535.0) - vec2f(1.0);
}

/** 勾配ノイズ。おおむね [-1, 1] */
fn gnoise(p: vec2f, seed: u32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(grad2(i, seed), f);
  let b = dot(grad2(i + vec2i(1, 0), seed), f - vec2f(1.0, 0.0));
  let c = dot(grad2(i + vec2i(0, 1), seed), f - vec2f(0.0, 1.0));
  let d = dot(grad2(i + vec2i(1, 1), seed), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 1.4142;
}

/**
 * 多重ノイズ。minWavelength より短い波長のオクターブは捨てる。
 * 画素や頂点の足元の大きさを渡すと、遠方で細部が暴れる（エイリアス）のを抑えられる。
 */
fn fbm(p0: vec2f, octaves: i32, wavelength0: f32, minWavelength: f32, seed: u32) -> f32 {
  var p = p0;
  var amp = 1.0;
  var wl = wavelength0;
  var sum = 0.0;
  var norm = 0.0;
  // オクターブ間で回転して格子の向きを散らす
  let rot = mat2x2f(0.8, 0.6, -0.6, 0.8);
  for (var i = 0; i < octaves; i++) {
    if (wl < minWavelength) { break; }
    sum += amp * gnoise(p / wl, seed + u32(i) * 0x9E37u);
    norm += amp;
    amp *= 0.5;
    wl *= 0.5;
    p = rot * p;
  }
  return select(sum / norm, 0.0, norm == 0.0);
}

/** 尾根の立つノイズ（山並み用）。[0, 1] */
fn ridged(p0: vec2f, octaves: i32, wavelength0: f32, minWavelength: f32, seed: u32) -> f32 {
  var p = p0;
  var amp = 1.0;
  var wl = wavelength0;
  var sum = 0.0;
  var norm = 0.0;
  var weight = 1.0;
  let rot = mat2x2f(0.8, 0.6, -0.6, 0.8);
  for (var i = 0; i < octaves; i++) {
    if (wl < minWavelength) { break; }
    var n = 1.0 - abs(gnoise(p / wl, seed + u32(i) * 0x7F4Au));
    n = n * n * weight;
    weight = clamp(n * 1.6, 0.0, 1.0);
    sum += amp * n;
    norm += amp;
    amp *= 0.5;
    wl *= 0.5;
    p = rot * p;
  }
  return select(sum / norm, 0.0, norm == 0.0);
}

fn smootherstep(e0: f32, e1: f32, x: f32) -> f32 {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
