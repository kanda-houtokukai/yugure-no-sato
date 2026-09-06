// 空の LUT（正距円筒）。毎フレーム compute で焼き、遠景の環境光と水面の映り込みが参照する。
// group(2) に束ねる。

@group(2) @binding(0) var skyLut: texture_2d<f32>;
@group(2) @binding(1) var skySamp: sampler;

fn skyLutUv(dir: vec3f) -> vec2f {
  let u = atan2(dir.x, dir.z) / TAU + 0.5;   // 方位（北 = 0.5）
  let v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;   // 天頂 = 0、地平線 = 0.5
  return vec2f(u, v);
}

/** LUT から空の放射輝度（太陽の円盤は含まない） */
fn skyRadianceLut(dir: vec3f) -> vec3f {
  return textureSampleLevel(skyLut, skySamp, skyLutUv(dir), 0.0).rgb;
}

@group(2) @binding(2) var<storage, read> skyIrr: array<vec4f, 3>;

/** 法線まわりの空の環境光。上向き面の放射照度を基準に、傾きで落とす（地面からの照り返しは未考慮） */
fn skyAmbient(n: vec3f) -> vec3f {
  let eUp = skyIrr[0].rgb;
  let up = clamp(n.y, -1.0, 1.0);
  return eUp / PI * (0.5 + 0.5 * up);
}
