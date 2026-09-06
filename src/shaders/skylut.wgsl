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

// ---- 遠景の溶け込み（大気散乱）の 3D LUT: (方位, 仰角, 距離) → 内散乱・透過率 ----
// 視線ごとの積分（DPR2 で 10.7ms）を起動時の焼き込み 1 回に置き換える。カメラ位置ごとに焼く。
@group(2) @binding(3) var aerialIn: texture_3d<f32>;
@group(2) @binding(4) var aerialTr: texture_3d<f32>;
@group(2) @binding(5) var<storage, read> sunLut: array<vec4f, 128>;

const AERIAL_EL_MIN: f32 = -0.6;     // 仰角の範囲 [rad]（地形は下向き、空は上向きに少し）
const AERIAL_EL_MAX: f32 = 0.6;
const AERIAL_D_MIN: f32 = 1.0;       // 距離 [m]（対数刻み）
const AERIAL_D_MAX: f32 = 20000.0;

fn aerialUvw(dir: vec3f, dist: f32) -> vec3f {
  let az = atan2(dir.x, dir.z) / TAU + 0.5;
  let el = clamp((asin(clamp(dir.y, -1.0, 1.0)) - AERIAL_EL_MIN) / (AERIAL_EL_MAX - AERIAL_EL_MIN), 0.0, 1.0);
  let d = clamp(log(max(dist, AERIAL_D_MIN) / AERIAL_D_MIN) / log(AERIAL_D_MAX / AERIAL_D_MIN), 0.0, 1.0);
  return vec3f(az, el, d);
}

struct Aerial { inscatter: vec3f, transmittance: vec3f };
fn aerialLut(dir: vec3f, dist: f32) -> Aerial {
  let uvw = aerialUvw(dir, dist);
  var a: Aerial;
  a.inscatter = textureSampleLevel(aerialIn, skySamp, uvw, 0.0).rgb;
  a.transmittance = textureSampleLevel(aerialTr, skySamp, uvw, 0.0).rgb;
  return a;
}

/** 高度 h [m] の点に届く太陽光（大気透過後）。0〜2000m の 1D LUT */
fn sunLightAt(h: f32) -> vec3f {
  let x = clamp(h / 2000.0, 0.0, 1.0) * 127.0;
  let i = i32(floor(x));
  let f = x - f32(i);
  return mix(sunLut[i].rgb, sunLut[min(i + 1, 127)].rgb, f);
}
