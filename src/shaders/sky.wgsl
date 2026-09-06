// 空。画素ごとに大気散乱を積分する（品質優先）。太陽の円盤もここで描く。
// 同じ積分を compute で正距円筒の LUT にも焼く（環境光・映り込み用）。

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VSOut;
  // reversed-Z: 深度 0 = 無限遠。地形が描かれていない画素だけ通る（greater-equal）
  out.pos = vec4f(corners[vi], 0.0, 1.0);
  out.ndc = corners[vi];
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let dir = viewRay(in.ndc);
  let sun = frame.sunDir.xyz;
  // 空は焼いた LUT（太陽が固定なら 1 回でよい）。太陽の円盤だけ解析的に足す
  var color = skyRadianceLut(dir);
  let mu = dot(dir, sun);
  let disc = sunDisc(mu, frame.sunDir.w);
  if (disc > 0.0) {
    // 円盤の放射輝度は太陽定数を立体角で割ったもの。ここでは相対値なので係数で調整。
    // 透過率はカメラ位置で reduce が計算した直射（skyIrr[2]）から取る
    let trans = skyIrr[2].rgb / SUN_E;
    color += SUN_E * 40.0 * disc * trans;
  }
  return vec4f(color, 1.0);
}

// ---- LUT 焼き ----
@group(0) @binding(1) var skyLutOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bakeSkyLut(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(skyLutOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let u = (f32(id.x) + 0.5) / f32(size.x);
  let v = (f32(id.y) + 0.5) / f32(size.y);
  let az = (u - 0.5) * TAU;
  let el = (0.5 - v) * PI;
  let dir = vec3f(sin(az) * cos(el), sin(el), cos(az) * cos(el));
  let a = atmosphereMarch(frame.camPos.xyz, dir, frame.sunDir.xyz, -1.0, 32, 8);
  textureStore(skyLutOut, vec2i(id.xy), vec4f(a.inscatter, 1.0));
}

// ---- LUT の半球積分（環境光） ----
// 4 方向のサンプル平均では、太陽周りの輝きを 1 サンプルが拾って環境光が跳ね上がる（実測）。
// cos 重みの半球積分で放射照度を出す。
@group(0) @binding(2) var skyLutIn: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> skyIrrOut: array<vec4f, 3>;

var<workgroup> partialUp: array<vec3f, 256>;
var<workgroup> partialMean: array<vec3f, 256>;

@compute @workgroup_size(256)
fn reduceSkyLut(@builtin(local_invocation_id) lid: vec3u) {
  let size = textureDimensions(skyLutIn);
  let x = i32(lid.x);
  var eUp = vec3f(0.0);
  var mean = vec3f(0.0);
  if (lid.x < size.x) {
    for (var y = 0; y < i32(size.y); y++) {
      let v = (f32(y) + 0.5) / f32(size.y);
      let el = (0.5 - v) * PI;
      let l = textureLoad(skyLutIn, vec2i(x, y), 0).rgb;
      let dOmega = cos(el) * (PI / f32(size.y)) * (TAU / f32(size.x));
      eUp += l * max(sin(el), 0.0) * dOmega;
      mean += l * dOmega;
    }
  }
  partialUp[lid.x] = eUp;
  partialMean[lid.x] = mean;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid.x < stride) {
      partialUp[lid.x] += partialUp[lid.x + stride];
      partialMean[lid.x] += partialMean[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    skyIrrOut[0] = vec4f(partialUp[0], 0.0);            // 上向き面の放射照度
    skyIrrOut[1] = vec4f(partialMean[0] / (4.0 * PI), 0.0); // 全天の平均放射輝度
    // 診断用: カメラ位置での直射（太陽に正対する面の放射照度）
    skyIrrOut[2] = vec4f(SUN_E * sunTransmittance(toPlanet(frame.camPos.xyz), frame.sunDir.xyz, 8), 0.0);
  }
}

// ---- 遠景 LUT と太陽光 LUT の焼き込み ----
@group(0) @binding(4) var aerialInOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var aerialTrOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<storage, read_write> sunLutOut: array<vec4f, 128>;

@compute @workgroup_size(4, 4, 4)
fn bakeAerial(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(aerialInOut);
  if (id.x >= size.x || id.y >= size.y || id.z >= size.z) { return; }
  let az = ((f32(id.x) + 0.5) / f32(size.x) - 0.5) * TAU;
  let el = AERIAL_EL_MIN + (f32(id.y) + 0.5) / f32(size.y) * (AERIAL_EL_MAX - AERIAL_EL_MIN);
  let dist = AERIAL_D_MIN * exp((f32(id.z) + 0.5) / f32(size.z) * log(AERIAL_D_MAX / AERIAL_D_MIN));
  let dir = vec3f(sin(az) * cos(el), sin(el), cos(az) * cos(el));
  // 地面を突き抜けないよう、下向きの視線でも大気の積分は距離で打ち切る（地形が距離を渡す）
  let a = atmosphereMarch(frame.camPos.xyz, dir, frame.sunDir.xyz, dist, 12, 3);
  textureStore(aerialInOut, vec3i(id), vec4f(a.inscatter, 1.0));
  textureStore(aerialTrOut, vec3i(id), vec4f(a.transmittance, 1.0));
}

@compute @workgroup_size(64)
fn bakeSunLut(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= 128u) { return; }
  let h = f32(id.x) / 127.0 * 2000.0;
  let t = sunTransmittance(toPlanet(vec3f(frame.camPos.x, h, frame.camPos.z)), frame.sunDir.xyz, 12);
  sunLutOut[id.x] = vec4f(SUN_E * t, 0.0);
}

// ---- 風の場の焼き込み（毎フレーム） ----
@group(0) @binding(7) var windOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bakeWind(@builtin(global_invocation_id) id: vec3u) {
  let n = u32(frame.wind.w);
  if (id.x >= n || id.y >= n) { return; }
  let p = frame.wind.xy + (vec2f(f32(id.x), f32(id.y)) + 0.5) * frame.wind.z;
  let w = windField(p, frame.params.x);
  textureStore(windOut, vec2i(id.xy), vec4f(w.bend, w.strength, 0.0));
}
