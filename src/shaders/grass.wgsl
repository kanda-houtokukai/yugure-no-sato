// 草。畦・あぜ道の脇・草地・丘の斜面に生える。稲より雑多で背丈も不揃い。
// 仕組みは稲と同じ（compute 生成 → drawIndirect、距離 2 段）。3 節構成（共通 / 生成 / 描画）。
// ==== SECTION: common ====
struct GrassInstance {
  pos: vec4f,     // xyz = 根元, w = 乱数の種
  attr: vec4f,    // x = 大きさ, y = 回転, z = 日向/日陰, w = 種類（0 = 青草, 1 = 枯れ茎）
  deform: vec4f,  // xy = 倒れ方向, z = 倒れ量, w = 予備
};
const GRASS_NEAR_R: f32 = 30.0;
const GRASS_MID_R: f32 = 120.0;
const GRASS_NEAR_STEP: f32 = 0.35;
const GRASS_MID_STEP: f32 = 1.2;
const GRASS_MAX: u32 = 120000u;
// ==== SECTION: spawn ====
struct GrassArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};
@group(3) @binding(5) var<storage, read_write> grassNear: array<GrassInstance>;
@group(3) @binding(6) var<storage, read_write> grassMid: array<GrassInstance>;
@group(3) @binding(7) var<storage, read_write> grassArgs: array<GrassArgs, 2>;

fn grassOutsideView(p: vec2f) -> bool {
  let d = p - frame.camPos.xz;
  let dist = length(d);
  if (dist < 3.0) { return false; }
  let f = normalize(frame.camForward.xz);
  let halfFov = atan(frame.camUp.w * frame.params.z) + 0.25;
  return dot(d / dist, f) < cos(halfFov);
}

/** 生える密度（0..1）。種別と、まだらのノイズで決める */
fn grassDensity(p: vec2f, b: BakedLight) -> f32 {
  let mottle = 0.5 + 0.5 * gnoise(p / 7.0, 401u);   // まだら（patch は WGSL の予約語）
  switch (b.kind) {
    case 2u: { return 1.0; }                                         // 畦: 密
    case 0u: { return select(0.6 + 0.4 * mottle, 1.0, b.ridgeDist < 1.8 && b.ridgeDist > 0.8); }   // 草地。道の脇は密
    case 5u: { return 0.35 + 0.5 * mottle; }                          // 丘の斜面: まだら
    case 7u: { return 0.06 + 0.30 * mottle * mottle; }                // 集落の敷地: まばら（隅にだけ茂る）
    case 8u: { return 0.04 * mottle; }                                // 境内: ほとんど生えない
    default: { return 0.0; }                                         // 田の泥・道の上・川床・山には生えない
  }
}

fn grassSpawn(cell: vec2i, step: f32, lod: u32, seed: u32) {
  let h0 = hash2f(cell, seed);
  let h1 = hash2f(cell, seed + 1u);
  let h2 = hash2f(cell, seed + 2u);
  let h3 = hash2f(cell, seed + 3u);
  let p = (vec2f(cell) + vec2f(h0, h1)) * step;
  let dist = distance(p, frame.camPos.xz);
  let dither = (h2 - 0.5) * 6.0;
  if (lod == 0u && dist > GRASS_NEAR_R + dither) { return; }
  if (lod == 1u && (dist <= GRASS_NEAR_R + dither || dist > GRASS_MID_R)) { return; }
  if (grassOutsideView(p)) { return; }
  let b = bakedLight(p);
  if (h3 > grassDensity(p, b)) { return; }
  if (b.normal.y < 0.55) { return; }   // 急斜面には生えない
  let y = sampleHeight(heightLevelFor(p), p);
  let df = deformAt(p);
  var inst: GrassInstance;
  inst.pos = vec4f(p.x, y - df.sink, p.y, h0);
  let dry = select(0.0, 1.0, hash2f(cell, seed + 4u) < 0.18);
  inst.attr = vec4f(0.45 + 0.9 * h1 * h1, h2 * TAU, b.shadow, dry);
  inst.deform = vec4f(df.bend, length(df.bend), 0.0);
  let idx = atomicAdd(&grassArgs[lod].instanceCount, 1u);
  if (idx < GRASS_MAX) {
    if (lod == 0u) { grassNear[idx] = inst; } else { grassMid[idx] = inst; }
  }
}

@compute @workgroup_size(8, 8)
fn spawnGrassNear(@builtin(global_invocation_id) id: vec3u) {
  let n = 176u;   // 0.35m × 176 = 61.6m（半径 30m）
  if (id.x >= n || id.y >= n) { return; }
  let c = vec2i(floor(frame.camPos.xz / GRASS_NEAR_STEP));
  grassSpawn(c + vec2i(id.xy) - vec2i(i32(n / 2u)), GRASS_NEAR_STEP, 0u, 501u);
}

@compute @workgroup_size(8, 8)
fn spawnGrassMid(@builtin(global_invocation_id) id: vec3u) {
  let n = 208u;   // 1.2m × 208 = 250m（半径 120m）
  if (id.x >= n || id.y >= n) { return; }
  let c = vec2i(floor(frame.camPos.xz / GRASS_MID_STEP));
  grassSpawn(c + vec2i(id.xy) - vec2i(i32(n / 2u)), GRASS_MID_STEP, 1u, 502u);
}

// ==== SECTION: draw ====
@group(3) @binding(5) var<storage, read> grassNear: array<GrassInstance>;
@group(3) @binding(6) var<storage, read> grassMid: array<GrassInstance>;

struct GVSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) shade: f32,
  @location(4) dry: f32,
};

const G_BLADES: u32 = 5u;
const G_SEGS: u32 = 2u;
const G_VERTS_PER_BLADE: u32 = G_SEGS * 6u;
const GRASS_NEAR_VERTS: u32 = G_BLADES * G_VERTS_PER_BLADE;   // 60

fn grassBladeCenter(base: vec3f, dirXZ: vec2f, height: f32, s: f32, bend: vec2f, flat: vec3f) -> vec3f {
  let lean = 0.45 * s * s;
  let b = bend * (1.0 - 0.85 * flat.z) + flat.xy * 1.1;
  return base + vec3f(dirXZ.x * lean * height + b.x * height * s * s, height * s * (1.0 - 0.2 * s) * (1.0 - 0.7 * flat.z * s), dirXZ.y * lean * height + b.y * height * s * s);
}

@vertex
fn vsGrassNear(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GVSOut {
  let inst = grassNear[ii];
  let blade = vi / G_VERTS_PER_BLADE;
  let r = vi % G_VERTS_PER_BLADE;
  let seg = r / 6u;
  let cornerIdx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[r % 6u];
  let sideSign = select(-1.0, 1.0, (cornerIdx & 1u) == 1u);
  let up = f32(seg + (cornerIdx >> 1u)) / f32(G_SEGS);

  let seedU = u32(inst.pos.w * 65535.0);
  let bh = hash1f(i32(blade), seedU);
  let bh2 = hash1f(i32(blade) + 17, seedU);
  let ang = inst.attr.y + f32(blade) * (TAU / f32(G_BLADES)) + (bh - 0.5) * 1.2;
  let dirXZ = vec2f(cos(ang), sin(ang));
  // 背丈は株ごと・葉ごとに不揃い（0.15〜0.6m）。枯れ茎は細く長い
  let dry = inst.attr.w;
  let height = mix(0.16, 0.58, inst.attr.x * (0.6 + 0.6 * bh2)) * mix(1.0, 1.3, dry);
  let base = inst.pos.xyz + vec3f(dirXZ.x, 0.0, dirXZ.y) * (0.03 * bh2);

  // 風: 共有の場に、株ごとの位相差とひらひらを足す（稲ほど揃えない）
  let t = frame.params.x;
  let wind = windAt(inst.pos.xz);
  let flutter = 0.12 * sin(t * (3.0 + 2.5 * bh) + inst.pos.w * 40.0 + f32(blade));
  let bend = wind.bend * (0.35 + 0.5 * bh2) + vec2f(-WIND_DIR.y, WIND_DIR.x) * flutter * wind.strength;

  let flat = inst.deform.xyz;
  let c = grassBladeCenter(base, dirXZ, height, up, bend, flat);
  let cNext = grassBladeCenter(base, dirXZ, height, min(up + 0.25, 1.0), bend, flat);
  let tangent = normalize(cNext - c + vec3f(0.0, 1e-4, 0.0));
  let side = normalize(cross(vec3f(0.0, 1.0, 0.0), vec3f(dirXZ.x, 0.0, dirXZ.y)));
  let width = mix(0.006, 0.011, bh) * mix(1.0, 0.5, dry) * (1.0 - up * up * 0.9);
  let world = c + side * (sideSign * width);

  var out: GVSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.normal = normalize(cross(side, tangent));
  out.uv = vec2f(sideSign, up);
  out.shade = inst.attr.z;
  out.dry = dry;
  return out;
}

@vertex
fn vsGrassMid(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GVSOut {
  let inst = grassMid[ii];
  let quad = vi / 6u;
  let corner = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[vi % 6u];
  let sx = select(-1.0, 1.0, (corner & 1u) == 1u);
  let up = f32(corner >> 1u);
  let ang = inst.attr.y + f32(quad) * (PI * 0.5);
  let side = vec3f(cos(ang), 0.0, sin(ang));
  let wind = windAt(inst.pos.xz);
  let height = mix(0.2, 0.55, inst.attr.x);
  let flat = inst.deform.xyz;
  let world = inst.pos.xyz + side * (sx * 0.35) + vec3f(0.0, height * up * (1.0 - 0.7 * flat.z), 0.0)
    + vec3f(wind.bend.x, 0.0, wind.bend.y) * (0.5 * up * up * height) * (1.0 - 0.85 * flat.z)
    + vec3f(flat.x, 0.0, flat.y) * (1.1 * up * up * height);
  var out: GVSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.normal = normalize(cross(side, vec3f(0.0, 1.0, 0.0)));
  out.uv = vec2f(sx, up);
  out.shade = inst.attr.z;
  out.dry = inst.attr.w;
  return out;
}

fn grassColor(uvY: f32, dry: f32) -> vec3f {
  let green = mix(vec3f(0.07, 0.20, 0.04), vec3f(0.26, 0.42, 0.10), uvY);
  let brown = mix(vec3f(0.30, 0.24, 0.10), vec3f(0.55, 0.45, 0.20), uvY);
  return mix(green, brown, dry);
}

@fragment
fn fsGrassNear(in: GVSOut) -> @location(0) vec4f {
  let trans = mix(vec3f(0.30, 0.60, 0.12), vec3f(0.65, 0.50, 0.18), in.dry);
  let color = leafShade(in.world, in.normal, grassColor(in.uv.y, in.dry), trans, in.shade, 0.8);
  return vec4f(color, 1.0);
}

@fragment
fn fsGrassMid(in: GVSOut) -> @location(0) vec4f {
  let x = in.uv.x;
  let y = in.uv.y;
  // 不揃いな筋: 2 つの周期の重ね合わせで葉の本数と高さを散らす
  let strands = 0.5 + 0.5 * sin(x * 17.0 + sin(x * 5.0 + in.dry * 3.0) * 2.5);
  let heights = 0.55 + 0.45 * sin(x * 9.0 + 1.7);
  if (strands < 0.35 + 0.5 * y || y > heights) { discard; }
  let trans = mix(vec3f(0.30, 0.60, 0.12), vec3f(0.65, 0.50, 0.18), in.dry);
  let color = leafShade(in.world, in.normal, grassColor(y, in.dry) * 0.85, trans, in.shade, 1.2);
  return vec4f(color, 1.0);
}
