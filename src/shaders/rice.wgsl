// 稲。夏の青々とした、まだ背の低い株（膝丈 45cm 前後、穂なし）。
// 生成: compute がカメラ周りの (u,v) 格子を走査し、焼いた材質 1 タップで田の泥だけを選んで追記（drawIndirect）。
// 描画: 近距離は株（葉 7 枚を頂点シェーダが生成、頂点バッファなし）、中距離は 3×3 株を束ねた交差板。
// 遠距離は water.wgsl が稲の面の色に溶かす。

// このファイルは 3 節から成る。TS 側が「共通＋生成」を compute モジュール、「共通＋描画」を描画モジュールに組む。
// 1 モジュールにまとめると、頂点シェーダが read_write の storage を参照する（不可）等で検証に落ちる（実測）。
// ==== SECTION: common ====
struct RiceInstance {
  pos: vec4f,     // xyz = 株元の世界座標, w = 乱数の種
  attr: vec4f,    // x = 大きさ, y = 回転, z = 予備, w = 予備
};
const RICE_ROW: f32 = 0.30;      // 条間 [m]
const RICE_HILL: f32 = 0.22;     // 株間 [m]
const RICE_NEAR_R: f32 = 40.0;   // 株で描く距離
const RICE_MID_R: f32 = 150.0;   // 束で描く距離
const RICE_MID_STEP: f32 = 0.9;  // 束の間隔（3×3 株）
const RICE_HEIGHT: f32 = 0.46;
const RICE_MAX: u32 = 160000u;
// ==== SECTION: spawn ====
struct DrawArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};

// bind group は最大 4 つ（0..3）なので、焼いた光・材質と同じ group(3) に同居させる
@group(3) @binding(2) var<storage, read_write> riceNear: array<RiceInstance>;
@group(3) @binding(3) var<storage, read_write> riceMid: array<RiceInstance>;
@group(3) @binding(4) var<storage, read_write> riceArgs: array<DrawArgs, 2>;


/** 田の泥（区画の内側で畦から離れている）か */
fn isPaddyMud(p: vec2f, margin: f32) -> bool {
  let b = bakedLight(p);
  return b.kind == 1u && b.ridgeDist > margin;
}

/** 視野の外（水平面での判定、余裕つき）なら true */
fn outsideView(p: vec2f) -> bool {
  let d = p - frame.camPos.xz;
  let dist = length(d);
  if (dist < 3.0) { return false; }
  let f = normalize(frame.camForward.xz);
  let halfFov = atan(frame.camUp.w * frame.params.z) + 0.25;
  return dot(d / dist, f) < cos(halfFov);
}

fn spawnCommon(uv: vec2f, lod: u32, seed: u32) {
  // 田は谷座標 v ∈ [-125, 170] の帯にしかない。外なら何も計算しない
  if (uv.y < -125.0 || uv.y > 170.0) { return; }
  let w = fromValley(uv);
  let dist = distance(w, frame.camPos.xz);
  // LOD の境界は株ごとに ±4m ばらし、直線の帯にならないようにする
  let dither = (hash2f(vec2i(i32(floor(uv.x * 3.0)), i32(floor(uv.y * 3.0))), 311u) - 0.5) * 8.0;
  if (lod == 0u && dist > RICE_NEAR_R + dither) { return; }
  if (lod == 1u && (dist <= RICE_NEAR_R + dither || dist > RICE_MID_R)) { return; }
  if (outsideView(w)) { return; }
  if (!isPaddyMud(w, select(0.9, 1.4, lod == 1u))) { return; }
  let h = hash2f(vec2i(i32(floor(uv.x * 10.0)), i32(floor(uv.y * 10.0))), seed);
  let h2 = hash2f(vec2i(i32(floor(uv.x * 10.0)), i32(floor(uv.y * 10.0))), seed + 7u);
  // 植えた位置のわずかなずれ
  let jitter = (vec2f(h, h2) - 0.5) * 0.06;
  let pj = w + jitter;
  let y = sampleHeight(heightLevelFor(pj), pj);
  var inst: RiceInstance;
  inst.pos = vec4f(pj.x, y, pj.y, h);
  // z = 株元の日向/日陰（頂点ごとにテクスチャを引かずに済ませる）
  inst.attr = vec4f(0.8 + 0.4 * h2, h * TAU, bakedLight(pj).shadow, 0.0);
  let idx = atomicAdd(&riceArgs[lod].instanceCount, 1u);
  if (idx < RICE_MAX) {
    if (lod == 0u) { riceNear[idx] = inst; } else { riceMid[idx] = inst; }
  }
}

/** 近距離の株: 条間 0.3m × 株間 0.22m の格子（谷座標 (u,v) に沿う＝区画の向きに揃う） */
@compute @workgroup_size(8, 8)
fn spawnNear(@builtin(global_invocation_id) id: vec3u) {
  let nu = 272u;   // 0.30m × 272 = 81.6m（半径 40m を覆う）
  let nv = 368u;   // 0.22m × 368 = 81.0m
  if (id.x >= nu || id.y >= nv) { return; }
  let c = toValley(frame.camPos.xz);
  let u = (floor(c.x / RICE_ROW) + f32(id.x) - f32(nu) * 0.5) * RICE_ROW;
  let v = (floor(c.y / RICE_HILL) + f32(id.y) - f32(nv) * 0.5) * RICE_HILL;
  spawnCommon(vec2f(u, v), 0u, 301u);
}

/** 中距離の束: 0.9m 格子 */
@compute @workgroup_size(8, 8)
fn spawnMid(@builtin(global_invocation_id) id: vec3u) {
  let n = 336u;   // 0.9m × 336 = 302m 四方（半径 150m を覆う）
  if (id.x >= n || id.y >= n) { return; }
  let c = toValley(frame.camPos.xz);
  let u = (floor(c.x / RICE_MID_STEP) + f32(id.x) - f32(n) * 0.5) * RICE_MID_STEP;
  let v = (floor(c.y / RICE_MID_STEP) + f32(id.y) - f32(n) * 0.5) * RICE_MID_STEP;
  spawnCommon(vec2f(u, v), 1u, 302u);
}

// ==== SECTION: draw ====
@group(3) @binding(2) var<storage, read> riceNear: array<RiceInstance>;
@group(3) @binding(3) var<storage, read> riceMid: array<RiceInstance>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,        // x = 葉の幅方向 -1..1, y = 根元 0 → 先端 1
  @location(3) shade: f32,       // 株元の日向/日陰
};

const BLADES: u32 = 7u;
const SEGS: u32 = 3u;
const VERTS_PER_BLADE: u32 = SEGS * 6u;
const NEAR_VERTS: u32 = BLADES * VERTS_PER_BLADE;   // 126

/** 葉 1 枚の、根元からの高さ比 s (0..1) における中心線と幅 */
fn bladeCenter(base: vec3f, dirXZ: vec2f, height: f32, s: f32, wind: Wind, t: f32) -> vec3f {
  // 外側へ反りながら立ち上がる。風は先端ほど強く風下へ倒す
  let lean = 0.35 * s * s;
  let windBend = wind.bend * (0.55 + 0.25 * sin(t * 2.3 + base.x * 3.1 + base.z * 2.7)) * s * s;
  return base + vec3f(dirXZ.x * lean * height + windBend.x * height, height * s * (1.0 - 0.15 * s), dirXZ.y * lean * height + windBend.y * height);
}

@vertex
fn vsNear(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let inst = riceNear[ii];
  let blade = vi / VERTS_PER_BLADE;
  let r = vi % VERTS_PER_BLADE;
  let seg = r / 6u;
  let corner = r % 6u;   // 2 三角形 = 6 頂点（0,1,2 / 2,1,3 の並び）
  let cornerIdx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[corner];
  let sideSign = select(-1.0, 1.0, (cornerIdx & 1u) == 1u);
  let up = f32(seg + (cornerIdx >> 1u)) / f32(SEGS);

  let seed = inst.pos.w;
  let bh = hash1f(i32(blade), u32(seed * 65535.0));
  let ang = inst.attr.y + f32(blade) * (TAU / f32(BLADES)) + (bh - 0.5) * 0.6;
  let dirXZ = vec2f(cos(ang), sin(ang));
  let height = RICE_HEIGHT * inst.attr.x * (0.75 + 0.35 * bh);
  let base = inst.pos.xyz + vec3f(dirXZ.x, 0.0, dirXZ.y) * 0.02;

  let t = frame.params.x;
  let wind = windAt(inst.pos.xz);
  let c = bladeCenter(base, dirXZ, height, up, wind, t);
  let cNext = bladeCenter(base, dirXZ, height, min(up + 0.2, 1.0), wind, t);
  let tangent = normalize(cNext - c + vec3f(0.0, 1e-4, 0.0));
  let side = normalize(cross(vec3f(0.0, 1.0, 0.0), vec3f(dirXZ.x, 0.0, dirXZ.y)));
  let width = 0.009 * inst.attr.x * (1.0 - up * up);   // 先端へ細く
  let world = c + side * (sideSign * width);

  var out: VSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.normal = normalize(cross(side, tangent));
  out.uv = vec2f(sideSign, up);
  out.shade = inst.attr.z;
  return out;
}

/** 中距離の束: 交差する 2 枚の板（幅 0.9m・高さ 0.5m）。葉の形は fs で切り抜く */
@vertex
fn vsMid(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let inst = riceMid[ii];
  let quad = vi / 6u;
  let corner = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[vi % 6u];
  let sx = select(-1.0, 1.0, (corner & 1u) == 1u);
  let up = f32(corner >> 1u);
  let ang = inst.attr.y + f32(quad) * (PI * 0.5);
  let side = vec3f(cos(ang), 0.0, sin(ang));
  let wind = windAt(inst.pos.xz);
  let height = 0.5 * inst.attr.x;
  let world = inst.pos.xyz + side * (sx * 0.5) + vec3f(0.0, height * up, 0.0)
    + vec3f(wind.bend.x, 0.0, wind.bend.y) * (0.6 * up * up * height);
  var out: VSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.normal = normalize(cross(side, vec3f(0.0, 1.0, 0.0)));
  out.uv = vec2f(sx, up);
  out.shade = inst.attr.z;
  return out;
}

/** 葉の色と逆光の透け。稲・草で共通の考え方 */
fn leafShade(world: vec3f, n0: vec3f, albedo: vec3f, transColor: vec3f, shade: f32, thickness: f32) -> vec3f {
  let toCam = frame.camPos.xyz - world;
  let v = normalize(toCam);
  let sun = frame.sunDir.xyz;
  // 薄い葉は両面。視線側に法線を向ける
  let n = select(n0, -n0, dot(n0, v) < 0.0);
  let sunLight = sunLightAt(world.y) * shade;
  let ambient = skyAmbient(n);
  let diffuse = max(dot(n, sun), 0.0);
  // 透過: 太陽が葉の裏にあり、視線がそちらを向くほど強い（逆光の縁が光る）
  let back = max(dot(-n, sun), 0.0);
  let toward = pow(max(dot(v, -sun), 0.0), 3.0);
  let trans = (0.12 * back + 0.30 * toward * back) / thickness;
  var color = albedo * (diffuse * sunLight + ambient) + transColor * trans * sunLight;
  let dist = length(toCam);
  let air = aerialLut(-v, dist);
  return color * air.transmittance + air.inscatter;
}

@fragment
fn fsNear(in: VSOut) -> @location(0) vec4f {
  // 根元は暗く、先端へ明るい黄緑
  let g = mix(vec3f(0.05, 0.14, 0.03), vec3f(0.12, 0.34, 0.07), in.uv.y);
  let color = leafShade(in.world, in.normal, g, vec3f(0.22, 0.55, 0.10), in.shade, 1.0);
  return vec4f(color, 1.0);
}

@fragment
fn fsMid(in: VSOut) -> @location(0) vec4f {
  // 板を葉の形に切り抜く: 縦の筋（株ごとの葉）と、上へ行くほど細くなる形
  let x = in.uv.x;
  let y = in.uv.y;
  let strands = 0.5 + 0.5 * sin(x * 21.0 + sin(x * 7.0) * 2.0);
  let taper = 1.0 - y * y;
  if (strands * taper < 0.42 + 0.5 * y) { discard; }
  // 板は面で日を受けて株より明るくなりがちなので、色を落として揃える
  let g = mix(vec3f(0.04, 0.11, 0.025), vec3f(0.08, 0.24, 0.05), y);
  let color = leafShade(in.world, in.normal, g, vec3f(0.18, 0.45, 0.08), in.shade, 1.6);
  return vec4f(color, 1.0);
}
