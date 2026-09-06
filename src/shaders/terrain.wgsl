// 地形。リング状メッシュを頂点シェーダが vertex_index から起こし、高さは world.wgsl の数式。
// 頂点バッファは持たない。1 回の draw call で 16km 先の山並みまで描く。

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) footprint: f32,   // この頂点付近のメッシュ間隔 [m]
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
  let r0 = frame.ring.x;
  let k = frame.ring.y;
  let sectors = u32(frame.ring.w);
  let center = frame.center.xy;

  var p: vec2f;
  var r: f32;
  if (vid == 0u) {
    p = center;
    r = 0.0;
  } else {
    let idx = vid - 1u;
    let ring = idx / sectors;
    let sec = idx % sectors;
    r = r0 * pow(k, f32(ring));
    let a = f32(sec) * TAU / f32(sectors);
    p = center + r * vec2f(cos(a), sin(a));
  }
  let footprint = max(0.05, r * TAU / f32(sectors));
  let h = terrainHeight(p, footprint * 2.0);
  let world = vec3f(p.x, h, p.y);

  var out: VSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.footprint = footprint;
  return out;
}

struct FSOut {
  @location(0) color: vec4f,
};

@fragment
fn fs(in: VSOut) -> FSOut {
  let p = in.world.xz;
  let toCam = frame.camPos.xyz - in.world;
  let dist = length(toCam);
  let viewDir = toCam / dist;

  // 画素の足元の大きさ。遠方ほど細部を落として暴れを抑える
  let pixelSize = dist * frame.camUp.w * 2.0 / frame.center.w;
  let minWl = max(pixelSize * 2.0, 0.04);

  var out: FSOut;
  let dbg = frame.params.w;
  if (dbg == 4.0) { out.color = vec4f(0.3, 0.5, 0.2, 1.0); return out; }   // 計測用: フラグメントの仕事を全部飛ばす

  // 材質・法線・日向/日陰はすべて焼いたもの（画素ごとに数式を評価しない）
  let baked = bakedLight(p);
  let n = select(baked.normal, sampleNormal(p, pixelSize), dbg == 7.0);   // 計測用 dbg=7: 双三次で直接計算
  let albedo = baked.albedo;
  let sun = frame.sunDir.xyz;
  let ndl = max(dot(n, sun), 0.0);

  // 太陽光: 高度ごとの透過後の色（1D LUT）× 地形の影
  let sunLight = sunLightAt(in.world.y);
  let shadow = select(baked.shadow, terrainShadow(in.world, sun), dbg == 8.0);   // 計測用 dbg=8: 影を行進で直接計算
  let ambient = skyAmbient(n);
  var color = albedo * (ndl * shadow * sunLight + ambient);

  // 遠くの田（110m 超）は水面と株を描かず、地形の側で稲の面として描く（水面は 170m 超で discard する）。
  // 風の場で明暗が渡る＝田の面を風が渡る画。近くは水面＋株が上に重なる
  if (baked.kind == 1u && dist > 110.0 && dbg != 12.0) {
    let canopyBlend = smootherstep(110.0, 170.0, dist);
    let wind = windAt(p);
    let riceAlbedo = vec3f(0.11, 0.30, 0.07);
    let lit = 0.55 + 0.45 * wind.strength;
    let toward = pow(max(dot(viewDir, -sun), 0.0), 3.0);
    let canopy = riceAlbedo * (lit * max(sun.y, 0.0) * 4.0 * sunLight * shadow + ambient)
      + vec3f(0.22, 0.55, 0.10) * toward * 0.25 * sunLight * shadow * (0.5 + 0.5 * wind.strength);
    color = mix(color, canopy, canopyBlend);
  }

  // 遠景の溶け込み（3D LUT）
  let air = aerialLut(-viewDir, dist);
  color = color * air.transmittance + air.inscatter;

  let sKind = baked.kind;
  let sHeight = in.world.y;
  // デバッグ表示: params.w == 1 で法線、2 で材質種別、3 で高さ
  if (dbg == 1.0) { color = n * 0.5 + 0.5; }
  else if (dbg == 2.0) {
    let k = f32(sKind);
    color = vec3f(fract(k * 0.37 + 0.1), fract(k * 0.61 + 0.5), fract(k * 0.83 + 0.2));
  } else if (dbg == 3.0) { color = vec3f(fract(sHeight), fract(sHeight * 0.1), fract(sHeight * 0.01)); }

  out.color = vec4f(color, 1.0);
  return out;
}
