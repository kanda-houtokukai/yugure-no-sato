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
  let pixelSize = dist * frame.camUp.w * 2.0 / 720.0;
  let minWl = max(pixelSize * 2.0, 0.04);

  var out: FSOut;
  let dbg = frame.params.w;
  if (dbg == 4.0) { out.color = vec4f(0.3, 0.5, 0.2, 1.0); return out; }   // 計測用: フラグメントの仕事を全部飛ばす

  // 材質は数式を画素ごとに 1 回だけ評価（これは軽い）。法線は焼いた高さテクスチャから
  let s = terrainSurface(p, minWl);
  let n = sampleNormal(p, pixelSize);

  let albedo = terrainAlbedo(p, s);
  let sun = frame.sunDir.xyz;
  let ndl = max(dot(n, sun), 0.0);

  // 仮の光（次の段階で大気散乱に置き換える）
  let sunColor = vec3f(1.0, 0.62, 0.36) * 4.0;
  let skyAmbient = vec3f(0.30, 0.40, 0.65) * (0.55 + 0.45 * n.y) * 0.5;
  var color = albedo * (ndl * sunColor + skyAmbient);

  // 仮の距離霞
  let fog = 1.0 - exp(-dist * 0.00035);
  let fogColor = vec3f(0.85, 0.55, 0.35);
  color = mix(color, fogColor, fog);

  // デバッグ表示: params.w == 1 で法線、2 で材質種別、3 で高さ
  if (dbg == 1.0) { color = n * 0.5 + 0.5; }
  else if (dbg == 2.0) {
    let k = f32(s.kind);
    color = vec3f(fract(k * 0.37 + 0.1), fract(k * 0.61 + 0.5), fract(k * 0.83 + 0.2));
  } else if (dbg == 3.0) { color = vec3f(fract(s.height), fract(s.height * 0.1), fract(s.height * 0.01)); }

  out.color = vec4f(color, 1.0);
  return out;
}
