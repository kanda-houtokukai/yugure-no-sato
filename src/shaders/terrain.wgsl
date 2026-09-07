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
  // 数式の高さ（正本）に、変形の場の沈みを足す。正本そのものは変えない
  let h = terrainHeight(p, footprint * 2.0) - deformAt(p).sink;
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

/**
 * 地面の細かい凹凸。高さテクスチャは 0.25m テクセルなので、それより細かい起伏は法線で足す。
 * 高さそのものは変えない（変形の場や当たり判定と食い違わせないため）。近距離でのみ効かせる。
 */
fn detailBump(p: vec2f, kind: u32) -> f32 {
  switch (kind) {
    // 太陽が仰角 3.5° と低いので、わずかな傾きが大きな明暗になる。振幅は控えめに
    case 3u, 7u: {   // 土の道・集落の敷地: 踏み固めた土の細かい凹凸
      return 0.0035 * gnoise(p * 6.0, 701u) + 0.0016 * gnoise(p * 17.0, 702u);
    }
    case 1u: {       // 田の泥: 濡れた泥のうねり（粗く、なだらか）
      return 0.0030 * gnoise(p * 4.5, 703u);
    }
    case 8u: {       // 境内: 玉砂利の粒
      return 0.0014 * gnoise(p * 24.0, 704u) + 0.0009 * gnoise(p * 55.0, 705u);
    }
    case 4u: {       // 川床: 石まじり
      return 0.0040 * gnoise(p * 8.0, 706u);
    }
    default: {       // 草地・畦・丘: 株の根元の起伏
      return 0.0050 * gnoise(p * 3.5, 707u) + 0.0024 * gnoise(p * 11.0, 708u);
    }
  }
}

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
  var n = select(baked.normal, sampleNormal(p, pixelSize), dbg == 7.0);   // 計測用 dbg=7: 双三次で直接計算
  // 変形の沈みで法線も傾ける（高さだけ変えると、へこんでいるのに陰影が平らなまま）
  let sg = deformSinkGradient(p);
  n = normalize(vec3f(n.x + sg.x * n.y, n.y, n.z + sg.y * n.y));
  // 細部の凹凸で法線を傾ける（近距離のみ。遠くでは画素より細かくなりエイリアスになる）
  let detailFade = 1.0 - smootherstep(7.0, 28.0, dist);
  if (detailFade > 0.0 && dbg != 17.0) {   // 計測用 dbg=17: 細部の凹凸なし
    let e = max(0.035, pixelSize);
    let b0 = detailBump(p, baked.kind);
    let bx = detailBump(p + vec2f(e, 0.0), baked.kind);
    let bz = detailBump(p + vec2f(0.0, e), baked.kind);
    n = normalize(n + vec3f((b0 - bx) / e, 0.0, (b0 - bz) / e) * detailFade);
  }

  let df = deformAt(p);
  var albedo = baked.albedo;
  // 踏み固めた土は湿って暗く見える（道の沈みは 2cm 程度で陰影だけでは見えにくい）
  if (baked.kind == 3u || baked.kind == 0u || baked.kind == 2u) { albedo *= 1.0 - 0.45 * clamp(df.sink / 0.02, 0.0, 1.0); }
  let sun = frame.sunDir.xyz;
  let ndl = max(dot(n, sun), 0.0);

  // 太陽光: 高度ごとの透過後の色（1D LUT）× 地形の影
  let sunLight = sunLightAt(in.world.y);
  let shadow = select(baked.shadow, terrainShadow(in.world, sun), dbg == 8.0);   // 計測用 dbg=8: 影を行進で直接計算
  // 遮蔽による陰り: 畦・道の際は両側に壁があり、草に覆われた地面は葉に遮られて空が見えにくい
  var ao = 0.60 + 0.40 * smootherstep(0.0, 1.1, baked.ridgeDist);
  if (baked.kind == 0u || baked.kind == 2u || baked.kind == 5u) { ao *= 0.78; }
  if (baked.kind == 1u) { ao *= 0.86; }   // 田の泥は稲に囲まれる
  if (dbg == 18.0) { ao = 1.0; }          // 計測・比較用 dbg=18: 遮蔽なし
  let ambient = skyAmbient(n);
  var color = albedo * (ndl * shadow * sunLight + ambient * ao);

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
