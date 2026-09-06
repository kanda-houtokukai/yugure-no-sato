// 木。テンプレートメッシュ（数値から生成）をインスタンス描画。幹は動かず、枝葉だけが風で揺れる。

struct TVSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) kind: f32,
  @location(4) shade: f32,
  @location(5) sway: f32,
};

@vertex
fn vsTree(
  @location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f,
  @location(3) kind: f32, @location(4) sway: f32,
  @location(5) iPos: vec4f, @location(6) iAttr: vec4f,
) -> TVSOut {
  // iPos: xyz = 根元, w = 大きさ。iAttr: x = 回転, y = 日向/日陰, z = 種, w = 予備
  let c = cos(iAttr.x);
  let s = sin(iAttr.x);
  let local = position * iPos.w;
  var p = vec3f(local.x * c - local.z * s, local.y, local.x * s + local.z * c);
  var n = vec3f(normal.x * c - normal.z * s, normal.y, normal.x * s + normal.z * c);
  let base = iPos.xyz;
  // 風: 枝葉だけ。揺れ幅は高さ（sway）で増える。株ごとの位相差つき
  let wind = windAt(base.xz);
  let t = frame.params.x;
  let flutter = sin(t * 1.7 + iAttr.w * 6.28 + p.y * 0.4) * 0.06;
  let offset = (wind.bend * (0.45 * sway) + vec2f(-WIND_DIR.y, WIND_DIR.x) * flutter * wind.strength * sway) * iPos.w;
  p += vec3f(offset.x, 0.0, offset.y);
  let world = base + p;

  var out: TVSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.normal = n;
  out.uv = uv;
  out.kind = kind;
  out.shade = bakedLight(base.xz).shadow;   // 根元の日向/日陰
  out.sway = sway;
  return out;
}

@fragment
fn fsTree(in: TVSOut) -> @location(0) vec4f {
  let kind = u32(round(in.kind));
  var color: vec3f;
  if (kind == 0u) {
    // 幹・枝: 樹皮
    let toCam = frame.camPos.xyz - in.world;
    let v = normalize(toCam);
    let n = select(in.normal, -in.normal, dot(in.normal, v) < 0.0);
    let sun = frame.sunDir.xyz;
    let bark = vec3f(0.22, 0.16, 0.11) * (0.8 + 0.4 * gnoise(vec2f(in.uv.x * 12.0, in.world.y * 3.0), 601u));
    color = bark * (max(dot(n, sun), 0.0) * sunLightAt(in.world.y) * in.shade + skyAmbient(n));
    let air = aerialLut(-v, length(toCam));
    color = color * air.transmittance + air.inscatter;
  } else if (kind == 1u) {
    // 杉の葉: 板を針葉の房の形に切り抜く（根元は太く、先端へ房がまばらに）
    let x = in.uv.x * 2.0 - 1.0;   // -1..1（板の幅方向）
    let y = in.uv.y;               // 0 根元 → 1 先端
    let nz = gnoise(vec2f(x * 6.0 + y * 3.0, y * 9.0), 611u) * 0.5 + 0.5;
    let envelope = (1.0 - abs(x)) * (0.55 + 0.45 * (1.0 - y * y));
    if (nz * envelope < 0.13 + 0.16 * y) { discard; }
    // 樹冠の内側（根元側）は暗い
    let occ = 0.45 + 0.55 * y;
    color = leafShade(in.world, in.normal, vec3f(0.05, 0.12, 0.05) * occ, vec3f(0.12, 0.30, 0.08), in.shade, 2.5);
  } else {
    // 広葉樹の葉塊: 丸い輪郭の中に、葉の房（高周波）の隙間を空けて板に見せない
    let d = length(in.uv - 0.5) * 2.0;
    let big = gnoise(vec2f(in.uv.x * 5.0, in.uv.y * 5.0), 621u) * 0.5 + 0.5;
    let fine = gnoise(vec2f(in.uv.x * 19.0, in.uv.y * 19.0), 622u) * 0.5 + 0.5;
    let leafy = 0.55 * big + 0.45 * fine;
    if (d > 0.55 + 0.45 * big || leafy < 0.30 + 0.25 * d) { discard; }
    let occ = 0.55 + 0.45 * smoothstep(0.9, 0.2, d);
    color = leafShade(in.world, in.normal, vec3f(0.08, 0.20, 0.05) * occ, vec3f(0.28, 0.55, 0.12), in.shade, 1.2);
  }
  return vec4f(color, 1.0);
}
