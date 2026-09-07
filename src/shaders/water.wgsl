// 水面。田の水と小川。不透明な板にせず、空の映り込み（Fresnel）と浅い水底の透け（濁り）を両立する。
// 頂点は CPU が cellQuery / riverQuery の結果から組む（位置は世界座標、y = 水面高さ）。

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
};

@vertex
fn vs(@location(0) position: vec3f) -> VSOut {
  var out: VSOut;
  out.pos = frame.viewProj * vec4f(position - frame.camPos.xyz, 1.0);
  out.world = position;
  return out;
}

/**
 * さざ波。風で立つ細かい波を正弦の和で。時刻は frame.params.x（固定なら絵も固定）。
 * footprint（画素の足元 [m]）より短い波は落とす。落とさないと画素より細かい波がモアレになる（実測）
 */
fn rippleNormal(p: vec2f, footprint: f32) -> vec3f {
  let t = frame.params.x;
  var dx = 0.0;
  var dz = 0.0;
  let waves = array<vec4f, 4>(
    vec4f(0.91, 0.41, 1.40, 1.9),    // 方向 xz, 波長 [m], 速度
    vec4f(-0.35, 0.94, 0.80, 1.3),
    vec4f(0.60, -0.80, 0.45, 0.9),
    vec4f(-0.90, -0.44, 0.24, 0.6),
  );
  let amps = array<f32, 4>(0.0016, 0.0011, 0.0007, 0.0004);
  for (var i = 0; i < 4; i++) {
    let w = waves[i];
    let visible = smootherstep(1.2, 5.0, w.z / footprint);
    if (visible <= 0.0) { continue; }
    let k = TAU / w.z;
    let phase = dot(p, w.xy) * k + t * w.w;
    let slope = amps[i] * k * cos(phase) * visible;
    dx += w.x * slope;
    dz += w.y * slope;
  }
  // ごく細かいさざ波（間近でだけ見える）
  let fine = smootherstep(1.2, 4.0, 0.11 / footprint);
  if (fine > 0.0) {
    let kf = TAU / 0.11;
    let ph = dot(p, vec2f(0.72, -0.69)) * kf + t * 0.42;
    let sl = 0.00018 * kf * cos(ph) * fine;
    dx += 0.72 * sl;
    dz += -0.69 * sl;
  }
  // 風の場（焼いたもの 1 タップ）で波を強弱させる。稲を渡る風と同じ場。
  // さらに低周波の「凪の帯」を掛けて、鏡のような面と風で乱れた面を混在させる
  let calm = 0.45 + 0.55 * gnoise(p / 38.0 + vec2f(t * 0.018, 0.0), 93u);
  let gust = (0.10 + 1.7 * windAt(p).strength) * calm;
  return normalize(vec3f(-dx * gust, 1.0, -dz * gust));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let toCam = frame.camPos.xyz - in.world;
  let dist = length(toCam);
  // 遠くの田は地形側が稲の面として描く。ここで捨てて重い経路（さざ波・映り込み）を通さない
  if (dist > 170.0) { discard; }
  let v = toCam / dist;
  let sun = frame.sunDir.xyz;
  // 画素の足元。水面をかすめる視線では奥行き方向に伸びる
  let footprint = dist * frame.camUp.w * 2.0 / frame.center.w / max(abs(v.y), 0.05);
  var n = select(rippleNormal(in.world.xz, footprint), vec3f(0.0, 1.0, 0.0), frame.params.w == 16.0);   // 計測用 dbg=16: さざ波なし
  // 変形の場: 足が起こした波紋（法線の傾き）と濁り、沈んだ泥
  let df = deformAt(in.world.xz);
  let rg = rippleGradient(in.world.xz);
  n = normalize(vec3f(n.x - rg.x, n.y, n.z - rg.y));

  // ---- 映り込み ----
  let r = reflect(-v, n);
  let rUp = vec3f(r.x, max(r.y, 0.02), r.z);   // 水面より下は映らないので地平線へ折り返す
  var refl = skyRadianceLut(normalize(rUp));
  let baked = bakedLight(in.world.xz);
  // 畦・道の線上は水ではない。遠くで畦の幾何が消えても、ここで穴を開けて区画の網目を保つ
  if (baked.ridgeDist < 0.6) { discard; }
  // 太陽の映り込み（円盤）。畦の影の中では消える
  let mu = dot(normalize(rUp), sun);
  let disc = sunDisc(mu, frame.sunDir.w);
  if (disc > 0.0) {
    refl += SUN_E * 40.0 * disc * (skyIrr[2].rgb / SUN_E) * baked.shadow;
  }

  // ---- 透け（浅く濁った水） ----
  let cosTheta = clamp(dot(n, v), 0.0, 1.0);
  let f0 = 0.02;
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
  // 水底の泥は太陽と空で照らされる（法線は上向きとして）
  let sunLight = sunLightAt(in.world.y) * max(sun.y, 0.0) * baked.shadow;
  let ambient = skyAmbient(vec3f(0.0, 1.0, 0.0));
  // 濡れた泥は乾いた泥より暗い
  let mud = vec3f(0.12, 0.10, 0.07) * (sunLight + ambient);
  // 濁り: 水中で散った光の色。深さ 0.12m の田なので泥の色が半分ほど残る
  let murk = vec3f(0.20, 0.18, 0.11) * ambient * 0.6;
  // 畦の際は浅い。水深が小さいほど水底の泥が透けて暖色になる
  let shallow = 1.0 - smootherstep(0.0, 2.4, baked.ridgeDist);
  let depth = mix(0.12, 0.025, shallow) + df.sink;
  let depthFactor = exp(-vec3f(2.4, 2.0, 3.4) * depth * 2.0 / max(cosTheta, 0.1));
  var under = mix(murk, mud * depthFactor, 0.55);
  // 濁り: 舞い上がった泥の色が水中で散る
  let stirred = vec3f(0.30, 0.24, 0.13) * ambient * 0.9;
  under = mix(under, stirred, df.turbidity * 0.75);
  // 浅い縁では水底の泥の色が前に出る
  under = mix(under, mud * 1.35 + murk * 0.4, shallow * 0.55);

  // 畦の際は水面も空が見えにくい（映り込みが弱まり、水底の色が出る）
  let edgeAo = 0.55 + 0.45 * smootherstep(0.0, 1.4, baked.ridgeDist);
  var color = mix(under * edgeAo, refl, fresnel * (0.35 + 0.65 * edgeAo));

  // 遠景の溶け込み（3D LUT）
  let air = aerialLut(-v, dist);
  color = color * air.transmittance + air.inscatter;

  let dbg = frame.params.w;
  if (dbg == 1.0) { color = n * 0.5 + 0.5; }
  if (dbg == 9.0) { color = vec3f(4.0, 0.0, 4.0); }   // 診断用: 水面画素をマゼンタで塗る
  return vec4f(color, 1.0);
}
