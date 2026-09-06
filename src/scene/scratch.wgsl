// 検証用の仮の景色。市松模様の地面と空のグラデーションだけ。
// フェーズ1で丸ごと捨てる前提なので作り込まない。
// ここでの役割は「自己修正ループが崩れを検出できるか」を試すための被写体であること。

struct Camera {
  eye: vec4f,      // xyz = 視点位置
  forward: vec4f,  // xyz = 視線方向（正規化済み）
  right: vec4f,    // xyz = 画面右方向
  up: vec4f,       // xyz = 画面上方向
  params: vec4f,   // x = tan(fov/2), y = アスペクト比, z = 時刻(秒), w = 未使用
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // 画面全体を覆う三角形1枚。頂点バッファは使わない
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VSOut;
  out.pos = vec4f(corners[vi], 0.0, 1.0);
  out.ndc = corners[vi];
  return out;
}

fn skyColor(dir: vec3f) -> vec3f {
  let h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3f(0.62, 0.70, 0.82);
  let zenith = vec3f(0.16, 0.32, 0.62);
  return mix(horizon, zenith, smoothstep(0.5, 1.0, h));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let tanHalfFov = cam.params.x;
  let aspect = cam.params.y;

  let dir = normalize(
    cam.forward.xyz
    + cam.right.xyz * (in.ndc.x * aspect * tanHalfFov)
    + cam.up.xyz * (in.ndc.y * tanHalfFov)
  );

  var color = skyColor(dir);

  // y = 0 の地面と交差するか
  if (dir.y < -1.0e-4 && cam.eye.y > 0.0) {
    let dist = -cam.eye.y / dir.y;
    let hit = cam.eye.xyz + dir * dist;

    // 1m 角の市松。fract(x) = x - floor(x) なので負の座標でも 0.0 / 0.5 に落ちる
    let cell = floor(hit.x) + floor(hit.z);
    let parity = fract(cell * 0.5);
    let dark = vec3f(0.12, 0.13, 0.15);
    let light = vec3f(0.78, 0.76, 0.72);
    let ground = select(light, dark, parity < 0.25);

    // 遠方はエイリアスが暴れて輝度統計を濁すので空へ溶かす
    let fade = clamp(dist / 120.0, 0.0, 1.0);
    color = mix(ground, skyColor(dir), fade * fade);
  }

  return vec4f(color, 1.0);
}
