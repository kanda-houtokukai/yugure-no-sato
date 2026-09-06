// 空。段階1では仮のグラデーション＋太陽の円盤。段階2で大気散乱に置き換える。

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

fn skyRadiance(dir: vec3f) -> vec3f {
  let sun = frame.sunDir.xyz;
  let h = clamp(dir.y, -0.05, 1.0);
  let zenith = vec3f(0.10, 0.22, 0.55);
  let horizon = vec3f(0.95, 0.55, 0.30);
  var c = mix(horizon, zenith, smootherstep(0.0, 0.5, h));
  let cosS = dot(dir, sun);
  c += vec3f(1.0, 0.7, 0.4) * pow(max(cosS, 0.0), 24.0) * 0.8;
  if (cosS > frame.sunDir.w) { c += vec3f(20.0, 12.0, 6.0); }
  return c;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let dir = viewRay(in.ndc);
  return vec4f(skyRadiance(dir), 1.0);
}
