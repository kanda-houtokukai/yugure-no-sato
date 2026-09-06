// HDR（rgba16float）を露出調整して表示形式へ。

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;

struct VSOut { @builtin(position) pos: vec4f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VSOut;
  out.pos = vec4f(corners[vi], 0.0, 1.0);
  return out;
}

fn acesFitted(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2i(in.pos.xy), 0).rgb;
  let exposed = c * frame.params.y;
  let mapped = acesFitted(exposed);
  // 出力先は bgra8unorm（sRGB 変換なし）なので、ここでガンマを掛ける
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
