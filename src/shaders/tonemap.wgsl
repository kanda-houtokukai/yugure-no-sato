// HDR（rgba16float）を露出調整して表示形式へ。

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var hdrSamp: sampler;

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

/** 露出→トーンマップ→ガンマ。FXAA はこの結果（表示値）で判断する */
fn display(uv: vec2f) -> vec3f {
  let c = textureSampleLevel(hdr, hdrSamp, uv, 0.0).rgb;
  return pow(acesFitted(c * frame.params.y), vec3f(1.0 / 2.2));
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

/**
 * FXAA（輪郭の平滑化）。MSAA は Chrome/Metal で 1 フレーム 200ms になり使えなかった（実測）ので、
 * 表示値の輝度差から輪郭の向きを推定し、その向きに沿って混ぜる古典的な方法で代える。
 */
@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let res = frame.center.zw;
  let px = 1.0 / res;
  let uv = in.pos.xy * px;
  let rgbM = display(uv);
  let rgbNW = display(uv + vec2f(-1.0, -1.0) * px);
  let rgbNE = display(uv + vec2f(1.0, -1.0) * px);
  let rgbSW = display(uv + vec2f(-1.0, 1.0) * px);
  let rgbSE = display(uv + vec2f(1.0, 1.0) * px);
  let lM = luma(rgbM);
  let lNW = luma(rgbNW); let lNE = luma(rgbNE); let lSW = luma(rgbSW); let lSE = luma(rgbSE);
  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  // 輝度差が小さい平坦部は触らない
  if (lMax - lMin < max(0.04, lMax * 0.125)) { return vec4f(rgbM, 1.0); }

  var dir = vec2f(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
  let dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * (1.0 / 8.0), 1.0 / 128.0);
  let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2f(-8.0), vec2f(8.0)) * px;

  let rgbA = 0.5 * (display(uv + dir * (1.0 / 3.0 - 0.5)) + display(uv + dir * (2.0 / 3.0 - 0.5)));
  let rgbB = rgbA * 0.5 + 0.25 * (display(uv + dir * -0.5) + display(uv + dir * 0.5));
  let lB = luma(rgbB);
  let outRgb = select(rgbB, rgbA, lB < lMin || lB > lMax);
  return vec4f(outRgb, 1.0);
}
