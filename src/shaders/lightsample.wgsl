// 焼いた法線・日向/日陰の参照（描画側）。group(3) に束ねる。rgba16float: r,g = 法線 xz、b = 太陽の可視度、a = 畦・道の中心線までの距離

@group(3) @binding(0) var lightTex: texture_2d_array<f32>;
@group(3) @binding(1) var matTex: texture_2d_array<f32>;

struct BakedLight { normal: vec3f, shadow: f32, ridgeDist: f32, albedo: vec3f, kind: u32 };

fn bakedLight(p: vec2f) -> BakedLight {
  let li = heightLevelFor(p);
  let lv = hmLevels.l[li];
  let uv = (p - lv.xy) / (lv.z * lv.w);
  let t = textureSampleLevel(lightTex, hmSamp, uv, li, 0.0);
  let m = textureSampleLevel(matTex, hmSamp, uv, li, 0.0);
  let y = sqrt(max(0.0, 1.0 - dot(t.rg, t.rg)));
  var out: BakedLight;
  out.normal = normalize(vec3f(t.r, y, t.g));
  out.shadow = t.b;
  out.ridgeDist = t.a;
  out.albedo = m.rgb;
  out.kind = u32(round(m.a * 10.0));
  return out;
}
