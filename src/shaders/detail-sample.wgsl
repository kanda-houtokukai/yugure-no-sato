// 焼いた細部タイルの参照（描画側）。group(1) binding 3。周期 8m で繰り返す。

@group(1) @binding(3) var detailTex: texture_2d_array<f32>;

/** 地面の細部による法線の傾き（勾配）。1 タップ */
fn detailGradient(p: vec2f, kind: u32) -> vec2f {
  let uv = p / DETAIL_TILE_M;
  return textureSampleLevel(detailTex, hmSamp, uv, detailLayer(kind), 0.0).rg;
}
