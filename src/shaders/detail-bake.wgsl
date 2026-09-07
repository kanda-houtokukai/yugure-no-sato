// 細部タイルの焼き込み（起動時に 1 回）。rg = 勾配（d/dx, d/dz）。

@group(0) @binding(0) var<uniform> bakeDetailLayer: u32;
@group(0) @binding(1) var outDetail: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bakeDetail(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(outDetail);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let n = f32(size.x);
  let texel = DETAIL_TILE_M / n;
  let p = (vec2f(f32(id.x), f32(id.y)) + 0.5) * texel;
  let layer = i32(bakeDetailLayer);
  let e = texel;
  let b0 = detailBumpTile(p, layer);
  let bx = detailBumpTile(p + vec2f(e, 0.0), layer);
  let bz = detailBumpTile(p + vec2f(0.0, e), layer);
  textureStore(outDetail, vec2i(id.xy), layer, vec4f((b0 - bx) / e, (b0 - bz) / e, 0.0, 0.0));
}
