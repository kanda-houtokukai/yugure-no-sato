// 高さテクスチャの生成。正本（world.wgsl）を起動時に評価して、近・中・遠の3段に焼く。
// 画素ごとの法線はこのテクスチャから取る。数式を画素ごとに3回評価すると GPU の占有率が
// 崖を越えて 30ms 以上かかる（フェーズ1で実測）ため、評価は焼く時の1回に集約する。

struct LevelParams {
  origin: vec2f,   // テクスチャ (0,0) の世界 xz
  texel: f32,      // 1テクセルの大きさ [m]
  size: f32,       // 1辺のテクセル数
};

@group(0) @binding(0) var<uniform> level: LevelParams;
@group(0) @binding(1) var outHeight: texture_storage_2d_array<r32float, write>;
@group(0) @binding(2) var<uniform> layer: u32;

@compute @workgroup_size(8, 8)
fn fillLevel(@builtin(global_invocation_id) id: vec3u) {
  let n = u32(level.size);
  if (id.x >= n || id.y >= n) { return; }
  let p = level.origin + (vec2f(f32(id.x), f32(id.y)) + 0.5) * level.texel;
  // 足元（minWl）をテクセルに合わせ、テクセルより細かい起伏は落とす（焼き込み時のエイリアス防止）
  let h = terrainHeight(p, level.texel * 2.0);
  textureStore(outHeight, vec2i(i32(id.x), i32(id.y)), i32(layer), vec4f(h, 0.0, 0.0, 0.0));
}
