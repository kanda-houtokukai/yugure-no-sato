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
// x = 書き込む層、y = 行の開始位置（焼き直しを数フレームに分けるため）
@group(0) @binding(2) var<uniform> layer: vec4u;

@compute @workgroup_size(8, 8)
fn fillLevel(@builtin(global_invocation_id) gid: vec3u) {
  let id = vec3u(gid.x, gid.y + layer.y, gid.z);
  let n = u32(level.size);
  if (id.x >= n || id.y >= n) { return; }
  let p = level.origin + (vec2f(f32(id.x), f32(id.y)) + 0.5) * level.texel;
  // 足元（minWl）をテクセルに合わせ、テクセルより細かい起伏は落とす（焼き込み時のエイリアス防止）
  let h = terrainHeight(p, level.texel * 2.0);
  textureStore(outHeight, vec2i(i32(id.x), i32(id.y)), i32(layer.x), vec4f(h, 0.0, 0.0, 0.0));
}

// ---- 法線と日向/日陰の焼き込み ----
// 高さテクスチャ（group 1）を読んで、法線（rg16float: x,z）と太陽の可視度（r8unorm）を段ごとに書く。
// 画素側はそれぞれ 1 タップで済む。太陽は固定なので影は起動時の 1 回でよい。

// x = 書き込む層、y = 行の開始位置、z = 高さを読む層、w = 1 なら z の層を明示して読む
// （w = 0 は起動時の焼き込み。点を含む一番細かい段から法線を取る従来どおりの経路）
@group(2) @binding(0) var<uniform> bakeLayer: vec4u;
// rg16float / r8unorm は storage 書き込み非対応（検証エラーで判明）。rgba16float 1 枚に
// 法線 xz と太陽の可視度をまとめる
@group(2) @binding(1) var outLight: texture_storage_2d_array<rgba16float, write>;
// 材質: rgb = アルベド, a = 種別/8。画素ごとの数式評価（DPR2 で 24ms）をなくす
@group(2) @binding(2) var outMaterial: texture_storage_2d_array<rgba8unorm, write>;
// 焼く段の原点・テクセル・辺の長さ。hmLevels を見ないのは、焼き直し中の近景段が
// まだ hmLevels に載っていないから（載せるのは焼き終えてから）
@group(2) @binding(3) var<uniform> bakeLevel: vec4f;

@compute @workgroup_size(8, 8)
fn bakeNormalShadow(@builtin(global_invocation_id) gid: vec3u) {
  let id = vec3u(gid.x, gid.y + bakeLayer.y, gid.z);
  // 段の原点は uniform で明示的に渡す。焼き直し中の近景段は hmLevels にまだ載っていない
  let lv = bakeLevel;
  let n = u32(lv.w);
  if (id.x >= n || id.y >= n) { return; }
  let p = lv.xy + (vec2f(f32(id.x), f32(id.y)) + 0.5) * lv.z;
  let src = i32(bakeLayer.z);
  var nrm: vec3f;
  var h: f32;
  if (bakeLayer.w == 1u) {
    // 近景段の焼き直し。まだ hmLevels に載っていない層を明示して読む
    nrm = sampleNormalAt(src, lv, p, lv.z);
    h = sampleHeightAt(src, lv, p);
  } else {
    // 起動時。中景・遠景の法線は「点を含む一番細かい段」から取る（フェーズ1からの挙動）
    nrm = sampleNormal(p, lv.z);
    h = sampleHeight(src, p);
  }
  let shade = terrainShadow(vec3f(p.x, h, p.y), frame.sunDir.xyz);
  // a = 畦・道の中心線までの距離。水面がその線上を discard して、遠くでも区画の網目を保つ
  let surf = terrainSurface(p, lv.z * 2.0);
  textureStore(outLight, vec2i(id.xy), i32(bakeLayer.x), vec4f(nrm.x, nrm.z, shade, min(surf.ridgeDist, 50.0)));
  textureStore(outMaterial, vec2i(id.xy), i32(bakeLayer.x), vec4f(terrainAlbedo(p, surf), f32(surf.kind) / 16.0));
}
