// 仕上げの前処理: 光のにじみ（bloom）。
// HDR を 1/4 に縮小して明るい部分を抜き、横・縦にぼかす。3 パスとも 1/4 解像度なので軽い。

@group(1) @binding(0) var postIn: texture_2d<f32>;
@group(1) @binding(1) var postOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var postSamp: sampler;

/**
 * 明るい部分の抽出と 1/8 縮小。
 * 8×8 の全画素を読むと 64 タップになるので、線形補間のサンプラで 4×4 点だけ読む
 * （1 タップが 2×2 の平均になるので 8×8 を覆える）。
 */
@compute @workgroup_size(8, 8)
fn bloomDown(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(postOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let src = vec2f(textureDimensions(postIn));
  var sum = vec3f(0.0);
  for (var dy = 0u; dy < 4u; dy++) {
    for (var dx = 0u; dx < 4u; dx++) {
      let uv = (vec2f(f32(id.x * 8u + dx * 2u), f32(id.y * 8u + dy * 2u)) + 1.0) / src;
      let c = textureSampleLevel(postIn, postSamp, uv, 0.0).rgb;
      // 閾値は「掛ける」でなく「引く」。重みで掛けると、閾値をわずかに超えた広い空が
      // そのままの明るさで通り、空全体がにじんでしまう（実測）。引き算なら超えた分だけが残る
      sum += max(c * frame.params.y - vec3f(1.1), vec3f(0.0));
    }
  }
  textureStore(postOut, vec2i(id.xy), vec4f(sum / 16.0, 1.0));
}

/**
 * 分離ガウシアン。σ ≈ 4 テクセル（1/4 解像度なので画面上は 16 画素相当）。
 * 向きは uniform でなくエントリポイントで分ける（同じフレーム内で uniform を書き換えると
 * 両方のパスに新しい値が適用されてしまう＝横ぼかしまで縦になる）。
 */
fn blurAlong(id: vec3u, dir: vec2i) {
  let size = textureDimensions(postOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let weights = array<f32, 9>(0.0276, 0.0663, 0.1238, 0.1802, 0.2042, 0.1802, 0.1238, 0.0663, 0.0276);
  var sum = vec3f(0.0);
  for (var i = -4; i <= 4; i++) {
    let q = clamp(vec2i(id.xy) + dir * (i * 3), vec2i(0), vec2i(size) - vec2i(1));
    sum += textureLoad(postIn, q, 0).rgb * weights[i + 4];
  }
  textureStore(postOut, vec2i(id.xy), vec4f(sum, 1.0));
}

@compute @workgroup_size(8, 8)
fn bloomBlurH(@builtin(global_invocation_id) id: vec3u) { blurAlong(id, vec2i(1, 0)); }

@compute @workgroup_size(8, 8)
fn bloomBlurV(@builtin(global_invocation_id) id: vec3u) { blurAlong(id, vec2i(0, 1)); }
