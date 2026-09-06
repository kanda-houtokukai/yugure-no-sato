// CPU が世界の値を要るときに使う compute。正本（world.wgsl）を直接評価して読み戻す。

@group(0) @binding(0) var<storage, read> qPoints: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> qHeights: array<f32>;

@compute @workgroup_size(64)
fn heightQuery(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&qPoints)) { return; }
  qHeights[i] = terrainHeight(qPoints[i], 0.05);
}
