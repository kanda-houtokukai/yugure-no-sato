// 変形の場の読み戻し（検証用）。任意の世界座標で沈み・倒れ・濁り・波紋を返す。
@group(3) @binding(0) var<storage, read> dqPoints: array<vec2f>;
@group(3) @binding(1) var<storage, read_write> dqOut: array<vec4f>;

@compute @workgroup_size(64)
fn deformQuery(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&dqPoints)) { return; }
  let d = deformAt(dqPoints[id.x]);
  dqOut[id.x] = vec4f(d.sink, d.bend.x, d.bend.y, select(d.turbidity, -1.0, !d.inside));
  // 波紋は 2 本目の要素に（同じ配列の後半）
  dqOut[id.x + arrayLength(&dqPoints)] = vec4f(d.ripple, 0.0, 0.0, 0.0);
}
