// CPU が世界の値を要るときに使う compute。正本（world.wgsl）を直接評価して読み戻す。

@group(0) @binding(0) var<storage, read> qPoints: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> qHeights: array<f32>;

@compute @workgroup_size(64)
fn heightQuery(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&qPoints)) { return; }
  qHeights[i] = terrainHeight(qPoints[i], 0.05);
}

// ---- 区画の形の問い合わせ（水面メッシュ用） ----
// 格子索引 (i, j) ごとに、その区画が「合併の左端（アンカー）かつ田んぼ」なら
// 水面高さと 9×3 の格子点（世界 xz）を返す。CPU はこれをそのまま三角形に組む。
// 1 区画 = 64 floats: [0] flags(1=水面あり) [1] level [2..55] 27 点の (x,z)

struct CellRange { i0: i32, j0: i32, ni: i32, nj: i32 };
@group(0) @binding(2) var<uniform> cellRange: CellRange;
@group(0) @binding(3) var<storage, read_write> cellOut: array<f32>;

const CELL_STRIDE: u32 = 64u;
const GRID_U: i32 = 9;
const GRID_V: i32 = 3;

@compute @workgroup_size(64)
fn cellQuery(@builtin(global_invocation_id) id: vec3u) {
  let n = u32(cellRange.ni * cellRange.nj);
  if (id.x >= n) { return; }
  let i = cellRange.i0 + i32(id.x) % cellRange.ni;
  let j = cellRange.j0 + i32(id.x) / cellRange.ni;
  let base = id.x * CELL_STRIDE;

  cellOut[base] = 0.0;
  if (!existsB(i, j) || j == 0) { return; }
  var iEnd = i + 1;
  for (var k = 0; k < 4; k++) {
    if (existsB(iEnd, j)) { break; }
    iEnd += 1;
  }
  // 区画の中心を正本で解決し、この (i, j) がアンカーである田んぼだけ採用
  let centerUV = vec2f(0.5 * (f32(i) + f32(iEnd)) * SU, 0.5 * (lineABase(j) + lineABase(j + 1)));
  let c = resolveCell(centerUV);
  if (!c.isPaddy || c.i != i || c.j != j) { return; }

  cellOut[base] = 1.0;
  cellOut[base + 1u] = c.level;
  for (var t = 0; t < GRID_V; t++) {
    for (var s = 0; s < GRID_U; s++) {
      let fs = f32(s) / f32(GRID_U - 1);
      let ft = f32(t) / f32(GRID_V - 1);
      // u は v に、v は u に依存するので 3 回の不動点反復で落ち着かせる
      var v = mix(lineABase(j), lineABase(j + 1), ft);
      var u = mix(f32(i) * SU, f32(iEnd) * SU, fs);
      for (var k = 0; k < 3; k++) {
        u = mix(lineB(i, v), lineB(iEnd, v), fs);
        v = mix(lineA(j, u), lineA(j + 1, u), ft);
      }
      let w = fromValley(vec2f(u, v));
      let o = base + 2u + u32(t * GRID_U + s) * 2u;
      cellOut[o] = w.x;
      cellOut[o + 1u] = w.y;
    }
  }
}

// ---- 視点の吸着（畦や道の上に立たせる） ----
// 入力 (family, index, x, z, offset)。family 0 = 縦線 B（x を線に合わせる）、1 = 横線 A（z を線に合わせる）
// 出力 (x, z)
@group(0) @binding(4) var<storage, read> snapIn: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> snapOut: array<vec2f>;
@group(0) @binding(8) var<storage, read> snapOffset: array<f32>;

@compute @workgroup_size(64)
fn snapQuery(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&snapIn)) { return; }
  let q = snapIn[id.x];
  let index = i32(q.y);
  var x = q.z;
  var z = q.w;
  let offset = snapOffset[id.x];
  if (q.x < 0.5) {
    // 縦線: u = lineB(index, v)、v は x に依存するので反復
    for (var k = 0; k < 3; k++) {
      let v = z - riverZ(x);
      x = lineB(index, v) + offset;
    }
  } else {
    let v = lineA(index, x) + offset;
    z = v + riverZ(x);
  }
  snapOut[id.x] = vec2f(x, z);
}

// ---- 川の中心線（水面の帯用） ----
@group(0) @binding(6) var<storage, read> riverX: array<f32>;
@group(0) @binding(7) var<storage, read_write> riverZOut: array<f32>;

@compute @workgroup_size(64)
fn riverQuery(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&riverX)) { return; }
  riverZOut[id.x] = riverZ(riverX[id.x]);
}
