// 世界の形の正本。高さ・田んぼ区画・川・あぜ道はすべてここの数式で決まる。
// TypeScript 側には複製しない。CPU が値を要るときは query.wgsl の compute で計算して読み戻す。
//
// 座標: x = 東, z = 北, y = 上。谷は x 方向に長く、川は x に沿って蛇行する。
// 谷座標 (u, v): u = x、v = 川からの横方向距離（z - riverZ(x)）。区画の格子はこの座標で組む。

// ---------- 川 ----------
const RIVER_HALF: f32 = 2.6;     // 水面の半幅 [m]
const RIVER_LEVEL: f32 = -1.0;   // 水面の高さ [m]
const RIVER_BAND: f32 = 6.5;     // 川の帯の半幅。この内側には区画を作らない

fn riverZ(x: f32) -> f32 {
  return 14.0 * sin(x * 0.0213) + 7.0 * sin(x * 0.0517 + 1.7) + 3.0 * sin(x * 0.113 + 0.4);
}
fn toValley(p: vec2f) -> vec2f { return vec2f(p.x, p.y - riverZ(p.x)); }
fn fromValley(uv: vec2f) -> vec2f { return vec2f(uv.x, uv.y + riverZ(uv.x)); }

fn riverBed(v: f32) -> f32 {
  let a = abs(v);
  let q = a / RIVER_HALF;
  let channel = RIVER_LEVEL - 0.7 + 0.7 * q * q;                      // 中央 -1.7、縁で -1.0
  let bank = mix(RIVER_LEVEL, 1.2, smootherstep(RIVER_HALF, RIVER_BAND, a)); // 土手へ上がる
  return select(channel, bank, a > RIVER_HALF);
}

// ---------- 谷底の基準面と盆地の形 ----------
/** 田んぼの高さの基準になる、ごく緩い面 */
fn valleyFloor(uv: vec2f) -> f32 {
  return 0.45 + 0.0055 * abs(uv.y) + 0.0006 * uv.x + 0.35 * gnoise(fromValley(uv) / 140.0, 11u);
}

/** 盆地の縁を 1 とする超楕円距離。北側は南側より広い */
fn basinT(uv: vec2f) -> f32 {
  let wN = 175.0 + 35.0 * gnoise(vec2f(uv.x / 260.0, 3.1), 21u);
  let wS = 130.0 + 30.0 * gnoise(vec2f(uv.x / 230.0, 7.7), 22u);
  let w = select(wS, wN, uv.y > 0.0);
  let l = 520.0 + 70.0 * gnoise(vec2f(uv.y / 210.0, 1.3), 23u);
  let a = abs(uv.y) / w;
  let b = abs(uv.x) / l;
  let a4 = a * a * a * a;
  let b4 = b * b * b * b;
  return sqrt(sqrt(a4 + b4));
}

/** 集落を囲む緩やかな丘。盆地の縁（t=1）から 600m ほどかけて 70m まで上がる */
fn hills(p: vec2f, t: f32, minWl: f32) -> f32 {
  let rise = 70.0 * pow(smootherstep(0.95, 3.6, t), 1.5);
  let bumps = (4.0 + 8.0 * smootherstep(1.0, 2.5, t)) * fbm(p, 5, 180.0, minWl, 31u)
    * smootherstep(0.85, 1.4, t);
  return rise + bumps;
}

/** 山並み。盆地の縁から離れるほど高く、層を重ねて遠景ほど大きい尾根にする */
fn mountains(p: vec2f, t: f32, minWl: f32) -> f32 {
  // 北側の盆地半幅 ≈175m なので、t=6 ≈ 1km、t=14 ≈ 2.4km、t=24 ≈ 4.2km、t=40 ≈ 7km
  let near = smootherstep(6.0, 14.0, t);
  let far = smootherstep(12.0, 24.0, t);
  let vfar = smootherstep(22.0, 40.0, t);
  var h = 0.0;
  if (near > 0.0) { h += near * (80.0 + 300.0 * ridged(p, 5, 1400.0, minWl, 41u)); }
  if (far > 0.0) { h += far * (120.0 + 520.0 * ridged(p + vec2f(3100.0, -1700.0), 5, 4200.0, minWl, 42u)); }
  if (vfar > 0.0) { h += vfar * (200.0 + 800.0 * ridged(p + vec2f(-5100.0, 2700.0), 4, 9000.0, minWl, 43u)); }
  return h;
}

/** 神社の丘（北側、主要なあぜ道の突き当たり）。石段・杉並木はフェーズ2 */
fn shrineHill(uv: vec2f) -> f32 {
  let d = length((uv - vec2f(0.0, 262.0)) / vec2f(75.0, 60.0));
  return 22.0 * (1.0 - smootherstep(0.0, 1.0, d));
}

// ---------- 田んぼの区画 ----------
const SU: f32 = 26.0;   // 縦線（川を横切る向きの線）の間隔 [m]
const SV: f32 = 17.0;   // 横線（川に沿う線）の間隔 [m]

/** 縦線 i の u 座標。低周波でうねる */
fn lineB(i: i32, v: f32) -> f32 {
  return f32(i) * SU + 2.2 * gnoise(vec2f(v / 70.0, f32(i) * 3.17), 51u);
}
/** 横線 j の v 座標。行 0 は川の帯（[-RIVER_BAND, +RIVER_BAND)）で、区画にしない */
fn lineABase(j: i32) -> f32 {
  return select(-RIVER_BAND + f32(j) * SV, RIVER_BAND + f32(j - 1) * SV, j >= 1);
}
fn lineA(j: i32, u: f32) -> f32 {
  return lineABase(j) + 1.6 * gnoise(vec2f(u / 60.0, f32(j) * 2.71), 52u);
}
/** 行 j で縦線 i が存在するか。存在しなければ左右の区画が合併する */
fn existsB(i: i32, j: i32) -> bool {
  return hash2f(vec2i(i, j), 53u) > 0.30;
}

struct Cell {
  i: i32,        // 合併後の左端（アンカー）
  j: i32,
  iEnd: i32,     // 右端の縦線
  isPaddy: bool,
  level: f32,    // 水面の高さ
  ridgeDist: f32,// 最寄りの畦の中心線までの距離
  centerUV: vec2f,
};

fn quantize(x: f32, step: f32) -> f32 { return floor(x / step + 0.5) * step; }

fn resolveCell(uv: vec2f) -> Cell {
  let u = uv.x;
  let v = uv.y;

  // 行 j
  var j: i32;
  if (v >= 0.0) { j = 1 + i32(floor((v - RIVER_BAND) / SV)); } else { j = i32(floor((v + RIVER_BAND) / SV)); }
  if (v < lineA(j, u)) { j -= 1; } else if (v >= lineA(j + 1, u)) { j += 1; }

  // 列 i（ずれを直してから、存在しない縦線を左へ辿って合併）
  var i = i32(floor(u / SU));
  if (u < lineB(i, v)) { i -= 1; } else if (u >= lineB(i + 1, v)) { i += 1; }
  for (var k = 0; k < 4; k++) {
    if (existsB(i, j)) { break; }
    i -= 1;
  }
  var iEnd = i + 1;
  for (var k = 0; k < 4; k++) {
    if (existsB(iEnd, j)) { break; }
    iEnd += 1;
  }

  let uL = lineB(i, v);
  let uR = lineB(iEnd, v);
  let vB = lineA(j, u);
  let vT = lineA(j + 1, u);
  let ridgeDist = min(min(u - uL, uR - u), min(v - vB, vT - v));

  let centerUV = vec2f(0.5 * (f32(i) + f32(iEnd)) * SU, 0.5 * (lineABase(j) + lineABase(j + 1)));
  let inRiverBand = j == 0;
  let inBasin = basinT(centerUV) < 0.93;
  let isPaddy = inBasin && !inRiverBand;

  let base = quantize(valleyFloor(centerUV), 0.12);
  let level = base + (hash2f(vec2i(i, j), 54u) - 0.5) * 0.05;

  var c: Cell;
  c.i = i; c.j = j; c.iEnd = iEnd;
  c.isPaddy = isPaddy;
  c.level = level;
  c.ridgeDist = max(ridgeDist, 0.0);
  c.centerUV = centerUV;
  return c;
}

// ---------- あぜ道 ----------
struct PathHit { dist: f32, halfWidth: f32, top: f32, on: bool };

/** 主要な一本（縦線 0 に沿って集落側＝南から神社側＝北へ）と、横に走る副道 2 本 */
fn nearestPath(uv: vec2f, floorH: f32) -> PathHit {
  var best: PathHit;
  best.dist = 1e9; best.halfWidth = 1.1; best.top = floorH + 0.55; best.on = false;

  // 主道
  if (uv.y > -190.0 && uv.y < 225.0) {
    let d = abs(uv.x - lineB(0, uv.y));
    if (d < best.dist) { best.dist = d; best.halfWidth = 1.1; best.top = floorH + 0.55; }
  }
  // 副道（横線 4 と -3 に沿う）
  if (uv.x > -210.0 && uv.x < 170.0) {
    let d = abs(uv.y - lineA(4, uv.x));
    if (d < best.dist) { best.dist = d; best.halfWidth = 0.8; best.top = floorH + 0.45; }
  }
  if (uv.x > -150.0 && uv.x < 230.0) {
    let d = abs(uv.y - lineA(-3, uv.x));
    if (d < best.dist) { best.dist = d; best.halfWidth = 0.8; best.top = floorH + 0.45; }
  }
  best.on = best.dist < best.halfWidth;
  return best;
}

// ---------- 高さと材質 ----------
const KIND_GROUND: u32 = 0u;
const KIND_MUD: u32 = 1u;
const KIND_RIDGE: u32 = 2u;
const KIND_PATH: u32 = 3u;
const KIND_RIVERBED: u32 = 4u;
const KIND_HILL: u32 = 5u;
const KIND_MOUNTAIN: u32 = 6u;

struct Surface {
  height: f32,
  kind: u32,
  t: f32,        // 盆地距離
  wet: f32,      // 0..1 濡れ（田の泥）
};

fn terrainSurface(p: vec2f, minWl: f32) -> Surface {
  let uv = toValley(p);
  let floorH = valleyFloor(uv);
  let t = basinT(uv);

  var s: Surface;
  s.t = t;
  s.wet = 0.0;
  s.kind = KIND_GROUND;

  var h = floorH;
  if (t > 0.85) { h += hills(p, t, minWl); }
  if (t > 5.9) { h += mountains(p, t, minWl); }
  if (t < 1.6 && abs(uv.x) < 200.0) { h += shrineHill(uv); }
  // 自然地の細かな起伏（近くでは 0.3m の波長まで）
  h += 0.30 * fbm(p, 6, 11.0, minWl, 61u);

  if (t > 5.9) { s.kind = KIND_MOUNTAIN; } else if (t > 1.0) { s.kind = KIND_HILL; }

  // 畦や土手のような細い盛り上がりは、評価する足元（minWl）より細いと網目に拾われて
  // ぎざぎざになる。足元に合わせて幅を広げ、体積が変わらないよう高さを下げる。
  // 画素側は足元が小さいので本来の形で評価され、陰影は鮮明に出る。
  // 盛り上がりがメッシュ間隔（≈ minWl/2）の 3〜4 倍にまたがるまで広げる
  let widen = max(1.0, minWl * 1.75 / 0.8);

  if (t < 1.05) {
    let c = resolveCell(uv);
    if (c.isPaddy) {
      let mud = c.level - 0.12;
      let crest = floorH + 0.42 + 0.06 * gnoise(p / 1.5, 71u);
      let ridge = mud + (crest - mud) / widen * (1.0 - smootherstep(0.0, 0.8 * widen, c.ridgeDist));
      h = max(mud, ridge);
      s.kind = select(KIND_RIDGE, KIND_MUD, c.ridgeDist > 0.75);
      s.wet = select(0.0, 1.0, c.ridgeDist > 0.75);
    }
  }

  // あぜ道は周囲より少し高い土手
  if (t < 1.3) {
    let path = nearestPath(uv, floorH);
    let hw = path.halfWidth * widen;
    let pathH = path.top - (path.top - h) * smootherstep(hw * 0.6, hw + 0.7 * widen, path.dist);
    if (path.dist < hw + 0.7 * widen && pathH > h) {
      h = pathH;
      if (path.on) { s.kind = KIND_PATH; s.wet = 0.0; }
    }
  }

  // 川の掘り込み
  if (abs(uv.y) < RIVER_BAND) {
    let bed = riverBed(uv.y);
    if (bed < h) {
      h = bed;
      s.kind = KIND_RIVERBED;
      s.wet = smootherstep(RIVER_HALF + 0.8, RIVER_HALF - 0.3, abs(uv.y));
    }
  }

  s.height = h;
  return s;
}

fn terrainHeight(p: vec2f, minWl: f32) -> f32 {
  return terrainSurface(p, minWl).height;
}

/** 材質の反射率（アルベド）。夕暮れの逆光では大半が暗く沈むが、日の当たる斜面で効く */
fn terrainAlbedo(p: vec2f, s: Surface) -> vec3f {
  let n = 0.5 + 0.5 * gnoise(p / 3.0, 81u);
  switch (s.kind) {
    case 1u: { return mix(vec3f(0.24, 0.19, 0.13), vec3f(0.31, 0.25, 0.17), n); }        // 泥
    case 2u: { return mix(vec3f(0.22, 0.30, 0.10), vec3f(0.34, 0.36, 0.15), n); }        // 畦の草
    case 3u: { return mix(vec3f(0.46, 0.38, 0.26), vec3f(0.56, 0.47, 0.32), n); }        // 土の道
    case 4u: { return mix(vec3f(0.22, 0.20, 0.16), vec3f(0.32, 0.29, 0.24), n); }        // 川床
    case 5u: { return mix(vec3f(0.12, 0.20, 0.08), vec3f(0.22, 0.30, 0.11), n); }        // 丘（草と林）
    case 6u: { return mix(vec3f(0.09, 0.14, 0.09), vec3f(0.16, 0.20, 0.13), n); }        // 山（林）
    default: { return mix(vec3f(0.20, 0.30, 0.09), vec3f(0.32, 0.38, 0.14), n); }        // 草地
  }
}
