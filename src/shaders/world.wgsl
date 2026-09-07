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

// ---------- 峠 ----------
// 里の北東を尾根が塞ぎ、一点だけ低い鞍部がある。そこへ九十九折りの道を上げる。
// 丘（t=0.95→3.6 で 70m）は勾配 9° 程度で緩く、そのままでは折り返す理由が生まれない。
// 尾根を足して斜面を 20° 級にすることで、九十九折りが必然になる。
const PASS_U: f32 = 210.0;    // 鞍部の u（東西）
const PASS_V: f32 = 395.0;    // 尾根の稜線の v（川からの距離）

fn passRidge(uv: vec2f) -> f32 {
  let d = (uv.y - PASS_V) / 150.0;
  let ridge = exp(-d * d * 1.6);
  // 鞍部: u = PASS_U のあたりだけ低い
  let sN = (uv.x - PASS_U) / 120.0;
  let notch = 1.0 - 0.62 * exp(-sN * sN * 1.4);
  return 78.0 * ridge * notch;
}

/**
 * 細かな起伏を含まない、なめらかな地面。峠道の路面の高さに使う。
 * 実際の地面（fbm の凹凸つき）を路面にすると、道が波打って山道に見えない。
 */
fn smoothLand(uv: vec2f) -> f32 {
  let t = basinT(uv);
  let outlet = smootherstep(-320.0, -620.0, uv.x) * (1.0 - smootherstep(110.0, 260.0, abs(uv.y)));
  var h = valleyFloor(uv);
  if (t > 0.85) { h += 70.0 * pow(smootherstep(0.95, 3.6, t), 1.5) * (1.0 - 0.7 * outlet); }
  h += passRidge(uv);
  return h;
}

/** 線分までの距離。任意の向きの道に使う（lineA / lineB の格子に乗らないもの） */
fn segDist(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let ap = p - a;
  let hh = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
  return length(ap - ab * hh);
}

/** 峠道の折り返しの点（谷座標）。0 = 麓、5 = 鞍部の見晴らし場 */
fn passNode(i: i32) -> vec2f {
  switch (i) {
    case 0: { return vec2f(150.0, 235.0); }
    case 1: { return vec2f(268.0, 268.0); }
    case 2: { return vec2f(152.0, 300.0); }
    case 3: { return vec2f(272.0, 330.0); }
    case 4: { return vec2f(160.0, 358.0); }
    default: { return vec2f(210.0, 384.0); }
  }
}

struct PassRoad { dist: f32, h: f32, t: f32 };

/**
 * 峠道。折れ線までの距離と、そこでの路面の高さを返す。
 *
 * 路面の高さを「近くの地面」から取ると、尾根の鞍部の形をなぞって道が波打つ（実測）。
 * 麓と鞍部の高さを両端に取り、道のりに比例して上げる＝一定勾配にする。これが人が付けた道の形。
 * 幅の方向には水平にならす（切土・盛土）ので、斜面に埋まることも浮くこともない。
 */
fn passRoad(uv: vec2f) -> PassRoad {
  var out: PassRoad;
  out.dist = 1e9; out.t = 0.0;
  // 各区間の長さを先に足して、道のり全体を出す
  var total = 0.0;
  for (var i = 0; i < 5; i++) { total += length(passNode(i + 1) - passNode(i)); }
  var acc = 0.0;
  for (var i = 0; i < 5; i++) {
    let a = passNode(i);
    let b = passNode(i + 1);
    let ab = b - a;
    let ap = uv - a;
    let hh = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    let d = length(ap - ab * hh);
    if (d < out.dist) {
      out.dist = d;
      out.t = (acc + hh * length(ab)) / total;
    }
    acc += length(ab);
  }
  let hFoot = smoothLand(passNode(0));
  let hTop = smoothLand(passNode(5));
  out.h = mix(hFoot, hTop, out.t) + 0.10;
  return out;
}

/** 神社の丘（北側、主要なあぜ道の突き当たり）。石段・杉並木はフェーズ2 */
fn shrineHill(uv: vec2f) -> f32 {
  let d = length((uv - vec2f(0.0, 262.0)) / vec2f(75.0, 60.0));
  return 22.0 * (1.0 - smootherstep(0.0, 1.0, d));
}

// ---------- 整地された平地（建物を置く土地） ----------
// 集落の敷地と神社の境内。田より一段高く、水はけのため平坦。ここに田・稲は来ない。
// 主道と参道が貫くので、家屋・鳥居・石段（フェーズ5）を置く場所になる。

const YARD_C: vec2f = vec2f(0.0, -132.0);       // 集落の敷地の中心（谷座標）
const YARD_H: vec2f = vec2f(66.0, 26.0);        // 半幅
const YARD_FEATHER: f32 = 7.5;                  // 縁の傾斜の幅 [m]。広すぎると田との段差（0.85m）が読めない
const PRECINCT_C: vec2f = vec2f(0.0, 270.0);    // 神社の境内
const PRECINCT_H: vec2f = vec2f(34.0, 24.0);
const PRECINCT_FEATHER: f32 = 8.0;              // 石段が上がる斜面の幅
// 峠の見晴らし場。上りきったところに立ち止まる理由を置く（東屋・大石）
const LOOKOUT_C: vec2f = vec2f(214.0, 379.0);
const LOOKOUT_H: vec2f = vec2f(10.0, 7.0);
const LOOKOUT_FEATHER: f32 = 6.0;

/** 角丸の矩形の内外。1 = 内側、0 = 外側。縁は低周波ノイズで乱して人工的な直線を避ける */
fn flatMask(uv: vec2f, center: vec2f, half: vec2f, feather: f32, seed: u32) -> f32 {
  let d = abs(uv - center) - half;
  let outside = length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
  // 縁を乱す（直線的な造成に見せない）。大小 2 つの波を重ねる
  let wobble = 4.2 * gnoise(uv / 21.0, seed) + 1.4 * gnoise(uv / 7.0, seed + 5u);
  return 1.0 - smootherstep(-feather, feather, outside + wobble);
}

/** 敷地の高さ: 隣の田より約 1m 高い平場 */
fn yardLevel() -> f32 { return valleyFloor(YARD_C) + 0.85; }

/** 峠の見晴らし場の高さ: 峠道の路面と同じ高さ（道から段差なく入れる） */
fn lookoutLevel() -> f32 { return smoothLand(passNode(5)) + 0.12; }

/** 境内の高さ: 丘の頂をならした平場 */
fn precinctLevel() -> f32 {
  let p = fromValley(PRECINCT_C);
  return valleyFloor(PRECINCT_C) + hills(p, basinT(PRECINCT_C), 8.0) + shrineHill(PRECINCT_C) + 0.25;
}

struct FlatSite { w: f32, level: f32, kind: u32 };

fn flatSite(uv: vec2f) -> FlatSite {
  var f: FlatSite;
  f.w = 0.0; f.level = 0.0; f.kind = KIND_GROUND;
  let wy = flatMask(uv, YARD_C, YARD_H, YARD_FEATHER, 131u);
  if (wy > 0.0) { f.w = wy; f.level = yardLevel(); f.kind = KIND_YARD; }
  let wp = flatMask(uv, PRECINCT_C, PRECINCT_H, PRECINCT_FEATHER, 132u);
  if (wp > f.w) { f.w = wp; f.level = precinctLevel(); f.kind = KIND_PRECINCT; }
  let wl = flatMask(uv, LOOKOUT_C, LOOKOUT_H, LOOKOUT_FEATHER, 133u);
  if (wl > f.w) { f.w = wl; f.level = lookoutLevel(); f.kind = KIND_TRAIL; }
  return f;
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
  // 整地した敷地・境内には田を作らない（水が敷地に流れ込まないよう、縁にも余裕を取る）
  let onFlat = flatSite(centerUV).w > 0.15;
  let isPaddy = inBasin && !inRiverBand && !onFlat;

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

/**
 * あぜ道の網目。区画の線に沿って通す（畦そのものを踏み固めた道）。
 * - 主道: 縦線 0 に沿って集落側（南）から神社の丘（北）へ抜ける一本。川は浅瀬で渡る（橋はフェーズ2）
 * - 副道: 横線 4 と -3 に沿って谷を東西に走る 2 本
 * - 枝道: 縦線 -4 と 5 に沿って副道と主道をつなぐ 2 本
 */
fn considerPath(best: ptr<function, PathHit>, d: f32, halfWidth: f32, top: f32) {
  if (d < (*best).dist) { (*best).dist = d; (*best).halfWidth = halfWidth; (*best).top = top; }
}

fn nearestPath(uv: vec2f, floorH: f32) -> PathHit {
  var best: PathHit;
  best.dist = 1e9; best.halfWidth = 1.1; best.top = floorH + 0.55; best.on = false;

  // 主道。境内（v≈246〜294）まで通す。途中で切ると、石段の斜面が草地のままになる
  if (uv.y > -190.0 && uv.y < 288.0) {
    considerPath(&best, abs(uv.x - lineB(0, uv.y)), 1.1, floorH + 0.55);
  }
  // 副道（横線 4 と -3 に沿う）
  if (uv.x > -210.0 && uv.x < 170.0) {
    considerPath(&best, abs(uv.y - lineA(4, uv.x)), 0.8, floorH + 0.45);
  }
  if (uv.x > -150.0 && uv.x < 230.0) {
    considerPath(&best, abs(uv.y - lineA(-3, uv.x)), 0.8, floorH + 0.45);
  }
  // 枝道（縦線 -4 と 5 に沿う。川の帯は跨がない）
  if (uv.y > -110.0 && uv.y < -RIVER_BAND + 1.0) {
    considerPath(&best, abs(uv.x - lineB(-4, uv.y)), 0.7, floorH + 0.42);
  }
  if (uv.y > RIVER_BAND - 1.0 && uv.y < 150.0) {
    considerPath(&best, abs(uv.x - lineB(5, uv.y)), 0.7, floorH + 0.42);
  }
  // 峠へ向かう枝道。主道（u≈0）の v=150 あたりから北東へ、峠の麓（150, 235）まで
  if (uv.x > -10.0 && uv.x < 165.0 && uv.y > 140.0 && uv.y < 250.0) {
    considerPath(&best, segDist(uv, vec2f(4.0, 150.0), vec2f(150.0, 235.0)), 0.85, floorH + 0.45);
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
const KIND_YARD: u32 = 7u;        // 集落の敷地（踏み固められた土）
const KIND_PRECINCT: u32 = 8u;    // 神社の境内（玉砂利まじりの土）
const KIND_STEPS: u32 = 9u;       // 石段（境内へ上がる斜面）
const KIND_TRAIL: u32 = 10u;      // 山道（石まじりの踏み固めた土）

struct Surface {
  height: f32,
  kind: u32,
  t: f32,        // 盆地距離
  wet: f32,      // 0..1 濡れ（田の泥）
  ridgeDist: f32,// 畦・道の中心線までの距離（田の外では 100）
};

fn terrainSurface(p: vec2f, minWl: f32) -> Surface {
  let uv = toValley(p);
  let floorH = valleyFloor(uv);
  let t = basinT(uv);

  var s: Surface;
  s.t = t;
  s.wet = 0.0;
  s.kind = KIND_GROUND;
  s.ridgeDist = 100.0;

  // 谷の出口: 川が西へ抜ける切れ目。ここを低くして、そこに夕日が沈むようにする
  // （盆地周りの丘 70m・山 400m は仰角 3.5° の太陽を隠してしまう。計算は台帳参照）
  let outlet = smootherstep(-320.0, -620.0, uv.x) * (1.0 - smootherstep(110.0, 260.0, abs(uv.y)));

  var h = floorH;
  if (t > 0.85) { h += hills(p, t, minWl) * (1.0 - 0.7 * outlet); }
  h += passRidge(uv);   // 峠の尾根
  if (t > 5.9) { h += mountains(p, t, minWl) * (1.0 - 0.85 * outlet); }
  if (t < 1.6 && abs(uv.x) < 200.0) { h += shrineHill(uv); }
  // 自然地の細かな起伏（近くでは 0.3m の波長まで）
  h += 0.30 * fbm(p, 6, 11.0, minWl, 61u);

  if (t > 5.9) { s.kind = KIND_MOUNTAIN; } else if (t > 1.0) { s.kind = KIND_HILL; }

  // 畦や土手のような細い盛り上がりは、評価する足元（minWl）より細いと網目に拾われて
  // ぎざぎざになる。足元に合わせて幅を広げ、体積が変わらないよう高さを下げる。
  // 画素側は足元が小さいので本来の形で評価され、陰影は鮮明に出る。
  // 畦や土手のような細い盛り上がりは、評価する足元（minWl ≈ メッシュ間隔の 2 倍）より細いと
  // 網目に拾われて鋸歯になる。足元に応じて幅を広げる（上限 3 倍。それ以上広げると区画の内部まで
  // 持ち上がって水面が地形に潜る＝bird 視点で水が消えた実測）。さらに遠くでは幾何ごと消す。
  // 遠くの畦の網目は、焼いた ridgeDist を水面が discard することで保つ。
  let widen = clamp(minWl * 2.5 / 0.8, 1.0, 3.0);
  let bumpScale = 1.0 - smootherstep(3.0, 6.0, minWl);

  if (t < 1.05) {
    let c = resolveCell(uv);
    if (c.isPaddy) {
      let mud = c.level - 0.12;
      let crest = floorH + 0.42 + 0.06 * gnoise(p / 1.5, 71u);
      let ridge = mud + (crest - mud) * bumpScale * (1.0 - smootherstep(0.0, 0.8 * widen, c.ridgeDist));
      h = max(mud, ridge);
      s.kind = select(KIND_RIDGE, KIND_MUD, c.ridgeDist > 0.75);
      s.wet = select(0.0, 1.0, c.ridgeDist > 0.75);
      s.ridgeDist = c.ridgeDist;
    }
  }

  // 整地: 自然な高さを平場へ寄せる。田の判定の後、道の前（道は敷地の上を通る）
  let site = flatSite(uv);
  if (site.w > 0.0) {
    // 完全な平面にはしない。踏み固めた土のわずかな起伏を残す
    let siteH = site.level + 0.06 * fbm(p, 3, 6.0, minWl, 141u);
    h = mix(h, siteH, site.w);
    if (site.w > 0.35) {
      s.kind = site.kind;
      s.wet = 0.0;
      s.ridgeDist = 100.0;
    }
  }

  // あぜ道は周囲より少し高い土手。敷地の上では敷地の高さを基準にする
  // （基準を谷底のままにすると、一段高い敷地に道が埋もれて消える）
  // その場の地面の高さを下回らせない。谷底基準のままだと、丘を上る参道で
  // 道の高さが地面より低くなり、道が一切現れない（実測）
  let pathBase = max(mix(floorH, site.level, site.w), h);
  // 参道は盆地の外（t > 1.3）へ出るので、その範囲も通す。
  // 条件を t < 1.3 だけにすると、境内へ上がる斜面が草地のままになる
  let onApproach = abs(uv.x) < 22.0 && uv.y > 140.0 && uv.y < 292.0;
  if (t < 1.3 || onApproach) {
    let path = nearestPath(uv, pathBase);
    let hw = path.halfWidth * widen;
    // 道の断面は場所で変える。田の間は土手（周囲より高い）、集落の敷地と境内の中は地面と同面。
    // 土手のまま集落を貫くと道が堤防になり、家がその下に沈んで見える（実機の指摘）。
    // site.w は敷地の縁で滑らかに 0→1 になるので、移行も自動的に滑らか
    let riseScale = 1.0 - site.w * 0.94;
    let rise = (path.top - h) * riseScale;
    let pathH = h + rise * bumpScale * (1.0 - smootherstep(hw * 0.6, hw + 0.7 * widen, path.dist));
    if (path.dist < hw + 0.7 * widen) {
      h = max(h, pathH);
      // 道の上は土。土手の斜面（道の縁から 0.7m）は草だが、同面になる敷地の中では作らない
      if (path.on) {
        s.kind = KIND_PATH; s.wet = 0.0;
      } else if (site.w < 0.45) {
        s.kind = KIND_RIDGE; s.wet = 0.0;
      }
    }
    s.ridgeDist = min(s.ridgeDist, path.dist);
  }

  // 峠道: 斜面を水平に削って段にする（切土・盛土）。土手にすると 20° の斜面では
  // 片側が宙に浮き、反対側が山に埋まる。折り返しの道は「段」でなければ成立しない
  if (uv.y > 200.0 && uv.y < 430.0 && uv.x > 100.0 && uv.x < 330.0) {
    let road = passRoad(uv);
    if (road.dist < 8.5) {
      let w = 1.0 - smootherstep(3.0, 8.5, road.dist);
      h = mix(h, road.h, w);
      if (road.dist < 2.4) { s.kind = KIND_TRAIL; s.wet = 0.0; }
      s.ridgeDist = min(s.ridgeDist, road.dist);
    }
  }

  // 石段: 境内へ上がる斜面（v=240〜254）で、参道の高さを段状に量子化する。
  // 別メッシュで載せると、道の土手や地形の S 字と競合して埋まる（実測で解決できなかった）。
  // 地形そのものを段にすれば、原理的に埋まりも浮きもしない
  if (uv.y > 239.0 && uv.y < 255.0) {
    let dx = abs(uv.x - lineB(0, uv.y));
    if (dx < 2.6) {
      let stepH = 0.33;
      let stepped = floor(h / stepH) * stepH + 0.04;
      // 縁は滑らかに地形へ戻す。縁石を max で足すと、地形が低い側で壁のようにそびえる（実測）
      let core = 1.0 - smootherstep(1.5, 2.3, dx);
      h = mix(h, stepped, core);
      if (dx < 2.1) { s.kind = KIND_STEPS; s.wet = 0.0; s.ridgeDist = 100.0; }
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
    case 7u: { return mix(vec3f(0.33, 0.28, 0.21), vec3f(0.46, 0.40, 0.30), n); }        // 集落の敷地（乾いた踏み固めの土。田の泥と分ける）
    case 8u: { return mix(vec3f(0.33, 0.31, 0.27), vec3f(0.45, 0.42, 0.37), n); }        // 境内（玉砂利まじり）
    case 9u: { return mix(vec3f(0.22, 0.215, 0.20), vec3f(0.34, 0.33, 0.31), n); }       // 石段
    case 10u: {                                                                          // 山道（石まじり）
      let stone = 0.5 + 0.5 * gnoise(p * 1.6, 83u);
      return mix(vec3f(0.34, 0.30, 0.24), vec3f(0.50, 0.45, 0.37), n) * (0.86 + 0.26 * stone);
    }
    default: { return mix(vec3f(0.20, 0.30, 0.09), vec3f(0.32, 0.38, 0.14), n); }        // 草地
  }
}
