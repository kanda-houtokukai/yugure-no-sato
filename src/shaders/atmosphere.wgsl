// 大気散乱（単一散乱）。Rayleigh + Mie + オゾン吸収。
// 空・遠景の溶け込み・太陽光の色・水面に映る空、すべてこのモデルから出す。
// 単位は m。放射輝度は相対値（露出で調整する）。

const EARTH_R: f32 = 6371000.0;
const ATMO_TOP: f32 = 60000.0;
const RAYLEIGH_H: f32 = 8000.0;
const MIE_H: f32 = 1200.0;
const BETA_R: vec3f = vec3f(5.802e-6, 13.558e-6, 33.1e-6);
// 夏の夕方の靄。標準（3.996e-6）より濃くして、遠景が溶けるようにする
const BETA_M: f32 = 2.3e-5;
const BETA_M_ABS: f32 = 0.5;    // 夏の靄は吸収性（煤・有機物）。直射を落とし、空の白飛びを抑える
const BETA_O: vec3f = vec3f(0.650e-6, 1.881e-6, 0.085e-6);
const MIE_G: f32 = 0.80;
const SUN_E: vec3f = vec3f(1.0, 0.97, 0.94) * 24.0;
// 多重散乱の近似。単一散乱だけだと夕方の空が暗すぎ、直射との比が 28:1 になる（実測）。
// 現実は 3〜6:1。各点で等方に散る二次光を、その点の太陽透過率に比例させて足す
const MULTI_SCATTER: f32 = 2.0;

struct Atmo {
  inscatter: vec3f,
  transmittance: vec3f,
};

/** 谷底に溜まる夏の夕暮れの靄。標高が低いほど濃い（遠景に層を作る） */
const VALLEY_FOG_H: f32 = 75.0;
const VALLEY_FOG_AMOUNT: f32 = 3.0;

fn densities(h: f32) -> vec3f {
  // x = Rayleigh, y = Mie（高層の靄＋谷底の靄）, z = オゾン（25km を中心に幅 30km の三角）
  let hh = max(h, 0.0);
  let mie = exp(-hh / MIE_H) + VALLEY_FOG_AMOUNT * exp(-hh / VALLEY_FOG_H);
  return vec3f(exp(-hh / RAYLEIGH_H), mie, max(0.0, 1.0 - abs(hh - 25000.0) / 15000.0));
}

fn phaseRayleigh(mu: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + mu * mu); }
fn phaseMie(mu: f32) -> f32 {
  let g = MIE_G;
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

/** 球（半径 r）と光線の交点のうち遠い方の距離。交わらなければ -1 */
fn raySphereFar(o: vec3f, d: vec3f, r: f32) -> f32 {
  let b = dot(o, d);
  let c = dot(o, o) - r * r;
  let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  return -b + sqrt(disc);
}
fn raySphereNear(o: vec3f, d: vec3f, r: f32) -> f32 {
  let b = dot(o, d);
  let c = dot(o, o) - r * r;
  let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  let t = -b - sqrt(disc);
  return select(-1.0, t, t > 0.0);
}

/** 惑星中心座標へ。y を高度とみなす（曲率は球で扱う） */
fn toPlanet(world: vec3f) -> vec3f { return vec3f(world.x, EARTH_R + max(world.y, 0.0), world.z); }

/** 点 pos（惑星座標）から太陽方向への透過率 */
fn sunTransmittance(pos: vec3f, sunDir: vec3f, steps: i32) -> vec3f {
  let tEnd = raySphereFar(pos, sunDir, EARTH_R + ATMO_TOP);
  if (tEnd <= 0.0) { return vec3f(0.0); }
  // 地面に遮られるなら 0
  if (raySphereNear(pos, sunDir, EARTH_R) > 0.0) { return vec3f(0.0); }
  var od = vec3f(0.0);
  var tPrev = 0.0;
  for (var i = 1; i <= steps; i++) {
    let f = f32(i) / f32(steps);
    let t = tEnd * f * f;
    let p = pos + sunDir * (0.5 * (t + tPrev));
    let h = length(p) - EARTH_R;
    od += densities(h) * (t - tPrev);
    tPrev = t;
  }
  return exp(-(BETA_R * od.x + BETA_M * (1.0 + BETA_M_ABS) * od.y + BETA_O * od.z));
}

/**
 * 視線に沿った散乱の積分。origin は世界座標、maxDist まで（負なら大気の上端まで）。
 */
fn atmosphereMarch(originWorld: vec3f, dir: vec3f, sunDir: vec3f, maxDist: f32, viewSteps: i32, lightSteps: i32) -> Atmo {
  let o = toPlanet(originWorld);
  var tEnd = raySphereFar(o, dir, EARTH_R + ATMO_TOP);
  let tGround = raySphereNear(o, dir, EARTH_R);
  if (tGround > 0.0) { tEnd = min(tEnd, tGround); }
  if (maxDist > 0.0) { tEnd = min(tEnd, maxDist); }

  let mu = dot(dir, sunDir);
  let pR = phaseRayleigh(mu);
  let pM = phaseMie(mu);

  var od = vec3f(0.0);
  var inscatter = vec3f(0.0);
  var tPrev = 0.0;
  for (var i = 1; i <= viewSteps; i++) {
    let f = f32(i) / f32(viewSteps);
    let t = tEnd * f * f;
    let dt = t - tPrev;
    let p = o + dir * (0.5 * (t + tPrev));
    let h = length(p) - EARTH_R;
    let dens = densities(h);
    od += dens * dt;
    let tView = exp(-(BETA_R * od.x + BETA_M * (1.0 + BETA_M_ABS) * od.y + BETA_O * od.z));
    let tSun = sunTransmittance(p, sunDir, lightSteps);
    let scatter = BETA_R * dens.x * pR + vec3f(BETA_M * dens.y * pM);
    // 二次光は Rayleigh（青）に限る。Mie にも掛けると天頂まで白く濁る（実測）
    let iso = BETA_R * dens.x * (MULTI_SCATTER / (4.0 * PI));
    inscatter += (scatter + iso) * tSun * tView * dt;
    tPrev = t;
  }
  var a: Atmo;
  a.inscatter = inscatter * SUN_E;
  a.transmittance = exp(-(BETA_R * od.x + BETA_M * (1.0 + BETA_M_ABS) * od.y + BETA_O * od.z));
  return a;
}

/** 太陽の円盤（周辺減光つき）。視線と太陽のなす角の cos を渡す */
fn sunDisc(mu: f32, cosRadius: f32) -> f32 {
  if (mu < cosRadius) { return 0.0; }
  // 円盤中心からの相対距離 0..1
  let x = clamp((1.0 - mu) / (1.0 - cosRadius), 0.0, 1.0);
  let limb = 1.0 - 0.6 * (1.0 - sqrt(max(0.0, 1.0 - x * x)));
  return limb;
}
