// 建物。テンプレートメッシュ（数値から生成）をインスタンス描画する。
// 材質は頂点の kind で分け、板の目地・瓦の列・漆喰のむらは uv（m 単位）から数式で作る。

struct BVSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,      // m 単位
  @location(3) kind: f32,
  @location(4) shade: f32,     // 建物の足元の日向/日陰
};

@vertex
fn vsBuilding(
  @location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f,
  @location(3) kind: f32, @location(4) aux: f32,   // aux は建物では未使用（木と頂点形式を揃えている）
  @location(5) iPos: vec4f, @location(6) iAttr: vec4f,
) -> BVSOut {
  // iPos: xyz = 建つ位置, w = 大きさ。iAttr: x = 向き(rad), y = 予備, z = 予備, w = 種
  let c = cos(iAttr.x);
  let s = sin(iAttr.x);
  let local = position * iPos.w;
  let p = vec3f(local.x * c - local.z * s, local.y, local.x * s + local.z * c);
  let n = vec3f(normal.x * c - normal.z * s, normal.y, normal.x * s + normal.z * c);
  let world = iPos.xyz + p;
  var out: BVSOut;
  out.pos = frame.viewProj * vec4f(world - frame.camPos.xyz, 1.0);
  out.world = world;
  out.normal = n;
  out.uv = uv * iPos.w;
  out.kind = kind;
  out.shade = bakedLight(iPos.xz).shadow;
  // aux は木の揺れ重み用。建物では使わない
  return out;
}

/** 面に沿った適当な接線（法線を面内で振るのに使う） */
fn tangentOf(n: vec3f) -> vec3f {
  let axis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);   // ref は WGSL の予約語
  return normalize(cross(n, axis));
}

/**
 * 茅葺き。分厚い束を葺き重ねた面。毛羽立ちで法線を細かく散らし、光を拡散して吸う。
 * 瓦と違って硬い反射を持たないので、逆光では縁が透けるように明るむ。
 */
fn thatchShade(uv: vec2f, n: ptr<function, vec3f>, grain: f32) -> vec3f {
  // 葺き足（軒と平行の段）。段の位置を段ごとにずらして、規則的な横縞に見せない
  let step = 0.55;
  let jitter = 0.18 * gnoise(vec2f(uv.x * 0.9, floor(uv.y / step)), 826u);
  let row = fract((uv.y + jitter) / step);
  let lip = smootherstep(0.0, 0.14, row) * (1.0 - smootherstep(0.14, 0.45, row));
  // 毛羽立ち: 茅の 1 本ずつ。流れ方向に細長い
  let strand = gnoise(vec2f(uv.x * 130.0, uv.y * 9.0), 821u);
  let strand2 = gnoise(vec2f(uv.x * 46.0, uv.y * 4.0), 825u);
  let fuzz = gnoise(uv * 70.0, 822u);
  let t = tangentOf(*n);
  let b = cross(*n, t);
  *n = normalize(*n
    + t * (strand * 0.55 + strand2 * 0.30 + fuzz * 0.22)
    + b * (lip * -0.22 + fuzz * 0.18 + strand2 * 0.12));
  let tone = gnoise(uv * 1.1, 823u) * 0.5 + 0.5;
  // 経年で褪せた茅。彩度は灰寄りに 1 段だけ（下げすぎると濡れた藁に見える）。
  // 瓦や木との分離は、色でなく「面は沈み、縁が透けて光る」反射の質で付ける
  var col = mix(vec3f(0.070, 0.064, 0.050), vec3f(0.122, 0.112, 0.088), tone * 0.55 + grain * 0.45);
  col *= 0.86 + 0.22 * (1.0 - lip);
  col *= 0.88 + 0.24 * (strand2 * 0.5 + 0.5);
  return col;
}

/** 瓦の凹凸。桟瓦は流れ方向に波打ち、軒と平行に段が並ぶ */
fn tileDetail(uv: vec2f, n: vec3f, tangentX: vec3f, tangentY: vec3f) -> vec3f {
  let wave = TAU / 0.30;            // 波の間隔 30cm
  let row = 0.27;                   // 段の間隔 27cm
  let dx = 0.085 * wave * cos(uv.x * wave);
  // 段: 下端に小さな立ち上がり
  let f = fract(uv.y / row);
  let dy = -0.5 * smootherstep(0.0, 0.12, f) * (1.0 - smootherstep(0.12, 0.3, f));
  return normalize(n + tangentX * dx * 0.12 + tangentY * dy * 0.35);
}

@fragment
fn fsBuilding(in: BVSOut) -> @location(0) vec4f {
  let kind = u32(round(in.kind));
  let toCam = frame.camPos.xyz - in.world;
  let dist = length(toCam);
  let v = toCam / dist;
  var n = select(in.normal, -in.normal, dot(in.normal, v) < 0.0);
  let sun = frame.sunDir.xyz;

  var albedo: vec3f;
  var ao = 1.0;
  // 反射の質を材質ごとに変える。西日を正面から受けると、明るさだけの差では全部オレンジに寄る
  var rough = 1.0;      // 1 = 完全拡散、小さいほど硬く光る
  var thatchRim = 0.0;  // 茅の縁の透け
  var spec = 0.0;       // 鏡面の強さ
  let grain = gnoise(in.uv * 7.0, 811u) * 0.5 + 0.5;

  if (kind == 0u) {          // 木の柱・梁: 経年で黒ずんだ木。繊維方向に細く光る
    albedo = mix(vec3f(0.070, 0.050, 0.034), vec3f(0.125, 0.090, 0.060), grain);
    // 木目: 繊維（uv.y）に沿った筋
    let fiber = gnoise(vec2f(in.uv.x * 26.0, in.uv.y * 2.5), 814u);
    albedo *= 0.80 + 0.30 * (fiber * 0.5 + 0.5);
    n = normalize(n + vec3f(fiber * 0.05, 0.0, 0.0));
    ao = 0.85;
    spec = 0.035; rough = 0.42;
  } else if (kind == 1u) {   // 板壁: 縦板の目地と木目
    let seam = abs(fract(in.uv.x / 0.24) - 0.5) * 2.0;
    let joint = 1.0 - smootherstep(0.80, 1.0, seam);
    let fiber = gnoise(vec2f(in.uv.x * 30.0, in.uv.y * 1.8), 815u);
    albedo = mix(vec3f(0.062, 0.045, 0.030), vec3f(0.108, 0.078, 0.050), grain) * (0.42 + 0.58 * joint);
    albedo *= 0.82 + 0.28 * (fiber * 0.5 + 0.5);
    // 板ごとにわずかに面を振る（平らな一枚板に見せない）
    let plank = floor(in.uv.x / 0.24);
    n = normalize(n + vec3f(0.0, 0.0, 0.0) + tangentOf(n) * (hash1f(i32(plank), 816u) - 0.5) * 0.10);
    ao = 0.88;
    spec = 0.028; rough = 0.5;
  } else if (kind == 2u) {   // 土壁まじりの古い漆喰: むらと鏝の跡。拡散のみ
    let mottle = gnoise(in.uv * 2.0, 812u) * 0.5 + 0.5;
    let trowel = gnoise(in.uv * vec2f(9.0, 5.0), 817u);
    albedo = mix(vec3f(0.185, 0.172, 0.150), vec3f(0.275, 0.258, 0.228), mottle);
    albedo *= 0.88 + 0.18 * (trowel * 0.5 + 0.5);
    // 鏝の跡で面をわずかに波打たせる（平らな板に見せない）
    n = normalize(n + tangentOf(n) * trowel * 0.075 + cross(n, tangentOf(n)) * gnoise(in.uv * vec2f(5.0, 9.0), 818u) * 0.075);
    ao = 0.95;
  } else if (kind == 3u) {   // 瓦: 濃い灰青。硬く光る
    let tx = normalize(cross(n, vec3f(0.0, 0.0, 1.0)) + vec3f(1e-4, 0.0, 0.0));
    let ty = normalize(cross(n, tx));
    n = tileDetail(in.uv, n, tx, ty);
    let tone = gnoise(in.uv * 1.6, 813u) * 0.5 + 0.5;
    albedo = mix(vec3f(0.052, 0.062, 0.084), vec3f(0.095, 0.108, 0.135), tone);
    spec = 0.30; rough = 0.22;
  } else if (kind == 4u) {   // 基礎石・床下の陰
    albedo = mix(vec3f(0.060, 0.056, 0.051), vec3f(0.115, 0.108, 0.100), grain);
    ao = 0.38;
    spec = 0.03; rough = 0.7;
  } else if (kind == 5u) {   // 縁側の床板: 踏まれて磨かれ、木目に沿って光る
    let seam = abs(fract(in.uv.x / 0.20) - 0.5) * 2.0;
    let fiber = gnoise(vec2f(in.uv.x * 24.0, in.uv.y * 2.0), 819u);
    albedo = mix(vec3f(0.115, 0.085, 0.055), vec3f(0.180, 0.135, 0.088), grain) * (0.55 + 0.45 * (1.0 - smootherstep(0.85, 1.0, seam)));
    albedo *= 0.85 + 0.25 * (fiber * 0.5 + 0.5);
    ao = 0.72;
    spec = 0.07; rough = 0.30;
  } else if (kind == 6u) {   // 格子窓: 桟の隙間は室内の闇
    let gx = abs(fract(in.uv.x / 0.11) - 0.5) * 2.0;
    let gy = abs(fract(in.uv.y / 0.30) - 0.5) * 2.0;
    let bar = max(smootherstep(0.55, 0.95, gx), smootherstep(0.72, 0.98, gy));
    albedo = mix(vec3f(0.008, 0.007, 0.006), vec3f(0.095, 0.072, 0.048), bar);
    ao = mix(0.18, 0.8, bar);
    spec = 0.03 * bar; rough = 0.6;
  } else if (kind == 8u) {   // 茅葺き: ざらついて光を吸う（正反射なし）。毛羽立った先端だけ逆光で透ける
    albedo = thatchShade(in.uv, &n, grain);
    ao = 0.9;
    thatchRim = 1.0;
  } else if (kind == 9u) {   // 芝棟: 棟に生えた草
    let g = gnoise(in.uv * 9.0, 824u) * 0.5 + 0.5;
    albedo = mix(vec3f(0.055, 0.085, 0.030), vec3f(0.105, 0.150, 0.055), g);
    n = normalize(n + tangentOf(n) * (g - 0.5) * 0.35);
    ao = 0.75;
  } else if (kind == 10u) {  // 作物の葉: 逆光で透ける。夏の菜園なので濃く若い緑
    let vein = gnoise(in.uv * vec2f(38.0, 7.0), 831u);
    let tone = gnoise(in.uv * 3.0, 832u) * 0.5 + 0.5;
    albedo = mix(vec3f(0.060, 0.125, 0.036), vec3f(0.115, 0.215, 0.062), tone);
    albedo *= 0.86 + 0.24 * (vein * 0.5 + 0.5);
    n = normalize(n + tangentOf(n) * vein * 0.12);
    ao = 0.82;
    thatchRim = 1.6;   // 葉は薄いので、茅より強く透ける
    spec = 0.05; rough = 0.35;
  } else if (kind == 11u) {  // 竹: 淡い黄緑。節が一定間隔で入り、丸い面が硬く光る
    let node = abs(fract(in.uv.y / 0.31) - 0.5) * 2.0;
    let ring = smootherstep(0.86, 1.0, node);
    let tone = gnoise(in.uv * vec2f(2.0, 0.8), 833u) * 0.5 + 0.5;
    albedo = mix(vec3f(0.135, 0.140, 0.070), vec3f(0.215, 0.220, 0.115), tone);
    albedo *= 1.0 - 0.34 * ring;
    ao = 0.9;
    spec = 0.13; rough = 0.26;
  } else if (kind == 12u) {  // 藁・干し草: 乾いて色が抜けた金色。繊維が逆光で光る
    let strand = gnoise(in.uv * vec2f(90.0, 6.0), 834u);
    let tone = gnoise(in.uv * 2.4, 835u) * 0.5 + 0.5;
    albedo = mix(vec3f(0.145, 0.115, 0.058), vec3f(0.235, 0.190, 0.098), tone);
    albedo *= 0.82 + 0.30 * (strand * 0.5 + 0.5);
    n = normalize(n + tangentOf(n) * strand * 0.42 + cross(n, tangentOf(n)) * gnoise(in.uv * 40.0, 836u) * 0.18);
    ao = 0.85;
    thatchRim = 1.2;
  } else if (kind == 13u) {  // 割った薪の木口: 白木。年輪が同心に走る
    let d = length(in.uv - vec2f(0.0, 0.0));
    let ringN = gnoise(vec2f(d * 46.0, 0.0), 837u);
    albedo = mix(vec3f(0.190, 0.155, 0.108), vec3f(0.285, 0.240, 0.172), gnoise(in.uv * 5.0, 838u) * 0.5 + 0.5);
    albedo *= 0.86 + 0.26 * (ringN * 0.5 + 0.5);
    ao = 0.8;
    spec = 0.02; rough = 0.6;
  } else if (kind == 14u) {  // 耕した土: 掘り返して黒い。塊がごろごろしている
    let clump = gnoise(in.uv * 16.0, 839u);
    let coarse = gnoise(in.uv * 4.5, 840u) * 0.5 + 0.5;
    // 掘り返した土。敷地の乾いた土（0.33〜0.46）より暗いが、黒くはしない。
    // 0.19〜0.31 では夕日の斜光（高度 3.5°）で上向きの面が真っ黒に潰れた（実測）
    albedo = mix(vec3f(0.255, 0.200, 0.140), vec3f(0.375, 0.300, 0.215), coarse);
    albedo *= 0.84 + 0.28 * (clump * 0.5 + 0.5);
    n = normalize(n + tangentOf(n) * clump * 0.32 + cross(n, tangentOf(n)) * gnoise(in.uv * 21.0, 841u) * 0.30);
    ao = 0.92;
  } else if (kind == 15u) {  // 農具の刃: 使い込んだ鉄。鈍く硬く光る
    let wear = gnoise(in.uv * 22.0, 842u) * 0.5 + 0.5;
    albedo = mix(vec3f(0.028, 0.026, 0.026), vec3f(0.058, 0.052, 0.048), wear);
    ao = 0.7;
    spec = 0.42; rough = 0.18;
  } else if (kind == 16u) {  // 薪の側面（樹皮）: 柱の黒い木より明るい、乾いた雑木
    let bark = gnoise(in.uv * vec2f(4.0, 34.0), 843u);
    albedo = mix(vec3f(0.105, 0.082, 0.058), vec3f(0.180, 0.142, 0.098), gnoise(in.uv * 3.0, 844u) * 0.5 + 0.5);
    albedo *= 0.80 + 0.34 * (bark * 0.5 + 0.5);
    n = normalize(n + tangentOf(n) * bark * 0.20);
    ao = 0.68;
    spec = 0.02; rough = 0.66;
  } else {                   // 石
    albedo = mix(vec3f(0.145, 0.138, 0.128), vec3f(0.235, 0.225, 0.210), grain);
    ao = 0.8;
    spec = 0.04; rough = 0.65;
  }

  let sunLight = sunLightAt(in.world.y) * in.shade;
  let ndl = max(dot(n, sun), 0.0);
  // 地面からの照り返し。skyAmbient は空だけを見ているので、垂直な壁が黒く沈む。
  // 夕日を受けた土の色を、下や横を向く面ほど強く足す
  let bounce = vec3f(0.34, 0.25, 0.16) * sunLight * 0.045 * (1.0 - max(n.y, 0.0));
  var color = albedo * (ndl * sunLight + skyAmbient(n) * ao + bounce);
  // 茅の縁の透け: 面は沈むが、毛羽立った先端は逆光で光る。
  // 草・葉と同じ理屈（太陽が裏にあり、視線がそちらを向くほど強い）。
  // これで彩度を落としすぎなくても瓦・木材から分離する
  if (thatchRim > 0.0) {
    let back = max(dot(-n, sun), 0.0);
    let toward = pow(max(dot(v, -sun), 0.0), 2.5);
    let edge = pow(1.0 - max(dot(n, v), 0.0), 2.0);   // 視線とすれる面（軒先の断面・けらば）
    let rim = (0.55 * back + 1.35 * toward * back) * (0.35 + 0.65 * edge);
    color += vec3f(0.62, 0.50, 0.30) * rim * sunLight * 0.20;
  }

  // 鏡面: 瓦は硬く光り、木は繊維方向に鈍く光り、漆喰と茅は光らない
  if (spec > 0.0) {
    let h = normalize(sun + v);
    let nh = max(dot(n, h), 0.0);
    let power = 2.0 / max(rough * rough * rough, 1e-3);
    // Blinn-Phong の正規化は (power+8)/(8π)。π を落とすと垂木や棟が白く飛ぶ（実測）
    color += sunLight * spec * pow(nh, power) * (power + 8.0) / (8.0 * PI) * ndl;
  }
  let air = aerialLut(-v, dist);
  color = color * air.transmittance + air.inscatter;
  return vec4f(color, 1.0);
}
