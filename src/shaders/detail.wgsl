// 地面の細部の凹凸を、材質ごとのタイル状の法線テクスチャに焼く。
// 画素ごとに gnoise を 6 回評価すると 1.4ms 掛かる（実測）。8m 周期のタイルに焼けば 1 タップで済む。
// 周期は 8m。2 種の周波数が混ざっているので、繰り返しは目立ちにくい。

const DETAIL_TILE_M: f32 = 8.0;   // タイルが覆う世界の大きさ [m]

/** 材質の種別 → タイルの層 */
fn detailLayer(kind: u32) -> i32 {
  switch (kind) {
    case 1u: { return 1; }              // 田の泥
    case 3u, 7u: { return 2; }          // 土の道・集落の敷地
    case 4u: { return 3; }              // 川床
    case 6u, 8u: { return 4; }          // 山・境内
    default: { return 0; }              // 草地・畦・丘
  }
}

/** 焼くときに使う起伏の式。層ごとに周波数と振幅が違う（8m 周期に収まる整数倍の波長） */
fn detailBumpTile(p: vec2f, layer: i32) -> f32 {
  // 太陽が仰角 3.5° と低いので、わずかな傾きが大きな明暗になる。振幅は控えめに
  switch (layer) {
    case 1: { return 0.0030 * gnoise(p * 4.0, 703u); }                                   // 泥
    case 2: { return 0.0035 * gnoise(p * 6.0, 701u) + 0.0016 * gnoise(p * 16.0, 702u); } // 土
    case 3: { return 0.0040 * gnoise(p * 8.0, 706u); }                                   // 川床
    case 4: { return 0.0014 * gnoise(p * 24.0, 704u) + 0.0009 * gnoise(p * 56.0, 705u); }// 山・境内
    default: { return 0.0050 * gnoise(p * 4.0, 707u) + 0.0024 * gnoise(p * 12.0, 708u); }// 草地
  }
}
