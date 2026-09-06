// 地形による日陰。焼いた高さテクスチャに沿って太陽方向へ進み、地面に遮られるかを見る。
// 数式評価ではなくテクスチャ参照なので、35 歩でも軽い。

fn terrainShadow(world: vec3f, sunDir: vec3f) -> f32 {
  if (sunDir.y <= 0.0) { return 0.0; }
  // 頂点の高さ（数式）とテクスチャの高さは数 cm 食い違う。太陽が低いとその差が
  // 手前 1m ほどの偽の影（縞）になるので、起点での差を光線側に足して打ち消す
  let h0 = sampleHeight(heightLevelFor(world.xz), world.xz);
  let delta = world.y - h0;
  // 畦（幅 1.6m）を跨ぎ越さないよう、最初の 2m は 0.25m 刻み、その先は等比で伸ばす
  var t = 0.25;
  var shade = 1.0;
  let bias = 0.03;
  for (var i = 0; i < 38; i++) {
    let p = world + sunDir * t;
    let li = heightLevelFor(p.xz);
    let h = sampleHeight(li, p.xz);
    let gap = p.y - delta + bias + t * 0.006 - h;
    if (gap < 0.0) { return 0.0; }
    shade = min(shade, gap / (t * 0.02));
    t = select(t * 1.3, t + 0.25, t < 2.0);
    if (t > 6000.0) { break; }
  }
  return clamp(shade, 0.0, 1.0);
}
