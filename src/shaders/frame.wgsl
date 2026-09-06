// 描画パイプラインが group(0) binding(0) で共有するフレーム定数。
// カメラ相対で描く（頂点位置から camPos を引いてから viewProj を掛ける）ので、
// 16km 先の山でも f32 の精度が足りる。

struct Frame {
  viewProj: mat4x4f,   // カメラ相対（平行移動なし）→ クリップ
  camPos: vec4f,       // xyz = 世界座標
  camForward: vec4f,
  camRight: vec4f,
  camUp: vec4f,        // w = tan(fov/2)
  sunDir: vec4f,       // xyz = 太陽へ向かう単位ベクトル, w = 太陽の視半径の cos
  params: vec4f,       // x = 時刻(秒), y = 露出, z = アスペクト比, w = デバッグ表示の切替
  ring: vec4f,         // x = r0, y = 隣接リング比 k, z = リング数, w = 扇形数
  center: vec4f,       // xy = リング中心の世界 xz, zw = 描画解像度 [px]
  wind: vec4f,         // xy = 風テクスチャの原点（世界 xz）, z = テクセル [m], w = 1辺のテクセル数
};

@group(0) @binding(0) var<uniform> frame: Frame;

/** 画面座標（NDC）から世界の視線方向 */
fn viewRay(ndc: vec2f) -> vec3f {
  let t = frame.camUp.w;
  return normalize(frame.camForward.xyz + frame.camRight.xyz * (ndc.x * frame.params.z * t) + frame.camUp.xyz * (ndc.y * t));
}
