// フェーズ1のシーン: 地形・空・（水面・あぜ道は段階的に追加）。
// 描く中身はここ、計測は harness/runner.ts。

import commonWgsl from '../shaders/common.wgsl?raw';
import noiseWgsl from '../shaders/noise.wgsl?raw';
import worldWgsl from '../shaders/world.wgsl?raw';
import frameWgsl from '../shaders/frame.wgsl?raw';
import terrainWgsl from '../shaders/terrain.wgsl?raw';
import skyWgsl from '../shaders/sky.wgsl?raw';
import tonemapWgsl from '../shaders/tonemap.wgsl?raw';
import queryWgsl from '../shaders/query.wgsl?raw';
import heightmapWgsl from '../shaders/heightmap.wgsl?raw';
import heightsampleWgsl from '../shaders/heightsample.wgsl?raw';
import atmosphereWgsl from '../shaders/atmosphere.wgsl?raw';
import skylutWgsl from '../shaders/skylut.wgsl?raw';
import shadowWgsl from '../shaders/shadow.wgsl?raw';
import lightsampleWgsl from '../shaders/lightsample.wgsl?raw';
import waterWgsl from '../shaders/water.wgsl?raw';
import windWgsl from '../shaders/wind.wgsl?raw';
import riceWgsl from '../shaders/rice.wgsl?raw';
import grassWgsl from '../shaders/grass.wgsl?raw';
import treeWgsl from '../shaders/tree.wgsl?raw';
import deformWgsl from '../shaders/deform.wgsl?raw';
import deformUpdateWgsl from '../shaders/deform-update.wgsl?raw';
import deformQueryWgsl from '../shaders/deform-query.wgsl?raw';
import { TREE_VERTEX_FLOATS, buildTreeVariants, planTrees } from './trees';

import type { DeviceBundle } from '../gpu/device';
import type { FrameContext, SceneRenderer, ShaderMessage } from '../harness/runner';
import { resolution } from '../harness/runner';
import { multiply, perspectiveReversedInfinite, viewRotation } from '../math/mat4';
import { cross, dirFromAzEl, dot, normalize, type Vec3 } from '../math/vec';
import type { View } from '../views';

/**
 * リングメッシュの構成。
 * 隣接リングの半径比 k = 1 + 2π/S にすると、扇形の幅（弧長）とリング間隔が常に等しく、
 * どの距離でも三角形がほぼ正三角形になる。近くは細かく、遠くは粗い、が自動で成り立つ。
 */
const RING_SECTORS = 512;
const RING_R0 = 0.5;      // 最内リングの半径 [m]
const RING_RMAX = 16000;  // 最外リングの半径 [m]。遠景の山並みまで同じメッシュで描く
const RING_K = 1 + (2 * Math.PI) / RING_SECTORS;
const RING_COUNT = Math.ceil(Math.log(RING_RMAX / RING_R0) / Math.log(RING_K)) + 1;
const NEAR = 0.05;
const SUN_ANGULAR_RADIUS_DEG = 0.27;

const FRAME_FLOATS = 56;
/** 変形の場: 1024² × 0.1m = 102.4m 四方をカメラ周りにトーラス状（wrap）に持つ。窓の外に出た変形は失われる */
const DEFORM_SIZE = 1024;
const DEFORM_TEXEL = 0.1;
/** 固定刻み。壁時計を使わないので筋書きは再現する */
const FIXED_DT = 1 / 60;
const MAX_STAMPS = 32;
/** 風の場のテクスチャ: 512² × 1.2m = 614m 四方をカメラ周りに毎フレーム焼く */
const WIND_SIZE = 512;
const WIND_TEXEL = 1.2;
/**
 * MSAA は使わない（既定 1）。Chrome/Metal で 4× MSAA にすると 1 フレーム 200ms になった（実測、DPR1 でも）。
 * 輪郭の平滑化はトーンマップ時の FXAA で行う。?msaa=4 で再実験できる
 */
const MSAA_SAMPLES = (() => { const v = Number(new URLSearchParams(location.search).get('msaa')); return v === 4 ? 4 : 1; })();

/**
 * 高さテクスチャの段構成。近・中・遠の 3 段、各 2048² の r32float（合計 48MB）。
 * テクセルは 0.25m / 1.6m / 12.8m。近景段だけ視線の 100m 前方を中心に置き、見えている範囲を多く覆う。
 * 画素ごとの法線はここから取る。数式を画素ごとに複数回評価すると GPU の占有率が崖を越えて
 * 30ms 以上かかる（実測）ため、評価は焼く時の 1 回に集約する。
 */
const HM_SIZE = 2048;
const HM_TEXELS = [0.25, 1.6, 12.8] as const;

/** 空の LUT（正距円筒）。空の描画・環境光・映り込みに使う。太陽が固定なので起動時に 1 回焼く */
const SKY_LUT_W = 512;
const SKY_LUT_H = 256;

/** 遠景の溶け込みの 3D LUT（方位 × 仰角 × 距離）。カメラ位置ごとに起動時に焼く */
const AERIAL_W = 64;
const AERIAL_H = 32;
const AERIAL_D = 32;

/** 水面メッシュに使う区画索引の範囲（盆地 |u|<484, 北 v<163, 南 v>-121 を余裕をもって覆う） */
const CELL_I0 = -22;
const CELL_NI = 45;
const CELL_J0 = -9;
const CELL_NJ = 22;
const CELL_STRIDE = 64;
const CELL_GRID_U = 9;
const CELL_GRID_V = 3;
/** 小川の帯: x の範囲と間隔、半幅（水面の半幅 2.6m より広く取り、土手の下に隠す） */
const RIVER_X0 = -900;
const RIVER_X1 = 900;
const RIVER_STEP = 4;
const RIVER_HALF = 3.2;
const RIVER_LEVEL = -1.0;

/** 稲: インスタンスの上限（compute が atomic で追記）と、頂点シェーダが生成する 1 株あたりの頂点数 */
const RICE_MAX = 160000;
/** インスタンス 1 件のバイト数（pos, attr, deform の 3 × vec4f） */
const INSTANCE_STRIDE = 48;
const RICE_NEAR_VERTS = 7 * 3 * 6;   // 葉 7 枚 × 3 節 × 6 頂点
const RICE_MID_VERTS = 2 * 6;        // 交差する板 2 枚
/** 草: 稲と同じ仕組み。葉 5 枚 × 2 節 */
const GRASS_MAX = 120000;
const GRASS_NEAR_VERTS = 5 * 2 * 6;
const GRASS_MID_VERTS = 2 * 6;

function buildRingIndices(rings: number, sectors: number): Uint32Array {
  const fan = sectors * 3;
  const quads = (rings - 1) * sectors * 6;
  const out = new Uint32Array(fan + quads);
  let o = 0;
  const at = (ring: number, sector: number) => 1 + ring * sectors + (sector % sectors);
  for (let j = 0; j < sectors; j++) {
    out[o++] = 0;
    out[o++] = at(0, j + 1);
    out[o++] = at(0, j);
  }
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < sectors; j++) {
      const a = at(i, j);
      const b = at(i, j + 1);
      const c = at(i + 1, j);
      const d = at(i + 1, j + 1);
      out[o++] = a; out[o++] = b; out[o++] = c;
      out[o++] = b; out[o++] = d; out[o++] = c;
    }
  }
  return out;
}

export class WorldScene implements SceneRenderer {
  private device!: GPUDevice;
  private view: View;
  private messages: ShaderMessage[] = [];

  private frameBuffer!: GPUBuffer;
  private frameData = new Float32Array(FRAME_FLOATS);
  private hdr!: GPUTexture;
  private hdrMsaa!: GPUTexture;
  private depth!: GPUTexture;
  private terrainPipeline!: GPURenderPipeline;
  private skyPipeline!: GPURenderPipeline;
  private tonemapPipeline!: GPURenderPipeline;
  private frameBindGroup!: GPUBindGroup;
  private tonemapBindGroup!: GPUBindGroup;
  private indexBuffer!: GPUBuffer;
  private indexCount = 0;
  private heightTex!: GPUTexture;
  private heightBindGroup!: GPUBindGroup;
  private heightBakeMs = 0;
  private skyLutTex!: GPUTexture;
  private skyLutPipeline!: GPUComputePipeline;
  private skyLutBakeBindGroup!: GPUBindGroup;
  private skyLutBindGroup!: GPUBindGroup;
  private skyReducePipeline!: GPUComputePipeline;
  private skyReduceBindGroup!: GPUBindGroup;
  private skyIrrBuffer!: GPUBuffer;
  private lightTex!: GPUTexture;
  private matTex!: GPUTexture;
  private lightBindGroup!: GPUBindGroup;
  private aerialInTex!: GPUTexture;
  private aerialTrTex!: GPUTexture;
  private sunLutBuffer!: GPUBuffer;
  private windTex!: GPUTexture;
  private windPipeline!: GPUComputePipeline;
  private windBindGroup!: GPUBindGroup;
  private lightBakeMs = 0;
  private waterPipeline!: GPURenderPipeline;
  private riceSpawnNear!: GPUComputePipeline;
  private riceSpawnMid!: GPUComputePipeline;
  private riceNearPipeline!: GPURenderPipeline;
  private riceMidPipeline!: GPURenderPipeline;
  private riceArgs!: GPUBuffer;
  private riceNearBuf!: GPUBuffer;
  private riceMidBuf!: GPUBuffer;
  private riceComputeBindGroup!: GPUBindGroup;
  private riceComputeLayout!: GPUBindGroupLayout;
  private riceCounts = { near: 0, mid: 0 };
  private grassSpawnNear!: GPUComputePipeline;
  private grassSpawnMid!: GPUComputePipeline;
  private grassNearPipeline!: GPURenderPipeline;
  private grassMidPipeline!: GPURenderPipeline;
  private grassArgs!: GPUBuffer;
  private grassNearBuf!: GPUBuffer;
  private grassMidBuf!: GPUBuffer;
  private grassCounts = { near: 0, mid: 0 };
  private treePipeline!: GPURenderPipeline;
  private treeDraws: { mesh: GPUBuffer; vertexCount: number; instances: GPUBuffer; instanceCount: number; name: string }[] = [];
  private treeStats = { total: 0, variants: [] as { name: string; vertices: number; instances: number }[] };

  // ---- 変形の場 ----
  private deformTex: GPUTexture[] = [];
  private rippleTex: GPUTexture[] = [];
  private deformPing = 0;
  private deformUpdatePipeline!: GPUComputePipeline;
  private deformUpdateBindGroups: GPUBindGroup[] = [];
  private deformQueryPipeline!: GPUComputePipeline;
  private probePoints: [number, number][] = [];
  private probeResults: { x: number; z: number; sink: number; bendX: number; bendZ: number; bend: number; turbidity: number; ripple: number; inside: boolean }[] = [];
  private skyLutBindGroups: GPUBindGroup[] = [];
  private deformParams!: GPUBuffer;
  private stampBuffer!: GPUBuffer;
  private pendingStamps: { x: number; z: number; radius: number; kind: number; dirX: number; dirZ: number; depth: number; bend: number }[] = [];
  private deformOrigin: [number, number] = [0, 0];
  private deformPrevOrigin: [number, number] = [0, 0];
  private frameIndex = 0;
  /** 歩き手の位置（変形を起こす主体）。段階2で移動が入るまでは視点の足元 */
  walker = { x: 0, z: 0, yawDeg: 0 };
  private waterBuffer!: GPUBuffer;
  private waterVertexCount = 0;
  private waterStats = { paddyCells: 0, riverSamples: 0, triangles: 0 };
  private queryModule!: GPUShaderModule;
  private cameraBasis = { forward: [0, 0, 1] as Vec3, right: [1, 0, 0] as Vec3, up: [0, 1, 0] as Vec3, tanHalfFov: 0.5 };

  /** 起動時に GPU へ問い合わせて確定した値（レポート用） */
  private resolved = {
    eyeGroundHeight: 0,
    /** 線に吸着した後の視点 xz */
    eyeXZ: [0, 0] as [number, number],
    eye: [0, 0, 0] as Vec3,
    sunDir: [0, 0, 0] as Vec3,
    /** 空の放射照度（上向き面）・全天平均放射輝度・直射（太陽に正対する面）。露出の判断材料 */
    light: { skyIrradianceUp: [0, 0, 0] as Vec3, skyMeanRadiance: [0, 0, 0] as Vec3, sunIrradiance: [0, 0, 0] as Vec3 },
  };

  constructor(view: View) {
    this.view = view;
  }

  private async makeModule(label: string, code: string): Promise<GPUShaderModule> {
    const module = this.device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    for (const m of info.messages) {
      this.messages.push({ module: label, type: m.type, message: m.message, lineNum: m.lineNum, linePos: m.linePos });
    }
    return module;
  }

  async init(bundle: DeviceBundle, canvasFormat: GPUTextureFormat): Promise<void> {
    const device = bundle.device;
    this.device = device;

    // 稲のインスタンスバッファ（group 3 に光・材質と同居させるので先に作る）
    this.riceNearBuf = device.createBuffer({ size: RICE_MAX * INSTANCE_STRIDE, usage: GPUBufferUsage.STORAGE });
    this.riceMidBuf = device.createBuffer({ size: RICE_MAX * INSTANCE_STRIDE, usage: GPUBufferUsage.STORAGE });
    this.riceArgs = device.createBuffer({ size: 32, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(this.riceArgs, 0, new Uint32Array([RICE_NEAR_VERTS, 0, 0, 0, RICE_MID_VERTS, 0, 0, 0]));
    this.grassNearBuf = device.createBuffer({ size: GRASS_MAX * INSTANCE_STRIDE, usage: GPUBufferUsage.STORAGE });
    this.grassMidBuf = device.createBuffer({ size: GRASS_MAX * INSTANCE_STRIDE, usage: GPUBufferUsage.STORAGE });
    this.grassArgs = device.createBuffer({ size: 32, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(this.grassArgs, 0, new Uint32Array([GRASS_NEAR_VERTS, 0, 0, 0, GRASS_MID_VERTS, 0, 0, 0]));

    const worldCode = commonWgsl + noiseWgsl + worldWgsl;
    const queryModule = await this.makeModule('query', worldCode + queryWgsl);
    this.queryModule = queryModule;
    const terrainModule = await this.makeModule(
      'terrain',
      worldCode + frameWgsl + heightsampleWgsl + atmosphereWgsl + skylutWgsl + windWgsl + deformWgsl + shadowWgsl + lightsampleWgsl + terrainWgsl,
    );
    const heightmapModule = await this.makeModule('heightmap', worldCode + frameWgsl + heightsampleWgsl + shadowWgsl + heightmapWgsl);
    const skyModule = await this.makeModule('sky', commonWgsl + noiseWgsl + frameWgsl + atmosphereWgsl + skylutWgsl + windWgsl + skyWgsl);
    const tonemapModule = await this.makeModule('tonemap', commonWgsl + frameWgsl.replace('@group(0) @binding(0) var<uniform> frame: Frame;', '') + tonemapWgsl);

    // --- 視点を線（畦・道）へ吸着し、地面高さを正本（world.wgsl）に問い合わせる ---
    let eyeXZ: [number, number] = [this.view.eye.x, this.view.eye.z];
    if (this.view.snap) {
      const sn = this.view.snap;
      const out = new Float32Array(await this.runQuery('snapQuery', [
        { binding: 4, data: new Float32Array([sn.family, sn.index, eyeXZ[0], eyeXZ[1]]), type: 'read-only-storage' },
        { binding: 8, data: new Float32Array([sn.offset]), type: 'read-only-storage' },
      ], { binding: 5, byteLength: 8 }, 1));
      eyeXZ = [out[0], out[1]];
    }
    this.resolved.eyeXZ = eyeXZ;
    const eyeGround = (await this.queryHeights(queryModule, [eyeXZ]))[0];
    this.resolved.eyeGroundHeight = eyeGround;

    // --- 高さテクスチャを焼く（リング中心 = 視点 xz） ---
    const fwd = dirFromAzEl(this.view.yawDeg, 0);
    const heightLayout = await this.bakeHeightmaps(heightmapModule, eyeXZ, [fwd[0] * 100, fwd[2] * 100]);

    // --- 水面メッシュ（区画の形と川の中心線を正本から読み戻して組む） ---
    await this.buildWater();
    // --- 木（テンプレートを数値から起こし、配置を正本に問い合わせて確定） ---
    await this.buildTrees();

    // --- 定数バッファ ---
    this.frameBuffer = device.createBuffer({
      size: FRAME_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.updateFrame();

    // --- 描画先 ---
    const WIDTH = resolution.width;
    const HEIGHT = resolution.height;
    this.hdr = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrMsaa = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      sampleCount: MSAA_SAMPLES,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depth = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'depth32float',
      sampleCount: MSAA_SAMPLES,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // --- 空の LUT ---
    this.skyLutTex = device.createTexture({
      size: [SKY_LUT_W, SKY_LUT_H],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const skyLutBakeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
      ],
    });
    this.skyLutPipeline = device.createComputePipeline({
      label: 'bakeSkyLut',
      layout: device.createPipelineLayout({ bindGroupLayouts: [skyLutBakeLayout] }),
      compute: { module: skyModule, entryPoint: 'bakeSkyLut' },
    });
    this.skyLutBakeBindGroup = device.createBindGroup({
      layout: skyLutBakeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.skyLutTex.createView() },
      ],
    });
    // LUT の半球積分（環境光）
    this.skyIrrBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const skyReduceLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.skyReducePipeline = device.createComputePipeline({
      label: 'reduceSkyLut',
      layout: device.createPipelineLayout({ bindGroupLayouts: [skyReduceLayout] }),
      compute: { module: skyModule, entryPoint: 'reduceSkyLut' },
    });
    this.skyReduceBindGroup = device.createBindGroup({
      layout: skyReduceLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 2, resource: this.skyLutTex.createView() },
        { binding: 3, resource: { buffer: this.skyIrrBuffer } },
      ],
    });

    // 遠景の溶け込み（3D LUT）と太陽光（1D LUT）
    this.aerialInTex = device.createTexture({
      size: [AERIAL_W, AERIAL_H, AERIAL_D], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aerialTrTex = device.createTexture({
      size: [AERIAL_W, AERIAL_H, AERIAL_D], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sunLutBuffer = device.createBuffer({ size: 128 * 16, usage: GPUBufferUsage.STORAGE });
    const aerialBakeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only', viewDimension: '3d' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only', viewDimension: '3d' } },
      ],
    });
    const aerialPipeline = device.createComputePipeline({
      label: 'bakeAerial',
      layout: device.createPipelineLayout({ bindGroupLayouts: [aerialBakeLayout] }),
      compute: { module: skyModule, entryPoint: 'bakeAerial' },
    });
    const aerialBakeBindGroup = device.createBindGroup({
      layout: aerialBakeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 4, resource: this.aerialInTex.createView() },
        { binding: 5, resource: this.aerialTrTex.createView() },
      ],
    });
    const sunBakeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const sunPipeline = device.createComputePipeline({
      label: 'bakeSunLut',
      layout: device.createPipelineLayout({ bindGroupLayouts: [sunBakeLayout] }),
      compute: { module: skyModule, entryPoint: 'bakeSunLut' },
    });
    const sunBakeBindGroup = device.createBindGroup({
      layout: sunBakeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 6, resource: { buffer: this.sunLutBuffer } },
      ],
    });
    {
      const encoder = device.createCommandEncoder();
      const a = encoder.beginComputePass();
      a.setPipeline(aerialPipeline);
      a.setBindGroup(0, aerialBakeBindGroup);
      a.dispatchWorkgroups(AERIAL_W / 4, AERIAL_H / 4, AERIAL_D / 4);
      a.end();
      const b = encoder.beginComputePass();
      b.setPipeline(sunPipeline);
      b.setBindGroup(0, sunBakeBindGroup);
      b.dispatchWorkgroups(2);
      b.end();
      device.queue.submit([encoder.finish()]);
    }

    // 風の場（毎フレーム焼く）
    this.windTex = device.createTexture({
      size: [WIND_SIZE, WIND_SIZE], format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const windBakeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
      ],
    });
    this.windPipeline = device.createComputePipeline({
      label: 'bakeWind',
      layout: device.createPipelineLayout({ bindGroupLayouts: [windBakeLayout] }),
      compute: { module: skyModule, entryPoint: 'bakeWind' },
    });
    this.windBindGroup = device.createBindGroup({
      layout: windBakeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 7, resource: this.windTex.createView() },
      ],
    });

    const vis = GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE;
    const skyLutLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: vis, texture: { sampleType: 'float' } },
        { binding: 1, visibility: vis, sampler: { type: 'filtering' } },
        { binding: 2, visibility: vis, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: vis, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: vis, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 5, visibility: vis, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: vis, texture: { sampleType: 'float' } },
        { binding: 7, visibility: vis, sampler: { type: 'filtering' } },
        { binding: 8, visibility: vis, texture: { sampleType: 'float' } },
        { binding: 9, visibility: vis, texture: { sampleType: 'float' } },
        { binding: 10, visibility: vis, sampler: { type: 'filtering' } },
      ],
    });
    // 変形の場（ping-pong の 2 組）
    for (let i = 0; i < 2; i++) {
      this.deformTex.push(device.createTexture({ size: [DEFORM_SIZE, DEFORM_SIZE], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING }));
      this.rippleTex.push(device.createTexture({ size: [DEFORM_SIZE, DEFORM_SIZE], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING }));
    }
    const deformSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
    const skySampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    const windSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    for (let i = 0; i < 2; i++) {
      this.skyLutBindGroups.push(device.createBindGroup({
        layout: skyLutLayout,
        entries: [
          { binding: 0, resource: this.skyLutTex.createView() },
          { binding: 1, resource: skySampler },
          { binding: 2, resource: { buffer: this.skyIrrBuffer } },
          { binding: 3, resource: this.aerialInTex.createView() },
          { binding: 4, resource: this.aerialTrTex.createView() },
          { binding: 5, resource: { buffer: this.sunLutBuffer } },
          { binding: 6, resource: this.windTex.createView() },
          { binding: 7, resource: windSampler },
          { binding: 8, resource: this.deformTex[i].createView() },
          { binding: 9, resource: this.rippleTex[i].createView() },
          { binding: 10, resource: deformSampler },
        ],
      }));
    }
    this.skyLutBindGroup = this.skyLutBindGroups[0];


    // --- リングメッシュのインデックス ---
    const indices = buildRingIndices(RING_COUNT, RING_SECTORS);
    this.indexCount = indices.length;
    this.indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indices);

    // --- パイプライン ---
    const frameLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
    });
    this.frameBindGroup = device.createBindGroup({
      layout: frameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } }],
    });
    // 法線・日向/日陰の焼き込み（高さテクスチャと太陽が確定してから）
    const lightLayout = await this.bakeNormalShadow(heightmapModule, frameLayout, heightLayout);
    const frameAndHeight = device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, skyLutLayout, lightLayout] });

    // 変形の場の更新 compute（読み i → 書き 1-i）
    this.deformParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.stampBuffer = device.createBuffer({ size: MAX_STAMPS * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const deformUpdateLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const deformModule = await this.makeModule('deformUpdate', commonWgsl + noiseWgsl + heightsampleWgsl + deformUpdateWgsl);
    this.deformUpdatePipeline = device.createComputePipeline({
      label: 'deformUpdate',
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, skyLutLayout, deformUpdateLayout] }),
      compute: { module: deformModule, entryPoint: 'deformUpdate' },
    });
    const deformQueryModule = await this.makeModule('deformQuery', commonWgsl + noiseWgsl + frameWgsl + heightsampleWgsl + skylutWgsl + windWgsl + deformWgsl + deformQueryWgsl);
    const dqLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.deformQueryPipeline = device.createComputePipeline({
      label: 'deformQuery',
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, skyLutLayout, dqLayout] }),
      compute: { module: deformQueryModule, entryPoint: 'deformQuery' },
    });
    for (let i = 0; i < 2; i++) {
      this.deformUpdateBindGroups.push(device.createBindGroup({
        layout: deformUpdateLayout,
        entries: [
          { binding: 0, resource: this.matTex.createView({ dimension: '2d-array' }) },
          { binding: 1, resource: this.deformTex[i].createView() },
          { binding: 2, resource: this.rippleTex[i].createView() },
          { binding: 3, resource: this.deformTex[1 - i].createView() },
          { binding: 4, resource: this.rippleTex[1 - i].createView() },
          { binding: 5, resource: { buffer: this.stampBuffer } },
          { binding: 6, resource: { buffer: this.deformParams } },
        ],
      }));
    }

    this.terrainPipeline = device.createRenderPipeline({
      label: 'terrain',
      layout: frameAndHeight,
      vertex: { module: terrainModule, entryPoint: 'vs' },
      fragment: { module: terrainModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: MSAA_SAMPLES },
    });
    const waterModule = await this.makeModule(
      'water',
      worldCode + frameWgsl + heightsampleWgsl + atmosphereWgsl + skylutWgsl + lightsampleWgsl + windWgsl + deformWgsl + waterWgsl,
    );
    this.waterPipeline = device.createRenderPipeline({
      label: 'water',
      layout: frameAndHeight,
      vertex: {
        module: waterModule,
        entryPoint: 'vs',
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      },
      fragment: { module: waterModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: MSAA_SAMPLES },
    });

    // --- 稲 ---
    {
      const riceCommon = worldCode + frameWgsl + heightsampleWgsl + atmosphereWgsl + skylutWgsl + lightsampleWgsl + windWgsl + deformWgsl;
      // rice.wgsl は 共通 / 生成 / 描画 の 3 節。頂点シェーダは read_write の storage を参照できないので別モジュールに組む
      const section = (name: string): string => {
        const start = riceWgsl.indexOf(`// ==== SECTION: ${name} ====`);
        const rest = riceWgsl.slice(start);
        const next = rest.indexOf('// ==== SECTION:', 10);
        return next < 0 ? rest : rest.slice(0, next);
      };
      const riceComputeModule = await this.makeModule('riceSpawn', riceCommon + section('common') + section('spawn'));
      const riceRenderModule = await this.makeModule('rice', riceCommon + section('common') + section('draw'));

      const computePL = device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, skyLutLayout, this.riceComputeLayout] });
      this.riceSpawnNear = device.createComputePipeline({ label: 'spawnNear', layout: computePL, compute: { module: riceComputeModule, entryPoint: 'spawnNear' } });
      this.riceSpawnMid = device.createComputePipeline({ label: 'spawnMid', layout: computePL, compute: { module: riceComputeModule, entryPoint: 'spawnMid' } });
      const renderPL = frameAndHeight;
      const riceDepth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
      this.riceNearPipeline = device.createRenderPipeline({
        label: 'riceNear', layout: renderPL,
        vertex: { module: riceRenderModule, entryPoint: 'vsNear' },
        fragment: { module: riceRenderModule, entryPoint: 'fsNear', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: riceDepth, multisample: { count: MSAA_SAMPLES },
      });
      this.riceMidPipeline = device.createRenderPipeline({
        label: 'riceMid', layout: renderPL,
        vertex: { module: riceRenderModule, entryPoint: 'vsMid' },
        fragment: { module: riceRenderModule, entryPoint: 'fsMid', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: riceDepth, multisample: { count: MSAA_SAMPLES },
      });
    }

    // --- 草（稲と同じ仕組み） ---
    {
      const common = worldCode + frameWgsl + heightsampleWgsl + atmosphereWgsl + skylutWgsl + lightsampleWgsl + windWgsl + deformWgsl;
      const section = (src: string, name: string): string => {
        const start = src.indexOf(`// ==== SECTION: ${name} ====`);
        const rest = src.slice(start);
        const next = rest.indexOf('// ==== SECTION:', 10);
        return next < 0 ? rest : rest.slice(0, next);
      };
      // leafShade は rice.wgsl の描画節にあるので、草の描画モジュールにも稲の描画節を含める
      const computeModule = await this.makeModule('grassSpawn', common + section(grassWgsl, 'common') + section(grassWgsl, 'spawn'));
      const renderModule = await this.makeModule('grass', common + section(riceWgsl, 'common') + section(riceWgsl, 'draw') + section(grassWgsl, 'common') + section(grassWgsl, 'draw'));
      const computePL = device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, skyLutLayout, this.riceComputeLayout] });
      this.grassSpawnNear = device.createComputePipeline({ label: 'spawnGrassNear', layout: computePL, compute: { module: computeModule, entryPoint: 'spawnGrassNear' } });
      this.grassSpawnMid = device.createComputePipeline({ label: 'spawnGrassMid', layout: computePL, compute: { module: computeModule, entryPoint: 'spawnGrassMid' } });
      const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
      this.grassNearPipeline = device.createRenderPipeline({
        label: 'grassNear', layout: frameAndHeight,
        vertex: { module: renderModule, entryPoint: 'vsGrassNear' },
        fragment: { module: renderModule, entryPoint: 'fsGrassNear', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depth, multisample: { count: MSAA_SAMPLES },
      });
      this.grassMidPipeline = device.createRenderPipeline({
        label: 'grassMid', layout: frameAndHeight,
        vertex: { module: renderModule, entryPoint: 'vsGrassMid' },
        fragment: { module: renderModule, entryPoint: 'fsGrassMid', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depth, multisample: { count: MSAA_SAMPLES },
      });
    }

    // --- 木 ---
    {
      const common = worldCode + frameWgsl + heightsampleWgsl + atmosphereWgsl + skylutWgsl + lightsampleWgsl + windWgsl + deformWgsl;
      const section = (src: string, name: string): string => {
        const start = src.indexOf(`// ==== SECTION: ${name} ====`);
        const rest = src.slice(start);
        const next = rest.indexOf('// ==== SECTION:', 10);
        return next < 0 ? rest : rest.slice(0, next);
      };
      const treeModule = await this.makeModule('tree', common + section(riceWgsl, 'common') + section(riceWgsl, 'draw') + treeWgsl);
      this.treePipeline = device.createRenderPipeline({
        label: 'tree', layout: frameAndHeight,
        vertex: {
          module: treeModule, entryPoint: 'vsTree',
          buffers: [
            {
              arrayStride: TREE_VERTEX_FLOATS * 4, stepMode: 'vertex',
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },
                { shaderLocation: 1, offset: 12, format: 'float32x3' },
                { shaderLocation: 2, offset: 24, format: 'float32x2' },
                { shaderLocation: 3, offset: 32, format: 'float32' },
                { shaderLocation: 4, offset: 36, format: 'float32' },
              ],
            },
            {
              arrayStride: 32, stepMode: 'instance',
              attributes: [
                { shaderLocation: 5, offset: 0, format: 'float32x4' },
                { shaderLocation: 6, offset: 16, format: 'float32x4' },
              ],
            },
          ],
        },
        fragment: { module: treeModule, entryPoint: 'fsTree', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
        multisample: { count: MSAA_SAMPLES },
      });
    }

    this.skyPipeline = device.createRenderPipeline({
      label: 'sky',
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, skyLutLayout] }),
      vertex: { module: skyModule, entryPoint: 'vs' },
      fragment: { module: skyModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater-equal' },
      multisample: { count: MSAA_SAMPLES },
    });

    const tonemapLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.tonemapPipeline = device.createRenderPipeline({
      label: 'tonemap',
      layout: device.createPipelineLayout({ bindGroupLayouts: [tonemapLayout] }),
      vertex: { module: tonemapModule, entryPoint: 'vs' },
      fragment: { module: tonemapModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    this.tonemapBindGroup = device.createBindGroup({
      layout: tonemapLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.hdr.createView() },
        { binding: 2, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }) },
      ],
    });

    // --- 光の数値を 1 回読み戻す（診断・露出の判断材料） ---
    {
      const encoder = device.createCommandEncoder();
      const lut = encoder.beginComputePass();
      lut.setPipeline(this.skyLutPipeline);
      lut.setBindGroup(0, this.skyLutBakeBindGroup);
      lut.dispatchWorkgroups(SKY_LUT_W / 8, SKY_LUT_H / 8);
      lut.end();
      const reduce = encoder.beginComputePass();
      reduce.setPipeline(this.skyReducePipeline);
      reduce.setBindGroup(0, this.skyReduceBindGroup);
      reduce.dispatchWorkgroups(1);
      reduce.end();
      const read = device.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(this.skyIrrBuffer, 0, read, 0, 48);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const v = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      read.destroy();
      this.resolved.light = {
        skyIrradianceUp: [v[0], v[1], v[2]],
        skyMeanRadiance: [v[4], v[5], v[6]],
        sunIrradiance: [v[8], v[9], v[10]],
      };
    }
  }

  /**
   * 高さテクスチャを compute で焼く。段ごとに原点をテクセル格子に揃える（後で視点が動いたとき
   * 揺れないため）。戻り値は描画側の bind group layout（group 1）。
   */
  private async bakeHeightmaps(module: GPUShaderModule, center: [number, number], l0Offset: [number, number]): Promise<GPUBindGroupLayout> {
    const device = this.device;
    const t0 = performance.now();
    this.heightTex = device.createTexture({
      size: [HM_SIZE, HM_SIZE, HM_TEXELS.length],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    const pipeline = device.createComputePipeline({
      label: 'fillLevel',
      layout: 'auto',
      compute: { module, entryPoint: 'fillLevel' },
    });

    const levelParams = new Float32Array(4 * HM_TEXELS.length);
    const encoder = device.createCommandEncoder();
    const scratch: GPUBuffer[] = [];
    HM_TEXELS.forEach((texel, i) => {
      const extent = texel * HM_SIZE;
      const cx = center[0] + (i === 0 ? l0Offset[0] : 0);
      const cz = center[1] + (i === 0 ? l0Offset[1] : 0);
      const ox = Math.floor((cx - extent / 2) / texel) * texel;
      const oz = Math.floor((cz - extent / 2) / texel) * texel;
      levelParams.set([ox, oz, texel, HM_SIZE], i * 4);

      const paramBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(paramBuf, 0, new Float32Array([ox, oz, texel, HM_SIZE]));
      const layerBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(layerBuf, 0, new Uint32Array([i, 0, 0, 0]));
      scratch.push(paramBuf, layerBuf);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramBuf } },
          { binding: 1, resource: this.heightTex.createView({ dimension: '2d-array' }) },
          { binding: 2, resource: { buffer: layerBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(HM_SIZE / 8, HM_SIZE / 8);
      pass.end();
    });
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    this.heightBakeMs = performance.now() - t0;
    for (const b of scratch) b.destroy();

    const levelsBuf = device.createBuffer({ size: levelParams.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(levelsBuf, 0, levelParams);
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      ],
    });
    this.heightBindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: levelsBuf } },
        { binding: 1, resource: this.heightTex.createView({ dimension: '2d-array' }) },
        { binding: 2, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }) },
      ],
    });
    return layout;
  }

  /** 焼いた高さから法線と日向/日陰を段ごとに焼く。画素側は 1 タップで読む */
  private async bakeNormalShadow(module: GPUShaderModule, frameLayout: GPUBindGroupLayout, heightLayout: GPUBindGroupLayout): Promise<GPUBindGroupLayout> {
    const device = this.device;
    const t0 = performance.now();
    const layers = HM_TEXELS.length;
    // rg16float / r8unorm は storage 書き込み非対応。rgba16float 1 枚に法線 xz と可視度をまとめる
    this.lightTex = device.createTexture({
      size: [HM_SIZE, HM_SIZE, layers], format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.matTex = device.createTexture({
      size: [HM_SIZE, HM_SIZE, layers], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const bakeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', access: 'write-only', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm', access: 'write-only', viewDimension: '2d-array' } },
      ],
    });
    const pipeline = device.createComputePipeline({
      label: 'bakeNormalShadow',
      layout: device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout, bakeLayout] }),
      compute: { module, entryPoint: 'bakeNormalShadow' },
    });
    const encoder = device.createCommandEncoder();
    const scratch: GPUBuffer[] = [];
    for (let i = 0; i < layers; i++) {
      const layerBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(layerBuf, 0, new Uint32Array([i, 0, 0, 0]));
      scratch.push(layerBuf);
      const bg = device.createBindGroup({
        layout: bakeLayout,
        entries: [
          { binding: 0, resource: { buffer: layerBuf } },
          { binding: 1, resource: this.lightTex.createView({ dimension: '2d-array' }) },
          { binding: 2, resource: this.matTex.createView({ dimension: '2d-array' }) },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.heightBindGroup);
      pass.setBindGroup(2, bg);
      pass.dispatchWorkgroups(HM_SIZE / 8, HM_SIZE / 8);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    this.lightBakeMs = performance.now() - t0;
    for (const b of scratch) b.destroy();

    // 描画用: 光・材質テクスチャ ＋ 稲インスタンス（read-only）。生成 compute 用は別レイアウト（read_write）
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.lightBindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: this.lightTex.createView({ dimension: '2d-array' }) },
        { binding: 1, resource: this.matTex.createView({ dimension: '2d-array' }) },
        { binding: 2, resource: { buffer: this.riceNearBuf } },
        { binding: 3, resource: { buffer: this.riceMidBuf } },
        { binding: 5, resource: { buffer: this.grassNearBuf } },
        { binding: 6, resource: { buffer: this.grassMidBuf } },
      ],
    });
    this.riceComputeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.riceComputeBindGroup = device.createBindGroup({
      layout: this.riceComputeLayout,
      entries: [
        { binding: 0, resource: this.lightTex.createView({ dimension: '2d-array' }) },
        { binding: 1, resource: this.matTex.createView({ dimension: '2d-array' }) },
        { binding: 2, resource: { buffer: this.riceNearBuf } },
        { binding: 3, resource: { buffer: this.riceMidBuf } },
        { binding: 4, resource: { buffer: this.riceArgs } },
        { binding: 5, resource: { buffer: this.grassNearBuf } },
        { binding: 6, resource: { buffer: this.grassMidBuf } },
        { binding: 7, resource: { buffer: this.grassArgs } },
      ],
    });
    return layout;
  }

  /**
   * 正本（query.wgsl）の compute を 1 回走らせて出力を読み戻す汎用の口。
   * 入力は binding 番号ごとに与える。出力は 1 本。
   */
  private async runQuery(
    entryPoint: string,
    inputs: { binding: number; data: ArrayBufferView; type: 'uniform' | 'read-only-storage' }[],
    output: { binding: number; byteLength: number },
    workgroups: number,
  ): Promise<ArrayBuffer> {
    const device = this.device;
    const pipeline = device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module: this.queryModule, entryPoint } });
    const buffers: GPUBuffer[] = [];
    const entries: GPUBindGroupEntry[] = [];
    for (const input of inputs) {
      const size = Math.max(16, Math.ceil(input.data.byteLength / 16) * 16);
      const buf = device.createBuffer({
        size,
        usage: (input.type === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, input.data.buffer, input.data.byteOffset, input.data.byteLength);
      buffers.push(buf);
      entries.push({ binding: input.binding, resource: { buffer: buf } });
    }
    const outSize = Math.max(16, Math.ceil(output.byteLength / 16) * 16);
    const outBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    entries.push({ binding: output.binding, resource: { buffer: outBuf } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outSize);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = readBuf.getMappedRange().slice(0, output.byteLength);
    readBuf.unmap();
    for (const b of buffers) b.destroy();
    outBuf.destroy();
    readBuf.destroy();
    return result;
  }

  /** 区画の形（cellQuery）と川の中心線（riverQuery）から水面の三角形を組む */
  private async buildWater(): Promise<void> {
    const cellCount = CELL_NI * CELL_NJ;
    const cells = new Float32Array(await this.runQuery('cellQuery', [
      { binding: 2, data: new Int32Array([CELL_I0, CELL_J0, CELL_NI, CELL_NJ]), type: 'uniform' },
    ], { binding: 3, byteLength: cellCount * CELL_STRIDE * 4 }, Math.ceil(cellCount / 64)));

    const verts: number[] = [];
    let paddyCells = 0;
    for (let c = 0; c < cellCount; c++) {
      const base = c * CELL_STRIDE;
      if (cells[base] < 0.5) continue;
      paddyCells++;
      const level = cells[base + 1];
      const pt = (sIdx: number, tIdx: number): [number, number] => {
        const o = base + 2 + (tIdx * CELL_GRID_U + sIdx) * 2;
        return [cells[o], cells[o + 1]];
      };
      for (let t = 0; t < CELL_GRID_V - 1; t++) {
        for (let sIdx = 0; sIdx < CELL_GRID_U - 1; sIdx++) {
          const a = pt(sIdx, t);
          const b = pt(sIdx + 1, t);
          const cc = pt(sIdx, t + 1);
          const d = pt(sIdx + 1, t + 1);
          verts.push(a[0], level, a[1], b[0], level, b[1], cc[0], level, cc[1]);
          verts.push(b[0], level, b[1], d[0], level, d[1], cc[0], level, cc[1]);
        }
      }
    }

    // 小川: 中心線に沿った帯
    const xs: number[] = [];
    for (let x = RIVER_X0; x <= RIVER_X1; x += RIVER_STEP) xs.push(x);
    const zs = new Float32Array(await this.runQuery('riverQuery', [
      { binding: 6, data: new Float32Array(xs), type: 'read-only-storage' },
    ], { binding: 7, byteLength: xs.length * 4 }, Math.ceil(xs.length / 64)));
    const side: [number, number][][] = [];
    for (let k = 0; k < xs.length; k++) {
      const kp = Math.max(0, k - 1);
      const kn = Math.min(xs.length - 1, k + 1);
      const tx = xs[kn] - xs[kp];
      const tz = zs[kn] - zs[kp];
      const len = Math.hypot(tx, tz) || 1;
      const nx = -tz / len;
      const nz = tx / len;
      side.push([[xs[k] + nx * RIVER_HALF, zs[k] + nz * RIVER_HALF], [xs[k] - nx * RIVER_HALF, zs[k] - nz * RIVER_HALF]]);
    }
    for (let k = 0; k < xs.length - 1; k++) {
      const [l0, r0] = side[k];
      const [l1, r1] = side[k + 1];
      verts.push(l0[0], RIVER_LEVEL, l0[1], r0[0], RIVER_LEVEL, r0[1], l1[0], RIVER_LEVEL, l1[1]);
      verts.push(r0[0], RIVER_LEVEL, r0[1], r1[0], RIVER_LEVEL, r1[1], l1[0], RIVER_LEVEL, l1[1]);
    }

    const data = new Float32Array(verts);
    this.waterVertexCount = data.length / 3;
    this.waterStats = { paddyCells, riverSamples: xs.length, triangles: this.waterVertexCount / 3 };
    this.waterBuffer = this.device.createBuffer({
      size: Math.max(16, data.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    if (data.byteLength > 0) this.device.queue.writeBuffer(this.waterBuffer, 0, data);
  }

  /** 木: テンプレートを数値から起こし、配置（谷座標）を正本への問い合わせで世界座標に直す */
  private async buildTrees(): Promise<void> {
    const device = this.device;
    const variants = buildTreeVariants();
    const plan = planTrees();

    // 吸着するもの（参道の並木）と、そのままの (u,v) のものを分けて問い合わせる
    const snapIdx = plan.map((t, i) => (t.snap ? i : -1)).filter((i) => i >= 0);
    const xz = plan.map((t) => [t.u, 0] as [number, number]);
    if (snapIdx.length > 0) {
      // 縦線への吸着: (family, index, x, z) で z は谷座標 v から川の分を後で足す。ここでは v → 仮の z = v として渡し、
      // snapQuery は z − riverZ(x) を v として扱うので、先に riverZ を足しておく
      const rz = new Float32Array(await this.runQuery('riverQuery', [
        { binding: 6, data: new Float32Array(snapIdx.map((i) => plan[i].u)), type: 'read-only-storage' },
      ], { binding: 7, byteLength: snapIdx.length * 4 }, Math.ceil(snapIdx.length / 64)));
      const q = new Float32Array(snapIdx.length * 4);
      const off = new Float32Array(snapIdx.length);
      snapIdx.forEach((i, k) => {
        const t = plan[i];
        q.set([0, t.snap!.index, t.u, t.v + rz[k]], k * 4);
        off[k] = t.snap!.offset;
      });
      const out = new Float32Array(await this.runQuery('snapQuery', [
        { binding: 4, data: q, type: 'read-only-storage' },
        { binding: 8, data: off, type: 'read-only-storage' },
      ], { binding: 5, byteLength: snapIdx.length * 8 }, Math.ceil(snapIdx.length / 64)));
      snapIdx.forEach((i, k) => { xz[i] = [out[k * 2], out[k * 2 + 1]]; });
    }
    const plainIdx = plan.map((t, i) => (t.snap ? -1 : i)).filter((i) => i >= 0);
    if (plainIdx.length > 0) {
      const rz = new Float32Array(await this.runQuery('riverQuery', [
        { binding: 6, data: new Float32Array(plainIdx.map((i) => plan[i].u)), type: 'read-only-storage' },
      ], { binding: 7, byteLength: plainIdx.length * 4 }, Math.ceil(plainIdx.length / 64)));
      plainIdx.forEach((i, k) => { xz[i] = [plan[i].u, plan[i].v + rz[k]]; });
    }
    const heights = await this.queryHeights(this.queryModule, xz);

    this.treeDraws = [];
    this.treeStats = { total: plan.length, variants: [] };
    variants.forEach((variant, vi) => {
      const mine = plan.map((t, i) => (t.variant === vi ? i : -1)).filter((i) => i >= 0);
      const inst = new Float32Array(Math.max(1, mine.length) * 8);
      mine.forEach((i, k) => {
        const t = plan[i];
        inst.set([xz[i][0], heights[i], xz[i][1], t.scale, t.rotation, 1, variant.species, (i * 0.618) % 1], k * 8);
      });
      const mesh = device.createBuffer({ size: variant.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(mesh, 0, variant.vertices);
      const instances = device.createBuffer({ size: inst.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(instances, 0, inst);
      const vertexCount = variant.vertices.length / TREE_VERTEX_FLOATS;
      this.treeDraws.push({ mesh, vertexCount, instances, instanceCount: mine.length, name: variant.name });
      this.treeStats.variants.push({ name: variant.name, vertices: vertexCount, instances: mine.length });
    });
  }

  /** 世界の方向 → 画素座標（画面外なら null） */
  dirToPixel(dir: Vec3): { x: number; y: number } | null {
    const { forward, right, up, tanHalfFov } = this.cameraBasis;
    const dz = dot(dir, forward);
    if (dz <= 1e-6) return null;
    const ndcX = dot(dir, right) / dz / (tanHalfFov * (resolution.width / resolution.height));
    const ndcY = dot(dir, up) / dz / tanHalfFov;
    return { x: ((ndcX + 1) / 2) * resolution.width, y: ((1 - ndcY) / 2) * resolution.height };
  }

  /** 画素座標 → 世界の方向 */
  pixelToDir(x: number, y: number): Vec3 {
    const { forward, right, up, tanHalfFov } = this.cameraBasis;
    const ndcX = (x / resolution.width) * 2 - 1;
    const ndcY = 1 - (y / resolution.height) * 2;
    const a = ndcX * (resolution.width / resolution.height) * tanHalfFov;
    const b = ndcY * tanHalfFov;
    return normalize([
      forward[0] + right[0] * a + up[0] * b,
      forward[1] + right[1] * a + up[1] * b,
      forward[2] + right[2] * a + up[2] * b,
    ]);
  }

  /** 平らな水面に映る太陽の方向（法線 +y の鏡像）と、その画素位置、地平線の画素行 */
  predictGlint(): { reflectedDir: Vec3; pixel: { x: number; y: number } | null; horizonY: number } {
    const s = this.resolved.sunDir;
    const reflectedDir: Vec3 = [s[0], -s[1], s[2]];
    const horizon = this.dirToPixel([this.cameraBasis.forward[0], 0, this.cameraBasis.forward[2]]);
    return { reflectedDir, pixel: this.dirToPixel(reflectedDir), horizonY: horizon ? horizon.y : resolution.height / 2 };
  }

  /** world.wgsl の高さを CPU から問い合わせる（正本は WGSL のみ、という約束の実装） */
  private async queryHeights(module: GPUShaderModule, points: [number, number][]): Promise<number[]> {
    const device = this.device;
    const n = points.length;
    const inBuf = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, new Float32Array(points.flat()));

    const pipeline = device.createComputePipeline({
      label: 'heightQuery',
      layout: 'auto',
      compute: { module, entryPoint: 'heightQuery' },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * 4);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = Array.from(new Float32Array(readBuf.getMappedRange().slice(0)));
    readBuf.unmap();
    inBuf.destroy(); outBuf.destroy(); readBuf.destroy();
    return result;
  }

  private updateFrame(): void {
    const v = this.view;
    const eye: Vec3 = [this.resolved.eyeXZ[0], this.resolved.eyeGroundHeight + v.eye.above, this.resolved.eyeXZ[1]];
    const forward = dirFromAzEl(v.yawDeg, v.pitchDeg);
    const worldUp: Vec3 = [0, 1, 0];
    const right = normalize(cross(forward, worldUp));
    const up = cross(right, forward);
    const aspect = resolution.width / resolution.height;
    const fov = (v.fovDeg * Math.PI) / 180;
    const viewProj = multiply(perspectiveReversedInfinite(fov, aspect, NEAR), viewRotation(forward, worldUp));
    const sunDir = dirFromAzEl(v.sunAzimuthDeg, v.sunElevationDeg);
    this.resolved.eye = eye;
    this.resolved.sunDir = sunDir;
    this.cameraBasis = { forward, right, up, tanHalfFov: Math.tan(fov / 2) };

    const f = this.frameData;
    f.set(viewProj, 0);
    f.set([eye[0], eye[1], eye[2], 0], 16);
    f.set([forward[0], forward[1], forward[2], 0], 20);
    f.set([right[0], right[1], right[2], 0], 24);
    f.set([up[0], up[1], up[2], Math.tan(fov / 2)], 28);
    f.set([sunDir[0], sunDir[1], sunDir[2], Math.cos((SUN_ANGULAR_RADIUS_DEG * Math.PI) / 180)], 32);
    f.set([v.time + this.frameIndex * FIXED_DT, v.exposure, aspect, v.debug], 36);
    f.set([RING_R0, RING_K, RING_COUNT, RING_SECTORS], 40);
    f.set([eye[0], eye[2], resolution.width, resolution.height], 44);
    const windExtent = WIND_SIZE * WIND_TEXEL;
    f.set([
      Math.floor((eye[0] - windExtent / 2) / WIND_TEXEL) * WIND_TEXEL,
      Math.floor((eye[2] - windExtent / 2) / WIND_TEXEL) * WIND_TEXEL,
      WIND_TEXEL, WIND_SIZE,
    ], 48);
    f.set([this.deformOrigin[0], this.deformOrigin[1], DEFORM_TEXEL, DEFORM_SIZE], 52);
    this.device.queue.writeBuffer(this.frameBuffer, 0, f);
  }

  /** 変形の場を世界座標で読み戻す（検証用）。現在の ping 側を読む */
  async probeDeform(points: [number, number][]): Promise<typeof this.probeResults> {
    const device = this.device;
    const n = points.length;
    if (n === 0) return [];
    const inBuf = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, new Float32Array(points.flat()));
    const outBuf = device.createBuffer({ size: n * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: n * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bg = device.createBindGroup({
      layout: this.deformQueryPipeline.getBindGroupLayout(3),
      entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.deformQueryPipeline);
    pass.setBindGroup(0, this.frameBindGroup);
    pass.setBindGroup(1, this.heightBindGroup);
    pass.setBindGroup(2, this.skyLutBindGroups[this.deformPing]);
    pass.setBindGroup(3, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * 32);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    inBuf.destroy(); outBuf.destroy(); readBuf.destroy();
    const out = points.map(([x, z], i) => ({
      x, z,
      sink: v[i * 4], bendX: v[i * 4 + 1], bendZ: v[i * 4 + 2],
      bend: Math.hypot(v[i * 4 + 1], v[i * 4 + 2]),
      turbidity: Math.max(0, v[i * 4 + 3]),
      ripple: v[(n + i) * 4],
      inside: v[i * 4 + 3] >= 0,
    }));
    this.probeResults = out;
    return out;
  }

  /** 撮影後に読む観測点（URL の probe=x,z;x,z）。世界座標。'eye' 相対も書ける: e+dx,dz */
  setProbePoints(points: [number, number][]): void {
    this.probePoints = points;
  }

  /** 踏み跡を予約する（次のフレームの更新で書き込まれる） */
  addStamp(x: number, z: number, radius: number, kind: number, dirX: number, dirZ: number, depth: number, bend: number): void {
    if (this.pendingStamps.length < MAX_STAMPS) this.pendingStamps.push({ x, z, radius, kind, dirX, dirZ, depth, bend });
  }

  /** 段階1の検証用: 視点の向きへ歩いた足跡の列を最初のフレームに置く（dbg=20） */
  private testStamps(): void {
    const wx = this.resolved.eyeXZ[0];
    const wz = this.resolved.eyeXZ[1];
    const yaw = (this.view.yawDeg * Math.PI) / 180;
    const fx = Math.sin(yaw), fz = Math.cos(yaw);
    const rx = Math.cos(yaw), rz = -Math.sin(yaw);
    for (let i = 0; i < 26; i++) {
      const d = 1.2 + i * 0.66;
      const side = (i % 2 === 0 ? -1 : 1) * 0.12;
      const x = wx + fx * d + rx * side;
      const z = wz + fz * d + rz * side;
      this.addStamp(x, z, 0.16, 0, fx, fz, 0.12, 0.9);
      this.addStamp(wx + fx * d, wz + fz * d, 0.45, 1, fx, fz, 0, 0.85);
    }
  }

  /** 変形の場を 1 歩進める（毎フレーム）。窓の原点は歩き手（＝視点）に追従 */
  private stepDeform(encoder: GPUCommandEncoder): void {
    const extent = DEFORM_SIZE * DEFORM_TEXEL;
    this.deformPrevOrigin = this.deformOrigin;
    this.deformOrigin = [
      Math.floor((this.walker.x - extent / 2) / DEFORM_TEXEL) * DEFORM_TEXEL,
      Math.floor((this.walker.z - extent / 2) / DEFORM_TEXEL) * DEFORM_TEXEL,
    ];
    if (this.frameIndex === 0) {
      // 最初のフレームは全面を 0 に（前の窓＝遠くの偽の窓にしておけば「新しく入った」扱いになる）
      this.deformPrevOrigin = [this.deformOrigin[0] + 1e6, this.deformOrigin[1] + 1e6];
      if (this.view.debug === 20) this.testStamps();
    }
    const stamps = new Float32Array(MAX_STAMPS * 8);
    this.pendingStamps.forEach((st, i) => stamps.set([st.x, st.z, st.radius, st.kind, st.dirX, st.dirZ, st.depth, st.bend], i * 8));
    this.device.queue.writeBuffer(this.stampBuffer, 0, stamps);
    this.device.queue.writeBuffer(this.deformParams, 0, new Float32Array([
      this.deformPrevOrigin[0], this.deformPrevOrigin[1], this.deformOrigin[0], this.deformOrigin[1],
      DEFORM_TEXEL, DEFORM_SIZE, FIXED_DT, this.pendingStamps.length,
    ]));
    this.pendingStamps = [];

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.deformUpdatePipeline);
    pass.setBindGroup(0, this.frameBindGroup);
    pass.setBindGroup(1, this.heightBindGroup);
    pass.setBindGroup(2, this.skyLutBindGroups[this.deformPing]);
    pass.setBindGroup(3, this.deformUpdateBindGroups[this.deformPing]);
    pass.dispatchWorkgroups(DEFORM_SIZE / 8, DEFORM_SIZE / 8);
    pass.end();
    this.deformPing = 1 - this.deformPing;
    this.skyLutBindGroup = this.skyLutBindGroups[this.deformPing];
  }

  render(ctx: FrameContext): void {
    const { encoder } = ctx;
    this.frameIndex = ctx.frameIndex;
    if (ctx.frameIndex === 0) this.walker = { x: this.resolved.eyeXZ[0], z: this.resolved.eyeXZ[1], yawDeg: this.view.yawDeg };
    // 変形の場を進めてから、フレーム定数（時刻・窓の原点）を書く
    this.stepDeform(encoder);
    this.updateFrame();
    const hdrView = MSAA_SAMPLES > 1 ? this.hdrMsaa.createView() : this.hdr.createView();
    const resolveView = this.hdr.createView();
    const depthView = this.depth.createView();
    const skipRice = this.view.debug === 13;   // 計測用: 稲の生成と描画を丸ごと飛ばす
    const skipSpawn = skipRice || this.view.debug === 14;   // 14: 生成だけ飛ばす
    const skipRiceDraw = skipRice || this.view.debug === 15;   // 15: 描画だけ飛ばす

    // -1. 風の場を焼く（毎フレーム。稲・水面・草・木が同じ場を 1 タップで読む）
    {
      const wind = encoder.beginComputePass();
      wind.setPipeline(this.windPipeline);
      wind.setBindGroup(0, this.windBindGroup);
      wind.dispatchWorkgroups(WIND_SIZE / 8, WIND_SIZE / 8);
      wind.end();
    }

    // 0. 稲のインスタンスをカメラ周りに生成（毎フレーム。フェーズ3で歩いても追従する）
    encoder.clearBuffer(this.riceArgs, 4, 4);
    encoder.clearBuffer(this.riceArgs, 20, 4);
    encoder.clearBuffer(this.grassArgs, 4, 4);
    encoder.clearBuffer(this.grassArgs, 20, 4);
    if (!skipSpawn) {
      const spawn = encoder.beginComputePass();
      spawn.setBindGroup(0, this.frameBindGroup);
      spawn.setBindGroup(1, this.heightBindGroup);
      spawn.setBindGroup(2, this.skyLutBindGroup);
      spawn.setBindGroup(3, this.riceComputeBindGroup);
      spawn.setPipeline(this.riceSpawnNear);
      spawn.dispatchWorkgroups(272 / 8, 368 / 8);
      spawn.setPipeline(this.riceSpawnMid);
      spawn.dispatchWorkgroups(336 / 8, 336 / 8);
      spawn.setPipeline(this.grassSpawnNear);
      spawn.dispatchWorkgroups(176 / 8, 176 / 8);
      spawn.setPipeline(this.grassSpawnMid);
      spawn.dispatchWorkgroups(208 / 8, 208 / 8);
      spawn.end();
    }

    // 1. 地形 → 水面 → 稲 を 1 つの render pass で（パスを分けると HDR/深度の読み書きが毎回掛かる。実測 2.3ms/パス）
    const scene = encoder.beginRenderPass({
      colorAttachments: [{ view: hdrView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...(ctx.tsBegin ? { timestampWrites: ctx.tsBegin } : {}),
    });
    scene.setBindGroup(0, this.frameBindGroup);
    scene.setBindGroup(1, this.heightBindGroup);
    scene.setBindGroup(2, this.skyLutBindGroup);
    scene.setBindGroup(3, this.lightBindGroup);
    // 1a. 地形（リングメッシュ、1 draw call）
    scene.setPipeline(this.terrainPipeline);
    scene.setIndexBuffer(this.indexBuffer, 'uint32');
    scene.drawIndexed(this.indexCount);
    // 1b. 水面（田と小川）
    if (this.waterVertexCount > 0) {
      scene.setPipeline(this.waterPipeline);
      scene.setVertexBuffer(0, this.waterBuffer);
      scene.draw(this.waterVertexCount);
    }
    // 1c. 稲（近距離の株と中距離の束。数は compute が決める → drawIndirect）
    if (!skipRiceDraw) {
      scene.setPipeline(this.riceNearPipeline);
      scene.drawIndirect(this.riceArgs, 0);
      scene.setPipeline(this.riceMidPipeline);
      scene.drawIndirect(this.riceArgs, 16);
      // 1d. 草
      scene.setPipeline(this.grassNearPipeline);
      scene.drawIndirect(this.grassArgs, 0);
      scene.setPipeline(this.grassMidPipeline);
      scene.drawIndirect(this.grassArgs, 16);
    }
    // 1e. 木（種ごとのテンプレートをインスタンス描画）
    scene.setPipeline(this.treePipeline);
    for (const d of this.treeDraws) {
      scene.setVertexBuffer(0, d.mesh);
      scene.setVertexBuffer(1, d.instances);
      scene.draw(d.vertexCount, d.instanceCount);
    }
    scene.end();

    // 2. 空（地形の無い画素だけ）
    const sky = encoder.beginRenderPass({
      // HDR へ描く最後のパス。ここで MSAA を解決して tonemap が読む hdr に落とす
      colorAttachments: [{ view: hdrView, ...(MSAA_SAMPLES > 1 ? { resolveTarget: resolveView } : {}), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    sky.setPipeline(this.skyPipeline);
    sky.setBindGroup(0, this.frameBindGroup);
    sky.setBindGroup(1, this.heightBindGroup);
    sky.setBindGroup(2, this.skyLutBindGroup);
    sky.draw(3);
    sky.end();

    // 3. 露出とトーンマップ → キャンバス
    const tonemap = encoder.beginRenderPass({
      colorAttachments: [{ view: ctx.target, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
      ...(ctx.tsEnd ? { timestampWrites: ctx.tsEnd } : {}),
    });
    tonemap.setPipeline(this.tonemapPipeline);
    tonemap.setBindGroup(0, this.tonemapBindGroup);
    tonemap.draw(3);
    tonemap.end();
  }

  async afterRun(): Promise<void> {
    await this.readRiceCounts();
    if (this.probePoints.length > 0) await this.probeDeform(this.probePoints);
  }

  /** 稲のインスタンス数（直近フレーム）を読み戻す。レポート用 */
  async readRiceCounts(): Promise<{ near: number; mid: number }> {
    const device = this.device;
    const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.riceArgs, 0, read, 0, 32);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    this.riceCounts = { near: Math.min(v[1], RICE_MAX), mid: Math.min(v[5], RICE_MAX) };
    const read2 = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(this.grassArgs, 0, read2, 0, 32);
    device.queue.submit([enc2.finish()]);
    await read2.mapAsync(GPUMapMode.READ);
    const g = new Uint32Array(read2.getMappedRange().slice(0));
    read2.unmap();
    read2.destroy();
    this.grassCounts = { near: Math.min(g[1], GRASS_MAX), mid: Math.min(g[5], GRASS_MAX) };
    return this.riceCounts;
  }

  describe(): unknown {
    return {
      view: this.view,
      resolved: this.resolved,
      heightmap: { size: HM_SIZE, texels: HM_TEXELS, format: 'r32float', bakeMs: this.heightBakeMs, normalShadowBakeMs: this.lightBakeMs },
      water: this.waterStats,
      rice: { ...this.riceCounts, max: RICE_MAX, nearVerts: RICE_NEAR_VERTS, midVerts: RICE_MID_VERTS },
      grass: { ...this.grassCounts, max: GRASS_MAX, nearVerts: GRASS_NEAR_VERTS, midVerts: GRASS_MID_VERTS },
      trees: this.treeStats,
      deform: { size: DEFORM_SIZE, texel: DEFORM_TEXEL, extentM: DEFORM_SIZE * DEFORM_TEXEL, fixedDt: FIXED_DT, origin: this.deformOrigin },
      probe: this.probeResults,
      resolution: { ...resolution, msaa: MSAA_SAMPLES },
      ring: {
        sectors: RING_SECTORS,
        rings: RING_COUNT,
        r0: RING_R0,
        rMax: RING_RMAX,
        ratio: RING_K,
        vertices: 1 + RING_COUNT * RING_SECTORS,
        triangles: this.indexCount / 3,
        drawCalls: 1,
      },
    };
  }

  shaderMessages(): ShaderMessage[] {
    return this.messages;
  }
}
