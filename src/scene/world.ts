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

import type { DeviceBundle } from '../gpu/device';
import type { FrameContext, SceneRenderer, ShaderMessage } from '../harness/runner';
import { HEIGHT, WIDTH } from '../harness/runner';
import { multiply, perspectiveReversedInfinite, viewRotation } from '../math/mat4';
import { cross, dirFromAzEl, normalize, type Vec3 } from '../math/vec';
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

const FRAME_FLOATS = 48;

/**
 * 高さテクスチャの段構成。近・中・遠の 3 段、各 2048² の r32float（合計 48MB）。
 * テクセルは 0.2m / 1.6m / 12.8m（8 倍ずつ）。いずれもリング中心（カメラ xz）を中心に置く。
 * 画素ごとの法線はここから取る。数式を画素ごとに複数回評価すると GPU の占有率が崖を越えて
 * 30ms 以上かかる（実測）ため、評価は焼く時の 1 回に集約する。
 */
const HM_SIZE = 2048;
const HM_TEXELS = [0.2, 1.6, 12.8] as const;

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

  /** 起動時に GPU へ問い合わせて確定した値（レポート用） */
  private resolved = { eyeGroundHeight: 0, eye: [0, 0, 0] as Vec3, sunDir: [0, 0, 0] as Vec3 };

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

    const worldCode = commonWgsl + noiseWgsl + worldWgsl;
    const queryModule = await this.makeModule('query', worldCode + queryWgsl);
    const terrainModule = await this.makeModule('terrain', worldCode + frameWgsl + heightsampleWgsl + terrainWgsl);
    const heightmapModule = await this.makeModule('heightmap', worldCode + heightmapWgsl);
    const skyModule = await this.makeModule('sky', commonWgsl + noiseWgsl + frameWgsl + skyWgsl);
    const tonemapModule = await this.makeModule('tonemap', commonWgsl + frameWgsl.replace('@group(0) @binding(0) var<uniform> frame: Frame;', '') + tonemapWgsl);

    // --- 視点の地面高さを正本（world.wgsl）に問い合わせる ---
    const eyeGround = (await this.queryHeights(queryModule, [[this.view.eye.x, this.view.eye.z]]))[0];
    this.resolved.eyeGroundHeight = eyeGround;

    // --- 高さテクスチャを焼く（リング中心 = 視点 xz） ---
    const heightLayout = await this.bakeHeightmaps(heightmapModule, [this.view.eye.x, this.view.eye.z]);

    // --- 定数バッファ ---
    this.frameBuffer = device.createBuffer({
      size: FRAME_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.updateFrame();

    // --- 描画先 ---
    this.hdr = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depth = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

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
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    this.frameBindGroup = device.createBindGroup({
      layout: frameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } }],
    });
    const frameOnly = device.createPipelineLayout({ bindGroupLayouts: [frameLayout] });
    const frameAndHeight = device.createPipelineLayout({ bindGroupLayouts: [frameLayout, heightLayout] });

    this.terrainPipeline = device.createRenderPipeline({
      label: 'terrain',
      layout: frameAndHeight,
      vertex: { module: terrainModule, entryPoint: 'vs' },
      fragment: { module: terrainModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    this.skyPipeline = device.createRenderPipeline({
      label: 'sky',
      layout: frameOnly,
      vertex: { module: skyModule, entryPoint: 'vs' },
      fragment: { module: skyModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater-equal' },
    });

    const tonemapLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
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
      ],
    });
  }

  /**
   * 高さテクスチャを compute で焼く。段ごとに原点をテクセル格子に揃える（後で視点が動いたとき
   * 揺れないため）。戻り値は描画側の bind group layout（group 1）。
   */
  private async bakeHeightmaps(module: GPUShaderModule, center: [number, number]): Promise<GPUBindGroupLayout> {
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
      const ox = Math.floor((center[0] - extent / 2) / texel) * texel;
      const oz = Math.floor((center[1] - extent / 2) / texel) * texel;
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
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, sampler: { type: 'filtering' } },
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
    const eye: Vec3 = [v.eye.x, this.resolved.eyeGroundHeight + v.eye.above, v.eye.z];
    const forward = dirFromAzEl(v.yawDeg, v.pitchDeg);
    const worldUp: Vec3 = [0, 1, 0];
    const right = normalize(cross(forward, worldUp));
    const up = cross(right, forward);
    const aspect = WIDTH / HEIGHT;
    const fov = (v.fovDeg * Math.PI) / 180;
    const viewProj = multiply(perspectiveReversedInfinite(fov, aspect, NEAR), viewRotation(forward, worldUp));
    const sunDir = dirFromAzEl(v.sunAzimuthDeg, v.sunElevationDeg);
    this.resolved.eye = eye;
    this.resolved.sunDir = sunDir;

    const f = this.frameData;
    f.set(viewProj, 0);
    f.set([eye[0], eye[1], eye[2], 0], 16);
    f.set([forward[0], forward[1], forward[2], 0], 20);
    f.set([right[0], right[1], right[2], 0], 24);
    f.set([up[0], up[1], up[2], Math.tan(fov / 2)], 28);
    f.set([sunDir[0], sunDir[1], sunDir[2], Math.cos((SUN_ANGULAR_RADIUS_DEG * Math.PI) / 180)], 32);
    f.set([v.time, v.exposure, aspect, v.debug], 36);
    f.set([RING_R0, RING_K, RING_COUNT, RING_SECTORS], 40);
    f.set([eye[0], eye[2], 0, 0], 44);
    this.device.queue.writeBuffer(this.frameBuffer, 0, f);
  }

  render(ctx: FrameContext): void {
    const { encoder } = ctx;
    const hdrView = this.hdr.createView();
    const depthView = this.depth.createView();

    // 1. 地形（HDR へ。深度は逆 Z で 0 クリア）
    const terrain = encoder.beginRenderPass({
      colorAttachments: [{ view: hdrView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...(ctx.tsBegin ? { timestampWrites: ctx.tsBegin } : {}),
    });
    terrain.setPipeline(this.terrainPipeline);
    terrain.setBindGroup(0, this.frameBindGroup);
    terrain.setBindGroup(1, this.heightBindGroup);
    terrain.setIndexBuffer(this.indexBuffer, 'uint32');
    terrain.drawIndexed(this.indexCount);
    terrain.end();

    // 2. 空（地形の無い画素だけ）
    const sky = encoder.beginRenderPass({
      colorAttachments: [{ view: hdrView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    sky.setPipeline(this.skyPipeline);
    sky.setBindGroup(0, this.frameBindGroup);
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

  describe(): unknown {
    return {
      view: this.view,
      resolved: this.resolved,
      heightmap: { size: HM_SIZE, texels: HM_TEXELS, format: 'r32float', bakeMs: this.heightBakeMs },
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
