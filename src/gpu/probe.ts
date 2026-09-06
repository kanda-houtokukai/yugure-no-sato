// WebGPU が実際に取れるかを機械確認する。
// navigator.gpu → requestAdapter → requestDevice まで通し、
// アダプタ情報・対応機能・上限値を構造化して返す。
//
// このファイルはフェーズ0の検証用だが、以降のフェーズでも
// 「デバイスをどう取るか」の唯一の入口として使い回す。

export interface ProbeReport {
  ok: boolean;
  stage: 'navigator.gpu' | 'requestAdapter' | 'requestDevice' | 'done';
  error: string | null;
  userAgent: string;
  /** 安全でない WebGPU（--enable-unsafe-webgpu）が要るかの判定材料 */
  isSecureContext: boolean;
  adapter: {
    info: Record<string, string>;
    isFallbackAdapter: boolean | null;
    features: string[];
    limits: Record<string, number>;
  } | null;
  device: {
    features: string[];
    limits: Record<string, number>;
  } | null;
  preferredCanvasFormat: string | null;
}

/**
 * GPUSupportedLimits は WebIDL インターフェイスなので、値はプロトタイプ上の
 * ゲッターとして生えている。for...in はプロトタイプ鎖の enumerable を辿るため
 * これで拾えるが、実装差で拾えない環境に備えて既知の名前でも読みにいく。
 */
const KNOWN_LIMIT_NAMES = [
  'maxTextureDimension1D',
  'maxTextureDimension2D',
  'maxTextureDimension3D',
  'maxTextureArrayLayers',
  'maxBindGroups',
  'maxBindingsPerBindGroup',
  'maxDynamicUniformBuffersPerPipelineLayout',
  'maxDynamicStorageBuffersPerPipelineLayout',
  'maxSampledTexturesPerShaderStage',
  'maxSamplersPerShaderStage',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxUniformBufferBindingSize',
  'maxStorageBufferBindingSize',
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
  'maxVertexBuffers',
  'maxBufferSize',
  'maxVertexAttributes',
  'maxVertexBufferArrayStride',
  'maxInterStageShaderVariables',
  'maxColorAttachments',
  'maxColorAttachmentBytesPerSample',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
] as const;

function limitsToObject(limits: GPUSupportedLimits | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  if (!limits) return out;
  const anyLimits = limits as unknown as Record<string, unknown>;
  for (const key in anyLimits) {
    const value = anyLimits[key];
    if (typeof value === 'number') out[key] = value;
  }
  for (const key of KNOWN_LIMIT_NAMES) {
    const value = anyLimits[key];
    if (typeof value === 'number') out[key] = value;
  }
  return out;
}

function featuresToArray(features: GPUSupportedFeatures | undefined): string[] {
  if (!features) return [];
  return [...features].sort();
}

function infoToObject(info: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (!info || typeof info !== 'object') return out;
  const anyInfo = info as Record<string, unknown>;
  for (const key in anyInfo) {
    const value = anyInfo[key];
    if (typeof value === 'string' || typeof value === 'number') out[key] = String(value);
  }
  // 主要4項目は for...in で拾えなかった場合に備えて名指しでも読む
  for (const key of ['vendor', 'architecture', 'device', 'description']) {
    const value = anyInfo[key];
    if (typeof value === 'string') out[key] = value;
  }
  return out;
}

function describeError(e: unknown): string {
  if (e instanceof Error) return `${e.name}: ${e.message}`;
  return String(e);
}

export async function probeWebGPU(): Promise<ProbeReport> {
  const report: ProbeReport = {
    ok: false,
    stage: 'navigator.gpu',
    error: null,
    userAgent: navigator.userAgent,
    isSecureContext: window.isSecureContext,
    adapter: null,
    device: null,
    preferredCanvasFormat: null,
  };

  const gpu = navigator.gpu;
  if (!gpu) {
    report.error =
      'navigator.gpu が undefined。WebGPU が無効。起動フラグ（--enable-unsafe-webgpu）か、' +
      'ヘッドレス／セキュアコンテキストの制約を疑う。';
    return report;
  }

  report.stage = 'requestAdapter';
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    report.error = `requestAdapter が例外: ${describeError(e)}`;
    return report;
  }
  if (!adapter) {
    report.error = 'requestAdapter が null を返した（利用可能なアダプタなし）。';
    return report;
  }

  report.adapter = {
    info: infoToObject((adapter as unknown as { info?: unknown }).info),
    isFallbackAdapter:
      typeof (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter === 'boolean'
        ? (adapter as unknown as { isFallbackAdapter: boolean }).isFallbackAdapter
        : null,
    features: featuresToArray(adapter.features),
    limits: limitsToObject(adapter.limits),
  };

  report.stage = 'requestDevice';
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice();
  } catch (e) {
    report.error = `requestDevice が例外: ${describeError(e)}`;
    return report;
  }

  report.device = {
    features: featuresToArray(device.features),
    limits: limitsToObject(device.limits),
  };

  try {
    report.preferredCanvasFormat = gpu.getPreferredCanvasFormat();
  } catch (e) {
    report.preferredCanvasFormat = `(取得失敗: ${describeError(e)})`;
  }

  // 検証だけが目的なので、掴んだデバイスはここで手放す
  device.destroy();

  report.stage = 'done';
  report.ok = true;
  return report;
}
