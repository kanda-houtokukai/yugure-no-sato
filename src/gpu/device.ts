// デバイス取得と GPU エラーの集約。
// GPU のエラーは黙って無視されるのが既定の挙動なので、
// uncapturederror と errorScope の両方を必ず張る。ここを通さずに device を取らない。

export interface GpuErrorRecord {
  source: 'uncapturederror' | 'errorScope' | 'deviceLost';
  scope: string | null;
  type: string;
  message: string;
}

export interface DeviceBundle {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** キャンバスに使う形式（bgra8unorm など） */
  format: GPUTextureFormat;
  /** timestamp-query が使えるか。使えなければ GPU 側のフレーム時間は取れない */
  hasTimestampQuery: boolean;
  /** 発生した GPU エラーの集積先。呼び出し側はこの配列を読む */
  errors: GpuErrorRecord[];
}

/** push した逆順に pop される。この順序を崩さないこと */
const SCOPES: readonly GPUErrorFilter[] = ['internal', 'out-of-memory', 'validation'];

export function pushErrorScopes(device: GPUDevice): void {
  for (const scope of SCOPES) device.pushErrorScope(scope);
}

export async function popErrorScopes(device: GPUDevice, sink: GpuErrorRecord[]): Promise<void> {
  for (let i = SCOPES.length - 1; i >= 0; i--) {
    const scope = SCOPES[i];
    const error = await device.popErrorScope();
    if (error) {
      sink.push({
        source: 'errorScope',
        scope,
        type: error.constructor?.name ?? 'GPUError',
        message: error.message,
      });
    }
  }
}

export async function acquireDevice(): Promise<DeviceBundle> {
  if (!navigator.gpu) throw new Error('navigator.gpu が無い（WebGPU 無効）');

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter が null（利用可能なアダプタなし）');

  const hasTimestampQuery = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestampQuery ? ['timestamp-query'] : [],
  });

  const errors: GpuErrorRecord[] = [];

  device.addEventListener('uncapturederror', (event) => {
    const error = (event as GPUUncapturedErrorEvent).error;
    errors.push({
      source: 'uncapturederror',
      scope: null,
      type: error.constructor?.name ?? 'GPUError',
      message: error.message,
    });
  });

  void device.lost.then((info) => {
    errors.push({
      source: 'deviceLost',
      scope: null,
      type: info.reason ?? 'unknown',
      message: info.message,
    });
  });

  return {
    adapter,
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    hasTimestampQuery,
    errors,
  };
}
