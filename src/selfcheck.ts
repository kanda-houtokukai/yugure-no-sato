// 自己点検。人間の目に頼らず、機械で取れる異常を数値にする。
// ここが本フェーズの狙いなので、判定の閾値は「なぜその値か」を添えて明示する。

import type { GpuErrorRecord } from './gpu/device';

export interface ImageStats {
  width: number;
  height: number;
  pixels: number;
  /** 0..1。真っ黒／真っ白の検出に使う */
  meanLuminance: number;
  varianceLuminance: number;
  stdDevLuminance: number;
  minLuminance: number;
  maxLuminance: number;
  /** 輝度 0.01 未満の比率。NaN は多くの場合 0 として書かれるのでここに出る */
  nearBlackRatio: number;
  /** 輝度 0.99 超の比率。Inf 由来の飽和はここに出る */
  nearWhiteRatio: number;
  extremeRatio: number;
  /** 16 段の輝度ヒストグラム（比率）。分布の偏りを人が見るため */
  histogram: number[];
}

export interface FrameTimeStats {
  available: boolean;
  reason: string | null;
  frames: number;
  avgMs: number | null;
  maxMs: number | null;
  minMs: number | null;
  /** GPU 側のみ。Chrome はタイムスタンプを丸めるので、実測した丸め幅を出す */
  quantumNs: number | null;
}

export interface Check {
  name: string;
  ok: boolean;
  detail: string;
}

const HIST_BINS = 16;

/**
 * copyTextureToBuffer で読み戻した生バイト列を解析する。
 * bytesPerRow は 256 の倍数に揃えられているため、行ごとに切り出す必要がある。
 */
export function analyzePixels(
  bytes: Uint8Array,
  width: number,
  height: number,
  bytesPerRow: number,
  format: GPUTextureFormat,
): ImageStats {
  // bgra8unorm はメモリ上 B,G,R,A の順。取り違えると輝度が別物になる
  const bgra = format.startsWith('bgra');
  const rOff = bgra ? 2 : 0;
  const gOff = 1;
  const bOff = bgra ? 0 : 2;

  const histogram = new Array<number>(HIST_BINS).fill(0);
  let sum = 0;
  let sumSq = 0;
  let min = 1;
  let max = 0;
  let nearBlack = 0;
  let nearWhite = 0;
  const pixels = width * height;

  for (let y = 0; y < height; y++) {
    const row = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const i = row + x * 4;
      const r = bytes[i + rOff] / 255;
      const g = bytes[i + gOff] / 255;
      const b = bytes[i + bOff] / 255;
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

      sum += lum;
      sumSq += lum * lum;
      if (lum < min) min = lum;
      if (lum > max) max = lum;
      if (lum < 0.01) nearBlack++;
      if (lum > 0.99) nearWhite++;

      const bin = Math.min(HIST_BINS - 1, (lum * HIST_BINS) | 0);
      histogram[bin]++;
    }
  }

  const mean = sum / pixels;
  const variance = Math.max(0, sumSq / pixels - mean * mean);

  return {
    width,
    height,
    pixels,
    meanLuminance: mean,
    varianceLuminance: variance,
    stdDevLuminance: Math.sqrt(variance),
    minLuminance: min,
    maxLuminance: max,
    nearBlackRatio: nearBlack / pixels,
    nearWhiteRatio: nearWhite / pixels,
    extremeRatio: (nearBlack + nearWhite) / pixels,
    histogram: histogram.map((n) => n / pixels),
  };
}

/** Chrome はタイムスタンプを丸める。実測値から丸め幅を割り出す（想定せず測る） */
export function detectQuantumNs(deltas: bigint[]): number | null {
  if (deltas.length === 0) return null;
  for (const q of [100000n, 10000n, 1000n, 100n]) {
    if (deltas.every((d) => d % q === 0n)) return Number(q);
  }
  return null;
}

export function summarizeMs(values: number[]): Pick<FrameTimeStats, 'avgMs' | 'maxMs' | 'minMs'> {
  if (values.length === 0) return { avgMs: null, maxMs: null, minMs: null };
  let sum = 0;
  let max = -Infinity;
  let min = Infinity;
  for (const v of values) {
    sum += v;
    if (v > max) max = v;
    if (v < min) min = v;
  }
  return { avgMs: sum / values.length, maxMs: max, minMs: min };
}

export interface CheckInput {
  image: ImageStats;
  gpuErrors: GpuErrorRecord[];
  shaderErrors: number;
  framesRendered: number;
  framesExpected: number;
  timedOut: boolean;
  gpu: FrameTimeStats;
  cpu: FrameTimeStats;
}

export function evaluateChecks(input: CheckInput): Check[] {
  const { image, gpuErrors, shaderErrors, framesRendered, framesExpected, timedOut, gpu, cpu } =
    input;

  const pct = (n: number) => `${(n * 100).toFixed(2)}%`;

  return [
    {
      name: '真っ黒でない',
      // 初期化に失敗した画面は平均輝度がほぼ 0 になる
      ok: image.meanLuminance > 0.02,
      detail: `平均輝度 ${image.meanLuminance.toFixed(4)}（> 0.02 を要求）`,
    },
    {
      name: '真っ白でない',
      ok: image.meanLuminance < 0.98,
      detail: `平均輝度 ${image.meanLuminance.toFixed(4)}（< 0.98 を要求）`,
    },
    {
      name: '一様な塗りつぶしでない',
      // クリアしただけの画面は分散がほぼ 0 になる。何かが描けていれば分散は立つ
      ok: image.stdDevLuminance > 0.01,
      detail: `輝度の標準偏差 ${image.stdDevLuminance.toFixed(4)}（> 0.01 を要求）`,
    },
    {
      name: '極端な輝度が支配的でない',
      // NaN / Inf が混ざると 0 か飽和に振り切れた画素が一気に増える
      ok: image.extremeRatio < 0.5,
      detail: `輝度 0.01 未満 ${pct(image.nearBlackRatio)} ／ 0.99 超 ${pct(image.nearWhiteRatio)}（合計 < 50% を要求）`,
    },
    {
      name: 'GPU エラーなし',
      ok: gpuErrors.length === 0,
      detail:
        gpuErrors.length === 0
          ? 'errorScope・uncapturederror ともに 0 件'
          : gpuErrors.map((e) => `[${e.source}/${e.scope ?? '-'}] ${e.message}`).join(' / '),
    },
    {
      name: 'シェーダのコンパイルエラーなし',
      ok: shaderErrors === 0,
      detail: `${shaderErrors} 件`,
    },
    {
      name: '規定フレーム数を描けた',
      ok: !timedOut && framesRendered >= framesExpected,
      detail: `${framesRendered} / ${framesExpected} フレーム${timedOut ? '（タイムアウト）' : ''}`,
    },
    {
      name: 'フレーム時間を計測できた',
      ok: cpu.available && cpu.avgMs !== null,
      detail: `CPU 平均 ${cpu.avgMs?.toFixed(3) ?? '-'} ms ／ GPU 平均 ${
        gpu.available ? `${gpu.avgMs?.toFixed(3) ?? '-'} ms` : `取得不可（${gpu.reason ?? '理由不明'}）`
      }`,
    },
  ];
}
