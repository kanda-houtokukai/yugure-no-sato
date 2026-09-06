// ゴールデンビュー（決め打ちの視点）。
// 毎回まったく同じ画角で撮れることが目的なので、値は固定。時刻もパラメータで固定する。

export interface View {
  readonly name: string;
  readonly eye: readonly [number, number, number];
  readonly target: readonly [number, number, number];
  readonly fovDeg: number;
  /** 秒。シェーダに渡す固定時刻。同じ値なら毎回同じ絵になる */
  readonly time: number;
}

export const VIEWS: Readonly<Record<string, View>> = {
  front: { name: 'front', eye: [0, 1.6, 7], target: [0, 1.0, 0], fovDeg: 55, time: 0 },
  bird: { name: 'bird', eye: [0, 26, 26], target: [0, 0, 0], fovDeg: 50, time: 0 },
  ground: { name: 'ground', eye: [0, 0.12, 4], target: [0, 0.1, 0], fovDeg: 65, time: 0 },
};

export const VIEW_NAMES = Object.keys(VIEWS);

function numbers(raw: string | null, count: number): number[] | null {
  if (!raw) return null;
  const parts = raw.split(',').map((s) => Number(s.trim()));
  if (parts.length !== count || parts.some((n) => !Number.isFinite(n))) return null;
  return parts;
}

/**
 * URL から視点を決める。
 * `?view=front` で名前指定。`?eye=x,y,z&target=x,y,z&fov=55&t=0` で個別に上書きもできる。
 */
export function viewFromUrl(search: string): View {
  const q = new URLSearchParams(search);
  const base = VIEWS[q.get('view') ?? 'front'] ?? VIEWS.front;

  const eye = numbers(q.get('eye'), 3);
  const target = numbers(q.get('target'), 3);
  const fov = Number(q.get('fov'));
  const t = Number(q.get('t'));

  return {
    name: q.get('view') ?? base.name,
    eye: eye ? [eye[0], eye[1], eye[2]] : base.eye,
    target: target ? [target[0], target[1], target[2]] : base.target,
    fovDeg: Number.isFinite(fov) && fov > 0 ? fov : base.fovDeg,
    time: Number.isFinite(t) ? t : base.time,
  };
}
