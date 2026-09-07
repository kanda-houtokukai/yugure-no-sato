// 歩き手。変形を起こす主体。人物はまだ描かない（フェーズ4）。
// 入力は「筋書き（フレーム番号 → 操作）」と「実操作（WASD＋マウス）」の 2 系統。固定刻みで進めるので筋書きは再現する。

export interface WalkerInput {
  forward: number;   // -1..1（W/S）
  strafe: number;    // -1..1（D/A）
  yawDelta: number;  // 度
  pitchDelta: number;
  run: boolean;
}

export const IDLE_INPUT: WalkerInput = { forward: 0, strafe: 0, yawDelta: 0, pitchDelta: 0, run: false };

export class Walker {
  x = 0;
  z = 0;
  y = 0;
  yawDeg = 0;
  /** カメラの見下ろし角（度）。歩き手の周りを回る三人称カメラ */
  pitchDeg = 12;
  private lastDirX = 0;
  private lastDirZ = 1;
  /** 直前の 1 フレームで進んだ距離 [m]。人物の歩容が使う */
  lastMoved = 0;
  /** 直前の入力が走りだったか */
  lastRunning = false;

  // 実寸（歩き 1.4m/s）だと画面では遅く感じる。体感に合わせて上げた（2026-09-07）
  static readonly WALK_SPEED = 2.2;   // m/s
  static readonly RUN_SPEED = 5.0;
  /** 歩幅。歩容（figure.ts）が足の運びと足跡の間隔を決めるのに使う */
  static readonly STEP_WALK = 0.85;
  static readonly STEP_RUN = 1.25;
  static readonly CAMERA_DIST = 3.0;
  static readonly PIVOT_HEIGHT = 1.2;

  /**
   * 1 フレーム進める。groundHeight で足元の高さに追従。
   * 足跡はここでは打たない。人物（figure.ts）の足が実際に着いた瞬間に打つ（フェーズ6 段階3）。
   */
  step(input: WalkerInput, dt: number, groundHeight: (x: number, z: number) => number): void {
    this.yawDeg += input.yawDelta;
    this.pitchDeg = Math.max(-8, Math.min(60, this.pitchDeg + input.pitchDelta));
    const yaw = (this.yawDeg * Math.PI) / 180;
    const fx = Math.sin(yaw), fz = Math.cos(yaw);
    const rx = Math.cos(yaw), rz = -Math.sin(yaw);
    let mx = fx * input.forward + rx * input.strafe;
    let mz = fz * input.forward + rz * input.strafe;
    const len = Math.hypot(mx, mz);
    this.lastMoved = 0;
    this.lastRunning = input.run;
    if (len > 1e-6) {
      mx /= len; mz /= len;
      const speed = input.run ? Walker.RUN_SPEED : Walker.WALK_SPEED;
      const dist = speed * dt;
      this.x += mx * dist;
      this.z += mz * dist;
      this.lastDirX = mx; this.lastDirZ = mz;
      this.lastMoved = dist;
    }
    this.y = groundHeight(this.x, this.z);
  }

  /** 三人称カメラ。歩き手の腰の高さを注視点に、後方・やや上から */
  camera(groundHeight: (x: number, z: number) => number, dist = Walker.CAMERA_DIST): { eye: [number, number, number]; forward: [number, number, number] } {
    const yaw = (this.yawDeg * Math.PI) / 180;
    const p = (this.pitchDeg * Math.PI) / 180;
    const fx = Math.sin(yaw), fz = Math.cos(yaw);
    const pivot: [number, number, number] = [this.x, this.y + Walker.PIVOT_HEIGHT, this.z];
    const d = dist;
    let ex = pivot[0] - fx * d * Math.cos(p);
    let ez = pivot[2] - fz * d * Math.cos(p);
    let ey = pivot[1] + d * Math.sin(p);
    // 地面にめり込まない
    const g = groundHeight(ex, ez);
    if (ey < g + 0.4) ey = g + 0.4;
    const fwd: [number, number, number] = [pivot[0] - ex, pivot[1] - ey, pivot[2] - ez];
    const l = Math.hypot(fwd[0], fwd[1], fwd[2]) || 1;
    return { eye: [ex, ey, ez], forward: [fwd[0] / l, fwd[1] / l, fwd[2] / l] };
  }

  direction(): [number, number] {
    return [this.lastDirX, this.lastDirZ];
  }
}

/**
 * 筋書き: フレーム番号 → 操作。verify の `walk` 視点が使う。
 * walk1: 道を北へ 3 秒 → 東へ向きを変えて田へ 3 秒（泥）→ 畦を越えて隣の田へ 2 秒 → 振り返って 1 秒（跡が見える）
 */
export function scriptInput(name: string, frame: number): WalkerInput {
  const on = (f: number): WalkerInput => ({ ...IDLE_INPUT, forward: f });
  switch (name) {
    // 歩く速さを 1.4 → 2.2 m/s に上げたので、同じ経路をたどるよう秒数を 0.64 倍にした
    case 'walk1': {
      if (frame < 115) return on(1);                                              // 道を北へ 4.2m
      if (frame < 175) return { ...IDLE_INPUT, forward: 1, yawDelta: 90 / 60 };   // 1 秒で 90° 右へ
      if (frame < 290) return on(1);                                              // 田を東へ 4.2m
      if (frame < 350) return { ...IDLE_INPUT, forward: 0.4, yawDelta: 180 / 60, pitchDelta: 30 / 60 };  // 振り返り、カメラを 30° 上げる
      return IDLE_INPUT;
    }
    default:
      return IDLE_INPUT;
  }
}
