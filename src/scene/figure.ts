// 歩く人。逆光でシルエットになる前提の、丸みのある形。
// モーションデータは持ち込まず、三角関数と 2 骨の逆運動学で毎フレーム姿勢を作り、
// メッシュをその場で組み直す（1 体しか居ないので、GPU スキニングより見通しがよい）。
//
// 一番大事なのは「接地している足がワールド座標で止まっていること」。
// 位相から足の位置を sin で作ると必ず滑るので、足は「踏んだ地点」を覚えて、
// 接地の間はそこに固定する。腰の高さと傾きは、左右の足の高さから決める。

import { MeshBuilder, type V3 } from './mesh';
import { Walker } from './walker';

// 材質の種別（building.wgsl の kind）
export const MAT_CLOTH = 17;    // 藍の野良着・もんぺ
export const MAT_HAT = 18;      // 麦わら帽子
export const MAT_SKIN = 19;     // 肌
export const MAT_TENUGUI = 20;  // 手ぬぐい（晒し木綿）

/** 体格 [m]。背丈 約 1.63m（昭和期以前の農村の成人） */
const THIGH = 0.42;
const SHIN = 0.38;
const ANKLE = 0.075;          // 踝から足裏まで
const HIP_HALF = 0.105;       // 骨盤の半幅
// 直立時の腰関節の高さ（足裏から）。脚が伸びきる高さ（0.87）に置くと、
// 接地の終わりに毎歩かならず脚長が足りなくなり、腰が上下に暴れる。膝を少し残す
const HIP_STAND = 0.815;
const PELVIS_CHEST = 0.47;
const CHEST_NECK = 0.055;   // 胸の上端から首の付け根。長いと鶴首になる（実測）
const NECK_HEAD = 0.105;
const HEAD_R = 0.098;
const SHOULDER_HALF = 0.158;   // 肩幅 32cm。広げると案山子に見える（実測）
const UPPER_ARM = 0.30;
const FOREARM = 0.27;
const REACH = THIGH + SHIN;

const clamp = (v: number, a: number, b: number): number => (v < a ? a : v > b ? b : v);
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
/** 角度差を -180..180 に畳む */
const wrapDeg = (d: number): number => {
  let x = d % 360;
  if (x > 180) x -= 360;
  if (x < -180) x += 360;
  return x;
};

export interface FootPlant {
  x: number;
  z: number;
  y: number;
  dirX: number;
  dirZ: number;
  /** 左足 = -1、右足 = +1 */
  side: number;
  /** 踏んだ地点が水（田の泥）か */
  inWater: boolean;
}

export interface FigureStep {
  /** 体の中心（腰の真下）の世界座標 */
  x: number;
  z: number;
  /** 進みたい向き（カメラの向き）[度] */
  yawDeg: number;
  /** カメラの見下ろし角 [度]。頭が少し追う */
  pitchDeg: number;
  /** この 1 フレームで進んだ距離 [m] */
  moved: number;
  running: boolean;
  dt: number;
  groundHeight: (x: number, z: number) => number;
  /** その地点が水かを返す */
  isWater: (x: number, z: number) => boolean;
}

interface Leg {
  /** 接地中の足の位置（世界座標で固定する） */
  px: number; py: number; pz: number;
  /** 直前に離した位置 */
  qx: number; qy: number; qz: number;
  swinging: boolean;
  /** 振り出しの進み 0..1。位相から直に引くと、旋回で早めに踏み替えたいとき破綻する */
  st: number;
  /** 接地の進み 0..1。踵を返すタイミングに使う */
  ct: number;
  /** 位相による振り出しを受け付ける状態か。着地したら下ろし、位相窓を出たら戻す。
   *  これが無いと、振り出しが窓より早く終わったとき同じ脚が続けて振り出す（実測） */
  armed: boolean;
  /** 踵の浮き [m] */
  heel: number;
  /** 現在の足の位置（描画に使う） */
  fx: number; fy: number; fz: number;
}

export class Figure {
  /** 歩容の位相 0..1（1 周で左右 1 歩ずつ） */
  private phase = 0;
  private legs: Leg[] = [
    { px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, swinging: false, st: 0, ct: 0, heel: 0, armed: true, fx: 0, fy: 0, fz: 0 },
    { px: 0, py: 0, pz: 0, qx: 0, qy: 0, qz: 0, swinging: false, st: 0, ct: 0, heel: 0, armed: true, fx: 0, fy: 0, fz: 0 },
  ];
  private bodyYaw = 0;
  private headYaw = 0;
  private headPitch = 0;
  private lean = 0;
  private time = 0;
  private started = false;
  private prevGround: number | null = null;
  /** 直近の歩容の状態（レポート・検証用） */
  stats = {
    hipY: 0, roll: 0, inWater: false, phase: 0,
    /** 接地中の足が 1 フレームで動いた最大距離 [m]。0 でなければ足が滑っている */
    stanceSlipMax: 0,
    /** 接地中の足の高さと地面の高さの差の最大 [m]。浮き・めり込みの検出 */
    footErrMax: 0,
    /** 走り全体での歩数と、そのうち水の中で踏んだ数 */
    plants: 0, plantsInWater: 0,
    /** 腰の高さの最小・最大（起伏に追随しているか） */
    hipMin: 1e9, hipMax: -1e9, hipMinT: 0, hipMinGround: 0, hipMinReach: 0,
    groundMin: 1e9, groundMax: -1e9,
    /** 足元の地面の 1 フレームあたりの変化の最大 [m]。近景段を焼き直した瞬間に
     *  地形が飛ぶと、ここに歩幅では説明できない値が出る */
    groundStepMax: 0, groundStepAtT: 0,
    /** 腰の高さ − 足元の地面。歩容だけの上下（地形の起伏を除く） */
    aboveMin: 1e9, aboveMax: -1e9,
    /** 腰の高さ − 支えている足。地形も歩幅の前後も除いた、純粋な上下動 */
    overFootMin: 1e9, overFootMax: -1e9,
    /** 最小になった瞬間の脚の状態（何が腰を下げているか） */
    lowWhy: '',
  };

  private mesh = new MeshBuilder();
  private packed = new Float32Array(0);

  /** 足が地面に着いた瞬間に貯まる。world.ts が取り出して足跡を打つ */
  readonly plants: FootPlant[] = [];

  /** 前フレームの接地足の位置（滑っていないことを数値で示すため） */
  private lastStance: [number, number][] = [[NaN, NaN], [NaN, NaN]];

  private legSide(i: number): number {
    return i === 0 ? -1 : 1;
  }

  /** 1 フレーム進める。戻り値は腰の高さ（世界座標） */
  step(s: FigureStep): void {
    this.time += s.dt;
    const stepLen = s.running ? Walker.STEP_RUN : Walker.STEP_WALK;
    const swingFrac = s.running ? 0.62 : 0.48;
    const speed = s.moved / Math.max(s.dt, 1e-6);
    const moving = s.moved > 1e-5;

    // 体の向きはカメラに遅れて追い、頭はもっと早く追う（＝見回すと頭が先に回る）
    const tauBody = moving ? 0.16 : 0.55;
    const tauHead = 0.09;
    const kb = 1 - Math.exp(-s.dt / tauBody);
    const kh = 1 - Math.exp(-s.dt / tauHead);
    if (!this.started) { this.bodyYaw = s.yawDeg; this.headYaw = s.yawDeg; }
    this.bodyYaw += wrapDeg(s.yawDeg - this.bodyYaw) * kb;
    this.headYaw += wrapDeg(s.yawDeg - this.headYaw) * kh;
    this.headPitch += (clamp(s.pitchDeg, -25, 45) - this.headPitch) * kh;
    // 走ると前傾。止まると戻る
    const leanTarget = clamp(speed * 0.035, 0, 0.20);
    this.lean += (leanTarget - this.lean) * (1 - Math.exp(-s.dt / 0.25));

    const yaw = (this.bodyYaw * Math.PI) / 180;
    const fx = Math.sin(yaw), fz = Math.cos(yaw);
    const rx = Math.cos(yaw), rz = -Math.sin(yaw);

    // 初回: 足を体の左右に置く
    if (!this.started) {
      this.started = true;
      for (let i = 0; i < 2; i++) {
        const sd = this.legSide(i) * HIP_HALF;
        const x = s.x + rx * sd, z = s.z + rz * sd;
        const y = s.groundHeight(x, z);
        const l = this.legs[i];
        l.px = x; l.py = y; l.pz = z; l.qx = x; l.qy = y; l.qz = z;
        l.fx = x; l.fy = y; l.fz = z; l.swinging = false;
      }
    }

    // 水の中では足の運びが重くなる: 位相の進みを落とし、足を高く上げる
    const wet = s.isWater(s.x, s.z);
    this.stats.inWater = wet;
    const cadence = wet ? 0.78 : 1.0;
    const lift = (s.running ? 0.20 : 0.10) * (wet ? 1.9 : 1.0);

    if (moving) this.phase = (this.phase + (s.moved * cadence) / (2 * stepLen)) % 1;
    this.stats.phase = this.phase;

    let slipMax = 0;
    for (let i = 0; i < 2; i++) {
      const l = this.legs[i];
      const sd = this.legSide(i) * HIP_HALF;
      // 足を置きたい理想の位置（体の前 半歩、左右へ骨盤の半幅）
      // 接地している間に体が進む距離の半分だけ前に置くと、足の振れが前後対称になる
      const ahead = moving ? stepLen * (1 - swingFrac) : 0;
      const tx = s.x + fx * ahead + rx * sd;
      const tz = s.z + fz * ahead + rz * sd;

      const off = i === 0 ? 0 : 0.5;
      const local = (this.phase - off + 1) % 1;
      // 踏んでいる場所が体から離れすぎたら、位相を待たずに踏み替える。
      // 待たせると脚が伸びきり、腰が沈む（旋回中に 0.6m 以上しゃがんだ実測がある）
      const stray = Math.hypot(l.px - (s.x + rx * sd), l.pz - (s.z + rz * sd));
      const otherOk = !this.legs[1 - i].swinging || this.legs[1 - i].st > 0.65;
      // その場で速く向きを変えると、両足とも踏み替えたい状況が起きる。
      // 片足ずつに縛ると脚が伸びきり、腰が 0.7m 沈んだ（実測）。深刻なときは同時を許す
      // 閾値は「歩行で自然に生じる前後の振れ」より大きく取る。
      // ahead（= stepLen×(1-swingFrac) ≒ 0.44m）より小さくすると、
      // 接地の終わりに毎回よけいな踏み替えが起きて左右が交互にならない（実測）
      if (local >= swingFrac) l.armed = true;   // 位相窓を出たので次の振り出しを受け付ける
      // 歩行で自然に生じる前後の振れは ahead（≒0.44m）まで。それを超えたら位相を待たずに踏み替える。
      // 歩き始めは両足が揃っているので、片脚の接地が 1 m 以上続いて脚が伸びきる（実測）
      const strayTrigger = (stray > 0.50 && otherOk) || stray > 0.66;
      const wantSwing = moving
        ? (local < swingFrac && l.armed) || strayTrigger
        : (stray > 0.36 && otherOk) || stray > 0.58;

      if (wantSwing && !l.swinging) {
        // 位相を待たずに踏み替えたときは、位相をこの脚に合わせ直す。
        // 合わせないと以後ずっと左右が揃わない
        if (strayTrigger && local >= swingFrac) this.phase = i === 0 ? 0 : 0.5;
        l.swinging = true;
        l.armed = false;
        l.st = 0;
        l.ct = 0;
        l.qx = l.px; l.qy = l.py; l.qz = l.pz;
      }
      if (l.swinging) {
        // 振り出しにかかる時間は、その速さで半歩を運ぶのに要る時間
        const dur = moving ? Math.max(0.12, (2 * stepLen * swingFrac) / Math.max(speed, 0.2)) : 0.32;
        l.st = Math.min(1, l.st + s.dt / dur);
        const t = l.st;
        // 前半はゆっくり離し、後半で素早く前へ出す
        const e = t * t * (3 - 2 * t);
        const cx = lerp(l.qx, tx, e), cz = lerp(l.qz, tz, e);
        const gy = s.groundHeight(cx, cz);
        l.fx = cx; l.fz = cz;
        l.fy = lerp(l.qy, gy, e) + lift * Math.sin(Math.PI * t);
        if (t >= 1) {
          // 着地。ここで初めて足跡を打つ（歩幅で機械的に打たない）
          l.swinging = false;
          l.px = cx; l.pz = cz; l.py = s.groundHeight(cx, cz);
          l.fx = l.px; l.fy = l.py; l.fz = l.pz;
          const wetHere = s.isWater(l.px, l.pz);
          this.plants.push({
            x: l.px, y: l.py, z: l.pz, dirX: fx, dirZ: fz,
            side: this.legSide(i), inWater: wetHere,
          });
          this.stats.plants++;
          if (wetHere) this.stats.plantsInWater++;
        }
      } else {
        // 接地中はワールド座標で完全に固定する（滑らせない）
        l.fx = l.px; l.fy = l.py; l.fz = l.pz;
        // 蹴り出しで踵が返る。これが無いと、歩幅 0.85m に対し脚長 0.87m では
        // 接地の終わりに脚が伸びきり、腰が 25cm も上下する（実測）
        const stanceDur = moving ? Math.max(0.14, (2 * stepLen * (1 - swingFrac)) / Math.max(speed, 0.2)) : 1e9;
        l.ct = Math.min(1, l.ct + s.dt / stanceDur);
        const rise = l.ct < 0.5 ? 0 : (l.ct - 0.5) / 0.5;
        l.heel = rise * rise * (3 - 2 * rise) * (s.running ? 0.13 : 0.095);
        const prev = this.lastStance[i];
        if (!Number.isNaN(prev[0])) slipMax = Math.max(slipMax, Math.hypot(l.fx - prev[0], l.fz - prev[1]));
        this.lastStance[i] = [l.fx, l.fz];
      }
      if (l.swinging) { this.lastStance[i] = [NaN, NaN]; l.heel = 0; }
    }
    this.stats.stanceSlipMax = Math.max(this.stats.stanceSlipMax, slipMax);
    // 接地中の足が地面から浮いていない/めり込んでいないこと
    for (let i = 0; i < 2; i++) {
      const l = this.legs[i];
      if (!l.swinging) {
        this.stats.footErrMax = Math.max(this.stats.footErrMax, Math.abs(l.fy - s.groundHeight(l.fx, l.fz)));
      }
    }

    // 腰の高さ: 両脚とも伸びきらない高さに収める（＝足が地面から離れない）
    const bob = moving ? 0.022 * Math.cos(4 * Math.PI * this.phase) : 0.006 * Math.sin(this.time * 1.5);
    // 支えているのは接地している足。振り出し中の足は持ち上がっているので、
    // これを含めると腰まで一緒に浮く（実測で 12cm 浮いた）
    let support = -1e9;
    for (const l of this.legs) if (!l.swinging) support = Math.max(support, l.fy);
    // 両足とも振り出し中（速い旋回でだけ起きる）は、体の足元の地面を支えとみなす
    if (support < -1e8) support = s.groundHeight(s.x, s.z);
    let hipY = support + HIP_STAND - ANKLE - bob;
    // 脚の伸びで腰を落とすのは自然だが、落とし過ぎは「しゃがみ」になる。18cm で止める
    const hipFloor = support + HIP_STAND - ANKLE - 0.18;
    const nominal = hipY;
    const limits: number[] = [];
    const dhs: number[] = [];
    for (let i = 0; i < 2; i++) {
      const l = this.legs[i];
      const sd = this.legSide(i) * HIP_HALF;
      const jx = s.x + rx * sd, jz = s.z + rz * sd;
      const dh = Math.hypot(jx - l.fx, jz - l.fz);
      const maxUp = REACH * 0.985;
      const dy = Math.sqrt(Math.max(0.0025, maxUp * maxUp - dh * dh));
      const lim = l.fy + ANKLE + l.heel + dy;
      limits.push(lim); dhs.push(dh);
      hipY = Math.min(hipY, lim);
    }
    hipY = Math.max(hipY, hipFloor);
    this.stats.hipY = hipY;
    const g = s.groundHeight(s.x, s.z);
    if (this.prevGround !== null) {
      const d = Math.abs(g - this.prevGround);
      if (d > this.stats.groundStepMax) {
        this.stats.groundStepMax = Number(d.toFixed(5));
        this.stats.groundStepAtT = Number(this.time.toFixed(2));
      }
    }
    this.prevGround = g;
    this.stats.groundMin = Math.min(this.stats.groundMin, g);
    this.stats.groundMax = Math.max(this.stats.groundMax, g);
    this.stats.aboveMin = Math.min(this.stats.aboveMin, hipY - g);
    this.stats.aboveMax = Math.max(this.stats.aboveMax, hipY - g);
    if (hipY - support < this.stats.overFootMin) {
      this.stats.overFootMin = hipY - support;
      this.stats.lowWhy = `nominal=${(nominal - support).toFixed(3)} `
        + this.legs.map((l, i) => `脚${i}${l.swinging ? '振' : '接'} dh=${dhs[i].toFixed(3)} 踵=${l.heel.toFixed(3)} 限=${(limits[i] - support).toFixed(3)}`).join(' | ');
    }
    this.stats.overFootMax = Math.max(this.stats.overFootMax, hipY - support);
    if (hipY < this.stats.hipMin) {
      this.stats.hipMin = hipY;
      this.stats.hipMinT = Number(this.time.toFixed(2));
      this.stats.hipMinGround = Number(s.groundHeight(s.x, s.z).toFixed(3));
      this.stats.hipMinReach = Number((hipY - support).toFixed(3));
    }
    this.stats.hipMax = Math.max(this.stats.hipMax, hipY);
    // 骨盤の傾き: 低い方の足へ傾く
    const roll = Math.atan2(this.legs[1].fy - this.legs[0].fy, 2 * HIP_HALF) * 0.5;
    this.stats.roll = roll;

    this.build(s, hipY, roll, fx, fz, rx, rz, s.running);
  }

  // ---- 姿勢からメッシュを組む ----

  private build(
    s: FigureStep, hipY: number, roll: number,
    fx: number, fz: number, rx: number, rz: number,
    running: boolean,
  ): void {
    const m = this.mesh;
    m.data.length = 0;   // 容量は保つ（毎フレーム組み直すので確保し直さない）
    const ox = s.x, oz = s.z, oy = s.groundHeight(s.x, s.z);
    // 図形はすべて「足元の地面」を原点にした局所座標で作る（インスタンスの位置に足元を渡す）
    const P = (x: number, y: number, z: number): V3 => [x - ox, y - oy, z - oz];

    const moving = s.moved > 1e-5;
    const speed = s.moved / Math.max(s.dt, 1e-6);
    const leanX = fx * this.lean, leanZ = fz * this.lean;

    // 骨盤: 傾き（roll）を左右方向に効かせる
    const hipDrop = Math.sin(roll) * HIP_HALF;
    const hipL: V3 = [ox + rx * -HIP_HALF, hipY - hipDrop, oz + rz * -HIP_HALF];
    const hipR: V3 = [ox + rx * HIP_HALF, hipY + hipDrop, oz + rz * HIP_HALF];
    const hipC: V3 = [(hipL[0] + hipR[0]) / 2, (hipL[1] + hipR[1]) / 2, (hipL[2] + hipR[2]) / 2];

    // --- 脚（2 骨の逆運動学。膝は前へ折れる） ---
    for (let i = 0; i < 2; i++) {
      const l = this.legs[i];
      const hip = i === 0 ? hipL : hipR;
      const ankle: V3 = [l.fx, l.fy + ANKLE + l.heel, l.fz];
      const knee = ik2(hip, ankle, THIGH, SHIN, [fx, 0.25, fz]);
      // もんぺ（太もも〜脛）。布なので脚より太く、膝で一度くびれる
      m.tube(P(hip[0], hip[1], hip[2]), P(knee[0], knee[1], knee[2]), 0.115, 0.088, 8, MAT_CLOTH);
      m.tube(P(knee[0], knee[1], knee[2]), P(ankle[0], ankle[1] + 0.055, ankle[2]), 0.088, 0.070, 8, MAT_CLOTH);
      // 裾を足首で絞る（もんぺ・脚絆）
      m.tube(P(ankle[0], ankle[1] + 0.075, ankle[2]), P(ankle[0], ankle[1] + 0.015, ankle[2]), 0.072, 0.058, 8, MAT_CLOTH);
      // 足: 踝から爪先へ。接地中は水平、振り出し中はつま先が下がる。
      // 細い筒だと棒に見えるので、足首から先は平たく大きくする（草鞋を履いた足）
      const toeDrop = l.swinging ? 0.045 : 0.0;
      const toe: V3 = [l.fx + fx * 0.150, l.fy + 0.026 - toeDrop, l.fz + fz * 0.150];
      const heel: V3 = [l.fx - fx * 0.060, l.fy + 0.034 + l.heel * 1.5, l.fz - fz * 0.060];
      m.tube(P(heel[0], heel[1], heel[2]), P(toe[0], toe[1], toe[2]), 0.062, 0.044, 6, MAT_SKIN);
      m.tube(P(ankle[0], ankle[1] + 0.02, ankle[2]), P(heel[0], heel[1], heel[2]), 0.070, 0.062, 6, MAT_SKIN);
    }

    // --- 胴 ---
    const chestYawDeg = this.bodyYaw + wrapDeg(this.headYaw - this.bodyYaw) * 0.42;
    const cy = (chestYawDeg * Math.PI) / 180;
    const cfx = Math.sin(cy), cfz = Math.cos(cy);
    const crx = Math.cos(cy), crz = -Math.sin(cy);
    const breathe = moving ? 0 : 0.008 * Math.sin(this.time * 1.5);
    const chest: V3 = [
      hipC[0] + leanX * PELVIS_CHEST,
      hipC[1] + PELVIS_CHEST + breathe,
      hipC[2] + leanZ * PELVIS_CHEST,
    ];
    // 胴。面を増やして丸くする（8 面だと箱の積み重ねに見える）。
    // 太さは実寸に寄せる（筒なので、正面から見て太く見えすぎない値にする）
    m.tube(P(hipC[0], hipC[1] - 0.04, hipC[2]), P(chest[0], chest[1], chest[2]), 0.135, 0.150, 12, MAT_CLOTH);
    // 肩: 上へすぼめて撫で肩にする。水平に張り出させると案山子になる
    m.tube(P(chest[0], chest[1] - 0.01, chest[2]),
      P(chest[0], chest[1] + 0.062, chest[2]), 0.163, 0.112, 12, MAT_CLOTH);
    // 野良着の裾。腰のあたりで止める（長いと僧衣、広いと袋に見える）
    m.tube(P(hipC[0], hipC[1] + 0.16, hipC[2]), P(hipC[0] - leanX * 0.10, hipC[1] - 0.055, hipC[2] - leanZ * 0.10),
      0.150, 0.182, 12, MAT_CLOTH);

    // --- 腕（脚と逆位相に振る） ---
    // 位相の対応: 左脚は phase 0 で最も後ろ、0.5 で最も前（振り出しが 0〜swingFrac）。
    // よって左腕は cos(2πphase)、右腕は -cos(2πphase) で「右足が前なら左腕が前」になる。
    // sin を使うと 1/4 周期ずれ、足と腕の関係が崩れる（実測で外していた）
    const armPhase = 2 * Math.PI * this.phase;
    // 遠景ではシルエットしか見えず、腕の振りが「歩いている」ことを伝える主な手がかりになる。
    // 実寸の歩行（±20°前後）より一段大きく振る
    const ampScale = clamp(speed / 2.2, 0.35, 1.6);
    // [m] 肘の前後の振れ幅。上腕の長さ（0.30m）に近づけると肩の角度が 60°を超え、
    // 行進に見える（実測）。歩行で肩角 ±30°、走りで ±45° に収まる値にする
    const swingAmp = (running ? 0.212 : 0.150) * ampScale;
    for (let i = 0; i < 2; i++) {
      const sd = this.legSide(i);
      // 肩の関節は胴の中に入れる。表面から出すと、腕と胴の間に隙間が見える
      const sh: V3 = [
        chest[0] + crx * sd * (SHOULDER_HALF - 0.045),
        chest[1] + 0.038,
        chest[2] + crz * sd * (SHOULDER_HALF - 0.045),
      ];
      const a = moving ? (i === 0 ? 1 : -1) * Math.cos(armPhase) * swingAmp
        : 0.012 * Math.sin(this.time * 1.5 + i * Math.PI);
      // 腕は肩を中心に振れる振り子。前後のずれを角度に直して長さを保つ
      // （y を決め打ちにすると、前に振るほど腕が伸びる）
      const t1 = Math.asin(clamp(a / UPPER_ARM, -0.92, 0.92));
      // 真後ろから見ると前後の振れは短縮して見えない。後ろへ振るときに外へ開かせて、
      // 三人称カメラ（ほぼ真後ろ）でも輪郭が動くようにする
      const outSwing = 0.070 + 0.040 * Math.max(0, -a / Math.max(swingAmp, 1e-3));
      const elbow: V3 = [
        sh[0] + cfx * Math.sin(t1) * UPPER_ARM + crx * sd * outSwing,
        sh[1] - Math.cos(t1) * UPPER_ARM,
        sh[2] + cfz * Math.sin(t1) * UPPER_ARM + crz * sd * outSwing,
      ];
      // 肘は常に少し曲げ、腕が前に出るほど深く折る。走ると深い
      const bend = (running ? 0.60 : 0.26) + Math.max(0, a / Math.max(swingAmp, 1e-3)) * (running ? 0.75 : 0.45);
      const t2 = t1 + bend;
      const wrist: V3 = [
        elbow[0] + cfx * Math.sin(t2) * FOREARM + crx * sd * (outSwing * 0.62),
        elbow[1] - Math.cos(t2) * FOREARM,
        elbow[2] + cfz * Math.sin(t2) * FOREARM + crz * sd * (outSwing * 0.62),
      ];
      const hand: V3 = [
        wrist[0] + cfx * Math.sin(t2) * 0.085,
        wrist[1] - Math.cos(t2) * 0.085,
        wrist[2] + cfz * Math.sin(t2) * 0.085,
      ];
      // 袖（肘まで）と、そこから先の腕
      m.tube(P(sh[0], sh[1], sh[2]), P(elbow[0], elbow[1], elbow[2]), 0.068, 0.050, 8, MAT_CLOTH);
      m.tube(P(elbow[0], elbow[1], elbow[2]), P(wrist[0], wrist[1], wrist[2]), 0.046, 0.034, 6, MAT_SKIN);
      m.tube(P(wrist[0], wrist[1], wrist[2]), P(hand[0], hand[1], hand[2]), 0.040, 0.026, 6, MAT_SKIN);
    }

    // --- 首・頭・帽子 ---
    const hy = (this.headYaw * Math.PI) / 180;
    const hfx = Math.sin(hy), hfz = Math.cos(hy);
    const neck: V3 = [chest[0], chest[1] + CHEST_NECK, chest[2]];
    const headPitchRad = (this.headPitch * Math.PI) / 180 * 0.35;
    const headC: V3 = [
      neck[0] + hfx * NECK_HEAD * Math.sin(-headPitchRad) + hfx * 0.012,
      neck[1] + NECK_HEAD * Math.cos(headPitchRad),
      neck[2] + hfz * NECK_HEAD * Math.sin(-headPitchRad) + hfz * 0.012,
    ];
    m.tube(P(neck[0], neck[1] - 0.03, neck[2]), P(headC[0], headC[1] - HEAD_R * 0.6, headC[2]), 0.058, 0.052, 8, MAT_SKIN);
    // 頭: 縦にわずかに長い塊（顔は作らない）
    ellipsoid(m, P(headC[0], headC[1], headC[2]), HEAD_R, HEAD_R * 1.14, HEAD_R * 0.98, 10, 6, MAT_SKIN);

    // 麦わら帽子: 浅い丸い山と、広く平たいつば。
    // 円錐にすると菅笠になって尖りすぎる（実測）。麦わら帽子は山が低く、つばが水平に近い
    const brimR = 0.225;
    const crownR = 0.130;
    const crownH = 0.070;
    const tilt = -0.022;   // ごくわずかに前下がり
    const brimY = headC[1] + HEAD_R * 0.42;
    const hcx = headC[0] + hfx * tilt, hcz = headC[2] + hfz * tilt;
    const n = 16;
    const ringPt = (r: number, a: number, y: number): V3 => [hcx + Math.cos(a) * r, y, hcz + Math.sin(a) * r];
    for (let k = 0; k < n; k++) {
      const a0 = (k / n) * Math.PI * 2, a1 = ((k + 1) / n) * Math.PI * 2;
      // つばの縁は編みのむらで少し波打つ
      const w0 = 0.010 * Math.sin(k * 2.1) + 0.006 * Math.sin(k * 5.3);
      const w1 = 0.010 * Math.sin((k + 1) * 2.1) + 0.006 * Math.sin((k + 1) * 5.3);
      const droop = 0.030;   // 外へ行くほどわずかに垂れる
      const iA = ringPt(crownR, a0, brimY), iB = ringPt(crownR, a1, brimY);
      const oA = ringPt(brimR + w0, a0, brimY - droop), oB = ringPt(brimR + w1, a1, brimY - droop);
      m.quad(P(iA[0], iA[1], iA[2]), P(iB[0], iB[1], iB[2]), P(oB[0], oB[1], oB[2]), P(oA[0], oA[1], oA[2]), MAT_HAT);
      // 山: 2 段に分けて丸みを出す
      const mA = ringPt(crownR * 0.72, a0, brimY + crownH * 0.72);
      const mB = ringPt(crownR * 0.72, a1, brimY + crownH * 0.72);
      m.quad(P(iA[0], iA[1], iA[2]), P(mA[0], mA[1], mA[2]), P(mB[0], mB[1], mB[2]), P(iB[0], iB[1], iB[2]), MAT_HAT);
      const top: V3 = [hcx, brimY + crownH, hcz];
      m.polygon([P(mA[0], mA[1], mA[2]), P(top[0], top[1], top[2]), P(mB[0], mB[1], mB[2])], MAT_HAT);
    }

    // 手ぬぐい: 帯に挟んで腰に下げる。首に掛けると肩に板が張り付いて見えた（実測）。
    // 薄い木綿なので逆光で白く透け、腰の位置と歩調の揺れが読める
    const tw = 0.045;
    const beltY = hipC[1] + 0.04;
    const tside = -1;   // 右腰
    const tbx = hipC[0] + crx * tside * 0.135, tbz = hipC[2] + crz * tside * 0.135;
    const sway = moving ? 0.030 * Math.sin(armPhase + 0.8) : 0.008 * Math.sin(this.time * 1.5);
    const tnA: V3 = [tbx - cfx * tw, beltY, tbz - cfz * tw];
    const tnB: V3 = [tbx + cfx * tw, beltY, tbz + cfz * tw];
    const drop = 0.26;
    m.quad(
      P(tnA[0], tnA[1], tnA[2]), P(tnB[0], tnB[1], tnB[2]),
      P(tnB[0] + crx * tside * 0.02 + cfx * sway, tnB[1] - drop, tnB[2] + crz * tside * 0.02 + cfz * sway),
      P(tnA[0] + crx * tside * 0.02 + cfx * sway, tnA[1] - drop * 0.94, tnA[2] + crz * tside * 0.02 + cfz * sway),
      MAT_TENUGUI,
    );

    // 詰め直す（Float32Array は容量が足りるときだけ使い回す）
    const need = m.data.length;
    if (this.packed.length < need) this.packed = new Float32Array(Math.ceil(need * 1.3));
    for (let i = 0; i < need; i++) this.packed[i] = m.data[i];
    this.vertexFloats = need;
  }

  private vertexFloats = 0;

  /** この フレームの姿勢のメッシュ（局所座標。原点は足元の地面） */
  meshData(): Float32Array {
    return this.packed.subarray(0, this.vertexFloats);
  }
}

/** 回転楕円体。頭に使う */
function ellipsoid(m: MeshBuilder, c: V3, rx: number, ry: number, rz: number, seg: number, ring: number, kind: number): void {
  const at = (i: number, j: number): V3 => {
    const th = (j / ring) * Math.PI;
    const ph = (i / seg) * Math.PI * 2;
    return [c[0] + rx * Math.sin(th) * Math.cos(ph), c[1] + ry * Math.cos(th), c[2] + rz * Math.sin(th) * Math.sin(ph)];
  };
  for (let j = 0; j < ring; j++) {
    for (let i = 0; i < seg; i++) {
      const a = at(i, j), b = at(i + 1, j), cc = at(i + 1, j + 1), d = at(i, j + 1);
      if (j === 0) m.polygon([a, cc, d], kind);
      else if (j === ring - 1) m.polygon([a, b, cc], kind);
      else m.quad(a, b, cc, d, kind);
    }
  }
}

/**
 * 2 骨の逆運動学。hip と foot を結び、長さ l1・l2 の骨で届かせる。
 * pole は関節が折れる向き（膝なら前）。
 */
function ik2(hip: V3, foot: V3, l1: number, l2: number, pole: V3): V3 {
  let dx = foot[0] - hip[0], dy = foot[1] - hip[1], dz = foot[2] - hip[2];
  let d = Math.hypot(dx, dy, dz);
  const maxD = (l1 + l2) * 0.995;
  if (d > maxD) { const k = maxD / d; dx *= k; dy *= k; dz *= k; d = maxD; }
  if (d < 1e-4) d = 1e-4;
  const ux = dx / d, uy = dy / d, uz = dz / d;
  const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
  // pole を骨の軸に直交させる
  const pd = pole[0] * ux + pole[1] * uy + pole[2] * uz;
  let px = pole[0] - ux * pd, py = pole[1] - uy * pd, pz = pole[2] - uz * pd;
  const pl = Math.hypot(px, py, pz);
  if (pl < 1e-5) { px = 0; py = 1; pz = 0; }
  else { px /= pl; py /= pl; pz /= pl; }
  return [hip[0] + ux * a + px * h, hip[1] + uy * a + py * h, hip[2] + uz * a + pz * h];
}
