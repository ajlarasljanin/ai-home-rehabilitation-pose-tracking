/**
 * Referentna JS implementacija pipeline-a iz rada
 * "Primena veštačke inteligencije u rehabilitaciji u kućnim uslovima"
 * i Python paketa pose_pipeline (landmarks, filter, FSM, kalibracija, kvalitet).
 */

export const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

export const VISIBILITY_THRESHOLD = 0.5;

export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26],
  [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
];

export class Landmark {
  constructor(x, y, z, visibility) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.visibility = visibility;
  }
}

export function jointAngle(a, vertex, b) {
  const ax = a.x - vertex.x;
  const ay = a.y - vertex.y;
  const az = a.z - vertex.z;
  const bx = b.x - vertex.x;
  const by = b.y - vertex.y;
  const bz = b.z - vertex.z;
  const na = Math.hypot(ax, ay, az);
  const nb = Math.hypot(bx, by, bz);
  if (na < 1e-9 || nb < 1e-9) return null;
  let cos = (ax * bx + ay * by + az * bz) / (na * nb);
  cos = Math.min(1, Math.max(-1, cos));
  return (Math.acos(cos) * 180) / Math.PI;
}

export function segmentLength(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

export function visibilityGate(landmarks, indices, threshold = VISIBILITY_THRESHOLD) {
  let minV = 1;
  for (const i of indices) {
    const lm = landmarks[i];
    if (!lm) return false;
    minV = Math.min(minV, lm.visibility);
  }
  return minV > threshold;
}

export function pickSide(landmarks, triples) {
  let best = null;
  let bestScore = -1;
  for (const t of triples) {
    const vis = t.map((i) => landmarks[i]?.visibility ?? 0);
    const score = Math.min(...vis);
    if (score > bestScore) {
      bestScore = score;
      best = t;
    }
  }
  return { indices: best, score: bestScore };
}

/** Kauzalni Butterworth 2. reda, fc=6 Hz, fs=30 Hz — sekcija 3.3, isti koeficijenti kao scipy. */
export class ButterworthCausal {
  constructor() {
    this.b = [0.20657208382614792, 0.41314416765229584, 0.20657208382614792];
    this.a = [1.0, -0.3695273773512415, 0.19581571265583314];
    this.ziTemplate = [0.7934279161738521, 0.01075637117031478];
    this.z0 = 0;
    this.z1 = 0;
    this.initialized = false;
  }

  reset(x0 = 0) {
    this.z0 = this.ziTemplate[0] * x0;
    this.z1 = this.ziTemplate[1] * x0;
    this.initialized = true;
  }

  step(x) {
    if (!this.initialized) this.reset(x);
    const y = this.b[0] * x + this.z0;
    this.z0 = this.b[1] * x - this.a[1] * y + this.z1;
    this.z1 = this.b[2] * x - this.a[2] * y;
    return y;
  }
}

export const SquatState = {
  START: "START",
  DOWN: "DOWN",
  HOLD: "HOLD",
  UP: "UP",
  COMPLETE: "COMPLETE",
};

export class RepetitionFSM {
  constructor(config) {
    this.config = {
      thetaStartActivation: 100,
      epsHysteresis: 10,
      thetaComplete: 160,
      epsOmega: 20,
      holdConfirmWindow: 5,
      holdConfirmRatio: 4 / 5,
      tMinHold: 0.15,
      tRefractory: 0.5,
      visibilityRequired: true,
      ...config,
    };
    this.reset();
  }

  get thetaDownExtreme() {
    return this.config.thetaStartActivation - this.config.epsHysteresis;
  }

  reset() {
    this.state = SquatState.START;
    this.repetitionCounter = 0;
    this.events = [];
    this.thetaPrev = null;
    this.timePrev = null;
    this.omega = 0;
    this.guardWindow = [];
    this.holdElapsed = 0;
    this.refractoryUntil = -1;
    this.lastRejectedShallow = false;
  }

  _majority() {
    const n = this.config.holdConfirmWindow;
    if (this.guardWindow.length < n) return false;
    const sum = this.guardWindow.reduce((a, b) => a + b, 0);
    return sum / n >= this.config.holdConfirmRatio;
  }

  step(thetaT, timestamp, gValid = true, frameIndex = 0) {
    const cfg = this.config;
    this.lastRejectedShallow = false;

    if (cfg.visibilityRequired && !gValid) {
      this.thetaPrev = thetaT;
      this.timePrev = timestamp;
      return this.state;
    }

    if (this.thetaPrev === null || this.timePrev === null) {
      this.thetaPrev = thetaT;
      this.timePrev = timestamp;
      return this.state;
    }

    const dt = timestamp - this.timePrev;
    if (dt <= 0) return this.state;

    this.omega = (thetaT - this.thetaPrev) / dt;
    const inRefractory = timestamp < this.refractoryUntil;

    if (this.state === SquatState.START) {
      if (!inRefractory && thetaT < cfg.thetaStartActivation) {
        this.state = SquatState.DOWN;
        this.guardWindow = [];
        this.holdElapsed = 0;
      }
    } else if (this.state === SquatState.DOWN) {
      const guard = Math.abs(this.omega) < cfg.epsOmega && thetaT <= this.thetaDownExtreme;
      this.guardWindow.push(guard);
      if (this.guardWindow.length > cfg.holdConfirmWindow) this.guardWindow.shift();
      if (this._majority()) {
        this.holdElapsed += dt;
        if (this.holdElapsed >= cfg.tMinHold) this.state = SquatState.HOLD;
      }
      if (this.omega > 0 && thetaT > cfg.thetaStartActivation) {
        this.lastRejectedShallow = true;
        this.state = SquatState.START;
        this.guardWindow = [];
        this.holdElapsed = 0;
      }
    } else if (this.state === SquatState.HOLD) {
      const guard = this.omega > 0 && thetaT > cfg.thetaStartActivation;
      this.guardWindow.push(guard);
      if (this.guardWindow.length > cfg.holdConfirmWindow) this.guardWindow.shift();
      if (this._majority()) {
        this.state = SquatState.UP;
        this.guardWindow = [];
      }
    } else if (this.state === SquatState.UP) {
      if (thetaT > cfg.thetaComplete) this.state = SquatState.COMPLETE;
    }

    if (this.state === SquatState.COMPLETE) {
      this.repetitionCounter += 1;
      this.events.push({ frameIndex, timestamp, theta: thetaT });
      this.refractoryUntil = timestamp + cfg.tRefractory;
      this.state = SquatState.START;
      this.guardWindow = [];
      this.holdElapsed = 0;
    }

    this.thetaPrev = thetaT;
    this.timePrev = timestamp;
    return this.state;
  }
}

export class AnthropometricCalibrator {
  constructor(minSamples = 45) {
    this.minSamples = minSamples;
    this.heights = [];
    this.thighs = [];
    this.shanks = [];
  }

  get nValid() {
    return this.heights.length;
  }

  get isCalibrated() {
    return this.nValid >= this.minSamples;
  }

  addFrame(landmarks) {
    const required = [LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE];
    const right = [LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE];
    const leftOk = visibilityGate(landmarks, required);
    const rightOk = visibilityGate(landmarks, right);
    const idx = leftOk ? required : rightOk ? right : null;
    if (!idx) return;
    const [shoulder, hip, knee, ankle] = idx.map((i) => landmarks[i]);
    const thigh = segmentLength(hip, knee);
    const shank = segmentLength(knee, ankle);
    const trunk = segmentLength(shoulder, hip);
    this.heights.push(trunk + thigh + shank);
    this.thighs.push(thigh);
    this.shanks.push(shank);
  }

  result() {
    const median = (arr) => {
      const s = [...arr].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)];
    };
    return {
      heightEstimate: median(this.heights),
      thighLength: median(this.thighs),
      shankLength: median(this.shanks),
      nSamples: this.nValid,
    };
  }
}

export function trunkLeanDeg(shoulder, hip) {
  const dx = hip.x - shoulder.x;
  const dy = hip.y - shoulder.y;
  return Math.abs((Math.atan2(dx, dy) * 180) / Math.PI);
}

export function kneeValgusDeg(hip, knee, ankle) {
  const ax = ankle.x - hip.x;
  const ay = ankle.y - hip.y;
  const kx = knee.x - hip.x;
  const ky = knee.y - hip.y;
  const lineLen = Math.hypot(ax, ay);
  const thigh = Math.hypot(kx, ky);
  if (lineLen < 1e-6 || thigh < 1e-6) return 0;
  const cross = ax * ky - ay * kx;
  const deviation = Math.abs(cross) / lineLen;
  const ratio = Math.min(1, deviation / thigh);
  return (Math.asin(ratio) * 180) / Math.PI;
}

export function cameraViewHint(landmarks) {
  const ls = landmarks[LM.LEFT_SHOULDER];
  const rs = landmarks[LM.RIGHT_SHOULDER];
  const lh = landmarks[LM.LEFT_HIP];
  const rh = landmarks[LM.RIGHT_HIP];
  if (!ls || !rs || !lh || !rh) return { kind: "unknown", label: "Telo nije u kadru" };
  const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  const torso = Math.hypot((ls.x + rs.x) / 2 - (lh.x + rh.x) / 2, (ls.y + rs.y) / 2 - (lh.y + rh.y) / 2);
  const ratio = torso > 1e-6 ? shoulderWidth / torso : 0;
  if (ratio < 0.18) return { kind: "side", label: "Bočni kadar — dobar za ugao kolena" };
  if (ratio > 0.55) return { kind: "front", label: "Čelni kadar — dobar za valgus kolena" };
  return { kind: "threequarter", label: "¾ kadar — dobar kompromis" };
}

export class HeuristicQuality {
  constructor(opts = {}) {
    this.kneeValgusThreshold = opts.kneeValgusThreshold ?? 8;
    this.trunkLeanThreshold = opts.trunkLeanThreshold ?? 15;
    this.minRom = opts.minRom ?? 70;
    this.checkValgus = opts.checkValgus ?? true;
    this.checkTrunk = opts.checkTrunk ?? true;
  }

  classify({ thetaSeq, valgusSeq, trunkSeq }) {
    if (!thetaSeq.length) return "nepoznato";
    const rom = Math.max(...thetaSeq) - Math.min(...thetaSeq);
    if (rom < this.minRom) return "nedovoljan_rom";
    if (this.checkValgus && valgusSeq.length && Math.max(...valgusSeq) > this.kneeValgusThreshold) {
      return "valgus_kolena";
    }
    if (this.checkTrunk && trunkSeq.length && Math.max(...trunkSeq) > this.trunkLeanThreshold) {
      return "nagib_torza";
    }
    return "ispravno";
  }
}

export const QUALITY_LABELS = {
  ispravno: "Pravilno izvođenje",
  nedovoljan_rom: "Nedovoljan opseg pokreta",
  valgus_kolena: "Koleno ide ka unutra",
  nagib_torza: "Ispravite položaj torza",
  nepoznato: "Nema dovoljno podataka",
};

export const EXERCISES = [
  {
    id: "squat-early",
    name: "Parcijalni čučanj",
    short: "Rana rehabilitacija kolena",
    useCase: "ACL / TKE — rana faza, ciljni ugao ~110°",
    body: "koleno",
    icon: "knee",
    description: "Kontrolisano spuštanje do parcijalne fleksije. FSM broji samo ako je dostignuta dubina i HOLD faza.",
    jointTriple: {
      left: [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE],
      right: [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
    },
    fsm: { thetaStartActivation: 130, epsHysteresis: 10, thetaComplete: 160, epsOmega: 20 },
    quality: { minRom: 40, checkValgus: true, checkTrunk: true },
    cues: {
      shallow: "Nedovoljna dubina čučnja",
      valgus: "Koleno ide ka unutra",
      trunk: "Ispravite položaj torza",
      occlusion: "Koleno nije dovoljno vidljivo",
      lighting: "Poboljšajte osvetljenje",
    },
    setup: "Stavite telefon na visini kuka, 2–3 m ispred. Celo telo u kadru, ¾ ili profil.",
  },
  {
    id: "squat-target70",
    name: "Čučanj — cilj 70°",
    short: "Protokol iz rada (θ_target = 70°)",
    useCase: "Kućna rehabilitacija kolena, progresivni ROM",
    body: "koleno",
    icon: "knee",
    description: "Isti FSM kao u radu: prag aktivacije 100°, ekstremum 90°, završetak 160°. Cilj je dublja fleksija.",
    jointTriple: {
      left: [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE],
      right: [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
    },
    fsm: { thetaStartActivation: 100, epsHysteresis: 10, thetaComplete: 160, epsOmega: 20 },
    quality: { minRom: 70, checkValgus: true, checkTrunk: true },
    cues: {
      shallow: "Nedovoljna dubina čučnja",
      valgus: "Koleno ide ka unutra",
      trunk: "Ispravite položaj torza",
      occlusion: "Koleno nije dovoljno vidljivo",
      lighting: "Poboljšajte osvetljenje",
    },
    setup: "Bočni ili ¾ kadar. Ne zaklanjajte koleno rukama.",
  },
  {
    id: "arm-curl",
    name: "Pregib ruke",
    short: "Fleksija lakta",
    useCase: "Vežbe ruke — ugao ramena–lakta–zgloba",
    body: "lakat",
    icon: "arm",
    description: "Brojanje pregiba: ispružena ruka → fleksija lakta → povratak. Ista ciklična logika dole–gore.",
    jointTriple: {
      left: [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST],
      right: [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
    },
    fsm: { thetaStartActivation: 110, epsHysteresis: 15, thetaComplete: 150, epsOmega: 25 },
    quality: { minRom: 50, checkValgus: false, checkTrunk: true, trunkLeanThreshold: 18 },
    cues: {
      shallow: "Nedovoljna fleksija lakta",
      valgus: "",
      trunk: "Ne naginjite trup — radi ruka, ne leđa",
      occlusion: "Lakat ili šaka nisu vidljivi",
      lighting: "Poboljšajte osvetljenje",
    },
    setup: "Čelni kadar, cela ruka u kadru, dlan ka kameri.",
  },
];

export function landmarksFromMediaPipe(poseLandmarks) {
  const out = {};
  if (!poseLandmarks) return out;
  poseLandmarks.forEach((lm, i) => {
    out[i] = new Landmark(lm.x, lm.y, lm.z ?? 0, lm.visibility ?? lm.presence ?? 0);
  });
  return out;
}

export class RehabPipeline {
  constructor(exercise) {
    this.exercise = exercise;
    this.filter = new ButterworthCausal();
    this.fsm = new RepetitionFSM(exercise.fsm);
    this.calibrator = new AnthropometricCalibrator(45);
    this.quality = new HeuristicQuality(exercise.quality);
    this.thetaRaw = 0;
    this.theta = 0;
    this.gValid = false;
    this.side = "left";
    this.usedIndices = [];
    this.valgus = 0;
    this.trunk = 0;
    this.view = { kind: "unknown", label: "" };
    this.repBuffer = { theta: [], valgus: [], trunk: [] };
    this.repQualities = [];
    this.thetaMin = 180;
    this.thetaMax = 0;
    this.startedAt = null;
    this.frameMs = 0;
  }

  reset() {
    this.filter = new ButterworthCausal();
    this.fsm.reset();
    this.calibrator = new AnthropometricCalibrator(45);
    this.repBuffer = { theta: [], valgus: [], trunk: [] };
    this.repQualities = [];
    this.thetaMin = 180;
    this.thetaMax = 0;
    this.startedAt = null;
  }

  process(landmarks, timestamp, frameIndex, frameMs = 0) {
    this.frameMs = frameMs;
    if (!this.startedAt) this.startedAt = timestamp;
    this.view = cameraViewHint(landmarks);

    if (!this.calibrator.isCalibrated) this.calibrator.addFrame(landmarks);

    const picked = pickSide(landmarks, [
      this.exercise.jointTriple.left,
      this.exercise.jointTriple.right,
    ]);
    this.usedIndices = picked.indices || [];
    this.side =
      picked.indices && picked.indices[0] === this.exercise.jointTriple.left[0] ? "left" : "right";
    this.gValid = visibilityGate(landmarks, this.usedIndices);

    let thetaRaw = this.thetaRaw;
    if (this.gValid) {
      const [a, v, b] = this.usedIndices.map((i) => landmarks[i]);
      const ang = jointAngle(a, v, b);
      if (ang != null) thetaRaw = ang;
    }
    this.thetaRaw = thetaRaw;
    this.theta = this.filter.step(thetaRaw);

    const ls = landmarks[LM.LEFT_SHOULDER];
    const rs = landmarks[LM.RIGHT_SHOULDER];
    const lh = landmarks[LM.LEFT_HIP];
    const rh = landmarks[LM.RIGHT_HIP];
    const shoulder = ls && rs ? (ls.visibility > rs.visibility ? ls : rs) : ls || rs;
    const hip = lh && rh ? (lh.visibility > rh.visibility ? lh : rh) : lh || rh;
    this.trunk = shoulder && hip ? trunkLeanDeg(shoulder, hip) : 0;

    if (this.exercise.quality.checkValgus && this.gValid) {
      const [a, v, b] = this.usedIndices.map((i) => landmarks[i]);
      this.valgus = kneeValgusDeg(a, v, b);
    } else {
      this.valgus = 0;
    }

    const prevCount = this.fsm.repetitionCounter;
    const prevState = this.fsm.state;
    let justCompleted = false;
    let justShallow = false;

    if (this.calibrator.isCalibrated) {
      this.fsm.step(this.theta, timestamp, this.gValid, frameIndex);
      this.thetaMin = Math.min(this.thetaMin, this.theta);
      this.thetaMax = Math.max(this.thetaMax, this.theta);
      this.repBuffer.theta.push(this.theta);
      this.repBuffer.valgus.push(this.valgus);
      this.repBuffer.trunk.push(this.trunk);
      justShallow = this.fsm.lastRejectedShallow && prevState === SquatState.DOWN;
      if (this.fsm.repetitionCounter > prevCount) {
        justCompleted = true;
        const q = this.quality.classify({
          thetaSeq: this.repBuffer.theta,
          valgusSeq: this.repBuffer.valgus,
          trunkSeq: this.repBuffer.trunk,
        });
        this.repQualities.push(q);
        this.repBuffer = { theta: [], valgus: [], trunk: [] };
      }
    }

    return {
      justCompleted,
      justShallow,
      quality: this.repQualities[this.repQualities.length - 1] || null,
    };
  }

  liveFeedback() {
    const cues = this.exercise.cues;
    if (!this.gValid) return { level: "red", text: cues.occlusion, code: "occlusion" };
    if (this.fsm.lastRejectedShallow) return { level: "red", text: cues.shallow, code: "shallow" };
    if (this.exercise.quality.checkValgus && this.valgus > (this.quality.kneeValgusThreshold || 8)) {
      return { level: "yellow", text: cues.valgus, code: "valgus" };
    }
    if (this.exercise.quality.checkTrunk && this.trunk > (this.quality.trunkLeanThreshold || 15)) {
      return { level: "yellow", text: cues.trunk, code: "trunk" };
    }
    if (this.fsm.state === SquatState.HOLD || this.fsm.state === SquatState.UP) {
      return { level: "green", text: "Dobro — završite ciklus", code: "ok" };
    }
    if (this.fsm.state === SquatState.DOWN) {
      return { level: "green", text: "Spuštanje… dostižite ciljnu dubinu", code: "down" };
    }
    return { level: "green", text: "Spreman — izvedite pokret", code: "ready" };
  }

  summary() {
    const qualities = this.repQualities;
    const good = qualities.filter((q) => q === "ispravno").length;
    const duration = this.startedAt == null ? 0 : 0;
    return {
      reps: this.fsm.repetitionCounter,
      good,
      anomalies: this.fsm.repetitionCounter - good,
      accuracy: this.fsm.repetitionCounter ? good / this.fsm.repetitionCounter : 0,
      thetaMin: this.thetaMin === 180 ? null : this.thetaMin,
      thetaMax: this.thetaMax === 0 ? null : this.thetaMax,
      rom: this.thetaMin === 180 ? null : this.thetaMax - this.thetaMin,
      qualities,
      calibrated: this.calibrator.isCalibrated,
      calibration: this.calibrator.isCalibrated ? this.calibrator.result() : null,
    };
  }
}
