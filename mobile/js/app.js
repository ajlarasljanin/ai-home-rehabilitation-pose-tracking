import {
  EXERCISES,
  RehabPipeline,
  QUALITY_LABELS,
  landmarksFromMediaPipe,
  SquatState,
} from "./pipeline.js";
import { createPoseLandmarker, startCamera, stopCamera, drawPose } from "./pose.js";

const STATE_LABELS = {
  START: "START",
  DOWN: "SPUŠTANJE",
  HOLD: "HOLD",
  UP: "PODIZANJE",
  COMPLETE: "CIKLUS",
};

const $ = (id) => document.getElementById(id);

const ui = {
  views: [...document.querySelectorAll(".view")],
};

let exercise = EXERCISES[1];
let pipeline = null;
let landmarker = null;
let stream = null;
let raf = 0;
let running = false;
let facingMode = "user";
let sessionStartedAt = 0;
let lastCue = "";
let lastCueAt = 0;
let frameIndex = 0;
let lastTs = 0;

function show(id) {
  ui.views.forEach((v) => v.classList.toggle("active", v.id === id));
}

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  const now = performance.now();
  if (text === lastCue && now - lastCueAt < 2500) return;
  lastCue = text;
  lastCueAt = now;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "sr-RS";
  u.rate = 1;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function renderExercises() {
  const root = $("exercise-list");
  root.innerHTML = "";
  EXERCISES.forEach((ex) => {
    const btn = document.createElement("button");
    btn.className = "exercise";
    btn.innerHTML = `<div class="meta">${ex.useCase}</div><b>${ex.name}</b><div>${ex.short}</div><small>${ex.description}</small>`;
    btn.onclick = () => {
      exercise = ex;
      $("setup-title").textContent = ex.name;
      $("setup-copy").textContent = ex.setup;
      show("view-setup");
    };
    root.appendChild(btn);
  });
}

async function startSession() {
  $("session-error").classList.add("hidden");
  $("session-error").textContent = "";
  show("view-session");
  pipeline = new RehabPipeline(exercise);
  frameIndex = 0;
  sessionStartedAt = performance.now();
  $("ex-name").textContent = exercise.name;

  try {
    if (!landmarker) {
      $("feedback").textContent = "Učitavanje MediaPipe modela…";
      landmarker = await createPoseLandmarker();
    }
    $("video").classList.toggle("mirror", facingMode === "user");
    stream = await startCamera($("video"), facingMode);
  } catch (err) {
    $("session-error").textContent =
      "Kamera nije dostupna. Dozvolite pristup kameri u Chrome-u, pa pokušajte ponovo. " +
      (err?.message || "");
    $("session-error").classList.remove("hidden");
    return;
  }

  running = true;
  lastTs = performance.now();
  loop();
}

function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  const video = $("video");
  const overlay = $("overlay");
  if (video.readyState < 2) return;

  const now = performance.now();
  const dtMs = now - lastTs;
  lastTs = now;

  overlay.width = overlay.clientWidth * devicePixelRatio;
  overlay.height = overlay.clientHeight * devicePixelRatio;
  const ctx = overlay.getContext("2d");

  let poseLandmarks = null;
  try {
    const result = landmarker.detectForVideo(video, now);
    poseLandmarks = result.landmarks?.[0] || null;
  } catch (e) {
    $("feedback").textContent = "Detekcija nije uspela na ovom frejmu.";
  }

  const landmarks = landmarksFromMediaPipe(poseLandmarks);
  const timestamp = (now - sessionStartedAt) / 1000;
  const evt = pipeline.process(landmarks, timestamp, frameIndex++, dtMs);
  const mirrored = facingMode === "user";
  drawPose(ctx, video, landmarks, pipeline.usedIndices, mirrored);

  const calOk = pipeline.calibrator.isCalibrated;
  $("cal-banner").classList.toggle("hidden", calOk);
  if (!calOk) {
    const n = pipeline.calibrator.nValid;
    $("cal-banner").textContent = `Kalibracija ${n}/45 — stojte uspravno, celo telo u kadru`;
  }

  const fb = pipeline.liveFeedback();
  $("semafor").className = `semafor ${fb.level}`;
  $("feedback").textContent = fb.text;
  $("count").textContent = String(pipeline.fsm.repetitionCounter);
  $("theta").textContent = `${pipeline.theta.toFixed(0)}°`;
  $("omega").textContent = `${pipeline.fsm.omega.toFixed(0)}°/s`;
  $("state").textContent = STATE_LABELS[pipeline.fsm.state] || pipeline.fsm.state;
  $("view-hint").textContent = pipeline.view.label;
  $("side-hint").textContent = pipeline.gValid
    ? `Prati ${pipeline.side === "left" ? "levu" : "desnu"} stranu`
    : "Čeka vidljive zglobove";
  $("fps").textContent = dtMs > 0 ? `${Math.round(1000 / dtMs)} FPS · ${dtMs.toFixed(0)} ms` : "";

  if (!calOk) return;

  if (evt.justCompleted) {
    const q = evt.quality;
    const msg = q === "ispravno" ? "Ponavljanje uspešno" : QUALITY_LABELS[q] || "Ciklus završen";
    speak(msg);
    if (navigator.vibrate) navigator.vibrate(q === "ispravno" ? [40, 40, 80] : 80);
  } else if (evt.justShallow) {
    speak(exercise.cues.shallow);
    if (navigator.vibrate) navigator.vibrate([30, 40, 30]);
  } else if (fb.level !== "green") {
    speak(fb.text);
  }
}

function stopLoop() {
  running = false;
  cancelAnimationFrame(raf);
  stopCamera($("video"));
}

function endSession() {
  stopLoop();
  const s = pipeline ? pipeline.summary() : null;
  const elapsed = (performance.now() - sessionStartedAt) / 1000;
  if (!s) {
    show("view-home");
    return;
  }
  $("sum-reps").textContent = s.reps;
  $("sum-good").textContent = s.good;
  $("sum-acc").textContent = `${Math.round(s.accuracy * 100)}%`;
  $("sum-rom").textContent = s.rom == null ? "—" : `${s.rom.toFixed(0)}°`;
  $("sum-min").textContent = s.thetaMin == null ? "—" : `${s.thetaMin.toFixed(0)}°`;
  $("sum-max").textContent = s.thetaMax == null ? "—" : `${s.thetaMax.toFixed(0)}°`;
  $("sum-time").textContent = `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, "0")}`;
  $("sum-ex").textContent = exercise.name;

  const list = $("sum-qualities");
  list.innerHTML = "";
  if (!s.qualities.length) {
    list.innerHTML = "<div class='quality-item'><span>Nema završenih ciklusa</span></div>";
  } else {
    s.qualities.forEach((q, i) => {
      const cls = q === "ispravno" ? "ok" : q === "nedovoljan_rom" ? "bad" : "warn";
      list.innerHTML += `<div class="quality-item"><span>Ponavljanje ${i + 1}</span><b class="${cls}">${QUALITY_LABELS[q]}</b></div>`;
    });
  }
  show("view-summary");
}

$("btn-start").onclick = () => show("view-exercises");
$("btn-about").onclick = () => show("view-about");
$("btn-install-help").onclick = () => show("view-install");
document.querySelectorAll("[data-back]").forEach((b) => {
  b.onclick = () => show(b.dataset.back);
});
$("btn-begin").onclick = () => startSession();
$("btn-flip").onclick = async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  $("video").classList.toggle("mirror", facingMode === "user");
  if (running) {
    stopCamera($("video"));
    try {
      stream = await startCamera($("video"), facingMode);
    } catch {
      $("feedback").textContent = "Promena kamere nije uspela.";
    }
  }
};
$("btn-end").onclick = () => endSession();
$("btn-again").onclick = () => show("view-exercises");
$("btn-home").onclick = () => show("view-home");

renderExercises();

let installPrompt = null;
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  $("btn-install-pwa").classList.remove("hidden");
});
$("btn-install-pwa").onclick = async () => {
  if (!installPrompt) {
    show("view-install");
    return;
  }
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  $("btn-install-pwa").classList.add("hidden");
};

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}
