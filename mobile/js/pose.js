import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

export async function createPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  try {
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  } catch {
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }
}

export async function startCamera(video, facingMode = "user") {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    },
  });
  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  await video.play();
  return stream;
}

export function stopCamera(video) {
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

export function drawPose(ctx, video, landmarks, highlight, mirrored) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!landmarks) return;

  const mapX = (x) => (mirrored ? 1 - x : x) * w;
  const mapY = (y) => y * h;

  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(46, 230, 194, 0.85)";
  ctx.fillStyle = "rgba(46, 230, 194, 0.95)";

  const connections = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [24, 26],
    [25, 27], [26, 28],
  ];
  ctx.beginPath();
  for (const [a, b] of connections) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    if (!pa || !pb) continue;
    ctx.moveTo(mapX(pa.x), mapY(pa.y));
    ctx.lineTo(mapX(pb.x), mapY(pb.y));
  }
  ctx.stroke();

  const highlightSet = new Set(highlight || []);
  for (const [i, lm] of Object.entries(landmarks)) {
    const idx = Number(i);
    const r = highlightSet.has(idx) ? 9 : 4;
    ctx.beginPath();
    ctx.fillStyle = highlightSet.has(idx) ? "#f5c542" : "rgba(255,255,255,0.9)";
    ctx.arc(mapX(lm.x), mapY(lm.y), r, 0, Math.PI * 2);
    ctx.fill();
  }
}

export { DrawingUtils, PoseLandmarker };
