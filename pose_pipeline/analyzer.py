"""
analyzer.py

MotionAnalyzer spaja detekciju poze, ugao, filter, FSM, overlay i plotove.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import AnthropometricCalibrator
from .estimator import PoseEstimator, mediapipe_available
from .evaluation import LatencyBenchmark
from .filtering import FilterMode, KinematicFilter
from .fsm import FSMConfig, SquatFSM, SquatState
from .interpolation import CausalAngleHold
from .landmarks import (
    KNEE_CHAIN_LEFT,
    KNEE_CHAIN_RIGHT,
    Landmark,
    knee_angle,
    normalize_skeleton,
    shoulder_angle,
    visibility_gate,
)
from .segmentation import CycleSegment, detect_body_momentum
from .visualization import TrafficLight, draw_overlay, save_thesis_figures, traffic_light_for


class FrameResult:
    __slots__ = (
        "frame_index", "timestamp", "theta_raw", "theta_filtered", "omega",
        "g_valid", "interpolated", "fsm_state", "repetition_counter",
        "cycle_completed", "theta_shoulder", "compensation", "traffic_light",
        "t_start", "t_end",
    )

    def __init__(
        self,
        frame_index: int,
        timestamp: float,
        theta_raw: float | None,
        theta_filtered: float | None,
        omega: float,
        g_valid: bool,
        interpolated: bool,
        fsm_state: SquatState,
        repetition_counter: int,
        cycle_completed: bool,
        theta_shoulder: float | None,
        compensation: bool,
        traffic_light: TrafficLight,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> None:
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.theta_raw = theta_raw
        self.theta_filtered = theta_filtered
        self.omega = omega
        self.g_valid = g_valid
        self.interpolated = interpolated
        self.fsm_state = fsm_state
        self.repetition_counter = repetition_counter
        self.cycle_completed = cycle_completed
        self.theta_shoulder = theta_shoulder
        self.compensation = compensation
        self.traffic_light = traffic_light
        self.t_start = t_start
        self.t_end = t_end


class MotionAnalyzer:
    """
    Sinhroni pipeline po frejmu (sekcija 6.2) plus offline izvestaji.

    Real-time grana nikad ne zove filtfilt ni centralnu razliku.
    """

    def __init__(
        self,
        fsm_config: FSMConfig | None = None,
        cutoff_hz: float = 6.0,
        sample_rate_hz: float = 30.0,
        side: str = "left",
        measure_latency: bool = True,
        enable_pose: bool = False,
        compensation_threshold_deg: float = 10.0,
        segments_dir: str | Path | None = None,
    ) -> None:
        self.side = side
        self.sample_rate_hz = sample_rate_hz
        self.compensation_threshold_deg = compensation_threshold_deg
        self.segments_dir = Path(segments_dir) if segments_dir else Path("output") / "lstm_segments"
        self._knee_chain = KNEE_CHAIN_LEFT if side == "left" else KNEE_CHAIN_RIGHT

        self.filter = KinematicFilter(
            mode=FilterMode.REALTIME,
            cutoff_hz=cutoff_hz,
            sample_rate_hz=sample_rate_hz,
        )
        self.offline_filter = KinematicFilter(
            mode=FilterMode.OFFLINE,
            cutoff_hz=cutoff_hz,
            sample_rate_hz=sample_rate_hz,
        )
        self.fsm = SquatFSM(config=fsm_config)
        self.calibrator = AnthropometricCalibrator()
        self.hold = CausalAngleHold()
        self.latency = LatencyBenchmark(target_fps=sample_rate_hz) if measure_latency else None
        self.estimator: PoseEstimator | None = None
        if enable_pose:
            self.estimator = PoseEstimator()

        self._results: list[FrameResult] = []
        self._landmarks_hist: list[dict[int, Landmark] | None] = []
        self.segments: list[CycleSegment] = []
        self._cycle_shoulder: list[float] = []

    def close(self) -> None:
        if self.estimator is not None:
            self.estimator.close()
            self.estimator = None

    def process_bgr(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> tuple[FrameResult, np.ndarray]:
        if self.estimator is None:
            if not mediapipe_available():
                raise ImportError("MediaPipe nije dostupan; predajte landmarks preko process_frame().")
            self.estimator = PoseEstimator()
        landmarks = self.estimator.process(frame_bgr)
        result = self.process_frame(landmarks, timestamp, frame_index, pose_detected=landmarks is not None)
        overlay = draw_overlay(
            frame_bgr,
            landmarks,
            result.theta_filtered,
            result.fsm_state,
            result.repetition_counter,
            result.traffic_light,
            compensation=result.compensation,
            omega_deg_s=result.omega,
        )
        return result, overlay

    def process_frame(
        self,
        landmarks: dict[int, Landmark] | None,
        timestamp: float,
        frame_index: int,
        pose_detected: bool | None = None,
    ) -> FrameResult:
        if self.latency is not None:
            with self.latency.measure("frame_total"):
                result = self._process_frame_impl(landmarks, timestamp, frame_index, pose_detected)
        else:
            result = self._process_frame_impl(landmarks, timestamp, frame_index, pose_detected)

        self._landmarks_hist.append(landmarks)
        self._results.append(result)
        if result.cycle_completed:
            self._emit_segment(result.t_start or timestamp, result.t_end or timestamp)
            self._cycle_shoulder.clear()
        return result

    def _process_frame_impl(
        self,
        landmarks: dict[int, Landmark] | None,
        timestamp: float,
        frame_index: int,
        pose_detected: bool | None,
    ) -> FrameResult:
        detected = landmarks is not None if pose_detected is None else pose_detected
        if landmarks is not None and not self.calibrator.is_calibrated:
            self.calibrator.add_frame(landmarks)

        g_valid = bool(landmarks is not None and visibility_gate(landmarks, self._knee_chain))
        theta_raw: float | None = None
        theta_shoulder: float | None = None
        if landmarks is not None:
            try:
                theta_raw = knee_angle(landmarks, side=self.side)
            except ValueError:
                g_valid = False
                theta_raw = None
            try:
                theta_shoulder = shoulder_angle(landmarks, side=self.side)
            except (ValueError, KeyError):
                theta_shoulder = None

        theta_held, used_hold = self.hold.step(theta_raw, g_valid)
        interpolated = used_hold and theta_held is not None

        theta_f: float | None = None
        omega = 0.0
        fsm_res = None
        if theta_held is not None:
            dt = None
            if self._results:
                dt = timestamp - self._results[-1].timestamp
            sample = self.filter.step(theta_held, dt=dt)
            theta_f = sample.theta_filtered
            omega = sample.omega
            fsm_res = self.fsm.evaluate(
                theta_f,
                timestamp,
                g_valid=g_valid,
                frame_index=frame_index,
                omega_t=omega,
            )

        state = fsm_res.state if fsm_res else self.fsm.state
        cycle_completed = bool(fsm_res.cycle_completed) if fsm_res else False

        if theta_shoulder is not None and (state != SquatState.START or cycle_completed):
            self._cycle_shoulder.append(theta_shoulder)

        compensation_live = detect_body_momentum(
            np.asarray(self._cycle_shoulder, dtype=np.float64),
            self.compensation_threshold_deg,
        ) if self._cycle_shoulder else False

        light = traffic_light_for(g_valid, state, compensation_live, detected)
        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            theta_raw=theta_raw,
            theta_filtered=theta_f,
            omega=omega,
            g_valid=g_valid,
            interpolated=interpolated,
            fsm_state=state,
            repetition_counter=self.fsm.repetition_counter,
            cycle_completed=cycle_completed,
            theta_shoulder=theta_shoulder,
            compensation=compensation_live,
            traffic_light=light,
            t_start=fsm_res.t_start if fsm_res else None,
            t_end=fsm_res.t_end if fsm_res else None,
        )

    def _emit_segment(self, t_start: float, t_end: float) -> CycleSegment | None:
        idxs = [i for i, r in enumerate(self._results) if t_start <= r.timestamp <= t_end]
        if not idxs:
            return None
        height = None
        if self.calibrator.is_calibrated:
            height = float(self.calibrator.result()["height_estimate"])

        coords = []
        thetas, omegas, shoulders, times = [], [], [], []
        for i in idxs:
            r = self._results[i]
            lm = self._landmarks_hist[i]
            if lm is None:
                coords.append(np.full((33, 3), np.nan))
            else:
                coords.append(normalize_skeleton(lm, height))
            thetas.append(r.theta_filtered if r.theta_filtered is not None else np.nan)
            omegas.append(r.omega)
            shoulders.append(r.theta_shoulder if r.theta_shoulder is not None else np.nan)
            times.append(r.timestamp)

        compensation = detect_body_momentum(
            np.asarray(shoulders), self.compensation_threshold_deg
        )
        seg = CycleSegment(
            rep_index=self.fsm.repetition_counter,
            t_start=t_start,
            t_end=t_end,
            timestamps=np.asarray(times),
            landmarks_norm=np.stack(coords, axis=0),
            theta_knee_deg=np.asarray(thetas),
            omega_deg_s=np.asarray(omegas),
            theta_shoulder_deg=np.asarray(shoulders),
            compensation_detected=compensation,
        )
        self.segments.append(seg)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        stem = self.segments_dir / f"rep_{seg.rep_index:03d}"
        seg.save(stem.with_suffix(".json"))
        np.save(stem.with_suffix(".npy"), seg.landmarks_norm)
        return seg

    @property
    def results(self) -> list[FrameResult]:
        return self._results

    def export_thesis_figures(
        self,
        out_dir: str | Path = "output/thesis_figures",
        dpi: int = 300,
    ) -> dict[str, Path]:
        """Offline: interpolacija + filtfilt + centralna razlika, zatim 300 DPI plotovi."""
        if not self._results:
            raise RuntimeError("Nema rezultata za plotovanje.")
        t = np.array([r.timestamp for r in self._results], dtype=np.float64)
        theta_raw = np.array(
            [np.nan if r.theta_raw is None else r.theta_raw for r in self._results],
            dtype=np.float64,
        )
        valid = np.array([r.g_valid for r in self._results], dtype=bool) & np.isfinite(theta_raw)
        kin = self.offline_filter.process_sequence(theta_raw, valid)
        return save_thesis_figures(
            t=t,
            theta_raw=kin.theta_interpolated,
            theta_filtered=kin.theta_filtered,
            omega=kin.omega,
            fsm_states=[r.fsm_state for r in self._results],
            complete_times=[e.timestamp for e in self.fsm.events],
            out_dir=out_dir,
            dpi=dpi,
        )

    def run_camera(self, camera_index: int = 0, window_name: str = "Squat FSM") -> None:
        """OpenCV petlja nad web-kamerom. ESC za izlaz."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python je potreban za run_camera().") from exc

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Kamera {camera_index} nije dostupna.")
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_index / self.sample_rate_hz
                _, overlay = self.process_bgr(frame, timestamp, frame_index)
                cv2.imshow(window_name, overlay)
                frame_index += 1
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()
