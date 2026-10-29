"""
live.py

OpenCV real-time prikaz: skelet, ugao kolena, FSM stanje, semafor.
ESC zatvara prozor.

    python live.py
    python live.py --camera 1
"""
from __future__ import annotations

import argparse

from pose_pipeline.analyzer import MotionAnalyzer
from pose_pipeline.estimator import mediapipe_available
from pose_pipeline.fsm import FSMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time squat FSM (MediaPipe Pose).")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fs", type=float, default=30.0)
    args = parser.parse_args()

    if not mediapipe_available():
        raise SystemExit("Instalirajte mediapipe i opencv-python da biste pokrenuli live.py.")

    analyzer = MotionAnalyzer(
        fsm_config=FSMConfig(),
        sample_rate_hz=args.fs,
        enable_pose=True,
        measure_latency=True,
    )
    analyzer.run_camera(camera_index=args.camera)


if __name__ == "__main__":
    main()
