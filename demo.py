"""
demo.py

End-to-end demonstracija nad sintetickim cucnjem -- bez kamere.
Generise 4 cucnja, propusta ih kroz MotionAnalyzer i snima 300 DPI
dijagrame za rad.

Pokretanje:
    python demo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_pipeline.analyzer import MotionAnalyzer
from pose_pipeline.fsm import FSMConfig
from pose_pipeline.landmarks import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    Landmark,
)


def synthetic_landmarks_for_angle(
    hip_knee_ankle_angle_deg: float,
    shoulder_swing_deg: float = 0.0,
) -> dict[int, Landmark]:
    """
    Minimalan skelet: rame, lakat, kuk, koleno, skocni zglob.
    Koleno je fiksirano; skocni zglob rotira tako da ugao kuk-koleno-skocni
    bude zadati theta. shoulder_swing_deg pomera lakat radi simulacije
    zamaha trupa.
    """
    hip = Landmark(0.5, 0.5, 0.0, 0.95)
    knee = Landmark(0.5, 0.9, 0.0, 0.95)
    shoulder = Landmark(0.5, 0.1, 0.0, 0.95)

    theta = np.radians(hip_knee_ankle_angle_deg)
    ankle_dx = 0.4 * np.sin(theta)
    ankle_dy = -0.4 * np.cos(theta)
    ankle = Landmark(0.5 + ankle_dx, 0.9 + ankle_dy, 0.0, 0.95)

    swing = np.radians(shoulder_swing_deg)
    elbow = Landmark(0.5 + 0.15 * np.sin(swing), 0.25, 0.0, 0.95)

    return {
        LEFT_SHOULDER: shoulder,
        LEFT_ELBOW: elbow,
        LEFT_HIP: hip,
        LEFT_KNEE: knee,
        LEFT_ANKLE: ankle,
    }


def main() -> None:
    fs = 30.0
    dt = 1 / fs
    n_reps = 4
    rep_duration_s = 1.5
    samples_per_rep = int(rep_duration_s / dt)

    theta_ref = 175.0
    theta_min = 70.0
    rng = np.random.default_rng(7)

    analyzer = MotionAnalyzer(
        fsm_config=FSMConfig(eps_omega=15.0),
        sample_rate_hz=fs,
        measure_latency=True,
        segments_dir=Path("output") / "lstm_segments",
    )

    descend_n = int(0.35 * samples_per_rep)
    hold_n = int(0.30 * samples_per_rep)
    ascend_n = samples_per_rep - descend_n - hold_n

    frame_index = 0
    timestamp = 0.0
    for _rep in range(n_reps):
        descend = theta_ref - (theta_ref - theta_min) * (
            0.5 - 0.5 * np.cos(np.linspace(0, np.pi, descend_n))
        )
        hold = np.full(hold_n, theta_min)
        ascend = theta_min + (theta_ref - theta_min) * (
            0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ascend_n))
        )
        one_rep = np.concatenate([descend, hold, ascend])
        one_rep_noisy = one_rep + rng.normal(0, 0.4, size=one_rep.shape)

        for theta_t in one_rep_noisy:
            landmarks = synthetic_landmarks_for_angle(float(theta_t))
            analyzer.process_frame(landmarks, timestamp=timestamp, frame_index=frame_index)
            timestamp += dt
            frame_index += 1

        for _ in range(int(0.4 * fs)):
            landmarks = synthetic_landmarks_for_angle(theta_ref)
            analyzer.process_frame(landmarks, timestamp=timestamp, frame_index=frame_index)
            timestamp += dt
            frame_index += 1

    print(f"Simulirano ponavljanja:   {n_reps}")
    print(f"FSM izbrojao ponavljanja: {analyzer.fsm.repetition_counter}")
    print(f"LSTM segmenata:           {len(analyzer.segments)}")
    print()

    if analyzer.calibrator.is_calibrated:
        cal = analyzer.calibrator.result()
        print("Antropometrijska kalibracija:")
        for k, v in cal.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print(f"Kalibracija nezavrsena: {analyzer.calibrator.n_valid_samples} uzoraka")
    print()

    if analyzer.latency is not None:
        report = analyzer.latency.report()
        print(f"Budzet po frejmu (30 FPS): {report['frame_budget_ms']:.2f} ms")
        print(
            "Izmereno srednje trajanje sinhronog dela po frejmu: "
            f"{report['mean_total_frame_ms']:.4f} ms"
        )
        print(f"Udeo frejmova preko budzeta: {report['over_budget_ratio']*100:.1f}%")

    fig_dir = Path("output") / "thesis_figures"
    try:
        paths = analyzer.export_thesis_figures(fig_dir, dpi=300)
        print()
        print(f"Dijagrami (300 DPI) sacuvani u {fig_dir.resolve()}:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    except ImportError as exc:
        print(f"Plotovi preskoceni ({exc})")


if __name__ == "__main__":
    main()
