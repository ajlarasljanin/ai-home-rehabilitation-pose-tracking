"""
segmentation.py

Izdvajanje normalizovane sekvence [t_start, t_end] za asinhronu LSTM
klasifikaciju (sekcija 5.6). Snimanje u JSON i NumPy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CycleSegment:
    rep_index: int
    t_start: float
    t_end: float
    timestamps: np.ndarray
    landmarks_norm: np.ndarray  # (T, 33, 3)
    theta_knee_deg: np.ndarray
    omega_deg_s: np.ndarray
    theta_shoulder_deg: np.ndarray
    compensation_detected: bool

    def to_dict(self) -> dict:
        return {
            "rep_index": self.rep_index,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "timestamps": self.timestamps.tolist(),
            "landmarks_norm": self.landmarks_norm.tolist(),
            "theta_knee_deg": self.theta_knee_deg.tolist(),
            "omega_deg_s": self.omega_deg_s.tolist(),
            "theta_shoulder_deg": self.theta_shoulder_deg.tolist(),
            "compensation_detected": self.compensation_detected,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npy":
            np.save(path, self.landmarks_norm)
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def detect_body_momentum(theta_shoulder_deg: np.ndarray, threshold_deg: float = 10.0) -> bool:
    """True ako peak-to-peak oscilacija ramena prelazi 10 deg (sekcija 3.4.3)."""
    seq = np.asarray(theta_shoulder_deg, dtype=np.float64)
    seq = seq[np.isfinite(seq)]
    if seq.size < 2:
        return False
    return float(np.nanmax(seq) - np.nanmin(seq)) > threshold_deg
