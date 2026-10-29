"""
calibration.py

Procena visine H i duzina segmenata L_seg iz prvih N validnih frejmova
(sekcija 3.4). Koriste se samo tacke sa V_i > 0.5; rezultat je medijana.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .landmarks import (
    Landmark,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    VISIBILITY_THRESHOLD,
    segment_length,
)


@dataclass
class AnthropometricCalibrator:
    """
    Prikuplja uzorke tokom pocetne (mirujuce, uspravne) faze i procenjuje:
      - ukupnu telesnu visinu H (sekcija 3.4.1)
      - duzinu potkolenice i natkolenice (sekcija 3.4.2, L_seg)

    Kalibracija se smatra zavrsenom kada broj VALIDNIH uzoraka (onih koji
    prolaze Visibility Score prag) dostigne `min_samples`.
    """

    min_samples: int = 45  # ~1.5s pri 30 FPS
    visibility_threshold: float = VISIBILITY_THRESHOLD

    _height_samples: list[float] = field(default_factory=list, init=False)
    _thigh_samples: list[float] = field(default_factory=list, init=False)
    _shank_samples: list[float] = field(default_factory=list, init=False)

    def add_frame(self, landmarks: dict[int, Landmark]) -> None:
        """Dodaje jedan frejm u bafer kalibracije, ako prolazi kontrolu kvaliteta."""
        required = [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]
        if not all(i in landmarks for i in required):
            return
        if any(landmarks[i].visibility <= self.visibility_threshold for i in required):
            return  # V_i <= 0.5, sekcija 7.3

        shoulder, hip, knee, ankle = (landmarks[i] for i in required)

        thigh = segment_length(hip, knee)
        shank = segment_length(knee, ankle)
        # H ~ trup + natkolenica + potkolenica (sekcija 3.4.1)
        trunk = segment_length(shoulder, hip)
        height_estimate = trunk + thigh + shank

        self._height_samples.append(height_estimate)
        self._thigh_samples.append(thigh)
        self._shank_samples.append(shank)

    @property
    def n_valid_samples(self) -> int:
        return len(self._height_samples)

    @property
    def is_calibrated(self) -> bool:
        return self.n_valid_samples >= self.min_samples

    def result(self) -> dict[str, float]:
        """Medijane H, L_natkolenice i L_potkolenice. Greska ako kalibracija nije gotova."""
        if not self.is_calibrated:
            raise RuntimeError(
                f"Kalibracija nije zavrsena: {self.n_valid_samples}/{self.min_samples} "
                "validnih uzoraka."
            )
        return {
            "height_estimate": float(np.median(self._height_samples)),
            "thigh_length": float(np.median(self._thigh_samples)),
            "shank_length": float(np.median(self._shank_samples)),
            "n_samples": self.n_valid_samples,
        }
