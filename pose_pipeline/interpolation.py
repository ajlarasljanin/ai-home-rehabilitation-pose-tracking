"""
interpolation.py

Linearna interpolacija nepouzdanih uzoraka ugla kada Visibility Score
padne na min(V_23, V_25, V_27) <= 0.5 (sekcija 3.1 / 7.3).

Real-time je strogo kauzalan: dok sledeci validan uzorak nije stigao,
koristi se drzanje poslednje validne vrednosti (zero-order hold). Offline
analiza popunjava rupe linearnom interpolacijom (np.interp) jer je cela
sekvenca vec dostupna.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CausalAngleHold:
    """Kauzalno popunjavanje rupe: drzi poslednji validan theta_t."""

    last_valid: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self.last_valid = None

    def step(self, theta_raw: float | None, g_valid: bool) -> tuple[float | None, bool]:
        """
        Returns
        -------
        theta, used_hold
            theta je None samo dok nije stigao nijedan validan uzorak.
            used_hold je True kada je vrednost drzana umesto merena.
        """
        if g_valid and theta_raw is not None and np.isfinite(theta_raw):
            self.last_valid = float(theta_raw)
            return self.last_valid, False
        if self.last_valid is None:
            return None, True
        return self.last_valid, True


def linear_interpolate_invalid(theta: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Offline linearna interpolacija uzoraka gde valid_mask == False.

    Rubovi pre prvog / posle poslednjeg validnog uzorka se ekstrapoliraju
    konstantom (np.interp ponašanje na granicama).
    """
    theta = np.asarray(theta, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    if theta.shape != valid.shape:
        raise ValueError("theta i valid_mask moraju biti iste duzine")
    if not np.any(valid):
        return theta
    x = np.arange(len(theta), dtype=np.float64)
    theta[~valid] = np.interp(x[~valid], x[valid], theta[valid])
    return theta
