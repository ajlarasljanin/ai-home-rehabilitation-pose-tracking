"""
filtering.py

Digitalni Butterworth niskopropusni filter 2. reda (sekcija 3.3).

Dva rezima, zbog kauzalnosti:

* REALTIME -- IIR korak po korak (lfilter), samo prosle vrednosti.
  FSM petlja sme da koristi iskljucivo ovaj rezim.

* OFFLINE -- dvostrano filtriranje nulte faze (scipy.signal.filtfilt)
  za naknadne izvestaje i plotove.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

from .interpolation import linear_interpolate_invalid
from .kinematics import causal_velocity_step, central_difference_velocity


class FilterMode(Enum):
    REALTIME = "realtime"
    OFFLINE = "offline"


@dataclass
class ButterworthCausal:
    """
    Kauzalni Butterworth LP 2. reda.

    Parametri iz sekcije 3.3.1: f_c = 6 Hz, f_s = 30 Hz, n = 2.
    """

    cutoff_hz: float = 6.0
    sample_rate_hz: float = 30.0
    order: int = 2

    _b: np.ndarray = field(init=False, repr=False)
    _a: np.ndarray = field(init=False, repr=False)
    _zi_template: np.ndarray = field(init=False, repr=False)
    _zi: np.ndarray | None = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        nyquist = 0.5 * self.sample_rate_hz
        if not (0 < self.cutoff_hz < nyquist):
            raise ValueError(
                f"cutoff_hz={self.cutoff_hz} mora biti u (0, {nyquist}) "
                f"(Nyquist granica pri f_s={self.sample_rate_hz} Hz)."
            )
        normal_cutoff = self.cutoff_hz / nyquist
        self._b, self._a = butter(self.order, normal_cutoff, btype="low", analog=False)
        self._zi_template = lfilter_zi(self._b, self._a)

    def reset(self, initial_value: float = 0.0) -> None:
        self._zi = self._zi_template * initial_value
        self._initialized = True

    def step(self, x_k: float) -> float:
        """
        Jedan uzorak, jednacina (10):
            y_k = b0*x_k + b1*x_{k-1} + b2*x_{k-2} - a1*y_{k-1} - a2*y_{k-2}
        """
        if not self._initialized:
            self.reset(initial_value=x_k)
        y, self._zi = lfilter(self._b, self._a, [x_k], zi=self._zi)
        return float(y[0])

    def filter_sequence_causal(self, x: np.ndarray) -> np.ndarray:
        self.reset(initial_value=float(x[0]))
        return np.array([self.step(float(v)) for v in x], dtype=np.float64)


def butter_coefficients(
    cutoff_hz: float = 6.0,
    sample_rate_hz: float = 30.0,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * sample_rate_hz
    b, a = butter(order, cutoff_hz / nyquist, btype="low", analog=False)
    return b, a


def zero_phase_filter(
    x: np.ndarray,
    cutoff_hz: float = 6.0,
    sample_rate_hz: float = 30.0,
    order: int = 2,
) -> np.ndarray:
    """
    Offline zero-phase (filtfilt), sekcija 3.3.1.
    Ne sme se koristiti u FSM petlji -- zahteva buduce uzorke.
    """
    x = np.asarray(x, dtype=np.float64)
    b, a = butter_coefficients(cutoff_hz, sample_rate_hz, order)
    padlen = min(3 * max(len(a), len(b)), len(x) - 1) if len(x) > 1 else 0
    return filtfilt(b, a, x, padlen=max(padlen, 0))


def moving_average(x: np.ndarray, window: int = 10) -> np.ndarray:
    """
    MA filter, jednacina (11). Samo vizuelno zagladivanje;
    analiticki proracuni se NE oslanjaju na MA (sekcija 3.3.2).
    """
    if window < 1:
        raise ValueError("window mora biti >= 1")
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


@dataclass
class FilteredSample:
    theta_filtered: float
    omega: float


@dataclass
class OfflineKinematics:
    theta_raw: np.ndarray
    theta_interpolated: np.ndarray
    theta_filtered: np.ndarray
    omega: np.ndarray
    valid_mask: np.ndarray
    dt: float


class KinematicFilter:
    """
    Jedinstveni interfejs za filtriranje ugla i racunanje ugaone brzine.

    REALTIME
        Butterworth IIR (kauzalni) + unazadna diferencija prvog reda:
            omega_t = (theta_t - theta_{t-1}) / dt     (jednacina 17)
    OFFLINE
        Linearna interpolacija nepouzdanih uzoraka, zatim filtfilt i
        centralna diferencija drugog reda:
            omega_k = (theta_{k+1} - theta_{k-1}) / (2 dt)   (jednacina 7)
    """

    def __init__(
        self,
        mode: FilterMode = FilterMode.REALTIME,
        cutoff_hz: float = 6.0,
        sample_rate_hz: float = 30.0,
        order: int = 2,
    ) -> None:
        self.mode = mode
        self.cutoff_hz = cutoff_hz
        self.sample_rate_hz = sample_rate_hz
        self.order = order
        self.dt = 1.0 / sample_rate_hz
        self._iir = ButterworthCausal(cutoff_hz, sample_rate_hz, order)
        self._theta_prev: float | None = None

    def reset(self) -> None:
        self._iir._initialized = False
        self._theta_prev = None

    def step(self, theta_raw: float, dt: float | None = None) -> FilteredSample:
        """Jedan kauzalni korak. Baca ValueError ako je mode OFFLINE."""
        if self.mode is not FilterMode.REALTIME:
            raise ValueError("KinematicFilter.step() je dozvoljen samo u REALTIME rezimu.")
        theta_f = self._iir.step(theta_raw)
        step_dt = self.dt if dt is None else dt
        if self._theta_prev is None or step_dt <= 0:
            omega = 0.0
        else:
            omega = causal_velocity_step(theta_f, self._theta_prev, step_dt)
        self._theta_prev = theta_f
        return FilteredSample(theta_filtered=theta_f, omega=omega)

    def process_sequence(
        self,
        theta_raw: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> OfflineKinematics:
        """Kompletna sekvenca: interpolacija + zero-phase + centralna razlika."""
        if self.mode is not FilterMode.OFFLINE:
            raise ValueError("process_sequence() je dozvoljen samo u OFFLINE rezimu.")
        theta_raw = np.asarray(theta_raw, dtype=np.float64)
        if valid_mask is None:
            valid_mask = np.isfinite(theta_raw)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
        theta_interp = linear_interpolate_invalid(theta_raw, valid_mask)
        theta_f = zero_phase_filter(
            theta_interp,
            cutoff_hz=self.cutoff_hz,
            sample_rate_hz=self.sample_rate_hz,
            order=self.order,
        )
        omega = central_difference_velocity(theta_f, self.dt)
        return OfflineKinematics(
            theta_raw=theta_raw,
            theta_interpolated=theta_interp,
            theta_filtered=theta_f,
            omega=omega,
            valid_mask=valid_mask,
            dt=self.dt,
        )
