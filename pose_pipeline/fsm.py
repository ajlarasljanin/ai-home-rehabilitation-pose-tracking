"""
fsm.py

Konacni automat za brojanje cucnjeva, sekcija 5 rada.

  START -> DOWN -> HOLD -> UP -> COMPLETE -> START

Pragovi (sekcija 5.4.1):
    theta_ref      ~ 170-180 deg  (neutralna ekstenzija)
    theta_down     = 100 deg      (ulazak u DOWN)
    theta_hold     = 90 deg       (theta_down - eps_hyst, validacija dna)
    theta_complete = 160 deg      (COMPLETE)
    eps_omega      = 15 deg/s     (zona nulte brzine)
    N = 5, delta = 4/5            (Majority-Vote)
    T_refractory   = 0.5 s        (anti-double count)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto


class SquatState(Enum):
    START = auto()
    DOWN = auto()
    HOLD = auto()
    UP = auto()
    COMPLETE = auto()


@dataclass
class FSMConfig:
    theta_start_activation: float = 100.0  # theta_down
    eps_hysteresis: float = 10.0
    theta_complete: float = 160.0
    eps_omega: float = 15.0
    hold_confirm_window: int = 5
    hold_confirm_ratio: float = 4.0 / 5.0
    t_min_hold: float = 0.15
    t_refractory: float = 0.5
    visibility_required: bool = True

    @property
    def theta_down_extreme(self) -> float:
        """Donji histerezicni prag: theta_down - eps_hyst = 90 deg."""
        return self.theta_start_activation - self.eps_hysteresis


@dataclass
class RepEvent:
    frame_index: int
    timestamp: float
    theta_at_complete: float
    t_start: float
    t_end: float


@dataclass
class FSMStepResult:
    state: SquatState
    omega_t: float
    repetition_counter: int
    cycle_completed: bool
    t_start: float | None
    t_end: float | None


class SquatFSM:
    """
    FSM petlja u realnom vremenu.

    Prima vec filtriran ugao kolena. Ugaonu brzinu racuna kauzalno
    (jednacina 17) -- nikad centralnom razlikom.
    Tranzicije se potvrdjuju Majority-Vote prozorom N=5, delta=4/5.
    """

    def __init__(self, config: FSMConfig | None = None) -> None:
        self.config = config or FSMConfig()
        self.state = SquatState.START
        self.repetition_counter = 0
        self.events: list[RepEvent] = []

        self._theta_prev: float | None = None
        self._time_prev: float | None = None
        self._guard_window: deque[bool] = deque(maxlen=self.config.hold_confirm_window)
        self._hold_elapsed_s: float = 0.0
        self._refractory_until: float = -1.0
        self._cycle_t_start: float | None = None
        self._last_omega: float = 0.0

    def reset(self) -> None:
        self.state = SquatState.START
        self.repetition_counter = 0
        self.events.clear()
        self._theta_prev = None
        self._time_prev = None
        self._guard_window.clear()
        self._hold_elapsed_s = 0.0
        self._refractory_until = -1.0
        self._cycle_t_start = None
        self._last_omega = 0.0

    def _majority_vote_satisfied(self) -> bool:
        """
        (1/N) * sum(g(t_i)) >= delta,  N=5, delta=4/5  (sekcija 5.2.3)
        """
        window = self.config.hold_confirm_window
        if len(self._guard_window) < window:
            return False
        return (sum(self._guard_window) / window) >= self.config.hold_confirm_ratio

    def _confirm(self, guard: bool) -> bool:
        self._guard_window.append(bool(guard))
        return self._majority_vote_satisfied()

    def _enter(self, new_state: SquatState) -> None:
        self.state = new_state
        self._guard_window.clear()
        self._hold_elapsed_s = 0.0

    def _complete_cycle(self, theta_t: float, timestamp: float, frame_index: int) -> None:
        t_start = self._cycle_t_start if self._cycle_t_start is not None else timestamp
        self.state = SquatState.COMPLETE
        self.repetition_counter += 1
        self.events.append(
            RepEvent(
                frame_index=frame_index,
                timestamp=timestamp,
                theta_at_complete=theta_t,
                t_start=t_start,
                t_end=timestamp,
            )
        )
        self._refractory_until = timestamp + self.config.t_refractory
        self._cycle_t_start = None
        self._enter(SquatState.START)

    def step(
        self,
        theta_t: float,
        timestamp: float,
        g_valid: bool = True,
        frame_index: int = 0,
        omega_t: float | None = None,
    ) -> SquatState:
        """Obradjuje jedan frejm; vraca trenutno stanje (COMPLETE je trenutan)."""
        self.evaluate(theta_t, timestamp, g_valid, frame_index, omega_t)
        return self.state

    def evaluate(
        self,
        theta_t: float,
        timestamp: float,
        g_valid: bool = True,
        frame_index: int = 0,
        omega_t: float | None = None,
    ) -> FSMStepResult:
        cfg = self.config
        cycle_completed = False
        t_end: float | None = None

        if cfg.visibility_required and not g_valid:
            self._theta_prev = theta_t
            self._time_prev = timestamp
            return FSMStepResult(
                state=self.state,
                omega_t=self._last_omega,
                repetition_counter=self.repetition_counter,
                cycle_completed=False,
                t_start=self._cycle_t_start,
                t_end=None,
            )

        if self._theta_prev is None or self._time_prev is None:
            self._theta_prev = theta_t
            self._time_prev = timestamp
            return FSMStepResult(
                state=self.state,
                omega_t=0.0,
                repetition_counter=self.repetition_counter,
                cycle_completed=False,
                t_start=self._cycle_t_start,
                t_end=None,
            )

        dt = timestamp - self._time_prev
        if dt <= 0:
            return FSMStepResult(
                state=self.state,
                omega_t=self._last_omega,
                repetition_counter=self.repetition_counter,
                cycle_completed=False,
                t_start=self._cycle_t_start,
                t_end=None,
            )

        if omega_t is None:
            omega_t = (theta_t - self._theta_prev) / dt
        self._last_omega = float(omega_t)

        in_refractory = timestamp < self._refractory_until

        if self.state == SquatState.START:
            guard = (not in_refractory) and (theta_t < cfg.theta_start_activation)
            if self._confirm(guard):
                self._enter(SquatState.DOWN)
                self._cycle_t_start = timestamp

        elif self.state == SquatState.DOWN:
            # Plitak pokret: povratak u ekstenziju bez HOLD -> reset bez inkrementa
            if theta_t > cfg.theta_complete:
                self._enter(SquatState.START)
                self._cycle_t_start = None
            else:
                guard = abs(omega_t) < cfg.eps_omega and theta_t <= cfg.theta_down_extreme
                if guard:
                    self._hold_elapsed_s += dt
                if self._confirm(guard) and self._hold_elapsed_s >= cfg.t_min_hold:
                    self._enter(SquatState.HOLD)

        elif self.state == SquatState.HOLD:
            # Izlazak: promena smera (omega > 0) i histerezni prag theta > 100
            guard = omega_t > 0.0 and theta_t > cfg.theta_start_activation
            if self._confirm(guard):
                self._enter(SquatState.UP)

        elif self.state == SquatState.UP:
            if self._confirm(theta_t > cfg.theta_complete):
                t_end = timestamp
                self._complete_cycle(theta_t, timestamp, frame_index)
                cycle_completed = True

        self._theta_prev = theta_t
        self._time_prev = timestamp
        return FSMStepResult(
            state=self.state,
            omega_t=self._last_omega,
            repetition_counter=self.repetition_counter,
            cycle_completed=cycle_completed,
            t_start=self.events[-1].t_start if cycle_completed else self._cycle_t_start,
            t_end=t_end,
        )
