"""
kinematics.py

Ugaona i linearna brzina, u dve varijante koje rad koristi u razlicitim
kontekstima (sekcija 5.2.1):

* central_difference_velocity  - tacnost O(dt^2), offline analiza (sekcija 3.2.1,
  jednacina 7), zahteva theta_{k+1} pa nije upotrebljiva u realnom vremenu za
  TRENUTNI frejm bez uvodjenja kasnjenja od jednog frejma.

* causal_velocity_step         - tacnost O(dt), koristi samo theta_t i theta_{t-1},
  upotrebljivo u FSM petlji u realnom vremenu (sekcija 5.2.1, jednacina 17).
"""
from __future__ import annotations

import numpy as np


def central_difference_velocity(theta: np.ndarray, dt: float) -> np.ndarray:
    """
    Offline centralna razlika, jednacina (7):
        omega_k = (theta_{k+1} - theta_{k-1}) / (2 * dt)

    Prvi i poslednji uzorak se racunaju jednostranom (forward/backward)
    razlikom prvog reda, jer za njih centralna razlika nije definisana.

    Parameters
    ----------
    theta : np.ndarray
        Vec filtrirani ugaoni signal (posle Butterworth filtera).
    dt : float
        Vremenski korak izmedju uzoraka (1/f_s).
    """
    theta = np.asarray(theta, dtype=np.float64)
    omega = np.empty_like(theta)
    omega[1:-1] = (theta[2:] - theta[:-2]) / (2 * dt)
    omega[0] = (theta[1] - theta[0]) / dt
    omega[-1] = (theta[-1] - theta[-2]) / dt
    return omega


def causal_velocity_step(theta_t: float, theta_t_minus_1: float, dt: float) -> float:
    """
    Kauzalna (unazadna) razlika, jednacina (17):
        omega_t = (theta_t - theta_{t-1}) / dt

    Koristi je FSM kontroler u realnom vremenu (sekcija 5.2.1) jer theta_{t+1}
    u trenutku obrade frejma t jos nije stigao sa kamere.
    """
    return (theta_t - theta_t_minus_1) / dt


def linear_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Linearna brzina proizvoljne tacke L_i, jednacina (8):
        v_i(t) = sqrt(vx^2 + vy^2 + vz^2)

    Parameters
    ----------
    positions : np.ndarray, shape (T, 3)
        Niz (x, y, z) pozicija kroz vreme (npr. filtrirane koordinate sake).
    dt : float
        Vremenski korak.
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocity_components = np.gradient(positions, dt, axis=0)
    return np.linalg.norm(velocity_components, axis=1)
