"""
visualization.py

OpenCV overlay (skelet, ugao, FSM, semafor) i matplotlib dijagrami
300 DPI za rad (sirovi vs. filtrirani, omega sa nula-presecima, FSM faze).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .fsm import SquatState
from .landmarks import POSE_CONNECTIONS, Landmark


def _load_cv2() -> Any:
    import cv2
    return cv2


def _load_plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class TrafficLight(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# BGR
_GREEN = (40, 180, 80)
_YELLOW = (0, 200, 255)
_RED = (40, 40, 220)
_WHITE = (245, 245, 245)
_BONE = (200, 200, 200)
_JOINT = (255, 180, 80)


def traffic_light_for(
    g_valid: bool,
    state: SquatState,
    compensation: bool,
    pose_detected: bool,
) -> TrafficLight:
    """
    Semafor iz sekcije 6.3: zeleno = validno, zuto = odstupanje,
    crveno = kriticna greska / nema poze.
    """
    if not pose_detected:
        return TrafficLight.RED
    if not g_valid or compensation:
        return TrafficLight.YELLOW
    return TrafficLight.GREEN


def draw_overlay(
    frame_bgr: np.ndarray,
    landmarks: dict[int, Landmark] | None,
    theta_deg: float | None,
    state: SquatState,
    repetition_counter: int,
    light: TrafficLight,
    compensation: bool = False,
    omega_deg_s: float | None = None,
) -> np.ndarray:
    """Iscrtava skelet, ugao, FSM stanje i semafor. Vraca kopiju frejma."""
    try:
        cv2 = _load_cv2()
    except ImportError as exc:
        raise ImportError("opencv-python nije instaliran.") from exc

    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    if landmarks is not None:
        pts: dict[int, tuple[int, int]] = {}
        for i, lm in landmarks.items():
            pts[i] = (int(lm.x * w), int(lm.y * h))
        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(canvas, pts[a], pts[b], _BONE, 2, cv2.LINE_AA)
        for i, p in pts.items():
            vis = landmarks[i].visibility
            color = _JOINT if vis > 0.5 else _RED
            cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)

    theta_txt = "--" if theta_deg is None else f"{theta_deg:5.1f}"
    omega_txt = "--" if omega_deg_s is None else f"{omega_deg_s:6.1f}"
    lines = [
        f"FSM: {state.name}",
        f"theta: {theta_txt} deg",
        f"omega: {omega_txt} deg/s",
        f"reps: {repetition_counter}",
    ]
    if compensation:
        lines.append("COMPENSATION: body momentum")
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 1, cv2.LINE_AA)
        y += 26

    cx = w - 40
    colors = {
        TrafficLight.RED: ((cx, 36), _RED),
        TrafficLight.YELLOW: ((cx, 72), _YELLOW),
        TrafficLight.GREEN: ((cx, 108), _GREEN),
    }
    for kind, (center, color) in colors.items():
        fill = color if kind is light else (50, 50, 50)
        cv2.circle(canvas, center, 14, fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 14, _WHITE, 1, cv2.LINE_AA)
    return canvas


def zero_crossings(omega: np.ndarray) -> np.ndarray:
    """Indeksi gde omega menja znak (sekcija 3.4, ROM ekstremi)."""
    omega = np.asarray(omega, dtype=np.float64)
    sign = np.sign(omega)
    sign[sign == 0] = 1
    return np.where(np.diff(sign) != 0)[0] + 1


def save_thesis_figures(
    t: np.ndarray,
    theta_raw: np.ndarray,
    theta_filtered: np.ndarray,
    omega: np.ndarray,
    fsm_states: list[SquatState] | np.ndarray,
    complete_times: list[float],
    out_dir: str | Path,
    dpi: int = 300,
) -> dict[str, Path]:
    """
    Tri komparativna dijagrama 300 DPI:
      1. sirovi vs. filtrirani ugao
      2. profil ugaone brzine sa nula-presecima
      3. FSM fazna segmentacija sa oznakama repeticija
    """
    try:
        plt = _load_plt()
    except ImportError as exc:
        raise ImportError("matplotlib nije instaliran.") from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t, theta_raw, color="#9aa3ad", lw=0.9, label=r"$\theta_{\mathrm{raw}}$")
    ax.plot(t, theta_filtered, color="#1f4e79", lw=1.6, label=r"$\theta_{\mathrm{filt}}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_title("Sirovi vs. filtrirani signal ugla kolena")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "theta_raw_vs_filtered.png"
    fig.savefig(p1, dpi=dpi)
    plt.close(fig)
    paths["theta"] = p1

    zc = zero_crossings(omega)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t, omega, color="#b85c38", lw=1.4, label=r"$\omega(t)$")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    if len(zc):
        ax.scatter(t[zc], omega[zc], c="#1f4e79", s=22, zorder=3, label="nula-preseci")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\omega$ [deg/s]")
    ax.set_title("Profil ugaone brzine sa nula-presecima")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "omega_zero_crossings.png"
    fig.savefig(p2, dpi=dpi)
    plt.close(fig)
    paths["omega"] = p2

    state_vals = np.array(
        [s.value if isinstance(s, SquatState) else int(s) for s in fsm_states],
        dtype=np.int32,
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, theta_filtered, color="#1f4e79", lw=1.4, label=r"$\theta_{\mathrm{filt}}$")
    colors = {
        SquatState.START.value: "#d9d9d9",
        SquatState.DOWN.value: "#9ecae1",
        SquatState.HOLD.value: "#fdae6b",
        SquatState.UP.value: "#a1d99b",
        SquatState.COMPLETE.value: "#74c476",
    }
    ymin, ymax = float(np.nanmin(theta_filtered)), float(np.nanmax(theta_filtered))
    i0 = 0
    for i in range(1, len(state_vals) + 1):
        if i == len(state_vals) or state_vals[i] != state_vals[i0]:
            ax.axvspan(
                t[i0],
                t[min(i, len(t) - 1)],
                color=colors.get(int(state_vals[i0]), "#eee"),
                alpha=0.35,
            )
            i0 = i
    for tc in complete_times:
        ax.axvline(tc, color="#d62728", lw=1.1, ls=":", label="COMPLETE" if tc == complete_times[0] else None)
    ax.set_ylim(ymin - 5, ymax + 5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_title("FSM fazna segmentacija sa oznakama repeticija")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = out_dir / "fsm_phase_segmentation.png"
    fig.savefig(p3, dpi=dpi)
    plt.close(fig)
    paths["fsm"] = p3
    return paths
