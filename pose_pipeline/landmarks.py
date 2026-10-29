"""
landmarks.py

MediaPipe Pose vraca 33 tacke (BlazePose topologija, Bazarevsky et al. 2020).
Modul definise indekse relevantne za cucanj i proracun unutrasnjeg ugla
preko skalarnog proizvoda 3D vektora, jednacine (1)-(6) u sekciji 3.1 rada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- MediaPipe Pose landmark indeksi (BlazePose, 33 tacke) ---
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

KNEE_CHAIN_LEFT = (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
KNEE_CHAIN_RIGHT = (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
SHOULDER_CHAIN_LEFT = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_HIP)

# Prag poverenja Visibility Score: G_valid = 1[min(V_i) > 0.5], sekcija 7.3
VISIBILITY_THRESHOLD = 0.5

# MediaPipe Pose veze za iscrtavanje skeleta (parovi indeksa).
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


@dataclass
class Landmark:
    """Jedna MediaPipe tacka: 3D pozicija (x, y, z) + Visibility Score."""

    x: float
    y: float
    z: float
    visibility: float

    def as_vector(self) -> np.ndarray:
        """L_i kao 3D pozicioni vektor (sekcija 3, posle Slike 4)."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def joint_angle(a: Landmark, vertex: Landmark, b: Landmark) -> float:
    """
    Unutrasnji ugao theta u tacki `vertex`, izmedju vektora ka `a` i ka `b`.

    Jednacine (1)-(6), sekcija 3.1:
        vec_a = L_a - L_vertex
        vec_b = L_b - L_vertex
        theta = arccos(clip((vec_a · vec_b) / (||vec_a|| ||vec_b||), -1, 1))

    Za koleno: a=kuk (23), vertex=koleno (25), b=skocni zglob (27).

    Returns
    -------
    float
        Ugao u stepenima, opseg [0, 180].
    """
    vec_a = a.as_vector() - vertex.as_vector()
    vec_b = b.as_vector() - vertex.as_vector()

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        raise ValueError("Degenerisan vektor segmenta tela (dva landmarka se poklapaju).")

    cos_theta = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def knee_angle(landmarks: dict[int, Landmark], side: str = "left") -> float:
    """Ugao u kolenu: L23-L25-L27 (levo) ili L24-L26-L28 (desno)."""
    if side == "left":
        hip, knee, ankle = KNEE_CHAIN_LEFT
    elif side == "right":
        hip, knee, ankle = KNEE_CHAIN_RIGHT
    else:
        raise ValueError("side mora biti 'left' ili 'right'")
    return joint_angle(landmarks[hip], landmarks[knee], landmarks[ankle])


def shoulder_angle(landmarks: dict[int, Landmark], side: str = "left") -> float:
    """
    Ugao ramena theta_shoulder = ∠(L_11, L_13, L_23) iz sekcije 3.4.3.

    Vertex je rame (L_11): vektori ka laktu (L_13) i kuku (L_23). Vertex u
    ramenu, a ne u laktu, jer oscilacija lakta meri sam pokret vežbe, a ne
    zamah trupa (body momentum).
    """
    if side == "left":
        shoulder, elbow, hip = LEFT_SHOULDER, LEFT_ELBOW, LEFT_HIP
    elif side == "right":
        shoulder, elbow, hip = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HIP
    else:
        raise ValueError("side mora biti 'left' ili 'right'")
    return joint_angle(landmarks[elbow], landmarks[shoulder], landmarks[hip])


def segment_length(p1: Landmark, p2: Landmark) -> float:
    """Euklidsko rastojanje L_seg (sekcija 3.4.2, jednacina 15)."""
    return float(np.linalg.norm(p1.as_vector() - p2.as_vector()))


def min_visibility(landmarks: dict[int, Landmark], indices: tuple[int, ...] | list[int]) -> float:
    """min(V_i) za zadati lanac zglobova; 0.0 ako neka tacka nedostaje."""
    scores: list[float] = []
    for i in indices:
        if i not in landmarks:
            return 0.0
        scores.append(landmarks[i].visibility)
    return min(scores) if scores else 0.0


def visibility_gate(
    landmarks: dict[int, Landmark],
    indices: list[int] | tuple[int, ...],
    threshold: float = VISIBILITY_THRESHOLD,
) -> bool:
    """
    G_valid iz sekcije 7.3:
        G_valid = 1[min(V_i) > threshold]
    Korisnicki uslov min(V_23, V_25, V_27) <= 0.5 oznacava nepouzdan frejm.
    """
    if not indices:
        return False
    return min_visibility(landmarks, indices) > threshold


def landmarks_to_array(landmarks: dict[int, Landmark], n_points: int = 33) -> np.ndarray:
    """Matrica (n_points, 4) sa kolonama x, y, z, visibility."""
    out = np.full((n_points, 4), np.nan, dtype=np.float64)
    for i, lm in landmarks.items():
        if 0 <= i < n_points:
            out[i] = (lm.x, lm.y, lm.z, lm.visibility)
    return out


def normalize_skeleton(
    landmarks: dict[int, Landmark],
    height_scale: float | None = None,
) -> np.ndarray:
    """
    Normalizovane 3D koordinate za LSTM: origin u sredini kukova,
    skala po procenjenoj visini (trup + natkolenica + potkolenica).
    Izlaz: (33, 3).
    """
    xyz = landmarks_to_array(landmarks)[:, :3]
    if LEFT_HIP in landmarks and RIGHT_HIP in landmarks:
        origin = 0.5 * (landmarks[LEFT_HIP].as_vector() + landmarks[RIGHT_HIP].as_vector())
    elif LEFT_HIP in landmarks:
        origin = landmarks[LEFT_HIP].as_vector()
    else:
        origin = np.zeros(3)
    xyz = xyz - origin
    if height_scale is None or height_scale < 1e-9:
        if all(i in landmarks for i in (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)):
            height_scale = (
                segment_length(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP])
                + segment_length(landmarks[LEFT_HIP], landmarks[LEFT_KNEE])
                + segment_length(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
            )
        else:
            height_scale = 1.0
    return xyz / max(height_scale, 1e-9)
