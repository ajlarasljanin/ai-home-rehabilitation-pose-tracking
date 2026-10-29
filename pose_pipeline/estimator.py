"""
estimator.py

PoseEstimator: Google MediaPipe Pose (BlazePose, 33 tacke).
MediaPipe se ucitava tesko (Windows), zato je import odlozen do konstrukcije.
"""
from __future__ import annotations

from types import TracebackType

import numpy as np

from .landmarks import Landmark


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


class PoseEstimator:
    """Omotac oko MediaPipe Pose: BGR frejm -> recnik 33 Landmark objekta."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe nije instaliran. Instalirajte: pip install mediapipe"
            ) from exc
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> dict[int, Landmark] | None:
        """
        Pokrece BlazePose inferenciju.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV slika (H, W, 3), BGR.

        Returns
        -------
        dict | None
            33 tacke ili None ako poza nije detektovana.
        """
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return None
        out: dict[int, Landmark] = {}
        for i, lm in enumerate(result.pose_landmarks.landmark):
            out[i] = Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
        return out

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
