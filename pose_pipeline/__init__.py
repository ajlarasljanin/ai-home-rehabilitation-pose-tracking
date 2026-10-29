"""
pose_pipeline
=============

Implementacija pipeline-a iz rada
"Primena vestacke inteligencije u rehabilitaciji u kucnim uslovima".

Javne klase:
    PoseEstimator, KinematicFilter, SquatFSM, MotionAnalyzer
"""

from .filtering import FilterMode, KinematicFilter
from .fsm import FSMConfig, SquatFSM, SquatState

__version__ = "1.0.0"


def __getattr__(name: str):
    """Odlozen import teskih modula (MediaPipe / OpenCV / matplotlib)."""
    if name in ("PoseEstimator",):
        from .estimator import PoseEstimator
        return PoseEstimator
    if name in ("MotionAnalyzer", "FrameResult"):
        from .analyzer import FrameResult, MotionAnalyzer
        return {"MotionAnalyzer": MotionAnalyzer, "FrameResult": FrameResult}[name]
    if name == "RealtimePipeline":
        from .pipeline import RealtimePipeline
        return RealtimePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PoseEstimator",
    "KinematicFilter",
    "FilterMode",
    "SquatFSM",
    "FSMConfig",
    "SquatState",
    "MotionAnalyzer",
    "RealtimePipeline",
    "FrameResult",
]
