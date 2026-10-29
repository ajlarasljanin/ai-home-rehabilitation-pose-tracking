"""
pipeline.py

RealtimePipeline je omotac oko MotionAnalyzer.
"""
from __future__ import annotations

from .analyzer import FrameResult, MotionAnalyzer
from .fsm import FSMConfig


class RealtimePipeline(MotionAnalyzer):
    """Sinhroni deo pipeline-a za jedan zglob (podrazumevano levo koleno)."""

    def __init__(
        self,
        fsm_config: FSMConfig | None = None,
        cutoff_hz: float = 6.0,
        sample_rate_hz: float = 30.0,
        measure_latency: bool = True,
    ) -> None:
        super().__init__(
            fsm_config=fsm_config,
            cutoff_hz=cutoff_hz,
            sample_rate_hz=sample_rate_hz,
            measure_latency=measure_latency,
            enable_pose=False,
        )


__all__ = ["FrameResult", "RealtimePipeline"]
