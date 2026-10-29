"""
evaluation.py

Tacnost brojanja, konfuziona matrica kvaliteta i latencija po frejmu
(budzet T_frame ~ 33.3 ms, sekcija 6.2).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix


def count_accuracy(predicted_counts: list[int], ground_truth_counts: list[int]) -> dict:
    """
    Poredi predikciju brojaca ponavljanja (po sesiji/snimku) sa rucno
    izbrojanim brojem ponavljanja. Vraca srednju apsolutnu gresku (MAE) i
    udeo sesija u kojima se brojac tacno poklopio.
    """
    predicted = np.asarray(predicted_counts)
    truth = np.asarray(ground_truth_counts)
    if predicted.shape != truth.shape:
        raise ValueError("predicted_counts i ground_truth_counts moraju biti iste duzine")
    errors = predicted - truth
    return {
        "mae": float(np.mean(np.abs(errors))),
        "exact_match_ratio": float(np.mean(errors == 0)),
        "mean_signed_error": float(np.mean(errors)),  # + = precenjuje, - = potcenjuje
        "n_sessions": int(len(truth)),
    }


def confusion_matrix_for_quality(predicted_labels: list[str], true_labels: list[str],
                                  class_order: list[str]) -> np.ndarray:
    """
    Konfuziona matrica za LSTM klasifikaciju kvaliteta pokreta (sekcija 5.6.2).
    Redovi = stvarna klasa, kolone = predvidjena klasa (sklearn konvencija).
    """
    return confusion_matrix(true_labels, predicted_labels, labels=class_order)


@dataclass
class LatencyBenchmark:
    """
    Meri trajanje pojedinacnih faza sinhronog dela pipeline-a (sekcija 6.2:
    akvizicija, lokalizacija repera, kinematicka transformacija+filtriranje,
    evaluacija FSM tranzicija) i poredi sumu sa budzetom T_frame.
    """

    target_fps: float = 30.0
    _stage_durations_ms: dict[str, list[float]] = field(default_factory=dict)

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / self.target_fps

    @contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._stage_durations_ms.setdefault(stage_name, []).append(elapsed_ms)

    def report(self) -> dict:
        """
        Vraca srednje/p95 trajanje po fazi, ukupno srednje trajanje frejma,
        i procenat frejmova koji su premasili budzet (frame_drop rizik,
        sekcija 6.2 rada).
        """
        per_stage = {
            stage: {
                "mean_ms": float(np.mean(durations)),
                "p95_ms": float(np.percentile(durations, 95)),
                "n": len(durations),
            }
            for stage, durations in self._stage_durations_ms.items()
        }
        if self._stage_durations_ms:
            n_frames = len(next(iter(self._stage_durations_ms.values())))
            total_per_frame = np.sum(
                [self._stage_durations_ms[s][:n_frames] for s in self._stage_durations_ms],
                axis=0,
            )
            over_budget_ratio = float(np.mean(total_per_frame > self.frame_budget_ms))
            mean_total_ms = float(np.mean(total_per_frame))
        else:
            over_budget_ratio = 0.0
            mean_total_ms = 0.0

        return {
            "per_stage": per_stage,
            "frame_budget_ms": self.frame_budget_ms,
            "mean_total_frame_ms": mean_total_ms,
            "over_budget_ratio": over_budget_ratio,
        }
