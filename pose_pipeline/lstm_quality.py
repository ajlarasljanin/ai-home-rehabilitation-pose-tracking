"""
lstm_quality.py

Klasifikacija kvaliteta jednog ponavljanja (sekcija 5.6).
LSTMQualityClassifier je PyTorch arhitektura; HeuristicQualityFallback
koristi pragove (ROM, valgus, nagib torza) dok nema treniranog modela.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


QUALITY_CLASSES = ["ispravno", "nedovoljan_rom", "valgus_kolena", "nagib_torza"]


if _TORCH_AVAILABLE:

    class LSTMQualityClassifier(nn.Module):
        """LSTM klasifikator za sekciju 5.6.2. Ulaz: (batch, T, n_features)."""

        def __init__(self, n_features: int, hidden_size: int = 64,
                     num_layers: int = 1, num_classes: int = len(QUALITY_CLASSES)):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, T, n_features)
            _, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]  # (batch, hidden_size) -- stanje poslednjeg sloja
            return self.classifier(last_hidden)


@dataclass
class HeuristicQualityFallback:
    """Procena kvaliteta pragovima: ROM, valgus kolena, nagib torza."""

    knee_valgus_threshold_deg: float = 8.0
    trunk_lean_threshold_deg: float = 15.0
    min_rom_deg: float = 70.0

    def classify(self, theta_sequence: np.ndarray, knee_frontal_deviation_deg: np.ndarray,
                 trunk_lean_deg: np.ndarray) -> str:
        rom = float(np.max(theta_sequence) - np.min(theta_sequence))
        if rom < self.min_rom_deg:
            return "nedovoljan_rom"
        if np.max(np.abs(knee_frontal_deviation_deg)) > self.knee_valgus_threshold_deg:
            return "valgus_kolena"
        if np.max(np.abs(trunk_lean_deg)) > self.trunk_lean_threshold_deg:
            return "nagib_torza"
        return "ispravno"
