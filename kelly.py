"""Fractional Kelly position sizing based on realized edge."""
import math
from collections import deque
from typing import Deque, Tuple


class KellySizer:
    def __init__(self, window: int = 50, fraction: float = 0.25,
                 cap_mult: float = 2.0, min_samples: int = 20):
        self.history: Deque[float] = deque(maxlen=window)
        self.fraction = fraction
        self.cap_mult = cap_mult
        self.min_samples = min_samples

    def record(self, r_multiple: float):
        self.history.append(float(r_multiple))

    def kelly_fraction(self) -> float:
        if len(self.history) < self.min_samples:
            return 0.0
        wins = [r for r in self.history if r > 0]
        losses = [-r for r in self.history if r < 0]
        if not wins or not losses:
            return 0.0
        p = len(wins) / len(self.history)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return 0.0
        b = avg_win / avg_loss
        f = (p * b - (1 - p)) / b
        return max(0.0, f)

    def size_multiplier(self, base_risk: float) -> float:
        """Multiplier applied to base risk (1.0 = use config default)."""
        f = self.kelly_fraction()
        if f <= 0:
            return 1.0
        scaled = f * self.fraction * 4.0  # normalize to ~1.0 when f=0.25
        return float(min(max(scaled, 0.25), self.cap_mult))
