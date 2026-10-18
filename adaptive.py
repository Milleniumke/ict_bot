"""
Online logistic learner — one per symbol.
Predicts P(win) for a setup from its feature vector.
Trains AFTER each closed trade. Persists to JSON.
"""
import json
import math
import os
import threading
from typing import Dict, Optional

import numpy as np

from features import FEATURE_NAMES, to_vector


def _sigmoid(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


class AdaptiveModel:
    def __init__(self, symbol: str, lr: float = 0.05, l2: float = 1e-4):
        self.symbol = symbol
        self.lr = lr
        self.l2 = l2
        self.w = np.zeros(len(FEATURE_NAMES), dtype=float)
        self.b = 0.0
        self.samples = 0

    def predict(self, features: dict) -> float:
        x = to_vector(features)
        z = float(np.dot(self.w, x) + self.b)
        return _sigmoid(z)

    def update(self, features: dict, won: bool):
        x = to_vector(features)
        y = 1.0 if won else 0.0
        p = self.predict(features)
        err = p - y
        # L2-regularized gradient descent step
        self.w -= self.lr * (err * x + self.l2 * self.w)
        self.b -= self.lr * err
        self.samples += 1

    def to_dict(self) -> dict:
        return {"w": self.w.tolist(), "b": self.b, "samples": self.samples,
                "features": FEATURE_NAMES}

    @classmethod
    def from_dict(cls, symbol: str, d: dict) -> "AdaptiveModel":
        m = cls(symbol)
        if d.get("features") == FEATURE_NAMES:
            m.w = np.array(d.get("w", [0.0] * len(FEATURE_NAMES)), dtype=float)
            m.b = float(d.get("b", 0.0))
            m.samples = int(d.get("samples", 0))
        return m


class AdaptiveRegistry:
    """Thread-safe, file-persisted store of AdaptiveModel per symbol."""

    def __init__(self, path: str = "adaptive.json"):
        self.path = path
        self._lock = threading.Lock()
        self._models: Dict[str, AdaptiveModel] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                raw = json.load(f)
            for sym, d in raw.items():
                self._models[sym] = AdaptiveModel.from_dict(sym, d)
        except Exception:
            self._models = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({s: m.to_dict() for s, m in self._models.items()}, f)
        os.replace(tmp, self.path)

    def get(self, symbol: str) -> AdaptiveModel:
        with self._lock:
            if symbol not in self._models:
                self._models[symbol] = AdaptiveModel(symbol)
            return self._models[symbol]

    def update(self, symbol: str, features: dict, won: bool):
        with self._lock:
            m = self._models.setdefault(symbol, AdaptiveModel(symbol))
            m.update(features, won)
            self._save()

    def ready(self, symbol: str, min_samples: int = 30) -> bool:
        return self.get(symbol).samples >= min_samples

    def predict(self, symbol: str, features: dict) -> Optional[float]:
        if not self.ready(symbol):
            return None
        return self.get(symbol).predict(features)
