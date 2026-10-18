import numpy as np
import pandas as pd
from adaptive import AdaptiveRegistry, AdaptiveModel
from kelly import KellySizer
from features import FEATURE_NAMES, to_vector, extract_features
from metrics import extended_metrics


def _synth_df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "volume": rng.integers(100, 1000, n),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


def test_feature_vector_length():
    assert len(to_vector({k: 0.5 for k in FEATURE_NAMES})) == len(FEATURE_NAMES)


def test_adaptive_learns():
    reg = AdaptiveRegistry(path="/tmp/_t_adaptive.json")
    feat = {k: 1.0 for k in FEATURE_NAMES}
    for _ in range(60):
        reg.update("X", feat, won=True)
    assert reg.get("X").predict(feat) > 0.6


def test_kelly_sane():
    k = KellySizer(window=20, min_samples=10)
    for r in [1.5, 2.0, 1.0, -1.0, 1.5, -1.0, 2.0, 1.0, 1.5, -1.0, 1.0, 1.5]:
        k.record(r)
    f = k.kelly_fraction()
    assert 0 <= f <= 1
    m = k.size_multiplier(0.01)
    assert 0.25 <= m <= 2.0


def test_metrics_smoke():
    class T:
        def __init__(self, pnl, r):
            self.pnl = pnl
            self.r_multiple = r
    trades = [T(100, 1.5), T(-50, -0.75), T(200, 2.0)]
    eq = [(pd.Timestamp("2024-01-01"), 10_000),
          (pd.Timestamp("2024-01-02"), 10_100),
          (pd.Timestamp("2024-01-03"), 10_050),
          (pd.Timestamp("2024-01-04"), 10_250)]
    m = extended_metrics(trades, eq)
    assert "sharpe" in m and "expectancy_r" in m
