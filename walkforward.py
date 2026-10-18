"""Walk-forward: train adaptive on window N, test on N+1."""
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from config import Config
from backtest import run_backtest
from adaptive import AdaptiveRegistry
from metrics import extended_metrics

log = logging.getLogger("walkforward")


def walk_forward(
    df: pd.DataFrame,
    cfg: Config,
    train_bars: int = 3000,
    test_bars: int = 1500,
    step_bars: int = 1500,
    symbol: str = "WF",
) -> dict:
    """Returns list of {window, train_metrics, test_metrics}."""
    results = []
    n = len(df)
    start = 0
    while start + train_bars + test_bars <= n:
        train = df.iloc[start: start + train_bars]
        test = df.iloc[start + train_bars: start + train_bars + test_bars]

        reg = AdaptiveRegistry(path=f"wf_{symbol}.json")
        cfg.adaptive_registry = reg

        # Train
        train_res = run_backtest(train, cfg)
        for t in train_res["trades"]:
            feats = getattr(t.signal, "features", {})
            if feats:
                reg.update(symbol, feats, t.pnl > 0)

        # Test
        test_res = run_backtest(test, cfg)
        test_metrics = extended_metrics(test_res["trades"], test_res["equity_curve"])
        train_metrics = extended_metrics(train_res["trades"], train_res["equity_curve"])

        results.append({
            "start_idx": start,
            "train_range": (str(train.index[0]), str(train.index[-1])),
            "test_range": (str(test.index[0]), str(test.index[-1])),
            "train": train_metrics,
            "test": test_metrics,
            "adaptive_samples": reg.get(symbol).samples,
        })
        log.info("WF window start=%s test_expectancy=%.3fR test_sharpe=%.2f",
                 start, test_metrics.get("expectancy_r", 0), test_metrics.get("sharpe", 0))
        start += step_bars

    return {"windows": results}
