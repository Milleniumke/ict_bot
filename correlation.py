"""Correlation guard — refuses to stack correlated trades."""
import numpy as np
import pandas as pd


class CorrelationGuard:
    def __init__(self, threshold: float = 0.7, lookback: int = 200):
        self.threshold = threshold
        self.lookback = lookback
        self.matrix: pd.DataFrame = pd.DataFrame()

    def update(self, price_series: dict):
        """price_series: {symbol: close_series}"""
        if not price_series:
            return
        closes = pd.DataFrame(price_series).dropna(how="all").tail(self.lookback)
        returns = closes.pct_change().dropna()
        if returns.empty:
            return
        self.matrix = returns.corr()

    def corr(self, a: str, b: str) -> float:
        if self.matrix.empty:
            return 0.0
        if a not in self.matrix.columns or b not in self.matrix.columns:
            return 0.0
        return float(self.matrix.loc[a, b])

    def is_safe(self, candidate: str, open_symbols: list) -> tuple:
        for sym in open_symbols:
            c = abs(self.corr(candidate, sym))
            if c >= self.threshold:
                return False, f"corr({candidate},{sym})={c:.2f}"
        return True, "ok"
