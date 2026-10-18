"""ICT structural primitives: swings, liquidity pools, sweeps, MSS, FVGs."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Swing:
    index: int
    price: float
    kind: str  # "high" | "low"


@dataclass
class LiquidityPool:
    level: float
    touches: int
    indices: List[int]
    kind: str


@dataclass
class Sweep:
    index: int
    level: float
    extreme: float
    direction: str  # "bullish" (swept lows) | "bearish" (swept highs)
    pool_touches: int


@dataclass
class MSS:
    index: int
    broken_swing_index: int
    broken_level: float
    direction: str
    body_atr: float


@dataclass
class FVG:
    index: int           # middle (displacement) bar
    top: float
    bottom: float
    direction: str
    size_atr: float


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------
def find_swings(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[Swing], List[Swing]]:
    highs, lows = [], []
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    for i in range(lookback, n - lookback):
        win_h = h[i - lookback : i + lookback + 1]
        win_l = l[i - lookback : i + lookback + 1]
        if h[i] == win_h.max() and np.sum(win_h == h[i]) == 1:
            highs.append(Swing(i, float(h[i]), "high"))
        if l[i] == win_l.min() and np.sum(win_l == l[i]) == 1:
            lows.append(Swing(i, float(l[i]), "low"))
    return highs, lows


# ---------------------------------------------------------------------------
# Liquidity pools (clusters of equal highs / equal lows)
# ---------------------------------------------------------------------------
def build_liquidity_pools(
    swings: List[Swing],
    atr_value: float,
    tol_mult: float,
    min_touches: int,
) -> List[LiquidityPool]:
    pools: List[LiquidityPool] = []
    tol = tol_mult * atr_value
    for s in swings:
        merged = False
        for p in pools:
            if abs(p.level - s.price) <= tol:
                p.level = (p.level * p.touches + s.price) / (p.touches + 1)
                p.touches += 1
                p.indices.append(s.index)
                merged = True
                break
        if not merged:
            pools.append(LiquidityPool(s.price, 1, [s.index], s.kind))
    return [p for p in pools if p.touches >= min_touches]


# ---------------------------------------------------------------------------
# Sweep detection
# ---------------------------------------------------------------------------
def detect_sweeps(df: pd.DataFrame, pools: List[LiquidityPool]) -> List[Sweep]:
    sweeps: List[Sweep] = []
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    for pool in pools:
        if not pool.indices:
            continue
        last_touch = max(pool.indices)
        for i in range(last_touch + 1, n):
            if pool.kind == "low":
                # Bullish sweep: wick below, close back above
                if l[i] < pool.level and c[i] > pool.level:
                    sweeps.append(Sweep(i, pool.level, float(l[i]), "bullish", pool.touches))
                    break
            else:
                # Bearish sweep: wick above, close back below
                if h[i] > pool.level and c[i] < pool.level:
                    sweeps.append(Sweep(i, pool.level, float(h[i]), "bearish", pool.touches))
                    break
    sweeps.sort(key=lambda s: s.index)
    return sweeps


# ---------------------------------------------------------------------------
# Market Structure Shift
# ---------------------------------------------------------------------------
def detect_mss(
    df: pd.DataFrame,
    sweep: Sweep,
    swing_highs: List[Swing],
    atr_series: pd.Series,
    body_mult: float,
    max_bars: int,
):
    if sweep.direction != "bullish":
        return None
    prior = [s for s in swing_highs if s.index < sweep.index]
    if not prior:
        return None
    target = prior[-1]

    o = df["open"].values
    c = df["close"].values
    end = min(sweep.index + max_bars + 1, len(df))
    for i in range(sweep.index, end):
        av = atr_series.iloc[i]
        if np.isnan(av) or av <= 0:
            continue
        body = abs(c[i] - o[i])
        if c[i] > target.price and body >= body_mult * av:
            return MSS(i, target.index, target.price, "bullish", float(body / av))
    return None


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------
def find_fvgs(
    df: pd.DataFrame,
    atr_series: pd.Series,
    min_size_atr: float,
    start: int = 2,
    end: int = None,
) -> List[FVG]:
    if end is None:
        end = len(df)
    h = df["high"].values
    l = df["low"].values
    fvgs: List[FVG] = []
    for i in range(max(start, 2), min(end, len(df))):
        av = atr_series.iloc[i]
        if np.isnan(av) or av <= 0:
            continue
        # Bullish FVG
        if l[i] > h[i - 2]:
            size = l[i] - h[i - 2]
            if size >= min_size_atr * av:
                fvgs.append(FVG(i - 1, float(l[i]), float(h[i - 2]), "bullish", float(size / av)))
        # Bearish FVG
        if h[i] < l[i - 2]:
            size = l[i - 2] - h[i]
            if size >= min_size_atr * av:
                fvgs.append(FVG(i - 1, float(l[i - 2]), float(h[i]), "bearish", float(size / av)))
    return fvgs
