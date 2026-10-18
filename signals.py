"""
Signal engine — ties structure + indicators + regime into a ranked setup.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from config import Config
from indicators import atr, adx, rolling_percentile_rank, htf_bias
from structure import (
    Sweep, MSS, FVG,
    find_swings, build_liquidity_pools, detect_sweeps, detect_mss, find_fvgs,
)


@dataclass
class Signal:
    direction: str
    entry_index: int
    entry_price: float
    stop: float
    risk: float
    targets: List[float]
    sweep: Sweep
    mss: MSS
    fvg: FVG
    confluence_score: int
    reasons: List[str] = field(default_factory=list)


def _entry_price_from_fvg(fvg: FVG, mode: str) -> float:
    if mode == "proximal":
        return fvg.top if fvg.direction == "bullish" else fvg.bottom
    if mode == "distal":
        return fvg.bottom if fvg.direction == "bullish" else fvg.top
    return (fvg.top + fvg.bottom) / 2.0


def _score(
    htf_ok: bool,
    sweep: Sweep,
    mss: MSS,
    fvg: FVG,
    in_session: bool,
) -> (int, List[str]):
    score = 0
    reasons: List[str] = []
    if htf_ok:
        score += 1
        reasons.append("HTF bullish bias aligned")
    if sweep.pool_touches >= 3:
        score += 1
        reasons.append(f"Strong liquidity pool ({sweep.pool_touches} touches)")
    if mss.body_atr >= 1.2:
        score += 1
        reasons.append(f"Displacement MSS ({mss.body_atr:.2f}x ATR)")
    if fvg.size_atr >= 0.30:
        score += 1
        reasons.append(f"Healthy FVG ({fvg.size_atr:.2f}x ATR)")
    if in_session:
        score += 1
        reasons.append("Inside killzone window")
    return score, reasons


def generate_signals(df: pd.DataFrame, cfg: Config) -> List[Signal]:
    if len(df) < max(100, cfg.swing_lookback * 4):
        return []

    atr_s = atr(df, cfg.atr_period)
    adx_s = adx(df, cfg.adx_period)
    atr_pct = rolling_percentile_rank(atr_s, window=200)
    htf_bias_s = htf_bias(df, cfg.htf_interval, cfg.htf_ema_period)

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    highs, lows = find_swings(df, cfg.swing_lookback)
    if len(lows) < 2 or len(highs) < 2:
        return []

    median_atr = atr_s.dropna().median()
    if np.isnan(median_atr) or median_atr <= 0:
        return []

    pools_low = build_liquidity_pools(lows, median_atr, cfg.equal_level_tol_atr, cfg.min_pool_touches)
    pools_high = build_liquidity_pools(highs, median_atr, cfg.equal_level_tol_atr, cfg.min_pool_touches)
    pools = pools_low + pools_high
    if not pools:
        return []

    sweeps = detect_sweeps(df, pools)

    signals: List[Signal] = []

    for sweep in sweeps:
        if sweep.direction != "bullish":
            continue

        mss = detect_mss(
            df, sweep, highs, atr_s,
            cfg.mss_body_atr_mult, cfg.mss_max_bars_after_sweep,
        )
        if mss is None:
            continue

        # FVGs created during the displacement leg
        fvgs = find_fvgs(
            df, atr_s, cfg.min_fvg_size_atr,
            start=sweep.index, end=mss.index + 4,
        )
        bull_fvgs = [f for f in fvgs if f.direction == "bullish"]
        if not bull_fvgs:
            continue
        fvg = sorted(bull_fvgs, key=lambda f: f.index)[-1]

        entry_price = _entry_price_from_fvg(fvg, cfg.entry_mode)
        atr_at_mss = atr_s.iloc[mss.index]
        if np.isnan(atr_at_mss):
            continue
        stop = sweep.extreme - cfg.stop_buffer_atr * atr_at_mss
        risk = entry_price - stop
        if risk <= 0:
            continue

        # Wait for pullback into FVG
        entry_bar = None
        end_search = min(mss.index + cfg.max_bars_wait_for_retrace + 1, len(df))
        for j in range(mss.index + 1, end_search):
            if df["low"].iloc[j] <= entry_price:
                entry_bar = j
                break
        if entry_bar is None:
            continue

        # Regime filters at entry bar
        adx_val = adx_s.iloc[entry_bar]
        if np.isnan(adx_val) or adx_val < cfg.adx_min_trend:
            continue
        atr_pct_val = atr_pct.iloc[entry_bar]
        if not np.isnan(atr_pct_val) and atr_pct_val > cfg.atr_pct_max:
            continue

        # HTF bias
        htf_ok = int(htf_bias_s.iloc[entry_bar]) == 1
        if cfg.require_htf_bias and not htf_ok:
            continue

        # Session window
        hour = df.index[entry_bar].hour
        in_session = cfg.session_start_utc <= hour < cfg.session_end_utc
        if cfg.enforce_session and not in_session:
            continue

        score, reasons = _score(htf_ok, sweep, mss, fvg, in_session)
        if score < cfg.min_confluence_score:
            continue

        targets = [entry_price + r * risk for r in cfg.tp_r]

        signals.append(
            Signal(
                direction="long",
                entry_index=entry_bar,
                entry_price=float(entry_price),
                stop=float(stop),
                risk=float(risk),
                targets=targets,
                sweep=sweep,
                mss=mss,
                fvg=fvg,
                confluence_score=score,
                reasons=reasons,
            )
        )

    return signals
