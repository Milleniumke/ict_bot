"""Direction-agnostic signal engine — long AND short."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from features import extract_features, FEATURE_NAMES
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
    direction: str           # "long" | "short"
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    risk: float
    targets: List[float]
    sweep: Sweep
    mss: MSS
    fvg: FVG
    confluence_score: int
    reasons: List[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    adaptive_p: float = 0.5


def _entry_from_fvg(fvg: FVG, mode: str) -> float:
    if fvg.direction == "bullish":
        if mode == "proximal": return fvg.top
        if mode == "distal":   return fvg.bottom
        return (fvg.top + fvg.bottom) / 2.0
    else:
        if mode == "proximal": return fvg.bottom
        if mode == "distal":   return fvg.top
        return (fvg.top + fvg.bottom) / 2.0


def _score(htf_ok: bool, sweep: Sweep, mss: MSS, fvg: FVG, in_session: bool) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    if htf_ok:
        score += 1
        reasons.append("HTF bias aligned")
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
    if len(df) < max(120, cfg.swing_lookback * 6):
        return []

    atr_s = atr(df, cfg.atr_period)
    adx_s = adx(df, cfg.adx_period)
    atr_pct = rolling_percentile_rank(atr_s, window=200)
    htf_s = htf_bias(df, cfg.htf_interval, cfg.htf_ema_period)

    highs, lows = find_swings(df, cfg.swing_lookback)
    if len(highs) < 2 or len(lows) < 2:
        return []

    median_atr = float(atr_s.dropna().median())
    if not median_atr or np.isnan(median_atr):
        return []

    pools_low  = build_liquidity_pools(lows,  median_atr, cfg.equal_level_tol_atr, cfg.min_pool_touches)
    pools_high = build_liquidity_pools(highs, median_atr, cfg.equal_level_tol_atr, cfg.min_pool_touches)
    pools = pools_low + pools_high
    if not pools:
        return []

    sweeps = detect_sweeps(df, pools)
    signals: List[Signal] = []

    for sweep in sweeps:
        direction = "long" if sweep.direction == "bullish" else "short"

        mss = detect_mss(
            df, sweep, highs, lows, atr_s,
            cfg.mss_body_atr_mult, cfg.mss_max_bars_after_sweep,
            direction=direction,
        )
        if mss is None:
            continue

        fvgs = find_fvgs(
            df, atr_s, cfg.min_fvg_size_atr,
            start=sweep.index, end=mss.index + 4,
        )
        wanted = "bullish" if direction == "long" else "bearish"
        cand = [f for f in fvgs if f.direction == wanted]
        if not cand:
            continue
        fvg = sorted(cand, key=lambda f: f.index)[-1]

        entry_price = _entry_from_fvg(fvg, cfg.entry_mode)
        atr_at_mss = atr_s.iloc[mss.index]
        if np.isnan(atr_at_mss):
            continue

        if direction == "long":
            stop = sweep.extreme - cfg.stop_buffer_atr * atr_at_mss
            risk = entry_price - stop
        else:
            stop = sweep.extreme + cfg.stop_buffer_atr * atr_at_mss
            risk = stop - entry_price
        if risk <= 0:
            continue

        # Wait for pullback into FVG
        end_search = min(mss.index + cfg.max_bars_wait_for_retrace + 1, len(df))
        entry_bar: Optional[int] = None
        for j in range(mss.index + 1, end_search):
            if direction == "long" and df["low"].iloc[j] <= entry_price:
                entry_bar = j; break
            if direction == "short" and df["high"].iloc[j] >= entry_price:
                entry_bar = j; break
        if entry_bar is None:
            continue

        # Regime gates
        a = adx_s.iloc[entry_bar]
        if np.isnan(a) or a < cfg.adx_min_trend:
            continue
        p = atr_pct.iloc[entry_bar]
        if not np.isnan(p) and p > cfg.atr_pct_max:
            continue

        # HTF bias
        htf_val = int(htf_s.iloc[entry_bar])
        htf_ok = (htf_val == 1 and direction == "long") or (htf_val == -1 and direction == "short")
        if cfg.require_htf_bias and not htf_ok:
            continue

        hour = df.index[entry_bar].hour
        in_session = cfg.session_start_utc <= hour < cfg.session_end_utc
        if cfg.enforce_session and not in_session:
            continue

        score, reasons = _score(htf_ok, sweep, mss, fvg, in_session)
        if score < cfg.min_confluence_score:
            continue

        if direction == "long":
            targets = [entry_price + r * risk for r in cfg.tp_r]
        else:
            targets = [entry_price - r * risk for r in cfg.tp_r]

        signals.append(Signal(
            direction=direction,
            entry_index=entry_bar,
            entry_time=df.index[entry_bar],
            entry_price=float(entry_price),
            stop=float(stop),
            risk=float(risk),
            targets=targets,
            sweep=sweep, mss=mss, fvg=fvg,
            confluence_score=score,
            reasons=reasons,
        ))

    # Augment with features + adaptive prediction
    registry = cfg.adaptive_registry  # attached at runtime, may be None
    for s in signals:
        try:
            s.features = extract_features(
                df, s.entry_index, s.direction, s.sweep, s.mss, s.fvg,
                atr_s, adx_s, atr_pct, htf_s,
                cfg.session_start_utc <= df.index[s.entry_index].hour < cfg.session_end_utc,
            )
            if registry is not None and s.features:
                p = registry.predict(cfg.symbol, s.features)
                if p is not None:
                    s.adaptive_p = p
        except Exception:
            pass

    return signals
    
