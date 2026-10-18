"""Minimal smoke tests — run: python -m pytest tests/ -q"""
import numpy as np
import pandas as pd
import pytest

from config import Config
from structure import find_swings, build_liquidity_pools, detect_sweeps, find_fvgs, detect_mss
from indicators import atr, adx
from signals import generate_signals


def synth(n=500, seed=7):
    rng = np.random.default_rng(seed)
    price = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    df = pd.DataFrame({
        "open": price + rng.normal(0, 0.0002, n),
        "high": price + np.abs(rng.normal(0, 0.0006, n)),
        "low":  price - np.abs(rng.normal(0, 0.0006, n)),
        "close": price,
        "volume": rng.integers(100, 1000, n),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


def test_atr_positive():
    df = synth()
    a = atr(df, 14).dropna()
    assert (a > 0).all()


def test_adx_bounds():
    df = synth()
    v = adx(df, 14).dropna()
    assert (v >= 0).all() and (v <= 100).all()


def test_swings_found():
    df = synth(1000)
    highs, lows = find_swings(df, 5)
    assert len(highs) > 0 and len(lows) > 0


def test_pools_have_min_touches():
    df = synth(1000)
    highs, lows = find_swings(df, 5)
    a = float(atr(df, 14).dropna().median())
    pools = build_liquidity_pools(lows, a, 0.15, 2)
    assert all(p.touches >= 2 for p in pools)


def test_pipeline_runs():
    df = synth(1500)
    cfg = Config()
    cfg.min_confluence_score = 2
    cfg.require_htf_bias = False
    sigs = generate_signals(df, cfg)
    assert isinstance(sigs, list)
    for s in sigs:
        assert s.risk > 0
        assert s.direction in ("long", "short")
