"""Bar-based volume proxies (no tick data needed)."""
import numpy as np
import pandas as pd


def volume_delta_proxy(df: pd.DataFrame, idx: int, window: int = 20) -> float:
    """Signed volume strength of the entry bar relative to recent window."""
    if idx < 1:
        return 0.0
    o = df["open"].iloc[idx]
    c = df["close"].iloc[idx]
    h = df["high"].iloc[idx]
    l = df["low"].iloc[idx]
    rng = max(h - l, 1e-9)
    body_dir = (c - o) / rng  # -1..1
    vol = float(df["volume"].iloc[idx])
    base = float(df["volume"].iloc[max(0, idx - window):idx].mean()) or 1.0
    return float(body_dir * (vol / base))


def vwap_rolling(df: pd.DataFrame, idx: int, window: int = 20) -> float:
    if idx < window:
        return float(df["close"].iloc[idx])
    seg = df.iloc[idx - window:idx + 1]
    tp = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    v = seg["volume"].replace(0, 1.0)
    return float((tp * v).sum() / v.sum())


def volume_confirms(df: pd.DataFrame, idx: int, direction: str) -> bool:
    d = volume_delta_proxy(df, idx)
    if direction == "long":
        return d > 0.15
    return d < -0.15
