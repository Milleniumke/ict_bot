"""Fixed-length, normalized feature vector for every signal."""
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "sweep_depth_atr",
    "pool_touches",
    "mss_body_atr",
    "fvg_size_atr",
    "fvg_fill_pct",
    "dist_sweep_to_entry_atr",
    "dist_entry_to_swing_atr",
    "volume_ratio",
    "rsi_norm",
    "atr_percentile",
    "adx_norm",
    "htf_aligned",
    "session_flag",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def extract_features(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    sweep,
    mss,
    fvg,
    atr_series: pd.Series,
    adx_series: pd.Series,
    atr_pct: pd.Series,
    htf_bias_series: pd.Series,
    in_session: bool,
) -> dict:
    """Returns a dict of normalized features, all roughly in [0,1] or [-1,1]."""
    av = float(atr_series.iloc[entry_idx])
    if not av or np.isnan(av):
        return {}

    entry_price = float(df["close"].iloc[entry_idx])
    entry_low = float(df["low"].iloc[entry_idx])
    entry_high = float(df["high"].iloc[entry_idx])

    # Sweep depth: how far beyond the pool the sweep went
    sweep_depth = abs(sweep.extreme - sweep.level) / av

    # FVG fill at entry
    if fvg.direction == "bullish":
        fvg_range = max(fvg.top - fvg.bottom, 1e-9)
        fill = (fvg.top - entry_low) / fvg_range
    else:
        fvg_range = max(fvg.top - fvg.bottom, 1e-9)
        fill = (entry_high - fvg.bottom) / fvg_range
    fill = float(np.clip(fill, 0.0, 1.5))

    # Distance sweep -> entry
    dist_sweep_entry = abs(entry_price - sweep.extreme) / av

    # Distance entry -> next opposing swing (room to run)
    if direction == "long":
        swing_hi = df["high"].iloc[max(0, entry_idx - 30):entry_idx + 1].max()
        dist_swing = max(0.0, (swing_hi - entry_price) / av) if swing_hi else 0.0
    else:
        swing_lo = df["low"].iloc[max(0, entry_idx - 30):entry_idx + 1].min()
        dist_swing = max(0.0, (entry_price - swing_lo) / av) if swing_lo else 0.0

    # Volume ratio
    vol_win = df["volume"].iloc[max(0, entry_idx - 20):entry_idx]
    vol_avg = float(vol_win.mean()) if len(vol_win) else 1.0
    vol_ratio = float(df["volume"].iloc[entry_idx]) / max(vol_avg, 1e-9)

    # RSI
    rsi_val = float(_rsi(df["close"]).iloc[entry_idx])
    rsi_norm = (rsi_val - 50.0) / 50.0

    atr_pct_val = float(atr_pct.iloc[entry_idx]) if not np.isnan(atr_pct.iloc[entry_idx]) else 0.5
    adx_val = float(adx_series.iloc[entry_idx]) if not np.isnan(adx_series.iloc[entry_idx]) else 0.0
    htf_val = int(htf_bias_series.iloc[entry_idx]) if not np.isnan(htf_bias_series.iloc[entry_idx]) else 0
    htf_ok = 1.0 if (htf_val == 1 and direction == "long") or (htf_val == -1 and direction == "short") else 0.0

    return {
        "sweep_depth_atr":          float(np.clip(sweep_depth, 0.0, 3.0) / 3.0),
        "pool_touches":             float(np.clip(sweep.pool_touches, 0, 6) / 6.0),
        "mss_body_atr":             float(np.clip(mss.body_atr, 0.0, 3.0) / 3.0),
        "fvg_size_atr":             float(np.clip(fvg.size_atr, 0.0, 2.0) / 2.0),
        "fvg_fill_pct":             fill,
        "dist_sweep_to_entry_atr":  float(np.clip(dist_sweep_entry, 0.0, 6.0) / 6.0),
        "dist_entry_to_swing_atr":  float(np.clip(dist_swing, 0.0, 12.0) / 12.0),
        "volume_ratio":             float(np.clip(vol_ratio, 0.0, 4.0) / 4.0),
        "rsi_norm":                 float(rsi_norm),
        "atr_percentile":           float(atr_pct_val),
        "adx_norm":                 float(np.clip(adx_val, 0.0, 100.0) / 100.0),
        "htf_aligned":              htf_ok,
        "session_flag":             1.0 if in_session else 0.0,
    }


def to_vector(features: dict) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in FEATURE_NAMES], dtype=float)
