"""Multi-timeframe confluence — reads 1H and 4H for context."""
import numpy as np
import pandas as pd


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def mtf_confluence(df_ltf: pd.DataFrame, entry_idx: int, direction: str) -> int:
    """Returns a bonus 0-3 based on 1H/4H alignment."""
    try:
        ltf_time = df_ltf.index[entry_idx]
    except Exception:
        return 0

    bonus = 0
    for rule in ("1h", "4h"):
        try:
            htf = _resample(df_ltf.iloc[: entry_idx + 1], rule)
        except Exception:
            continue
        if len(htf) < 30:
            continue
        ema = htf["close"].ewm(span=20, adjust=False).mean()
        last_close = float(htf["close"].iloc[-1])
        last_ema = float(ema.iloc[-1])
        if direction == "long" and last_close > last_ema:
            bonus += 1
        elif direction == "short" and last_close < last_ema:
            bonus += 1

    # Alignment between 1H and 4H trend
    try:
        h1 = _resample(df_ltf.iloc[: entry_idx + 1], "1h")
        h4 = _resample(df_ltf.iloc[: entry_idx + 1], "4h")
        if len(h1) > 20 and len(h4) > 20:
            e1 = h1["close"].ewm(span=20, adjust=False).mean()
            e4 = h4["close"].ewm(span=20, adjust=False).mean()
            b1 = 1 if h1["close"].iloc[-1] > e1.iloc[-1] else -1
            b4 = 1 if h4["close"].iloc[-1] > e4.iloc[-1] else -1
            if b1 == b4 and ((direction == "long" and b1 == 1) or (direction == "short" and b1 == -1)):
                bonus += 1
    except Exception:
        pass

    return min(bonus, 3)
