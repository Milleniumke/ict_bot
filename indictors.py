"""Technical indicators used across the bot."""
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = (
        100.0
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        / atr_
    )
    minus_di = (
        100.0
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        / atr_
    )
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def rolling_percentile_rank(series: pd.Series, window: int = 200) -> pd.Series:
    """Fast percentile rank of latest value within rolling window."""
    return series.rolling(window, min_periods=20).rank(pct=True)


def htf_bias(df_ltf: pd.DataFrame, htf_interval: str, ema_period: int) -> pd.Series:
    """
    Returns 1 when HTF close > EMA (bullish), -1 otherwise.
    Aligned to the LTF index via forward-fill.
    """
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    htf = df_ltf.resample(htf_interval).agg(ohlc).dropna()
    if len(htf) < ema_period:
        return pd.Series(0, index=df_ltf.index, dtype=int)
    ema = htf["close"].ewm(span=ema_period, adjust=False).mean()
    bias = np.where(htf["close"] > ema, 1, -1)
    bias_s = pd.Series(bias, index=htf.index)
    return bias_s.reindex(df_ltf.index, method="ffill").fillna(0).astype(int)
