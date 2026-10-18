"""
Central configuration for the ICT A+ Setup Hunter.
Tune these values to change behavior without touching logic.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ==== Data ====
    symbol: str = "EURUSD=X"
    interval: str = "15m"
    period: str = "60d"

    # ==== Swing / structure ====
    swing_lookback: int = 5
    equal_level_tol_atr: float = 0.15
    min_pool_touches: int = 2

    # ==== Market Structure Shift ====
    mss_body_atr_mult: float = 1.0
    mss_max_bars_after_sweep: int = 10

    # ==== Fair Value Gap ====
    min_fvg_size_atr: float = 0.15
    max_bars_wait_for_retrace: int = 50
    entry_mode: str = "midpoint"   # "proximal" | "midpoint" | "distal"

    # ==== Regime filters ====
    atr_period: int = 14
    adx_period: int = 14
    adx_min_trend: float = 20.0
    atr_pct_max: float = 0.85

    # ==== Risk ====
    account_balance: float = 10_000.0
    risk_per_trade_pct: float = 0.01        # 1% per trade
    stop_buffer_atr: float = 0.3
    tp_r: tuple = (1.5, 2.5, 4.0)
    tp_size: tuple = (0.5, 0.3, 0.2)

    # ==== Circuit breakers ====
    max_daily_dd_pct: float = 0.02
    max_total_dd_pct: float = 0.06
    max_consecutive_losses: int = 3
    max_concurrent_positions: int = 2

    # ==== Confluence ====
    min_confluence_score: int = 3           # 0..5

    # ==== Session (UTC) ====
    session_start_utc: int = 7              # London open
    session_end_utc: int = 20               # NY close
    enforce_session: bool = False

    # ==== Higher-timeframe bias ====
    htf_interval: str = "4h"
    htf_ema_period: int = 50
    require_htf_bias: bool = True

    # ==== Backtest ====
    max_hold_bars: int = 200
