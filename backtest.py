"""Bar-by-bar backtester with partial TP ladder + circuit breakers."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import Config
from signals import Signal, generate_signals
from risk import RiskManager


@dataclass
class Trade:
    signal: Signal
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    stop: float
    targets: List[float]
    exits: List[tuple] = field(default_factory=list)  # (price, fraction, reason)
    exit_time: Optional[pd.Timestamp] = None
    pnl: float = 0.0
    r_multiple: float = 0.0


def _simulate_trade(df: pd.DataFrame, sig: Signal, size: float, cfg: Config) -> Trade:
    entry_idx = sig.entry_index
    entry_price = sig.entry_price
    stop = sig.stop
    targets = sig.targets
    tp_sizes = cfg.tp_size

    t_hit = [False] * len(targets)
    remaining = 1.0
    pnl_per_unit = 0.0
    exits: List[tuple] = []

    end = min(entry_idx + cfg.max_hold_bars, len(df))
    exit_time = df.index[end - 1]

    for j in range(entry_idx + 1, end):
        h = df["high"].iloc[j]
        l = df["low"].iloc[j]

        # Conservative: stop is checked before TPs in the same bar
        if l <= stop:
            pnl_per_unit += (stop - entry_price) * remaining
            exits.append((stop, remaining, "stop"))
            remaining = 0.0
            exit_time = df.index[j]
            break

        for k, tp in enumerate(targets):
            if not t_hit[k] and h >= tp:
                frac = tp_sizes[k] if k < len(tp_sizes) else 0.0
                frac = min(frac, remaining)
                if frac <= 0:
                    continue
                pnl_per_unit += (tp - entry_price) * frac
                exits.append((tp, frac, f"tp{k + 1}"))
                remaining -= frac
                t_hit[k] = True

        if remaining <= 1e-9:
            exit_time = df.index[j]
            break
    else:
        # timeout exit at last close
        if remaining > 1e-9:
            last_close = df["close"].iloc[end - 1]
            pnl_per_unit += (last_close - entry_price) * remaining
            exits.append((last_close, remaining, "timeout"))
            exit_time = df.index[end - 1]

    pnl = pnl_per_unit * size
    denom = (entry_price - stop)
    r_mult = pnl_per_unit / denom if denom != 0 else 0.0
    return Trade(sig, size, entry_price, df.index[entry_idx], stop, targets,
                 exits, exit_time, pnl, r_mult)


def run_backtest(df: pd.DataFrame, cfg: Config) -> dict:
    signals = sorted(generate_signals(df, cfg), key=lambda s: s.entry_index)
    rm = RiskManager(cfg)
    trades: List[Trade] = []
    equity_curve = [(df.index[0], cfg.account_balance)]

    current_day = None
    for sig in signals:
        t = df.index[sig.entry_index]
        day = t.date()
        if current_day is None:
            current_day = day
        elif day != current_day:
            rm.roll_day()
            current_day = day

        ok, _ = rm.can_open(t)
        if not ok:
            continue
        size = rm.position_size(sig.entry_price, sig.stop)
        if size <= 0:
            continue

        rm.register_open()
        trade = _simulate_trade(df, sig, size, cfg)
        rm.register_close(trade.pnl)
        trades.append(trade)
        equity_curve.append((trade.exit_time, rm.state.balance))

    if not trades:
        return {"trades": [], "metrics": {"n_trades": 0}, "equity_curve": equity_curve}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / len(trades)
    avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = (sum(t.pnl for t in wins) / gross_loss) if gross_loss > 0 else float("inf")

    eq = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd_pct = float(dd.min() * 100)

    metrics = {
        "n_trades": len(trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "final_balance": float(rm.state.balance),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": max_dd_pct,
        "avg_r": float(np.mean([t.r_multiple for t in trades])),
    }
    return {"trades": trades, "metrics": metrics, "equity_curve": equity_curve}


def export_trades_csv(result: dict, path: str = "trades.csv"):
    rows = []
    for t in result["trades"]:
        rows.append({
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "stop": t.stop,
            "tp1": t.targets[0] if len(t.targets) > 0 else None,
            "tp2": t.targets[1] if len(t.targets) > 1 else None,
            "tp3": t.targets[2] if len(t.targets) > 2 else None,
            "score": t.signal.confluence_score,
            "size": t.size,
            "pnl": t.pnl,
            "r_multiple": t.r_multiple,
            "exit_time": t.exit_time,
            "exits": str(t.exits),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
