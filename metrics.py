"""Extended backtest metrics: Sharpe, Sortino, Calmar, expectancy, regime breakdown."""
import math
from typing import List
import numpy as np
import pandas as pd


def extended_metrics(trades: list, equity_curve: list, bars_per_year: int = 24 * 252) -> dict:
    if not trades:
        return {}
    pnls = np.array([t.pnl for t in trades], dtype=float)
    rs = np.array([t.r_multiple for t in trades], dtype=float)

    eq = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
    rets = eq.pct_change().dropna()
    if rets.std() > 0:
        sharpe = float(np.sqrt(bars_per_year) * rets.mean() / rets.std())
        downside = rets[rets < 0].std()
        sortino = float(np.sqrt(bars_per_year) * rets.mean() / downside) if downside > 0 else float("inf")
    else:
        sharpe = sortino = 0.0

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = float(-dd.min()) if len(dd) else 0.0
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] else 0.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    calmar = float(cagr / max_dd) if max_dd > 0 else float("inf")

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    expectancy = float(np.mean(rs)) if len(rs) else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cagr": cagr,
        "expectancy_r": expectancy,
        "total_return_pct": total_return * 100,
        "max_dd_pct": max_dd * 100,
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "avg_r_win": float(np.mean(rs[rs > 0])) if (rs > 0).any() else 0.0,
        "avg_r_loss": float(np.mean(rs[rs < 0])) if (rs < 0).any() else 0.0,
    }
