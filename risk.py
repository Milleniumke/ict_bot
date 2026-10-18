"""Position sizing + hard circuit breakers."""
from dataclasses import dataclass

from config import Config


@dataclass
class RiskState:
    balance: float
    peak_balance: float
    day_start_balance: float
    consecutive_losses: int = 0
    open_positions: int = 0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = RiskState(
            balance=cfg.account_balance,
            peak_balance=cfg.account_balance,
            day_start_balance=cfg.account_balance,
        )

    def can_open(self, current_time) -> (bool, str):
        s = self.state
        if s.halted:
            return False, s.halt_reason
        if s.open_positions >= self.cfg.max_concurrent_positions:
            return False, "max concurrent positions"
        if s.consecutive_losses >= self.cfg.max_consecutive_losses:
            s.halted = True
            s.halt_reason = f"consecutive losses >= {self.cfg.max_consecutive_losses}"
            return False, s.halt_reason

        daily_dd = (s.day_start_balance - s.balance) / s.day_start_balance
        if daily_dd >= self.cfg.max_daily_dd_pct:
            return False, "daily drawdown limit"

        total_dd = (s.peak_balance - s.balance) / s.peak_balance
        if total_dd >= self.cfg.max_total_dd_pct:
            s.halted = True
            s.halt_reason = "total drawdown limit"
            return False, s.halt_reason

        return True, "ok"

    def position_size(self, entry: float, stop: float) -> float:
        risk_amount = self.state.balance * self.cfg.risk_per_trade_pct
        per_unit = abs(entry - stop)
        if per_unit <= 0:
            return 0.0
        return risk_amount / per_unit

    def register_open(self):
        self.state.open_positions += 1

    def register_close(self, pnl: float):
        s = self.state
        s.open_positions = max(0, s.open_positions - 1)
        s.balance += pnl
        if s.balance > s.peak_balance:
            s.peak_balance = s.balance
        if pnl < 0:
            s.consecutive_losses += 1
        else:
            s.consecutive_losses = 0

    def roll_day(self):
        self.state.day_start_balance = self.state.balance
        self.state.halted = False
        self.state.halt_reason = ""
