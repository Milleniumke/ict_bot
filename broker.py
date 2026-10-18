"""Broker adapter — Paper (default) and MetaTrader 5 (optional)."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

log = logging.getLogger("broker")


@dataclass
class Order:
    symbol: str
    side: str            # "buy" | "sell"
    size: float
    entry: float
    stop: float
    targets: List[float] = field(default_factory=list)


@dataclass
class Fill:
    ticket: int
    order: Order
    fill_price: float
    time: str


class Broker(ABC):
    @abstractmethod
    def place(self, order: Order) -> Optional[Fill]: ...

    @abstractmethod
    def close(self, ticket: int) -> bool: ...

    @abstractmethod
    def positions(self) -> List[Fill]: ...


class PaperBroker(Broker):
    """Simulated fills — logs everything, never touches a real account."""

    def __init__(self):
        self._fills: List[Fill] = []
        self._next = 1

    def place(self, order: Order) -> Optional[Fill]:
        fill = Fill(
            ticket=self._next,
            order=order,
            fill_price=order.entry,
            time=datetime.now(timezone.utc).isoformat(),
        )
        self._next += 1
        self._fills.append(fill)
        log.info(
            "[PAPER] %s %s %.4f size=%.4f SL=%.5f TPs=%s",
            order.side.upper(), order.symbol, order.entry, order.size,
            order.stop, order.targets,
        )
        return fill

    def close(self, ticket: int) -> bool:
        for i, f in enumerate(self._fills):
            if f.ticket == ticket:
                self._fills.pop(i)
                log.info("[PAPER] closed ticket %s", ticket)
                return True
        return False

    def positions(self) -> List[Fill]:
        return list(self._fills)


class MT5Broker(Broker):
    """Thin wrapper around the MetaTrader5 python package. Requires MT5 terminal."""

    def __init__(self, login: int, password: str, server: str, path: str = ""):
        import MetaTrader5 as mt5  # local import — only if used
        self.mt5 = mt5
        if not mt5.initialize(path=path or None, login=login, password=password, server=server):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        log.info("MT5 connected: %s", mt5.terminal_info())

    def place(self, order: Order) -> Optional[Fill]:
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            log.error("no tick for %s", order.symbol)
            return None
        price = tick.ask if order.side == "buy" else tick.bid
        otype = mt5.ORDER_TYPE_BUY if order.side == "buy" else mt5.ORDER_TYPE_SELL
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": float(order.size),
            "type": otype,
            "price": float(price),
            "sl": float(order.stop),
            "tp": float(order.targets[-1]) if order.targets else 0.0,
            "deviation": 20,
            "magic": 20250101,
            "comment": "ICT-A+",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        r = mt5.order_send(req)
        if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("order_send failed: %s", r)
            return None
        return Fill(r.order, order, r.price, str(datetime.now(timezone.utc)))

    def close(self, ticket: int) -> bool:
        mt5 = self.mt5
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        p = pos[0]
        tick = mt5.symbol_info_tick(p.symbol)
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": 20250101,
            "comment": "ICT-A+ close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        r = mt5.order_send(req)
        return r is not None and r.retcode == mt5.TRADE_RETCODE_DONE

    def positions(self) -> List[Fill]:
        out = []
        for p in self.mt5.positions_get() or []:
            out.append(Fill(p.ticket, Order(p.symbol, "buy" if p.type == 0 else "sell",
                                            p.volume, p.price_open, p.sl, []),
                            p.price_open, str(datetime.now(timezone.utc))))
        return out
