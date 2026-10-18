"""SQLite trade & signal journal."""
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager

log = logging.getLogger("journal")

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry REAL, stop REAL, targets TEXT,
    confluence INTEGER,
    reasons TEXT,
    sweep_level REAL, sweep_extreme REAL,
    mss_level REAL, mss_body_atr REAL,
    fvg_bottom REAL, fvg_top REAL, fvg_size_atr REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    symbol TEXT,
    direction TEXT,
    entry_time TEXT, exit_time TEXT,
    entry REAL, stop REAL, size REAL,
    pnl REAL, r_multiple REAL,
    exits TEXT,
    status TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);

CREATE INDEX IF NOT EXISTS idx_sig_key ON signals(symbol, interval, ts, direction);
"""


class Journal:
    def __init__(self, path: str = "journal.sqlite"):
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_signal(self, symbol: str, interval: str, s) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                """INSERT INTO signals
                (ts, symbol, interval, direction, entry, stop, targets,
                 confluence, reasons, sweep_level, sweep_extreme,
                 mss_level, mss_body_atr, fvg_bottom, fvg_top, fvg_size_atr)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(s.entry_time), symbol, interval, s.direction,
                    s.entry_price, s.stop, json.dumps(list(s.targets)),
                    s.confluence_score, json.dumps(s.reasons),
                    s.sweep.level, s.sweep.extreme,
                    s.mss.broken_level, s.mss.body_atr,
                    s.fvg.bottom, s.fvg.top, s.fvg.size_atr,
                ),
            )
            return cur.lastrowid

    def log_trade(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_time: str,
        entry: float,
        stop: float,
        size: float,
        status: str = "open",
    ) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                """INSERT INTO trades
                (signal_id, symbol, direction, entry_time, entry, stop, size, status)
                VALUES (?,?,?,?,?,?,?,?)""",
                (signal_id, symbol, direction, entry_time, entry, stop, size, status),
            )
            return cur.lastrowid

    def close_trade(self, trade_id: int, exit_time, pnl: float, r_multiple: float, exits):
        with self._lock, self._conn() as c:
            c.execute(
                """UPDATE trades
                SET exit_time=?, pnl=?, r_multiple=?, exits=?, status='closed'
                WHERE id=?""",
                (str(exit_time), pnl, r_multiple, json.dumps(exits), trade_id),
            )
