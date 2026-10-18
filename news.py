"""Economic calendar blackout. Reads optional news.csv."""
import csv
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

log = logging.getLogger("news")


class NewsBlackout:
    def __init__(self, path: str = "news.csv", before_min: int = 30, after_min: int = 30):
        self.path = path
        self.before = timedelta(minutes=before_min)
        self.after = timedelta(minutes=after_min)
        self.events: List[Tuple[datetime, str, str]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                r = csv.DictReader(f)
                for row in r:
                    t = row.get("time_utc", "").strip()
                    impact = row.get("impact", "").strip().lower()
                    ccy = row.get("currency", "").strip().upper()
                    if not t or impact != "high":
                        continue
                    try:
                        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                    except Exception:
                        continue
                    self.events.append((dt, impact, ccy))
            log.info("news blackout loaded %d high-impact events", len(self.events))
        except Exception as e:
            log.warning("news load failed: %s", e)

    def blocked(self, when: datetime, symbol_ccy: str = "") -> bool:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        for dt, _impact, ccy in self.events:
            if symbol_ccy and ccy and symbol_ccy.upper() not in ccy and ccy not in symbol_ccy.upper():
                continue
            if dt - self.before <= when <= dt + self.after:
                return True
        return False
