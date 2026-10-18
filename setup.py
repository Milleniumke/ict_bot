"""Persistent signal dedup — survives restarts."""
import json
import os
import threading


class SignalState:
    def __init__(self, path: str = "state.json", max_entries: int = 5000):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._seen = set()
        self._order = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._order = list(data.get("seen", []))
            self._seen = set(self._order)
        except Exception:
            self._seen = set()
            self._order = []

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"seen": self._order[-self.max_entries:]}, f)
        os.replace(tmp, self.path)

    def has(self, key: str) -> bool:
        return key in self._seen

    def add(self, key: str):
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            self._order.append(key)
            if len(self._order) > self.max_entries:
                drop = self._order[:-self.max_entries]
                self._order = self._order[-self.max_entries:]
                self._seen -= set(drop)
            self._save()
