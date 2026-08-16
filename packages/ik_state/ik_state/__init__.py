"""Thread-safe in-process state store for development; durable deployments use Postgres."""

from dataclasses import dataclass, field
from threading import RLock


@dataclass
class StateStore:
    values: dict[str, object] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def get(self, key, default=None):
        with self._lock:
            return self.values.get(key, default)

    def set(self, key, value):
        with self._lock:
            self.values[key] = value

    def delete(self, key):
        with self._lock:
            self.values.pop(key, None)

    def snapshot(self):
        with self._lock:
            return dict(self.values)
