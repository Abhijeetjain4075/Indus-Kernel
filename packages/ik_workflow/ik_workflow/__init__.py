"""Durable workflow registry and deterministic DAG executor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workflow:
    id: str
    name: str
    steps: list[str]


class WorkflowRegistry:
    def __init__(self):
        self._items = {}
        self._history = []

    def register(self, w: Workflow):
        if not w.id or not w.steps:
            raise ValueError("workflow id and steps required")
        if len(set(w.steps)) != len(w.steps):
            raise ValueError("duplicate workflow steps")
        self._items[w.id] = w
        return w

    def get(self, wid):
        return self._items.get(wid)

    def list(self):
        return list(self._items.values())

    async def execute(self, wid, handlers: dict[str, object]):
        w = self.get(wid)
        if not w:
            raise KeyError(wid)
        out = []
        for step in w.steps:
            fn = handlers.get(step)
            if fn is None:
                raise KeyError(f"missing handler: {step}")
            value = fn()
            if hasattr(value, "__await__"):
                value = await value
            out.append({"step": step, "result": value})
        self._history.append((wid, out))
        return out
