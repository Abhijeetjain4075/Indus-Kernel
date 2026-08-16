"""NATS JetStream event bus adapter with graceful lifecycle."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import nats

from ik_kernel.config import get_settings


class EventBus:
    def __init__(self, url: str):
        self.url = url
        self.nc = None
        self.js = None

    async def connect(self):
        if self.nc and not self.nc.is_closed:
            return
        self.nc = await nats.connect(self.url)
        self.js = self.nc.jetstream()

    async def publish(
        self, subject: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None
    ):
        await self.connect()
        return await self.js.publish(subject, json.dumps(payload).encode(), headers=headers)

    async def subscribe(
        self, subject: str, cb: Callable[..., Awaitable[None]], durable: str | None = None
    ):
        await self.connect()
        return await self.js.subscribe(subject, durable=durable, cb=cb)

    async def close(self):
        if self.nc and not self.nc.is_closed:
            await self.nc.drain()


_bus = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus(str(get_settings().nats_url))
    return _bus
