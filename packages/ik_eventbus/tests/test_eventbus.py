"""Real tests for ik_eventbus."""
import asyncio
import pytest
from ik_eventbus import Event, EventBus


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_and_replay(self):
        bus = EventBus()
        try:
            eid = await bus.publish(Event(type="user.created", payload={"id": 1}))
            assert eid
            events = list(bus.replay())
            assert len(events) == 1
            assert events[0].type == "user.created"
            assert events[0].payload == {"id": 1}
        finally:
            bus.close()

    @pytest.mark.asyncio
    async def test_replay_by_type(self):
        bus = EventBus()
        try:
            await bus.publish(Event(type="a", payload={}))
            await bus.publish(Event(type="b", payload={}))
            await bus.publish(Event(type="a", payload={"v": 2}))
            a_events = list(bus.replay(event_type="a"))
            assert len(a_events) == 2
        finally:
            bus.close()

    @pytest.mark.asyncio
    async def test_event_idempotent(self):
        bus = EventBus()
        try:
            e = Event(type="x", payload={}, id="fixed-id")
            await bus.publish(e)
            await bus.publish(e)  # same id
            events = list(bus.replay())
            assert len(events) == 1
        finally:
            bus.close()
