"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ik_reasoning.types import ReasoningRequest, ReasoningResult


class BaseReasoningStrategy(ABC):
    name: str = "base"

    @abstractmethod
    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        """Run reasoning over the request and return a result."""
