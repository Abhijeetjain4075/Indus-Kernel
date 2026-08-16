"""Configuration primitives with immutable layered snapshots."""

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ConfigSnapshot:
    environment: str = "dev"
    debug: bool = False
    api_port: int = 8000

    def overlay(self, **values):
        return replace(self, **values)


def from_env() -> ConfigSnapshot:
    return ConfigSnapshot(
        os.getenv("INDUS_ENV", "dev"),
        os.getenv("INDUS_DEBUG", "false").lower() == "true",
        int(os.getenv("INDUS_API_PORT", "8000")),
    )


__version__ = "1.0.0"
