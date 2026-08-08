"""Seed data for development.

Creates:
- Default tenant (t-default)
- Default admin user
- Default roles
- Default models (gpt-4o-mini, gpt-4o, claude-3-5-sonnet)
- Default workflows
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add workspace to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


async def seed() -> None:
    """Insert seed data. Idempotent (uses INSERT ... ON CONFLICT)."""
    # In M0, this is a no-op. In M1, will insert default tenant + roles.
    print("M0: seed is a no-op (full seeding comes in M1)")


if __name__ == "__main__":
    asyncio.run(seed())
