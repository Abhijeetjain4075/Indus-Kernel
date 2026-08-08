"""Pytest configuration.

Adds the workspace packages to PYTHONPATH so `ik_*` packages are importable,
ensures asyncio mode is set, and provides shared fixtures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add the workspace root to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add each ik_* package to PYTHONPATH (so `import ik_kernel` etc. works
# without the workspace being installed via uv)
PACKAGES = ROOT / "packages"
if PACKAGES.exists():
    for pkg_dir in PACKAGES.iterdir():
        if pkg_dir.is_dir() and pkg_dir.name.startswith("ik_"):
            sys.path.insert(0, str(pkg_dir))

# Force test environment
os.environ.setdefault("INDUS_ENVIRONMENT", "test")
os.environ.setdefault("INDUS_LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Pin asyncio backend for anyio tests."""
    return "asyncio"
