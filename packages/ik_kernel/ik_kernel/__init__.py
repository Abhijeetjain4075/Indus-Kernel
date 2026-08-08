"""ik_kernel — Indus Kernel core.

The core package provides:
- App factory (FastAPI)
- CLI entrypoint
- Lifespan management
- Dependency injection
- Configuration loading
- OpenTelemetry bootstrap
- Error handling
"""

from ik_kernel.app import create_app
from ik_kernel.config import Settings, get_settings
from ik_kernel.version import __version__

__all__ = ["__version__", "create_app", "Settings", "get_settings"]
