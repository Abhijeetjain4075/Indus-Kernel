"""ik_kernel — Indus Kernel core.

The core package provides:
- App factory (FastAPI)
- CLI entrypoint
- Lifespan management
- Dependency injection
- Configuration loading
- OpenTelemetry bootstrap
- Error handling
- Orchestration layer (the control plane)
"""

# Lazy imports: app.py pulls in all routers, which is heavy.
# Submodules should be importable without triggering that.
from ik_kernel.version import __version__


def create_app(*args, **kwargs):
    """Lazy proxy for the FastAPI app factory."""
    from ik_kernel.app import create_app as _factory
    return _factory(*args, **kwargs)


def get_settings(*args, **kwargs):
    """Lazy proxy for settings."""
    from ik_kernel.config import get_settings as _get
    return _get(*args, **kwargs)


__all__ = ["__version__", "create_app", "get_settings", "Settings", "Orchestrator", "get_orchestrator"]


def __getattr__(name):
    """Lazy attribute access for Settings, Orchestrator, get_orchestrator."""
    if name == "Settings":
        from ik_kernel.config import Settings
        return Settings
    if name == "Orchestrator":
        from ik_kernel.orchestration.orchestrator import Orchestrator
        return Orchestrator
    if name == "get_orchestrator":
        from ik_kernel.orchestration.orchestrator import get_orchestrator as _go
        return _go
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
