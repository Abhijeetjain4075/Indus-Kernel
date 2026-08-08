"""IPython shell bootstrap for `ik-kernel shell`."""
from __future__ import annotations

# Imports for the interactive shell
from ik_kernel import __version__
from ik_kernel.app import app
from ik_kernel.config import get_settings
from ik_kernel.version import __version__ as version

print(f"Indus Kernel v{__version__} interactive shell")
print("Available objects: app, get_settings(), version")
print()
