"""Indus Kernel CLI entrypoint.

Usage:
    ik-kernel dev [--reload] [--debug]
    ik-kernel version
    ik-kernel hello
    ik-kernel shell

This is a thin wrapper over the FastAPI app + utility commands.
"""

from __future__ import annotations

import argparse
import sys


def cmd_dev(args: argparse.Namespace) -> int:
    """Start the kernel in development mode."""
    import uvicorn

    from ik_kernel.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "ik_kernel.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
    )
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    """Print the kernel version."""
    from ik_kernel.config import get_settings
    from ik_kernel.version import __version__

    s = get_settings()
    print(f"indus-kernel {__version__}")
    print(f"environment: {s.environment}")
    print(f"api: http://{s.api_host}:{s.api_port}")
    return 0


def cmd_hello(_: argparse.Namespace) -> int:
    """Call the hello-world agent via the running kernel."""
    import httpx

    from ik_kernel.config import get_settings

    s = get_settings()
    url = f"http://{s.api_host}:{s.api_port}{s.api_prefix}/agents/runs"
    r = httpx.post(url, json={"goal": "Hello from the Indus Kernel CLI!"}, timeout=30.0)
    r.raise_for_status()
    print(r.json())
    return 0


def cmd_shell(_: argparse.Namespace) -> int:
    """Open an IPython shell with the kernel loaded."""
    try:
        import IPython
    except ImportError:
        print("Install ipython: uv add ipython", file=sys.stderr)
        return 1
    from ik_kernel.app import app
    from ik_kernel.config import get_settings

    IPython.embed(
        header="Indus Kernel interactive shell. `app`, `get_settings()` available.",
        user_ns={"app": app, "get_settings": get_settings},
    )
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Run database migrations."""
    import subprocess

    cmd = ["alembic", "upgrade", "head" if not args.downgrade else "base"]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ik-kernel",
        description="Indus Kernel — the cognitive operating system CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dev
    p_dev = sub.add_parser("dev", help="Start the kernel in dev mode")
    p_dev.add_argument("--reload", action="store_true", help="Hot reload on file change")
    p_dev.add_argument("--debug", action="store_true", help="Enable debugpy")
    p_dev.set_defaults(func=cmd_dev)

    # version
    p_ver = sub.add_parser("version", help="Print version")
    p_ver.set_defaults(func=cmd_version)

    # hello
    p_hello = sub.add_parser("hello", help="Call the hello-world agent")
    p_hello.set_defaults(func=cmd_hello)

    # shell
    p_shell = sub.add_parser("shell", help="Open IPython shell with the kernel loaded")
    p_shell.set_defaults(func=cmd_shell)

    # migrate
    p_mig = sub.add_parser("migrate", help="Run database migrations")
    p_mig.add_argument("--downgrade", action="store_true", help="Downgrade instead of upgrade")
    p_mig.set_defaults(func=cmd_migrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
