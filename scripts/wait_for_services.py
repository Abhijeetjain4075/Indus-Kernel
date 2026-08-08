#!/usr/bin/env python3
"""Wait for docker compose services to be healthy.

Usage:
    python scripts/wait_for_services.py
    python scripts/wait_for_services.py --timeout 120
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Iterable


def check_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


SERVICES = [
    ("Postgres", "localhost", 5432),
    ("Redis", "localhost", 6379),
    ("NATS", "localhost", 4222),
    ("Qdrant", "localhost", 6333),
    ("Neo4j HTTP", "localhost", 7474),
    ("Neo4j Bolt", "localhost", 7687),
    ("Temporal", "localhost", 7233),
    ("Temporal UI", "localhost", 8080),
    ("LiteLLM", "localhost", 4000),
    ("Langfuse", "localhost", 3000),
    ("OTel Collector gRPC", "localhost", 4317),
    ("Prometheus", "localhost", 9091),
    ("Grafana", "localhost", 3001),
    ("Jaeger UI", "localhost", 16686),
    ("MinIO API", "localhost", 9000),
]


def wait_for_services(services: Iterable, timeout: float = 90.0, interval: float = 2.0) -> bool:
    """Wait for all services to be ready."""
    pending = list(services)
    start = time.time()
    while pending and (time.time() - start) < timeout:
        still_pending = []
        for name, host, port in pending:
            if not check_tcp(host, port):
                still_pending.append((name, host, port))
        if not still_pending:
            return True
        print(f"  waiting on: {', '.join(n for n, _, _ in still_pending)}")
        pending = still_pending
        time.sleep(interval)
    return not pending


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for docker compose services.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Max seconds to wait")
    args = parser.parse_args()

    print(f"==> Waiting for {len(SERVICES)} services (timeout={args.timeout}s)")
    ok = wait_for_services(SERVICES, timeout=args.timeout)
    if ok:
        print("==> All services ready")
        return 0
    else:
        print("==> Some services not ready (timeout)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
