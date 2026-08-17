"""ik_config — Configuration primitives with layered snapshots (M0, M1).

The kernel's configuration is a layered, immutable, validated
snapshot. Layers (in order of increasing precedence):
  1. Built-in defaults
  2. YAML files (config/base.yaml, config/{environment}.yaml)
  3. Environment variables (prefixed INDUS_)
  4. Explicit overrides (passed at construction time)

Each layer is merged into a single immutable snapshot. The merge
order is intentional: later layers override earlier ones. The
config is "frozen" — to change it, you build a new snapshot.

This module is the foundation for the kernel's reproducibility
invariant. A given commit + environment should yield a single,
deterministic configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

__version__ = "1.0.0"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _from_env(cls: type, prefix: str = "INDUS_") -> dict[str, Any]:
    """Read environment variables for each field of `cls`.

    Convention: field 'foo_bar' maps to env var 'INDUS_FOO_BAR'.
    """
    out: dict[str, Any] = {}
    for f in fields(cls):
        env_name = prefix + f.name.upper()
        if f.type is bool or f.type == "bool":
            out[f.name] = _env_bool(env_name, bool(f.default) if f.default is not None else False)
        elif f.type is int or f.type == "int":
            out[f.name] = _env_int(env_name, int(f.default) if f.default is not None else 0)
        elif f.type is float or f.type == "float":
            try:
                out[f.name] = float(os.getenv(env_name, f.default))
            except (TypeError, ValueError):
                out[f.name] = f.default
        else:
            out[f.name] = _env_str(env_name, str(f.default) if f.default is not None else "")
    return out


@dataclass(frozen=True)
class ConfigSnapshot:
    """An immutable layered configuration snapshot.

    Built once, frozen forever. To change a value, you build a new
    snapshot via `overlay(**overrides)`.
    """

    environment: str = "dev"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    database_url: str = ""
    redis_url: str = ""
    nats_url: str = ""
    otel_service_name: str = "indus-kernel"
    otel_exporter_otlp_endpoint: str = ""
    llm_default_provider: str = "indus"
    llm_fallback_providers: tuple[str, ...] = ("indus",)
    budget_max_cents_per_request: int = 100
    budget_max_cents_per_user_per_day: int = 10000
    cache_enabled: bool = True
    cache_ttl_s: int = 300
    workflow_max_concurrency: int = 8
    workflow_default_timeout_s: int = 30
    distributed_enabled: bool = False
    sandbox_default_timeout_s: int = 30
    sandbox_max_memory_mb: int = 1024
    telemetry_enabled: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def overlay(self, **values: Any) -> "ConfigSnapshot":
        """Return a new snapshot with the given fields overridden.

        Unknown fields raise ValueError. Use `extras` for ad-hoc
        values that don't have a typed field.
        """
        allowed = {f.name for f in fields(self)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **values)

    def with_extras(self, **values: Any) -> "ConfigSnapshot":
        """Merge values into the `extras` dict (shallow merge)."""
        merged = {**self.extras, **values}
        return replace(self, extras=merged)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def require(self, key: str) -> Any:
        """Require a value is set; raise if empty/None."""
        v = getattr(self, key, None)
        if v is None or v == "" or v == 0:
            raise ValueError(f"required config value not set: {key}")
        return v

    def is_production(self) -> bool:
        return self.environment in {"staging", "production"}

    def validate(self) -> None:
        """Validate the snapshot. Raises ValueError on invalid values."""
        if self.environment not in {"dev", "test", "staging", "production"}:
            raise ValueError(f"unknown environment: {self.environment}")
        if self.api_port < 1 or self.api_port > 65535:
            raise ValueError(f"api_port out of range: {self.api_port}")
        if self.is_production() and self.debug:
            raise ValueError("debug must be False in staging/production")
        if self.budget_max_cents_per_request < 0:
            raise ValueError("budget_max_cents_per_request must be >= 0")
        if self.workflow_default_timeout_s < 1:
            raise ValueError("workflow_default_timeout_s must be >= 1")
        if self.sandbox_max_memory_mb < 64:
            raise ValueError("sandbox_max_memory_mb must be >= 64")


def from_env(prefix: str = "INDUS_") -> ConfigSnapshot:
    """Build a snapshot from INDUS_* environment variables."""
    env_values = _from_env(ConfigSnapshot, prefix=prefix)
    snap = ConfigSnapshot(**env_values)
    snap.validate()
    return snap


def from_yaml(path: str | Path) -> ConfigSnapshot:
    """Load a snapshot from a YAML file. Optional dependency (pyyaml).

    Falls back to a from_env() snapshot if pyyaml is unavailable.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pyyaml required to load YAML config") from exc
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping")
    # Coerce tuples
    if "llm_fallback_providers" in data and isinstance(data["llm_fallback_providers"], list):
        data["llm_fallback_providers"] = tuple(data["llm_fallback_providers"])
    snap = ConfigSnapshot(**{k: v for k, v in data.items() if k in {f.name for f in fields(ConfigSnapshot)}})
    snap.validate()
    return snap


def load(
    env: str | None = None,
    config_file: str | None = None,
) -> ConfigSnapshot:
    """Load a config snapshot with the standard precedence.

    Order: defaults → YAML file → env vars → explicit `env` override.
    """
    snap = from_yaml(config_file) if config_file else ConfigSnapshot()
    env_snap = from_env()
    # Merge: YAML values fill in where env is at default
    merged: dict[str, Any] = {}
    for f in fields(ConfigSnapshot):
        env_val = getattr(env_snap, f.name)
        yaml_val = getattr(snap, f.name)
        # If env_val differs from its env-default (i.e. was actually set), use it
        if env_val != _env_default(f):
            merged[f.name] = env_val
        else:
            merged[f.name] = yaml_val
    if env is not None:
        merged["environment"] = env
    final = ConfigSnapshot(**merged)
    final.validate()
    return final


def _env_default(f: Any) -> Any:
    """Get the default value for a field as it would be read from env."""
    if f.type is bool or f.type == "bool":
        return bool(f.default) if f.default is not None else False
    if f.type is int or f.type == "int":
        return int(f.default) if f.default is not None else 0
    return str(f.default) if f.default is not None else ""


__all__ = [
    "ConfigSnapshot",
    "from_env",
    "from_yaml",
    "load",
]
