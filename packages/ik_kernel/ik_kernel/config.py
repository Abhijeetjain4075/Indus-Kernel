"""Indus Kernel configuration via Pydantic Settings.

Configuration is layered (in order of precedence, highest wins):
1. Per-tenant overrides (Postgres `config_overrides` table)
2. Per-instance overrides (YAML file at `$INDUS_CONFIG_PATH`)
3. Environment variables (prefix `INDUS_`)
4. Defaults (defined here)

In dev mode, the kernel reads from `.env` automatically via pydantic-settings.
In production, secrets come from HashiCorp Vault via the `ik_security` package.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level Indus Kernel settings."""

    model_config = SettingsConfigDict(
        env_prefix="INDUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Meta ----
    app_name: str = "indus-kernel"
    app_version: str = "0.11.0"
    environment: Literal["dev", "test", "staging", "production"] = "dev"
    debug: bool = False
    log_level: str = "INFO"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_rate_limit_per_minute: int = 120
    api_allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])
    api_keys: str = ""
    api_require_auth: bool = False
    api_max_body_bytes: int = 16 * 1024 * 1024
    jwt_secret: str | None = None
    jwt_expiration_minutes: int = 15
    webhook_secrets: dict[str, str] = Field(default_factory=dict)
    webhook_tolerance_s: int = 300

    # ---- Database ----
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://indus:indus@localhost:5432/indus"
    )
    database_pool_size: int = 20
    database_pool_max_overflow: int = 10
    database_echo: bool = False

    # ---- Redis ----
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_pool_size: int = 50

    # ---- Qdrant ----
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # ---- Neo4j ----
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "indus"

    # ---- NATS JetStream ----
    nats_url: str = "nats://localhost:4222"
    nats_stream_name: str = "indus-events"
    nats_replicas: int = 1

    # ---- Temporal ----
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "indus"
    temporal_task_queue: str = "indus-tasks"

    # ---- LLM Router ----
    litellm_proxy_url: str = "http://localhost:4000"
    litellm_master_key: str | None = None
    default_model: str = "indus/tiny-v0.3.0"
    indus_llm_checkpoint: str | None = None

    # ---- Vault (Secrets) ----
    vault_addr: str | None = None
    vault_token: str | None = None
    vault_path_prefix: str = "secret/indus"

    # ---- Telemetry ----
    otel_service_name: str = "indus-kernel"
    otel_exporter_otlp_endpoint: str | None = None
    otel_sample_ratio: float = 0.1
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # ---- Sandbox ----
    e2b_api_key: str | None = None
    wasm_cache_dir: str = "/tmp/indus-wasm-cache"

    # ---- LLM Defaults ----
    llm_default_max_tokens: int = 4096
    llm_default_temperature: float = 0.7
    llm_default_timeout_s: int = 60
    llm_cache_ttl_s: int = 86400
    llm_semantic_cache_threshold: float = 0.92

    # ---- Memory Defaults ----
    memory_consolidation_interval_s: int = 300
    memory_short_term_ttl_s: int = 604800  # 7 days
    memory_importance_threshold_promote: float = 0.4
    memory_importance_threshold_long: float = 0.7

    # ---- Reasoning Defaults ----
    reasoning_default_strategy: str = "auto"
    reasoning_max_concurrent: int = 100
    reasoning_self_consistency_n: int = 10
    reasoning_tot_branching: int = 4
    reasoning_tot_depth: int = 5

    # ---- Agent Defaults ----
    agent_default_max_cost_cents: int = 100
    agent_default_max_latency_s: int = 120
    agent_goa_top_k: int = 3

    # ---- Self-Improvement ----
    improvement_gepa_budget: Literal["light", "medium", "heavy"] = "medium"
    improvement_finetune_backend: str = "llama_factory+unsloth"

    # ---- Telemetry / Monitoring ----
    metrics_port: int = 9090

    # ---- Multi-tenancy ----
    multi_tenant: bool = True
    default_tenant_id: str = "t-default"
    production_require_dependencies: bool = False
    required_services: list[str] = Field(default_factory=list)
    strict_startup: bool = False

    # ---- Paths ----
    config_path: str | None = None
    plugins_dir: str = "./plugins"
    schemas_dir: str = "./schemas"

    @model_validator(mode="after")
    def validate_runtime_security(self) -> Settings:
        if self.environment in {"staging", "production"}:
            if self.debug:
                raise ValueError("debug must be false in staging/production")
            if not self.api_keys:
                raise ValueError("INDUS_API_KEYS is required in staging/production")
            if not self.jwt_secret or len(self.jwt_secret) < 32:
                raise ValueError("INDUS_JWT_SECRET (32+ chars) is required in staging/production")
            if "*" in self.api_cors_origins:
                raise ValueError("wildcard CORS is forbidden in staging/production")
            object.__setattr__(self, "api_require_auth", True)
            object.__setattr__(self, "production_require_dependencies", True)
            object.__setattr__(self, "strict_startup", True)
            if "*" in self.api_allowed_hosts:
                raise ValueError("wildcard allowed hosts are forbidden in staging/production")
            if any(
                v in {"indus", "indus-secret", "indus-dev-secret-change-in-prod"}
                for v in [self.neo4j_password, self.langfuse_secret_key or ""]
            ):
                raise ValueError("default development secrets are forbidden in staging/production")
        if self.indus_llm_checkpoint is not None and not Path(self.indus_llm_checkpoint).is_file():
            raise ValueError("INDUS_LLM_CHECKPOINT must point to an existing file")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
