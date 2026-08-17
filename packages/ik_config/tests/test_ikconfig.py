"""Tests for ik_config — real, no mocks."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import pytest

from ik_config import ConfigSnapshot, from_env, from_yaml, load


class TestConfigSnapshot:
    def test_defaults(self):
        s = ConfigSnapshot()
        assert s.environment == "dev"
        assert s.api_port == 8000
        assert s.telemetry_enabled is True

    def test_immutable(self):
        s = ConfigSnapshot()
        with pytest.raises(FrozenInstanceError):
            s.environment = "production"  # type: ignore

    def test_overlay_creates_new(self):
        s = ConfigSnapshot(environment="dev")
        s2 = s.overlay(environment="production")
        assert s.environment == "dev"  # original unchanged
        assert s2.environment == "production"

    def test_overlay_unknown_field(self):
        s = ConfigSnapshot()
        with pytest.raises(ValueError, match="unknown"):
            s.overlay(not_a_field=1)  # type: ignore

    def test_with_extras(self):
        s = ConfigSnapshot()
        s2 = s.with_extras(custom_key="value")
        assert s2.extras["custom_key"] == "value"
        assert "custom_key" not in s.extras

    def test_to_dict(self):
        s = ConfigSnapshot()
        d = s.to_dict()
        assert "environment" in d
        assert "api_port" in d

    def test_require(self):
        s = ConfigSnapshot()
        with pytest.raises(ValueError):
            s.require("database_url")
        assert s.require("environment") == "dev"

    def test_is_production(self):
        assert ConfigSnapshot(environment="production").is_production()
        assert ConfigSnapshot(environment="staging").is_production()
        assert not ConfigSnapshot(environment="dev").is_production()

    def test_validate_invalid_env(self):
        s = ConfigSnapshot(environment="weird")
        with pytest.raises(ValueError, match="environment"):
            s.validate()

    def test_validate_invalid_port(self):
        s = ConfigSnapshot(api_port=0)
        with pytest.raises(ValueError, match="api_port"):
            s.validate()
        s2 = ConfigSnapshot(api_port=99999)
        with pytest.raises(ValueError, match="api_port"):
            s2.validate()

    def test_validate_production_debug(self):
        s = ConfigSnapshot(environment="production", debug=True)
        with pytest.raises(ValueError, match="debug"):
            s.validate()

    def test_validate_negative_budget(self):
        s = ConfigSnapshot(budget_max_cents_per_request=-1)
        with pytest.raises(ValueError, match="budget"):
            s.validate()

    def test_validate_low_memory(self):
        s = ConfigSnapshot(sandbox_max_memory_mb=10)
        with pytest.raises(ValueError, match="memory"):
            s.validate()


class TestFromEnv:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("INDUS_ENVIRONMENT", "test")
        monkeypatch.setenv("INDUS_API_PORT", "9999")
        s = from_env()
        assert s.environment == "test"
        assert s.api_port == 9999

    def test_env_bool(self, monkeypatch):
        monkeypatch.setenv("INDUS_DEBUG", "true")
        s = from_env()
        assert s.debug is True

    def test_env_invalid_int(self, monkeypatch):
        monkeypatch.setenv("INDUS_API_PORT", "not_a_number")
        s = from_env()
        assert s.api_port == 8000  # falls back to default


class TestFromYaml:
    def test_load_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "environment: staging\n"
            "api_port: 9000\n"
            "llm_fallback_providers:\n"
            "  - indus\n"
            "  - openai\n"
        )
        s = from_yaml(str(yaml_path))
        assert s.environment == "staging"
        assert s.api_port == 9000
        assert "openai" in s.llm_fallback_providers

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            from_yaml("/nonexistent/config.yaml")

    def test_invalid_yaml_root(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("- one\n- two\n")
        with pytest.raises(ValueError):
            from_yaml(str(yaml_path))


class TestLoad:
    def test_load_with_env_override(self, monkeypatch, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("api_port: 7000\nenvironment: dev\n")
        monkeypatch.setenv("INDUS_API_PORT", "8888")
        s = load(config_file=str(yaml_path))
        # env wins
        assert s.api_port == 8888

    def test_load_explicit_env(self, tmp_path, monkeypatch):
        # Clear any leaked INDUS_* env vars from other tests
        for k in list(os.environ.keys()):
            if k.startswith("INDUS_"):
                monkeypatch.delenv(k, raising=False)
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("environment: dev\n")
        s = load(env="staging", config_file=str(yaml_path))
        assert s.environment == "staging"

    def test_load_no_yaml(self, monkeypatch):
        monkeypatch.setenv("INDUS_API_PORT", "1234")
        s = load()
        assert s.api_port == 1234
