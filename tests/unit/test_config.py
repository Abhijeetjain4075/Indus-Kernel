"""Unit tests for the config module."""
from __future__ import annotations

import pytest

from ik_kernel.config import Settings, get_settings


def test_default_settings():
    s = Settings()
    assert s.app_name == "indus-kernel"
    # environment is set by conftest.py to "test"
    assert s.environment in ("dev", "test")
    assert s.api_port == 8000
    assert s.api_prefix == "/api/v1"
    assert s.multi_tenant is True


def test_settings_caching():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_env_prefix():
    import os
    os.environ["INDUS_API_PORT"] = "9999"
    os.environ["INDUS_LOG_LEVEL"] = "DEBUG"
    # Cached settings from before won't see this
    # Force a re-read by clearing the cache
    get_settings.cache_clear()
    s = get_settings()
    # Note: pydantic-settings v2 reads env at instantiation, not from cache
    # The cache returns the previous instance. Use Settings() directly to test env.
    direct = Settings()
    assert direct.api_port == 9999
    assert direct.log_level == "DEBUG"
    # Clean up
    del os.environ["INDUS_API_PORT"]
    del os.environ["INDUS_LOG_LEVEL"]
    get_settings.cache_clear()
