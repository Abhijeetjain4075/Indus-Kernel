from ik_kernel.config import Settings
from ik_kernel.security import authenticate_api_key, parse_api_keys


def test_api_key_authentication() -> None:
    secret = "a" * 48
    s = Settings(environment="test", api_keys=f"k1:{secret}:tenant-a:admin:*")
    p = authenticate_api_key(f"k1.{secret}", s)
    assert p is not None
    assert p.tenant_id == "tenant-a"
    assert "admin" in p.roles


def test_api_key_invalid_secret() -> None:
    secret = "a" * 48
    s = Settings(environment="test", api_keys=f"k1:{secret}:tenant-a:admin:*")
    assert authenticate_api_key("k1.invalid-secret-that-is-long-enough", s) is None


def test_parser_does_not_store_plain_secret() -> None:
    secret = "b" * 48
    values = parse_api_keys(Settings(environment="test", api_keys=f"k1:{secret}:tenant-a:user:read"))
    assert secret not in repr(values)
