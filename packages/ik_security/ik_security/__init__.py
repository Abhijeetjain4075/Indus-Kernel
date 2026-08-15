"""Security primitives: fingerprints, constant-time comparison, scopes and capability policy."""
import hashlib,hmac,secrets
from dataclasses import dataclass
def fingerprint(value:str)->str: return hashlib.sha256(value.encode()).hexdigest()
def constant_time_equal(a:str,b:str)->bool: return hmac.compare_digest(a,b)
def generate_token(nbytes:int=32)->str: return secrets.token_urlsafe(nbytes)
@dataclass(frozen=True)
class Capability:
 name:str; scopes:frozenset[str]
def authorize(required:str,granted:set[str]|frozenset[str])->bool: return required in granted or "*" in granted
