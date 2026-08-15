"""API package: contracts and gateway helpers."""
from dataclasses import dataclass
@dataclass(frozen=True)
class APIInfo:
    name:str="indus-kernel"; version:str="0.11.0"; api_prefix:str="/api/v1"
__version__="1.0.0"
