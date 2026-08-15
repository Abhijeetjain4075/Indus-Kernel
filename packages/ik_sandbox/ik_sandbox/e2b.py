"""Optional E2B sandbox adapter. Never falls back to host execution."""
from . import SandboxUnavailable
class E2BSandbox:
 def __init__(self,api_key=None): self.api_key=api_key
 async def execute(self,command,policy):
  if not self.api_key: raise SandboxUnavailable("E2B API key is required")
  try: from e2b_code_interpreter import AsyncSandbox
  except ImportError as exc: raise SandboxUnavailable("install e2b-code-interpreter") from exc
  raise SandboxUnavailable("E2B command execution adapter requires an explicit image/runtime policy; host execution is forbidden")
