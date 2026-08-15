"""Sandbox policy boundary. Untrusted execution is fail-closed unless a verified backend is configured."""
from dataclasses import dataclass
@dataclass(frozen=True)
class SandboxPolicy:
 timeout_s:float=10.0; memory_mb:int=512; network:bool=False; max_output_bytes:int=1_000_000
 def validate(self):
  if self.timeout_s<=0 or self.timeout_s>300: raise ValueError("timeout_s out of range")
  if self.memory_mb<64: raise ValueError("memory_mb too low")
  return self
class SandboxUnavailable(RuntimeError): pass
class SandboxExecutor:
 def __init__(self,backend=None): self.backend=backend
 async def execute(self,command:list[str],policy:SandboxPolicy|None=None):
  (policy or SandboxPolicy()).validate()
  if not self.backend: raise SandboxUnavailable("No isolated executor configured; production must provide Firecracker/gVisor/E2B backend")
  return await self.backend.execute(command,policy or SandboxPolicy())
def execute(command:list[str],policy:SandboxPolicy|None=None):
 raise SandboxUnavailable("Direct local execution of untrusted commands is prohibited; configure an isolated backend")
