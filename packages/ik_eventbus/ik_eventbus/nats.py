"""Optional NATS JetStream adapter. Dependency is lazy and failures are explicit."""
class NATSUnavailable(RuntimeError): pass
class NATSJetStreamBus:
 def __init__(self,url="nats://localhost:4222",subject_prefix="indus"):
  self.url=url; self.subject_prefix=subject_prefix; self.nc=None; self.js=None
 async def connect(self):
  try:
   import nats
  except ImportError as exc: raise NATSUnavailable("install nats-py") from exc
  self.nc=await nats.connect(self.url); self.js=self.nc.jetstream(); return self
 async def publish(self,event_type,payload):
  if self.js is None: raise NATSUnavailable("connect() required")
  import json
  return await self.js.publish(f"{self.subject_prefix}.{event_type}",json.dumps(payload).encode())
 async def close(self):
  if self.nc: await self.nc.drain()
