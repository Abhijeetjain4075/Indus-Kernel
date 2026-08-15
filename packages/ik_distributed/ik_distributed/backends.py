"""Optional distributed backends. Local runtime remains the deterministic test backend."""
class BackendUnavailable(RuntimeError): pass
async def nats_connection(url):
 try: import nats
 except ImportError as exc: raise BackendUnavailable("install nats-py") from exc
 return await nats.connect(url)
async def temporal_client(target,namespace="default"):
 try: from temporalio.client import Client
 except ImportError as exc: raise BackendUnavailable("install temporalio") from exc
 return await Client.connect(target,namespace=namespace)
