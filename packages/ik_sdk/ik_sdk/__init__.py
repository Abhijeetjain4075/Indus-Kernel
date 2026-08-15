"""Typed Python SDK primitives for the Indus Kernel API."""
from dataclasses import dataclass
import httpx
@dataclass
class IndusClient:
    base_url:str
    api_key:str
    timeout:float=30.0
    def _client(self):
        return httpx.Client(base_url=self.base_url.rstrip("/"),headers={"Authorization":f"Bearer {self.api_key}"},timeout=self.timeout)
    def health(self):
        with self._client() as c: r=c.get("/healthz"); r.raise_for_status(); return r.json()
    def request(self,method:str,path:str,**kwargs):
        with self._client() as c: r=c.request(method,path,**kwargs); r.raise_for_status(); return r.json()
__all__=["IndusClient"]
