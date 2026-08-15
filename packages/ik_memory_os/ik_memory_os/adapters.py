"""Optional production adapters for Qdrant, Neo4j and Redis."""
class AdapterUnavailable(RuntimeError): pass
def qdrant_client(url,api_key=None):
 try: from qdrant_client import QdrantClient
 except ImportError as exc: raise AdapterUnavailable("install qdrant-client") from exc
 return QdrantClient(url=url,api_key=api_key)
def neo4j_driver(uri,user,password):
 try: from neo4j import AsyncGraphDatabase
 except ImportError as exc: raise AdapterUnavailable("install neo4j") from exc
 return AsyncGraphDatabase.driver(uri,auth=(user,password))
def redis_client(url):
 try: from redis.asyncio import Redis
 except ImportError as exc: raise AdapterUnavailable("install redis") from exc
 return Redis.from_url(url,decode_responses=True)
