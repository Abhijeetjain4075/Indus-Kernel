"""ik_wasm — WASM Plugin Runtime (Subsystem 40, new in v1.1.0).

Replaces the original Plugin Manager. Wasmtime + WASI 0.2 + Component Model
+ Extism (multi-language SDK) + Wassette (Microsoft, Wasmtime + OCI for MCP).

Capability-based security. About 1-3ms cold start. 15MB memory per instance.

Fully wired in M7.5.
"""

__version__ = "0.1.0"
