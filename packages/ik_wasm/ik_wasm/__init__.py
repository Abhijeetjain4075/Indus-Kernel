"""WASM execution boundary. Uses wasmtime when explicitly configured; never executes host binaries."""


class WasmExecutionUnavailable(RuntimeError):
    pass


def execute_module(
    module_bytes: bytes, entrypoint: str = "_start", stdin: bytes = b"", fuel: int = 1_000_000
):
    if not module_bytes:
        raise ValueError("module_bytes required")
    try:
        from wasmtime import Engine, Store, Module, Linker
    except ImportError as exc:
        raise WasmExecutionUnavailable("install wasmtime") from exc
    engine = Engine()
    store = Store(engine)
    store.set_fuel(fuel)
    module = Module(engine, module_bytes)
    linker = Linker(engine)
    instance = linker.instantiate(store, module)
    fn = instance.exports(store).get(entrypoint)
    if fn is None:
        raise WasmExecutionUnavailable(f"entrypoint not found: {entrypoint}")
    return fn(store)
