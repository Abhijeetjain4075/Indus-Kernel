"""Deterministic context assembly with explicit truncation semantics."""


def truncate_context(text: str, max_chars: int) -> str:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    return text if len(text) <= max_chars else text[-max_chars:]


def build_context(system: str, history: list[str], user: str, max_chars: int = 32000) -> str:
    return truncate_context("\n".join(p for p in [system, *history, user] if p), max_chars)
