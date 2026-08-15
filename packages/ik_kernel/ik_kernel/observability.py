"""Small tracing abstraction that never silently loses correlation IDs."""
from __future__ import annotations
from contextlib import contextmanager
from opentelemetry import trace
_tracer = trace.get_tracer("indus-kernel")
@contextmanager
def span(name: str, **attributes):
    with _tracer.start_as_current_span(name) as s:
        for k,v in attributes.items():
            if v is not None: s.set_attribute(k,str(v))
        yield s
def get_tracer(): return _tracer
