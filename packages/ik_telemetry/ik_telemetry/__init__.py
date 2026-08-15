"""Production telemetry facade with OpenTelemetry when installed and a safe local fallback."""
from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Iterator

__version__="1.0.0"
_configured=False

def setup_telemetry(settings_or_service_name="indus-kernel", endpoint:str|None=None)->bool:
    global _configured
    if _configured: return True
    service_name = getattr(settings_or_service_name, "otel_service_name", settings_or_service_name)
    if endpoint is None: endpoint = getattr(settings_or_service_name, "otel_exporter_otlp_endpoint", None)
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        provider=TracerProvider(resource=Resource.create({"service.name":service_name}))
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            except Exception:
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider); _configured=True; return True
    except Exception as exc:
        logging.getLogger(__name__).warning("OpenTelemetry unavailable; using stdlib telemetry: %s",exc)
        _configured=True; return False

def get_tracer(name:str="indus"):
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        class _Noop:
            @contextmanager
            def start_as_current_span(self,*args,**kwargs)->Iterator[None]: yield None
        return _Noop()
