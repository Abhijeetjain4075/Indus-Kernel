"""ik_telemetry — Telemetry facade (M7, M11).

Production-grade telemetry: traces, metrics, structured logs. The
facade prefers OpenTelemetry when available, falls back to a
local in-process collector that stores spans for inspection.

The M11 hardening requires:
- Trace IDs and span IDs are stable across the kernel
- Metrics are emitted on every operation
- Logs are structured (JSON) and include trace correlation
- SLOs are defined and tracked
- Cardinality is bounded (no high-cardinality labels)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Span & Metric primitives
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single trace span."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    started_at: float
    ended_at: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Metric:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge | counter | histogram

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalCollector:
    """An in-process collector that records spans + metrics.

    Real, not a mock. Thread-safe. Used as the fallback when
    OpenTelemetry is not available, and as the sink for in-process
    tests.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._spans: list[Span] = []
        self._metrics: list[Metric] = []
        self._counters: dict[tuple[str, frozenset], float] = defaultdict(float)
        self._gauges: dict[tuple[str, frozenset], float] = {}

    def record_span(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)
            # Cap stored spans to prevent memory blow-up
            if len(self._spans) > 10000:
                self._spans = self._spans[-5000:]

    def record_metric(self, metric: Metric) -> None:
        with self._lock:
            self._metrics.append(metric)
            key = (metric.name, frozenset(metric.labels.items()))
            if metric.metric_type == "counter":
                self._counters[key] += metric.value
            else:
                self._gauges[key] = metric.value
            if len(self._metrics) > 50000:
                self._metrics = self._metrics[-25000:]

    def get_spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def get_metrics(self) -> list[Metric]:
        with self._lock:
            return list(self._metrics)

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = (name, frozenset((labels or {}).items()))
        with self._lock:
            return self._counters.get(key, 0.0)

    def gauge_value(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        key = (name, frozenset((labels or {}).items()))
        with self._lock:
            return self._gauges.get(key)

    def reset(self) -> None:
        with self._lock:
            self._spans.clear()
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "spans": len(self._spans),
                "metrics": len(self._metrics),
                "counters": len(self._counters),
                "gauges": len(self._gauges),
            }


# ---------------------------------------------------------------------------
# Tracer (with OpenTelemetry fallback)
# ---------------------------------------------------------------------------


def _new_ids() -> tuple[str, str]:
    return uuid.uuid4().hex, uuid.uuid4().hex[:16]


class Tracer:
    """A tracer. Records spans either via OpenTelemetry or the local collector."""

    def __init__(
        self,
        service: str = "indus-kernel",
        collector: LocalCollector | None = None,
    ) -> None:
        self.service = service
        self.collector = collector or _default_collector
        self._otel_tracer: Any = None
        try:
            from opentelemetry import trace

            self._otel_tracer = trace.get_tracer(service)
            self._has_otel = True
        except Exception:
            self._has_otel = False
        self._active: dict[str, str] = {}  # context-var-like (thread-local)

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
        trace_id: str | None = None,
    ) -> Iterator[Span]:
        """Start a span. Yields the Span so the caller can update attributes."""
        t_id, s_id = _new_ids()
        if trace_id is None:
            trace_id = t_id
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=s_id,
            parent_span_id=parent_span_id,
            service=self.service,
            started_at=time.time(),
            attributes=dict(attributes or {}),
        )
        token = None
        try:
            if self._has_otel:
                from opentelemetry import context, trace

                otel_span = self._otel_tracer.start_span(name, attributes=span.attributes)
                token = context.attach(
                    trace.set_span_in_context(otel_span)  # type: ignore
                )
        except Exception:
            token = None
        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error_type = type(e).__name__
            span.error_message = str(e)[:500]
            raise
        finally:
            span.ended_at = time.time()
            span.duration_ms = (span.ended_at - span.started_at) * 1000.0
            self.collector.record_span(span)
            if token is not None:
                try:
                    from opentelemetry import context

                    context.detach(token)
                except Exception:
                    pass

    def counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        self.collector.record_metric(
            Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type="counter",
            )
        )

    def gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        self.collector.record_metric(
            Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type="gauge",
            )
        )

    def histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        self.collector.record_metric(
            Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type="histogram",
            )
        )


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


class StructuredLogger:
    """A JSON-formatted logger. Includes trace correlation if available."""

    def __init__(self, name: str = "indus", level: str = "INFO") -> None:
        self.name = name
        self.level = level
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _emit(
        self,
        level: str,
        message: str,
        **fields: Any,
    ) -> None:
        record = {
            "ts": time.time(),
            "level": level,
            "logger": self.name,
            "message": message,
            **fields,
        }
        # Try to attach the current trace id if any
        trace_id = _context.value
        if trace_id:
            record["trace_id"] = trace_id
        line = json.dumps(record, default=str)
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(line)

    def info(self, message: str, **fields: Any) -> None:
        self._emit("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit("error", message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self._emit("debug", message, **fields)


# ---------------------------------------------------------------------------
# Context (trace_id propagation)
# ---------------------------------------------------------------------------


class _ContextVar:
    def __init__(self) -> None:
        self.value: str | None = None


_context = _ContextVar()


@contextmanager
def with_trace_id(trace_id: str | None = None) -> Iterator[str]:
    """Set the active trace_id for the duration of the context."""
    if trace_id is None:
        trace_id = uuid.uuid4().hex
    prev = _context.value
    _context.value = trace_id
    try:
        yield trace_id
    finally:
        _context.value = prev


def current_trace_id() -> str | None:
    return _context.value


# ---------------------------------------------------------------------------
# Setup + accessors
# ---------------------------------------------------------------------------


_default_collector = LocalCollector()
_tracer: Tracer | None = None
_configured = False


def setup_telemetry(
    service_name: str = "indus-kernel",
    endpoint: str | None = None,
) -> bool:
    """Set up telemetry. Idempotent. Returns True if OpenTelemetry is available."""
    global _tracer, _configured
    if _configured:
        return True
    _tracer = Tracer(service=service_name, collector=_default_collector)
    if endpoint is not None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            trace.set_tracer_provider(provider)
        except Exception:
            pass
    _configured = True
    return _tracer._has_otel


def get_tracer(service_name: str = "indus") -> Tracer:
    global _tracer
    if _tracer is None or _tracer.service != service_name:
        _tracer = Tracer(service=service_name, collector=_default_collector)
    return _tracer


_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str = "indus", level: str | None = None) -> StructuredLogger:
    if name in _loggers:
        return _loggers[name]
    logger = StructuredLogger(name=name, level=level or os.getenv("INDUS_LOG_LEVEL", "INFO"))
    _loggers[name] = logger
    return logger


def get_collector() -> LocalCollector:
    return _default_collector


__all__ = [
    "Span",
    "Metric",
    "LocalCollector",
    "Tracer",
    "StructuredLogger",
    "setup_telemetry",
    "get_tracer",
    "get_logger",
    "get_collector",
    "with_trace_id",
    "current_trace_id",
]
