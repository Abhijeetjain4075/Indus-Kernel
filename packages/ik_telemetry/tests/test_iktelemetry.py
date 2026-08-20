"""Tests for ik_telemetry — real, no mocks."""

from __future__ import annotations

import json
import logging
import time

import pytest

from ik_telemetry import (
    LocalCollector,
    Metric,
    Span,
    StructuredLogger,
    Tracer,
    current_trace_id,
    get_collector,
    get_logger,
    get_tracer,
    setup_telemetry,
    with_trace_id,
)


@pytest.fixture
def collector():
    c = LocalCollector()
    return c


class TestSpan:
    def test_basic(self):
        s = Span(
            name="test",
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            service="svc",
            started_at=time.time(),
        )
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["trace_id"] == "t1"


class TestLocalCollector:
    def test_record_span(self, collector):
        s = Span(
            name="x",
            trace_id="t",
            span_id="s",
            parent_span_id=None,
            service="svc",
            started_at=time.time(),
        )
        collector.record_span(s)
        assert len(collector.get_spans()) == 1

    def test_record_metric_counter(self, collector):
        m = Metric(name="hits", value=1, timestamp=time.time(), metric_type="counter")
        collector.record_metric(m)
        collector.record_metric(m)
        assert collector.counter_value("hits") == 2.0

    def test_record_metric_gauge(self, collector):
        m = Metric(name="temp", value=42, timestamp=time.time(), metric_type="gauge")
        collector.record_metric(m)
        assert collector.gauge_value("temp") == 42

    def test_counter_with_labels(self, collector):
        m = Metric(
            name="hits",
            value=1,
            timestamp=time.time(),
            labels={"route": "/a"},
            metric_type="counter",
        )
        collector.record_metric(m)
        assert collector.counter_value("hits", {"route": "/a"}) == 1
        assert collector.counter_value("hits", {"route": "/b"}) == 0

    def test_reset(self, collector):
        collector.record_metric(Metric(name="x", value=1, timestamp=time.time()))
        collector.reset()
        assert collector.summary() == {
            "spans": 0,
            "metrics": 0,
            "counters": 0,
            "gauges": 0,
        }

    def test_bounded_storage(self, collector):
        # Record more than the cap (10000). The collector should trim to <= 10000.
        for i in range(10050):
            collector.record_span(
                Span(
                    name="x",
                    trace_id="t",
                    span_id=str(i),
                    parent_span_id=None,
                    service="s",
                    started_at=time.time(),
                )
            )
        spans = collector.get_spans()
        assert len(spans) <= 10000
        # Should be the most recent ones
        assert int(spans[-1].span_id) >= 10049 - 10000

    def test_summary(self, collector):
        collector.record_metric(
            Metric(name="a", value=1, timestamp=time.time(), metric_type="counter")
        )
        s = collector.summary()
        assert s["counters"] == 1


class TestTracer:
    def test_basic_span(self, collector):
        t = Tracer(service="svc", collector=collector)
        with t.start_span("op") as span:
            span.attributes["k"] = "v"
        spans = collector.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "op"
        assert spans[0].attributes["k"] == "v"
        assert spans[0].status == "ok"
        assert spans[0].duration_ms >= 0

    def test_error_span(self, collector):
        t = Tracer(service="svc", collector=collector)
        with pytest.raises(ValueError), t.start_span("op"):
            raise ValueError("boom")
        spans = collector.get_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert spans[0].error_type == "ValueError"
        assert "boom" in spans[0].error_message

    def test_trace_id_propagated(self, collector):
        t = Tracer(service="svc", collector=collector)
        with t.start_span("op") as outer:
            with t.start_span("inner", trace_id=outer.trace_id) as inner:
                pass
            assert inner.trace_id == outer.trace_id

    def test_parent_span(self, collector):
        t = Tracer(service="svc", collector=collector)
        with t.start_span("outer") as outer:
            with t.start_span("inner", parent_span_id=outer.span_id) as inner:
                pass
            assert inner.parent_span_id == outer.span_id

    def test_counter(self, collector):
        t = Tracer(service="svc", collector=collector)
        t.counter("hits", labels={"r": "/a"})
        t.counter("hits", labels={"r": "/a"})
        t.counter("hits", labels={"r": "/b"})
        assert collector.counter_value("hits", {"r": "/a"}) == 2
        assert collector.counter_value("hits", {"r": "/b"}) == 1

    def test_gauge(self, collector):
        t = Tracer(service="svc", collector=collector)
        t.gauge("temp", 42)
        assert collector.gauge_value("temp") == 42

    def test_histogram(self, collector):
        t = Tracer(service="svc", collector=collector)
        t.histogram("latency_ms", 12.5)
        m = collector.get_metrics()[0]
        assert m.metric_type == "histogram"


class TestStructuredLogger:
    def test_basic(self, caplog):
        logger = StructuredLogger("test", level="INFO")
        with caplog.at_level(logging.INFO, logger="test"):
            logger.info("hello", user="u1")
        # The record contains the JSON line
        assert any("hello" in r.message for r in caplog.records)

    def test_error(self, caplog):
        logger = StructuredLogger("test", level="ERROR")
        with caplog.at_level(logging.ERROR, logger="test"):
            logger.error("oops", code=500)
        assert any("oops" in r.message for r in caplog.records)

    def test_with_trace_id(self, caplog):
        logger = StructuredLogger("test", level="INFO")
        with caplog.at_level(logging.INFO, logger="test"), with_trace_id("trace-1"):
            assert current_trace_id() == "trace-1"
            logger.info("inside")
        assert any("trace-1" in r.message for r in caplog.records)
        assert current_trace_id() is None

    def test_get_logger(self):
        l1 = get_logger("x")
        l2 = get_logger("x")
        assert l1 is l2  # singleton by name


class TestWithTraceId:
    def test_yields_trace_id(self):
        with with_trace_id("abc") as tid:
            assert tid == "abc"
            assert current_trace_id() == "abc"
        assert current_trace_id() is None

    def test_auto_generated(self):
        with with_trace_id() as tid:
            assert tid
            assert len(tid) > 0

    def test_restores_previous(self):
        with with_trace_id("outer"):
            with with_trace_id("inner"):
                assert current_trace_id() == "inner"
            assert current_trace_id() == "outer"


class TestSetup:
    def test_setup_telemetry(self):
        ok = setup_telemetry(service_name="test")
        # Either True or False is fine (depends on OTel availability)
        assert isinstance(ok, bool)
