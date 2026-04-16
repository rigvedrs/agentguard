# Telemetry

agentguard supports two complementary observability models:

- **Built-in trace platform** for local durability and operator workflows
- **OpenTelemetry** for teams already using an external tracing backend

## Recommended split

- Use **SQLite tracing** when you want first-party persistence, CLI inspection, import/export, and the built-in dashboard.
- Use **JSONL tracing** when you need portable files for replay, generated tests, or legacy compatibility.
- Use **OpenTelemetry** when agentguard spans should appear inside Jaeger, Tempo, Honeycomb, Datadog, or another existing trace store.

These options are not mutually exclusive. A common production setup is:

1. persist traces to SQLite for agentguard-specific workflows
2. emit OpenTelemetry spans for central platform observability

## OpenTelemetry setup

```bash
pip install awesome-agentguard opentelemetry-sdk opentelemetry-exporter-otlp
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from agentguard import instrument_agentguard

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)
trace.set_tracer_provider(provider)

instrument_agentguard()
```

## What OTel gives you

agentguard emits:

- a parent `agentguard.tool_call` span
- guard check child spans
- events for retries, anomalous responses, budget exceedance, and circuit openings

## Built-in dashboard vs OTel

Use the built-in dashboard when you want fast local answers to:

- which sessions failed?
- which tools are slow?
- where did retries spike?
- which calls incurred cost?

Use OTel when you need:

- service-to-service distributed tracing
- central storage across many app instances
- existing alerting and SLO tooling
- cross-service correlation with your wider system
