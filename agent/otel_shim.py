"""
OpenTelemetry Shim for Hermes AIAgent.

Wraps an AIAgent instance and instruments all callbacks to produce:
  - Traces : one trace per run_conversation(), one span per tool call
  - Metrics: counters + latency histograms exported via OTLP

Usage::

    from run_agent import AIAgent
    from agent.otel_shim import OtelShim, wrapped_chat

    agent = AIAgent(model="anthropic/claude-sonnet-4")
    shim  = OtelShim(agent)

    response = wrapped_chat(agent, "your query here", shim)

Or step-by-step for gateway/cron integration::

    shim  = OtelShim(agent)
    agent = AIAgent(model="...", **shim.callbacks)

    shim.start_trace(user_message)
    try:
        result = agent.run_conversation(user_message=user_message)
        final  = result.get("final_response", "")
        shim.end_trace(final, success=True)
    except Exception as exc:
        shim.record_error(str(exc), type(exc).__name__)
        shim.end_trace(str(exc), success=False)
        raise

Enable via env vars::

    export OTEL_ENABLED=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export OTEL_SERVICE_NAME=hermes-agent
"""

from __future__ import annotations

import os
import time
import logging
import re
import json
from typing import Optional, Any
from datetime import datetime, timezone
#import fs
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# Ensure INFO logs are visible from the very first import — before setup_logging() runs
# in gateway. Only configure if no handlers are already present (let setup_logging own stdout).
if not logger.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(name)s: %(levelname)s %(message)s"))
    _h.setLevel(logging.INFO)
    logging.getLogger().addHandler(_h)
    logging.getLogger().setLevel(logging.INFO)

# -------------------------------------------------------------------------
# Config — all via environment variables
# -------------------------------------------------------------------------

OTEL_ENABLED                 = os.getenv("OTEL_ENABLED", "false").lower() in ("1", "true", "yes")
OTEL_EXPORTER_OTLP_ENDPOINT  = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTEL_EXPORTER_OTLP_HEADERS   = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
OTEL_SERVICE_NAME            = os.getenv("OTEL_SERVICE_NAME", "hermes-agent")
OTEL_SERVICE_VERSION         = os.getenv("OTEL_SERVICE_VERSION", "0.8.0")
OTEL_ENVIRONMENT             = os.getenv("OTEL_ENVIRONMENT", "development")
OTEL_PROFILE                 = os.getenv("OTEL_PROFILE", "")
OTEL_METRICS_INTERVAL_SECS   = int(os.getenv("OTEL_METRICS_INTERVAL_SECS", "10"))

logger.info(
    "OTEL_CONFIG: enabled=%s endpoint=%s service=%s",
    OTEL_ENABLED, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME,
)

# ---------------------------------------------------------------------------
# Module-level singleton — initialised once per process
# ---------------------------------------------------------------------------

_otel_initialised = False
_tracer: Optional[trace.Tracer] = None
_meter:  Optional[metrics.Meter] = None

# State names indexed by integer value for logging / trace attributes
_STATE_NAMES = ("idle", "thinking", "tool_executing", "waiting_for_user")

# Metrics instruments (created once, reused across all conversations)
_query_counter             = None
_tool_counter             = None
_query_latency_hist       = None
_tool_latency_hist        = None
_error_counter            = None
_active_conversations_gauge = None
_llm_latency_hist         = None
_llm_error_counter        = None
_state_gauge              = None
_state_duration_hist      = None


def _parse_headers(header_str: str) -> dict:
    """Parse 'key1=val1,key2=val2' into a dict. Empty string → {}."""
    out = {}
    for part in header_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ── State machine collector ─────────────────────────────────────────────────
# All active OtelShim instances register here so the observable gauge can
# report their current state values.  Thread-safe enough for production use.
# (Defined at module level so _init_otel() can reference them as closures.)
_state_shims: list = []
_state_shims_lock = __import__("threading").Lock()


def _register_state_shim(shim) -> None:
    with _state_shims_lock:
        if shim not in _state_shims:
            _state_shims.append(shim)


def _state_gather_callback(options) -> None:
    """Called by the observable gauge — yields one Observation per active shim.

    The callback must be a generator that yields Observation objects.
    Using Observation directly (not options.observe) for SDK compatibility.
    """
    from opentelemetry.metrics import Observation as _Obs
    with _state_shims_lock:
        for shim in _state_shims:
            try:
                state_name = _STATE_NAMES[shim._state] if shim._state < len(_STATE_NAMES) else str(shim._state)
                yield _Obs(
                    float(shim._state),
                    {
                        "conversation_id":    shim._conversation_id[-12:],
                        "conversation_state": state_name,
                        **shim._base_labels(),
                    },
                )
            except Exception:
                pass


def _init_otel() -> None:
    global _otel_initialised, _tracer, _meter
    global _query_counter, _tool_counter
    global _query_latency_hist, _tool_latency_hist
    global _error_counter, _active_conversations_gauge
    global _llm_latency_hist, _llm_error_counter
    global _state_gauge, _state_duration_hist

    if _otel_initialised:
        return
    _otel_initialised = True

    logger.info("OTEL_INIT: starting initialization — endpoint=%s", OTEL_EXPORTER_OTLP_ENDPOINT)

    resource = Resource.create({
        SERVICE_NAME:            OTEL_SERVICE_NAME,
        SERVICE_VERSION:         OTEL_SERVICE_VERSION,
        "deployment.environment": OTEL_ENVIRONMENT,
        "hermes.profile":         OTEL_PROFILE,
    })

    headers = _parse_headers(OTEL_EXPORTER_OTLP_HEADERS)
    use_tls = OTEL_EXPORTER_OTLP_ENDPOINT.startswith("https://")

    logger.info("OTEL_INIT: creating OTLP span exporter — endpoint=%s insecure=%s", OTEL_EXPORTER_OTLP_ENDPOINT, not use_tls)

    # ---- Traces ----
    trace_provider = TracerProvider(resource=resource)
    _trace_endpoint = OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/") + "/v1/traces"
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=_trace_endpoint,
            )
        )
    )
    trace.set_tracer_provider(trace_provider)
    _tracer = trace.get_tracer(__name__)

    # ---- Metrics ----
    _metric_endpoint = OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/") + "/v1/metrics"
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=_metric_endpoint,
        ),
        export_interval_millis=OTEL_METRICS_INTERVAL_SECS * 1000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(__name__)

    # Counters
    _query_counter = _meter.create_counter(
        name="hermes.queries.total",
        description="Total number of queries processed",
        unit="1",
    )
    _tool_counter = _meter.create_counter(
        name="hermes.tool_calls.total",
        description="Total number of tool calls",
        unit="1",
    )
    _error_counter = _meter.create_counter(
        name="hermes.errors.total",
        description="Total number of errors",
        unit="1",
    )

    # Histograms
    _query_latency_hist = _meter.create_histogram(
        name="hermes.query.latency_seconds",
        description="End-to-end query latency in seconds",
        unit="s",
    )
    _tool_latency_hist = _meter.create_histogram(
        name="hermes.tool.latency_seconds",
        description="Per-tool call latency in seconds",
        unit="s",
    )

    # UpDownCounter for active conversations
    _active_conversations_gauge = _meter.create_up_down_counter(
        name="hermes.active_conversations",
        description="Number of currently active conversations",
        unit="1",
    )

    # LLM latency histogram
    _llm_latency_hist = _meter.create_histogram(
        name="hermes.llm.latency_seconds",
        description="LLM API call latency in seconds",
        unit="s",
    )
    _llm_error_counter = _meter.create_counter(
        name="hermes.llm.errors.total",
        description="Total number of LLM API errors",
        unit="1",
    )

    # State gauge — observable gauge reporting current conversation state
    _state_gauge = _meter.create_observable_gauge(
        name="hermes.conversation.state",
        description="Current state of each active conversation (0=idle, 1=thinking, 2=tool_executing, 3=waiting_for_user)",
        unit="1",
        callbacks=[_state_gather_callback],
    )

    # State duration histogram — time spent in each state per conversation
    _state_duration_hist = _meter.create_histogram(
        name="hermes.conversation.state_duration_seconds",
        description="Time spent in each conversation state",
        unit="s",
    )

    logger.info(
        "OTEL_INIT: complete — tracer=%s meter=%s endpoint=%s",
        _tracer, _meter, OTEL_EXPORTER_OTLP_ENDPOINT,
    )


# ---------------------------------------------------------------------------
# OtelShim — the main wrapper class
# ---------------------------------------------------------------------------

def _no_op(*args, **kwargs) -> None:
    """Sentinel used when no agent is available (bg/btw agents are ephemeral)."""
    pass


def merge_callbacks(*callback_dicts) -> dict:
    """
    Merge multiple per-name callback dicts into one.

    For keys where multiple dicts have a value, all values are combined into
    a single list so no callback is lost.  This allows the OTel shim to sit
    alongside existing TUI/display callbacks without replacing them.

    Example::

        merged = merge_callbacks(
            {"tool_start_callback": cli._on_tool_start},
            shim.callbacks,
        )
        # → AIAgent(tool_start_callback=merged["tool_start_callback"])
    """
    from collections import defaultdict

    result: dict = {}
    by_name: dict = defaultdict(list)

    for d in callback_dicts:
        if not d:
            continue
        for name, fn in d.items():
            if fn is not None:
                by_name[name].append(fn)

    for name, fns in by_name.items():
        if len(fns) == 1:
            result[name] = fns[0]
        else:
            # Wrap all of them in a single dispatcher
            def make_dispatcher(fs):
                def dispatcher(*args, **kwargs):
                    for f in fs:
                        try:
                            f(*args, **kwargs)
                        except Exception:
                            logger.exception("callback %s.%s raised", name, f.__name__)
                return dispatcher
            result[name] = make_dispatcher(fns)

    return result


class OtelShim:
    """
    Wraps an AIAgent instance and wires every callback to produce
    OpenTelemetry traces and metrics.

    ``agent`` may be ``None`` — the shim still initialises the OTel SDK
    (once per process) so that metrics/traces are exported.  When agent is
    ``None``, the ``callbacks`` dict returns no-op stubs; pass the callbacks
    to ``AIAgent`` via ``merge_callbacks(shim.callbacks, your_callbacks)``::

        shim    = OtelShim(None)   # no agent needed just to initialise
        shim2   = OtelShim(agent)
        merged  = merge_callbacks(shim.callbacks, existing_tui_callbacks)
        agent   = AIAgent(model="...", **merged)

    Or use the two-step::

        shim   = OtelShim(agent)
        agent  = AIAgent(model="...", **shim.callbacks)   # replaces existing

    Prefer ``merge_callbacks`` when existing callbacks (TUI display) must be preserved.
    """

    def __init__(
        self,
        agent: Any = None,
        conversation_id: Optional[str] = None,
        extra_attributes: Optional[dict] = None,
    ):
        if OTEL_ENABLED:
            _init_otel()

        self._agent            = agent
        self._conversation_id = conversation_id or f"conv-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._extra           = extra_attributes or {}
        self._trace_active    = False
        self._tool_spans      = {}   # tool_call_id → span
        self._tool_start      = {}   # tool_call_id → start_time (perf_counter)
        self._query_start     : Optional[float] = None
        self._span            : Optional[Any]    = None
        # LLM API call tracking
        self._llm_span        : Optional[Any]    = None
        self._llm_start       : Optional[float]  = None
        self._llm_call_count  : int              = 0

        # ── State machine ──────────────────────────────────────────────────────
        # States:  idle=0 | thinking=1 | tool_executing=2 | waiting_for_user=3
        self._state       : int = 0
        self._state_start : float = time.perf_counter()  # when current state began
        self._state_start_ts : Optional[float] = None   # wall-clock start (for duration)

        # Register this shim's state getter with the module-level collector
        _register_state_shim(self)

        # If agent is None, callbacks are no-ops (bg/btw path: shim exists on HermesCLI
        # but individual AIAgent instances are created without going through _init_agent)
        if agent is None:
            self.callbacks = {k: _no_op for k in (
                "tool_start_callback", "tool_complete_callback",
                "thinking_callback", "step_callback", "status_callback",
                "clarify_callback", "stream_delta_callback",
                "pre_api_request_callback", "stream_complete_callback",
            )}
        else:
            self.callbacks = {
                "tool_start_callback":      self._on_tool_start,
                "tool_complete_callback":  self._on_tool_complete,
                "thinking_callback":       self._on_thinking,
                "step_callback":           self._on_step,
                "status_callback":         self._on_status,
                "clarify_callback":       self._on_clarify,
                "stream_delta_callback":   self._on_stream_delta,
                "pre_api_request_callback": self._on_pre_api_request,
                "stream_complete_callback": self._on_stream_complete,
            }

    @property
    def _otel_enabled(self) -> bool:
        """True when OTel is both enabled and successfully initialised."""
        return OTEL_ENABLED and _otel_initialised

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _set_state(self, new_state: int) -> None:
        """
        Transition to a new state, recording the duration spent in the previous one.

        States:
            0 = idle
            1 = thinking  (LLM API call in progress)
            2 = tool_executing
            3 = waiting_for_user  (clarify_callback fired, agent paused)
        """
        if not OTEL_ENABLED:
            return

        old_state = self._state

        # Record how long we were in the previous state
        if old_state != new_state:
            now       = time.perf_counter()
            wall_now = time.time()
            elapsed  = now - self._state_start

            if self._state_start_ts is not None:
                wall_elapsed = wall_now - self._state_start_ts
            else:
                wall_elapsed = elapsed

            state_name = _STATE_NAMES[old_state] if old_state < len(_STATE_NAMES) else str(old_state)

            # Histogram — time in this state for this conversation
            try:
                _state_duration_hist.record(
                    wall_elapsed,
                    {
                        "conversation.state": state_name,
                        "conversation.id":     self._conversation_id,
                        **self._base_labels(),
                    },
                )
            except Exception as e:
                logger.warning("_set_state: histogram record failed: %s", e)

            # Add a trace event for every state transition
            if self._span:
                self._span.add_event(
                    name=f"state.{state_name}.duration",
                    attributes={
                        "state.from":   state_name,
                        "state.to":     _STATE_NAMES[new_state] if new_state < len(_STATE_NAMES) else str(new_state),
                        "state.duration_ms": round(wall_elapsed * 1000, 2),
                        **self._base_labels(),
                    },
                )

            logger.debug(
                "OTEL_STATE: conv=%s %s → %s (%.0fms)",
                self._conversation_id, state_name,
                _STATE_NAMES[new_state] if new_state < len(_STATE_NAMES) else str(new_state),
                wall_elapsed * 1000,
            )

            self._state       = new_state
            self._state_start = now
            self._state_start_ts = wall_now

    # ------------------------------------------------------------------
    # Attribute helpers
    # ------------------------------------------------------------------

    def _base_labels(self, extra: Optional[dict] = None) -> dict:
        """
        Build a label dict with user.id, platform, and profile prepended.

        These three labels are the primary dimension for per-conversation
        observability.  They are attached to every metric and span so that
        dashboards can filter/slice by who sent the message, which platform
        it came from, and which Hermes profile was active.
        """
        labels = {
            "user.id":    self._extra.get("user.id", ""),
            "platform":   self._extra.get("platform", ""),
            "profile":    self._extra.get("profile", ""),
        }
        if extra:
            labels.update(extra)
        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_trace(self, user_message: str) -> None:
        """Call BEFORE run_conversation() to open the top-level span."""
        if not OTEL_ENABLED:
            return

        self._query_start  = time.perf_counter()
        self._trace_active = True

        base = self._base_labels()
        attrs = {
            "conversation.id":    self._conversation_id,
            "user.query":         user_message[:500],
            "user.query.length":  len(user_message),
            "agent.model":        getattr(self._agent, "model", "unknown"),
            **base,
            **self._extra,
            # spanmetrics connector sees these underscore names (dots are sanitized to underscores by Prometheus)
            "user_id":   base.get("user.id",   ""),
            "platform":  base.get("platform",  ""),
            "profile":   base.get("profile",   ""),
        }

        logger.info(
            "OTEL_START_TRACE: conv=%s query_len=%d OTEL_ENABLED=%s _otel_initialised=%s",
            self._conversation_id, len(user_message), OTEL_ENABLED, _otel_initialised,
        )

        if _tracer is None:
            logger.error("OTEL_START_TRACE: _tracer is None — _init_otel() likely failed, skipping trace")
            return

        self._span = _tracer.start_span(
            name=f"agent.query {self._conversation_id[-12:]}",
            attributes=attrs,
        )
        self._span.__enter__()

        _active_conversations_gauge.add(1, self._base_labels({"conversation.state": "active"}))
        _query_counter.add(1, self._base_labels({"query.type": _classify_query(user_message)}))

        # State: enter thinking (1) — first LLM call will follow shortly
        self._set_state(1)

    def end_trace(self, final_response: str, success: bool = True) -> None:
        """Call AFTER run_conversation() returns to close the span."""
        if not OTEL_ENABLED:
            logger.warning("OTEL_END_TRACE: early-exit — OTEL_ENABLED=%s", OTEL_ENABLED)
            return
        if not self._trace_active:
            logger.warning("OTEL_END_TRACE: early-exit — _trace_active=False (span=%s)", self._span)
            return

        elapsed = time.perf_counter() - self._query_start
        logger.info("OTEL_END_TRACE: conv=%s elapsed=%.3fs success=%s span=%s", self._conversation_id, elapsed, success, self._span)

        try:
            _query_latency_hist.record(elapsed, self._base_labels({"conversation.id": self._conversation_id}))
        except Exception as e:
            logger.warning("OTEL_END_TRACE: histogram record failed: %s", e)

        # Close any open LLM span
        if self._llm_span:
            try:
                self._llm_span.set_attribute("llm.error", "conversation_ended")
                self._llm_span.set_status(Status(StatusCode.ERROR, "Conversation ended"))
                self._llm_span.__exit__(None, None, None)
            except Exception:
                pass
            self._llm_span = None

        span = self._span
        self._span = None
        self._trace_active = False

        # Transition to idle — records the final state's duration before the trace closes
        self._set_state(0)

        if span:
            try:
                span.set_attribute("conversation.success", success)
                span.set_attribute("response.length", len(final_response) if final_response else 0)
                span.set_status(Status(StatusCode.OK if success else StatusCode.ERROR))
                span.__exit__(None, None, None)
            except Exception as e:
                logger.error("OTEL_END_TRACE: span.__exit__ failed: %s", e)

        try:
            _active_conversations_gauge.add(-1, self._base_labels({"conversation.state": "ended"}))
        except Exception as e:
            logger.warning("OTEL_END_TRACE: gauge update failed: %s", e)

        logger.debug("trace ended — latency=%.3fs success=%s", elapsed, success)

    def record_error(self, error_message: str, error_type: str = "generic") -> None:
        """Call on any error path to record it in traces and metrics."""
        if not OTEL_ENABLED:
            return
        _error_counter.add(1, self._base_labels({"error.type": error_type}))
        if self._span:
            self._span.record_exception(Exception(error_message))
            self._span.set_status(Status(StatusCode.ERROR, error_message))

    # ------------------------------------------------------------------
    # Internal callbacks — wired to AIAgent
    # ------------------------------------------------------------------

    def _on_tool_start(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict,
    ) -> None:
        if not OTEL_ENABLED:
            return

        self._set_state(2)

        start = time.perf_counter()
        self._tool_start[tool_call_id] = start

        _tool_counter.add(1, self._base_labels({
            "tool.name": tool_name,
            "tool.type": _classify_tool(tool_name),
        }))

        span = _tracer.start_span(
            name=f"tool.{tool_name}",
            attributes={
                "tool.call_id": tool_call_id,
                "tool.name":    tool_name,
                "tool.args":    _sanitise_args(tool_args),
                **self._base_labels(),
            },
        )
        span.__enter__()
        self._tool_spans[tool_call_id] = span

    def _on_tool_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict,
        function_result: str,
    ) -> None:
        if not OTEL_ENABLED:
            return

        start = self._tool_start.pop(tool_call_id, None)
        span  = self._tool_spans.pop(tool_call_id, None)
        latency = time.perf_counter() - start if start else 0.0

        _tool_latency_hist.record(latency, self._base_labels({"tool.name": tool_name}))

        if span:
            is_error = _is_error_result(function_result)
            span.set_attribute("tool.result_length", len(function_result) if function_result else 0)
            span.set_attribute("tool.error", is_error)
            span.set_status(Status(StatusCode.OK if not is_error else StatusCode.ERROR))
            span.__exit__(None, None, None)

            if is_error:
                _error_counter.add(1, self._base_labels({"error.type": "tool", "tool.name": tool_name}))

        # State: back to thinking (1) — agent will call LLM again next
        self._set_state(1)

    def _on_thinking(self, thinking: str) -> None:
        if not OTEL_ENABLED or not self._span:
            return
        self._span.add_event(
            name="model.thinking",
            attributes={
                "thinking.length": len(thinking),
                "thinking.preview": thinking[:200],
            },
        )

    def _on_step(self, *args, **kwargs) -> None:
        if not OTEL_ENABLED or not self._span:
            return
        # run_agent calls step_callback(iteration, prev_tools)
        step_info = args[0] if args else kwargs.get("step_info")
        try:
            step_str = str(step_info)[:300]
        except Exception:
            step_str = "<unstringable>"
        self._span.add_event(name="agent.step", attributes={"step.info": step_str})

    def _on_status(self, *args, **kwargs) -> None:
        if not OTEL_ENABLED or not self._span:
            return
        # run_agent calls status_callback(event_type, message)
        status_type = args[0] if args else kwargs.get("status_type") or "unknown"
        message = args[1] if len(args) > 1 else kwargs.get("message", "")
        self._span.add_event(
            name=f"status.{status_type}",
            attributes={"status.message": message[:500]},
        )
        if status_type in ("error", "warning"):
            _error_counter.add(1, self._base_labels({"status.type": status_type}))

    def _on_clarify(self, *args, **kwargs) -> None:
        if not OTEL_ENABLED or not self._span:
            return
        # run_agent calls clarify_callback(question, choices)
        question = args[0] if args else kwargs.get("question", "")
        choices = args[1] if len(args) > 1 else kwargs.get("choices", [])
        self._span.add_event(
            name="agent.clarify",
            attributes={
                "clarify.question":     question[:500],
                "clarify.choice_count": len(choices) if choices else 0,
            },
        )
        # State: waiting_for_user (3) — agent paused, awaiting user response
        self._set_state(3)

    def _on_stream_delta(self, *args, **kwargs) -> None:
        if not OTEL_ENABLED or not self._span:
            return
        # run_agent calls stream_delta_callback(string)
        delta_text = args[0] if args else kwargs.get("delta_text") or ""
        if not isinstance(delta_text, str):
            delta_text = str(delta_text) if delta_text is not None else ""
        self._span.add_event(
            name="stream.token",
            attributes={"token.length": len(delta_text)},
        )

    def _on_pre_api_request(
        self,
        model: str,
        provider: str,
        api_mode: str,
        message_count: int,
        tool_count: int,
        approx_input_tokens: int,
        api_call_number: int,
    ) -> None:
        """Called before every LLM API request (from pre_api_request plugin hook)."""
        if not OTEL_ENABLED:
            return
        self._llm_call_count += 1
        self._llm_start = time.perf_counter()
        span_name = f"llm.{api_mode} {self._llm_call_count}/{model}"
        self._llm_span = _tracer.start_span(
            name=span_name,
            attributes={
                "llm.model":       model,
                "llm.provider":    provider or "unknown",
                "llm.api_mode":    api_mode,
                "llm.call_number": self._llm_call_count,
                "llm.message_count": message_count,
                "llm.tool_count":  tool_count,
                "llm.input_tokens_approx": approx_input_tokens,
                **self._base_labels(),
            },
        )
        self._llm_span.__enter__()

    def _on_stream_complete(
        self,
        output_tokens: int,
        finish_reason: str,
        model: str,
        provider: str,
        error: Optional[str] = None,
    ) -> None:
        """Called when an LLM streaming response completes (from stream_consumer post_hook)."""
        if not OTEL_ENABLED:
            return
        elapsed = time.perf_counter() - self._llm_start if self._llm_start else 0.0

        if self._llm_span:
            self._llm_span.set_attribute("llm.output_tokens", output_tokens or 0)
            self._llm_span.set_attribute("llm.finish_reason", finish_reason or "unknown")
            if error:
                self._llm_span.set_attribute("llm.error", error)
                self._llm_span.set_status(Status(StatusCode.ERROR, error))
            else:
                self._llm_span.set_status(Status(StatusCode.OK))
            self._llm_span.__exit__(None, None, None)
            self._llm_span = None

        _llm_latency_hist.record(elapsed, self._base_labels({
            "llm.model":    model or "unknown",
            "llm.provider": provider or "unknown",
            "llm.finish_reason": finish_reason or "unknown",
        }))
        if error:
            _llm_error_counter.add(1, self._base_labels({
                "llm.model":    model or "unknown",
                "llm.provider": provider or "unknown",
                "error.type":   "llm",
            }))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def wrapped_chat(agent: Any, message: str, shim: OtelShim, **kwargs) -> str:
    """
    One-shot convenience: start trace → run_conversation → end trace → return.

    Use this instead of calling ``agent.run_conversation()`` directly when
    using ``OtelShim``::

        shim = OtelShim(agent)
        response = wrapped_chat(agent, "your query", shim)
    """
    shim.start_trace(message)
    try:
        result = agent.run_conversation(user_message=message, **kwargs)
        final_response = (
            result.get("final_response", "") if isinstance(result, dict)
            else str(result)
        )
        shim.end_trace(final_response, success=True)
        return final_response
    except Exception as exc:
        shim.record_error(str(exc), type(exc).__name__)
        shim.end_trace(str(exc), success=False)
        raise


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _classify_query(query: str) -> str:
    """Coarse intent label for query counter."""
    q = query.lower()
    if any(k in q for k in ("cob", "close of business", "end of day", "batch", "end-of-day")):
        return "cob"
    if any(k in q for k in ("status", "health", "check")):
        return "status"
    if any(k in q for k in ("run", "execute", "start", "trigger")):
        return "action"
    if any(k in q for k in ("report", "summary", "summarise")):
        return "report"
    return "general"


def _classify_tool(tool_name: str) -> str:
    """Coarse type label for tool counter."""
    prefixes = {
        "shell":   ("terminal", "bash", "shell", "process"),
        "file":    ("file", "read", "write", "patch", "search_files"),
        "web":     ("web", "http", "fetch", "browse", "scrape"),
        "delegate":("delegate", "spawn", "subagent"),
        "browser": ("browser", "click", "navigate", "type"),
        "mcp":     ("mcp_",),
    }
    for category, prefixes_list in prefixes.items():
        if any(tool_name.startswith(p) for p in prefixes_list):
            return category
    return "other"


_SECRET_RE = re.compile(r'("(?:api_key|token|secret|password|auth|credential|key)[^"]*"\s*:\s*")[^"]+(")', re.IGNORECASE)


def _sanitise_args(args: dict, max_len: int = 600) -> str:
    """JSON-encode args, truncate, redact common secret field values."""
    if not args:
        return "{}"
    try:
        raw = json.dumps(args, default=str, separators=(",", ":"))
    except Exception:
        raw = str(args)
    raw = _SECRET_RE.sub(r'\1***\2', raw)
    if len(raw) > max_len:
        raw = raw[:max_len] + "...[truncated]"
    return raw


def _is_error_result(result: str) -> bool:
    """Heuristic: does a tool result string look like an error?"""
    if not result:
        return False
    result_lower = result.lower()
    signals = (
        "error", "exception", "traceback", "failed", "failure",
        "invalid", "denied", "unauthorized", "timeout",
        "not found", "command failed", "returned non-zero", "refused",
        "no such file", "connection refused",
    )
    return any(sig in result_lower for sig in signals)
