"""Tests for agent/otel_shim.py — import fallback, merge_callbacks, and disabled-mode no-op."""

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("opentelemetry")


class TestMergeCallbacks:
    """Tests for merge_callbacks() — all existing callbacks must be preserved."""

    def test_empty_dicts_returns_empty(self):
        """No inputs → empty result."""
        from agent.otel_shim import merge_callbacks
        assert merge_callbacks() == {}
        assert merge_callbacks({}) == {}

    def test_single_dict_passthrough(self):
        """One dict with one entry is returned as-is."""
        from agent.otel_shim import merge_callbacks
        cb = lambda: None
        result = merge_callbacks({"tool_start_callback": cb})
        assert result["tool_start_callback"] is cb

    def test_two_disjoint_keys_both_present(self):
        """Two dicts with different keys → both included."""
        from agent.otel_shim import merge_callbacks
        cb1 = lambda: None
        cb2 = lambda: None
        result = merge_callbacks(
            {"tool_start_callback": cb1},
            {"tool_end_callback": cb2},
        )
        assert result["tool_start_callback"] is cb1
        assert result["tool_end_callback"] is cb2

    def test_same_key_runs_both(self):
        """Same key in two dicts → a dispatcher that calls both, original order."""
        from agent.otel_shim import merge_callbacks
        cb1 = MagicMock(name="cb1")
        cb2 = MagicMock(name="cb2")
        result = merge_callbacks(
            {"tool_start_callback": cb1},
            {"tool_start_callback": cb2},
        )
        dispatcher = result["tool_start_callback"]
        dispatcher()
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_three_callbacks_same_key_all_run(self):
        """Three callbacks under same key → all three fire."""
        from agent.otel_shim import merge_callbacks
        cb1 = MagicMock(name="cb1")
        cb2 = MagicMock(name="cb2")
        cb3 = MagicMock(name="cb3")
        result = merge_callbacks(
            {"tool_start_callback": cb1},
            {"tool_start_callback": cb2},
            {"tool_start_callback": cb3},
        )
        result["tool_start_callback"]()
        cb1.assert_called_once()
        cb2.assert_called_once()
        cb3.assert_called_once()

    def test_none_values_skipped(self):
        """None values in a dict are silently dropped."""
        from agent.otel_shim import merge_callbacks
        cb = MagicMock(name="cb")
        result = merge_callbacks(
            {"tool_start_callback": None},
            {"tool_start_callback": cb},
        )
        assert result["tool_start_callback"] is cb

    def test_none_in_single_dict_skipped(self):
        """A dict with only None values is dropped entirely for that key."""
        from agent.otel_shim import merge_callbacks
        cb = MagicMock(name="cb")
        result = merge_callbacks(
            {"tool_start_callback": None},
            {"tool_end_callback": cb},
        )
        assert "tool_start_callback" not in result
        assert result["tool_end_callback"] is cb

    def test_dispatcher_exception_caught_and_logged(self):
        """If one callback raises, the dispatcher continues to the next one."""
        from agent.otel_shim import merge_callbacks
        import logging

        cb1 = MagicMock(side_effect=RuntimeError("cb1 error"))
        cb2 = MagicMock(name="cb2")

        result = merge_callbacks(
            {"tool_start_callback": cb1},
            {"tool_start_callback": cb2},
        )
        # Must not raise
        result["tool_start_callback"]()
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_otel_callbacks_and_tui_callbacks_merged(self):
        """Simulate OTel shim callbacks + existing TUI callbacks — both fire."""
        from agent.otel_shim import merge_callbacks

        tui_cb = MagicMock(name="tui_cb")
        otel_cb = MagicMock(name="otel_cb")

        merged = merge_callbacks(
            {"tool_start_callback": tui_cb},      # TUI callback
            {"tool_start_callback": otel_cb},      # OTel shim callback
        )

        merged["tool_start_callback"]("tool_name", {"param": "value"})
        tui_cb.assert_called_once()
        otel_cb.assert_called_once()


class TestWrappedChatDisabledMode:
    """Tests for wrapped_chat when OTel is disabled — must not crash or emit traces."""

    def test_wrapped_chat_disabled_does_not_raise(self):
        """wrapped_chat must not raise when OTEL_ENABLED=false."""
        import os
        from agent.otel_shim import wrapped_chat

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "hello"}

        # Patch OTEL_ENABLED before the function runs
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}, clear=False):
            with patch("agent.otel_shim.OTEL_ENABLED", False):
                # Should not raise
                result = wrapped_chat(mock_agent, "hello", shim=MagicMock())

        assert result == "hello"
        mock_agent.run_conversation.assert_called_once()


class TestSpanParenting:
    """Ensure child spans are created under active root conversation span context."""

    def test_pre_api_request_span_uses_root_context(self):
        from agent.otel_shim import OtelShim

        tracer = MagicMock()
        llm_span = MagicMock()
        tracer.start_span.return_value = llm_span

        with patch("agent.otel_shim.OTEL_ENABLED", True), \
             patch("agent.otel_shim._otel_initialised", True), \
             patch("agent.otel_shim._tracer", tracer), \
             patch("agent.otel_shim.trace.set_span_in_context", return_value="ROOT_CTX"):
            shim = OtelShim(agent=None)
            shim._span = MagicMock()
            shim._on_pre_api_request(
                model="glm-5.1",
                provider="custom",
                api_mode="chat_completions",
                message_count=2,
                tool_count=1,
                approx_input_tokens=128,
                api_call_number=1,
            )

        tracer.start_span.assert_called_once()
        _, kwargs = tracer.start_span.call_args
        assert kwargs["context"] == "ROOT_CTX"

    def test_tool_span_uses_root_context(self):
        from agent.otel_shim import OtelShim

        tracer = MagicMock()
        tool_span = MagicMock()
        tracer.start_span.return_value = tool_span

        with patch("agent.otel_shim.OTEL_ENABLED", True), \
             patch("agent.otel_shim._otel_initialised", True), \
             patch("agent.otel_shim._tracer", tracer), \
             patch("agent.otel_shim._tool_counter", MagicMock()), \
             patch("agent.otel_shim._state_duration_hist", MagicMock()), \
             patch("agent.otel_shim.threading.Thread", return_value=MagicMock(start=MagicMock())), \
             patch("agent.otel_shim.trace.set_span_in_context", return_value="ROOT_CTX"):
            shim = OtelShim(agent=None)
            shim._span = MagicMock()
            shim._on_tool_start(
                tool_call_id="call_1",
                tool_name="terminal",
                tool_args={"command": "echo hi"},
            )

        tracer.start_span.assert_called_once()
        _, kwargs = tracer.start_span.call_args
        assert kwargs["context"] == "ROOT_CTX"
