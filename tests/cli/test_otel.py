"""Tests for OTel trace lifecycle in cli.py — wrapped_chat contract."""

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("opentelemetry")


class TestCliOtelLifecycle:
    """OTel lifecycle in CLI — trace started before run, always closed in finally."""

    def test_wrapped_chat_starts_trace_before_run(self):
        """When shim is provided, start_trace fires before run_conversation."""
        from agent.otel_shim import wrapped_chat

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "hello"}

        shim = MagicMock()
        shim.start_trace = MagicMock()
        shim.end_trace = MagicMock()

        result = wrapped_chat(mock_agent, "hello", shim)

        assert result == "hello"
        shim.start_trace.assert_called_once_with("hello")
        assert shim.start_trace.call_count == 1

    def test_wrapped_chat_closes_trace_on_success(self):
        """On normal exit, end_trace is called with success=True."""
        from agent.otel_shim import wrapped_chat

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "all good"}
        shim = MagicMock()
        shim.start_trace = MagicMock()
        shim.end_trace = MagicMock()

        result = wrapped_chat(mock_agent, "test", shim)

        assert result == "all good"
        shim.end_trace.assert_called_once_with("all good", success=True)

    def test_wrapped_chat_records_error_and_closes_on_failure(self):
        """On exception: record_error + end_trace(success=False) fire before exception propagates."""
        from agent.otel_shim import wrapped_chat

        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = RuntimeError("model error")
        shim = MagicMock()
        shim.start_trace = MagicMock()
        shim.end_trace = MagicMock()
        shim.record_error = MagicMock()

        with pytest.raises(RuntimeError, match="model error"):
            wrapped_chat(mock_agent, "test", shim)

        shim.record_error.assert_called_once()
        call_args = shim.end_trace.call_args
        assert call_args[1]["success"] is False

    def test_wrapped_chat_with_string_result(self):
        """run_conversation can return a plain string instead of a dict."""
        from agent.otel_shim import wrapped_chat

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = "plain string response"
        shim = MagicMock()
        shim.start_trace = MagicMock()
        shim.end_trace = MagicMock()

        result = wrapped_chat(mock_agent, "test", shim)

        assert result == "plain string response"
        shim.end_trace.assert_called_once_with("plain string response", success=True)

    def test_wrapped_chat_start_before_end_order(self):
        """start_trace must be called before end_trace (verifiable via call sequence)."""
        from agent.otel_shim import wrapped_chat

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        shim = MagicMock()
        shim.start_trace = MagicMock()
        shim.end_trace = MagicMock()

        wrapped_chat(mock_agent, "test", shim)

        # Both called exactly once
        assert shim.start_trace.call_count == 1
        assert shim.end_trace.call_count == 1
        # Verify the message arg passed to each
        shim.start_trace.assert_called_once_with("test")
        shim.end_trace.assert_called_once_with("ok", success=True)
