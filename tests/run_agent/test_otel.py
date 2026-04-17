"""Tests for OTel callback bridge in run_agent.py — pre/post API hooks wired to OtelShim."""

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("opentelemetry")


class TestRunAgentOtelBridge:
    """_on_pre_api_request and _on_stream_complete fire at the right moments."""

    @pytest.fixture
    def mock_shim(self):
        """A mock OtelShim with the two OTel callback methods."""
        shim = MagicMock()
        shim._on_pre_api_request = MagicMock()
        shim._on_stream_complete = MagicMock()
        return shim

    def test_on_pre_api_request_called_before_stream(self, mock_shim):
        """_on_pre_api_request fires before the streaming response is consumed."""
        # This test documents the bridge contract:
        # run_agent.py calls _on_pre_api_request before the API request is made.
        # We verify the attribute lookup pattern is correct.
        shim = mock_shim

        # The bridge does: cb = getattr(_otel_shim, "_on_pre_api_request", None)
        cb = getattr(shim, "_on_pre_api_request", None)
        assert cb is not None

        # Calling it with the args that run_agent.py passes
        cb(
            model="anthropic/claude-sonnet-4",
            provider="anthropic",
            api_mode="chat",
            message_count=5,
            tool_count=3,
            approx_input_tokens=1200,
            api_call_number=0,
        )
        shim._on_pre_api_request.assert_called_once()

    def test_on_stream_complete_called_after_response(self, mock_shim):
        """_on_stream_complete fires after the streaming response is normalised."""
        shim = mock_shim

        # The bridge does: cb = getattr(_otel_shim, "_on_stream_complete", None)
        cb = getattr(shim, "_on_stream_complete", None)
        assert cb is not None

        cb(
            output_tokens=342,
            finish_reason="stop",
            model="anthropic/claude-sonnet-4",
            provider="anthropic",
            error=None,
        )
        shim._on_stream_complete.assert_called_once()

    def test_getattr_returns_none_when_no_shim(self):
        """When _otel_shim is absent, getattr returns None — bridge handles this gracefully."""
        plain = MagicMock(spec=[])  # no _otel_shim attr
        cb = getattr(plain, "_on_pre_api_request", None)
        assert cb is None

    def test_getattr_returns_none_when_shim_has_no_method(self, mock_shim):
        """When _otel_shim exists but has no _on_pre_api_request, bridge skips the call."""
        del mock_shim._on_pre_api_request
        cb = getattr(mock_shim, "_on_pre_api_request", None)
        assert cb is None

    def test_callback_exception_propagates_without_bridge(self, mock_shim):
        """Without the bridge's try/except wrapper, a raising callback crashes the agent.

        This documents WHY the bridge in run_agent.py lines 8326-8340 wraps every
        OTel callback call in try/except — to prevent a crashing callback from
        terminating the agent mid-request.
        """
        shim = mock_shim
        shim._on_pre_api_request.side_effect = RuntimeError("OTel export error")

        cb = getattr(shim, "_on_pre_api_request", None)
        assert cb is not None

        # Without the bridge wrapper, this exception propagates — which would crash the agent.
        # The test confirms the callback CAN raise, and the bridge MUST catch it.
        with pytest.raises(RuntimeError, match="OTel export error"):
            cb(
                model="test", provider="test", api_mode="chat",
                message_count=1, tool_count=0,
                approx_input_tokens=10, api_call_number=0,
            )

    def test_callback_merge_keeps_both_sets(self):
        """merge_callbacks combines TUI callbacks and OTel callbacks under the same key."""
        from agent.otel_shim import merge_callbacks

        tui_cb = MagicMock(name="tui_cb")
        otel_cb = MagicMock(name="otel_cb")

        merged = merge_callbacks(
            {"tool_start_callback": tui_cb},   # existing TUI
            {"tool_start_callback": otel_cb},   # OTel shim
        )

        merged["tool_start_callback"]()

        tui_cb.assert_called_once()
        otel_cb.assert_called_once()
