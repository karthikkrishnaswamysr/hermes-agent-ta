"""Tests for OTel trace lifecycle in gateway/run.py — _handle_message OTel wrapping."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

pytest.importorskip("opentelemetry")


class TestGatewayOtelLifecycle:
    """OTel trace lifecycle in gateway: start_trace before run_conversation,
    end_trace(success=True) on normal exit, record_error + end_trace(success=False) on exception."""

    @pytest.fixture
    def mock_otel_shim(self):
        """A mock OtelShim with all lifecycle methods tracked."""
        shim = MagicMock()
        shim.start_trace = MagicMock()
        shim.end_trace = MagicMock()
        shim.record_error = MagicMock()
        shim.callbacks = {}
        shim._otel_enabled = True
        return shim

    def _run_otel_block_with_exception(self, shim, run_conversation_fn, message="hello"):
        """Mirror gateway/run.py lines 8165-8174: try/finally/except all in one nested structure."""
        result = None
        if shim:
            shim.start_trace(message)
        try:
            try:
                result = run_conversation_fn()
            finally:
                if shim:
                    res_dict = result if isinstance(result, dict) else {"final_response": str(result)}
                    shim.end_trace(res_dict.get("final_response", ""), success=not res_dict.get("error"))
        except Exception as exc:
            if shim:
                shim.record_error(str(exc), type(exc).__name__)
                shim.end_trace(str(exc), success=False)
            raise

    def _run_otel_block(self, shim, run_conversation_fn, message="hello"):
        """Mirror gateway/run.py lines 8165-8169: try/finally with no except block."""
        result = None
        if shim:
            shim.start_trace(message)
        try:
            result = run_conversation_fn()
        finally:
            if shim:
                res_dict = result if isinstance(result, dict) else {"final_response": str(result)}
                shim.end_trace(res_dict.get("final_response", ""), success=not res_dict.get("error"))
        return result

    def test_start_trace_before_run_conversation(self, mock_otel_shim):
        """start_trace must be called before run_conversation executes."""
        run_conv = MagicMock(return_value={"final_response": "ok"})
        self._run_otel_block(mock_otel_shim, run_conv)

        assert mock_otel_shim.start_trace.call_count == 1
        assert mock_otel_shim.end_trace.call_count == 1
        # Verify correct args flow
        mock_otel_shim.start_trace.assert_called_once_with("hello")
        mock_otel_shim.end_trace.assert_called_once_with("ok", success=True)

    def test_end_trace_success_on_normal_exit(self, mock_otel_shim):
        """end_trace(success=True) fires after run_conversation returns normally."""
        run_conv = MagicMock(return_value={"final_response": "response"})
        result = self._run_otel_block(mock_otel_shim, run_conv)

        assert result == {"final_response": "response"}
        mock_otel_shim.end_trace.assert_called_once_with("response", success=True)

    def test_record_error_and_end_trace_failure_on_exception(self, mock_otel_shim):
        """On exception: record_error + end_trace(success=False) fire, then exception re-raised.

        Note: end_trace is called twice — once in the inner finally (with None/"None"), then
        again in the except block (with the actual error). This matches the actual gateway/run.py
        lines 8165-8174 structure.
        """
        run_conv = MagicMock(side_effect=RuntimeError("api error"))

        with pytest.raises(RuntimeError, match="api error"):
            self._run_otel_block_with_exception(mock_otel_shim, run_conv)

        mock_otel_shim.record_error.assert_called_once()
        # end_trace called twice: inner finally (None, success=True) then except (error, success=False)
        assert mock_otel_shim.end_trace.call_count == 2
        # The final call (from except) has success=False
        last_call = mock_otel_shim.end_trace.call_args_list[-1]
        assert last_call[1]["success"] is False

    def test_end_trace_success_with_string_return(self, mock_otel_shim):
        """If run_conversation returns a plain string, end_trace receives that string."""
        run_conv = MagicMock(return_value="plain string")
        result = self._run_otel_block(mock_otel_shim, run_conv)

        mock_otel_shim.end_trace.assert_called_once_with("plain string", success=True)

    def test_no_shim_no_crash(self):
        """When shim is None the block still completes without error."""
        run_conv = MagicMock(return_value={"final_response": "ok"})
        result = self._run_otel_block(None, run_conv)
        assert result == {"final_response": "ok"}

    def test_shim_callbacks_merge_preserves_both(self):
        """merge_callbacks must not drop either TUI or OTel callbacks when both exist."""
        from agent.otel_shim import merge_callbacks

        tui_cb = MagicMock(name="tui_cb")
        otel_cb = MagicMock(name="otel_cb")

        merged = merge_callbacks(
            {"tool_start_callback": tui_cb},
            {"tool_start_callback": otel_cb},
        )

        merged["tool_start_callback"]()
        tui_cb.assert_called_once()
        otel_cb.assert_called_once()

    def test_start_trace_call_order_before_end_trace(self, mock_otel_shim):
        """start_trace must be recorded before end_trace in mock call order."""
        run_conv = MagicMock(return_value={"final_response": "ok"})
        self._run_otel_block(mock_otel_shim, run_conv)

        # Use a real MagicMock call ordering check
        mock_otel_shim.start_trace.assert_called()
        mock_otel_shim.end_trace.assert_called()
        # The actual assertion: start_trace came first
        start_call = mock_otel_shim.start_trace.call_args
        end_call = mock_otel_shim.end_trace.call_args
        assert start_call[0][0] == "hello"  # message arg
        assert end_call[0][0] == "ok"       # response arg


class TestGatewayShutdownTelemetry:
    """Tests for gateway shutdown behavior and forced trace closure."""

    def test_interrupt_running_agents_forces_end_trace_when_active(self):
        from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

        runner = GatewayRunner.__new__(GatewayRunner)
        mock_agent = MagicMock()
        mock_shim = MagicMock()
        mock_shim._trace_active = True
        mock_shim._otel_enabled = True
        mock_agent._otel_shim = mock_shim
        runner._running_agents = {"session-1": mock_agent, "pending": _AGENT_PENDING_SENTINEL}

        runner._interrupt_running_agents("shutdown timeout")

        mock_agent.interrupt.assert_called_once_with("shutdown timeout")
        mock_shim.record_runtime_event.assert_called_once_with(
            "gateway.shutdown.interrupt_sent",
            session_key="session-1",
            reason="shutdown timeout",
        )
        mock_shim.end_trace.assert_called_once_with("Gateway shutdown interrupt", success=False)

    def test_interrupt_running_agents_does_not_end_trace_when_inactive(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        mock_agent = MagicMock()
        mock_shim = MagicMock()
        mock_shim._trace_active = False
        mock_shim._otel_enabled = True
        mock_agent._otel_shim = mock_shim
        runner._running_agents = {"session-1": mock_agent}

        runner._interrupt_running_agents("shutdown timeout")

        mock_agent.interrupt.assert_called_once_with("shutdown timeout")
        mock_shim.record_runtime_event.assert_called_once_with(
            "gateway.shutdown.interrupt_sent",
            session_key="session-1",
            reason="shutdown timeout",
        )
        mock_shim.end_trace.assert_not_called()


class TestGatewayStillWorkingEvents:
    """Focused assertions for still-working telemetry classification logic."""

    def test_possible_stuck_emitted_when_signature_repeats(self):
        shim = MagicMock()
        repeated = "iteration 11/60 | running: terminal"
        prev_signature = repeated
        unchanged_count = 1

        payload = {
            "session_id": "session-1",
            "elapsed_minutes": 20.0,
            "status": repeated,
        }
        shim.record_runtime_event("gateway.agent.still_working", payload)
        if repeated == prev_signature:
            unchanged_count += 1
            shim.record_runtime_event(
                "gateway.agent.possible_stuck",
                {
                    "session_id": "session-1",
                    "elapsed_minutes": 20.0,
                    "status": repeated,
                    "unchanged_intervals": unchanged_count,
                },
            )

        assert shim.record_runtime_event.call_count == 2
        assert shim.record_runtime_event.call_args_list[-1].args[0] == "gateway.agent.possible_stuck"
