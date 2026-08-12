"""
Tests for CircuitBreaker logging behavior on the call() failure path.
"""

import logging

import pytest

from code_puppy.mcp_.circuit_breaker import CircuitBreaker


class TestCircuitBreakerCallLogging:
    """Test cases verifying CircuitBreaker.call() logs before re-raising."""

    @pytest.mark.asyncio
    async def test_call_logs_warning_and_reraises_on_failure(self, caplog):
        """CircuitBreaker.call() should log a warning capturing the exception,
        then still propagate that same exception to the caller.
        """
        breaker = CircuitBreaker(failure_threshold=5, success_threshold=2, timeout=60)

        async def failing_func():
            raise ValueError("boom")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="boom"):
                await breaker.call(failing_func)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("boom" in r.getMessage() for r in warning_records)
