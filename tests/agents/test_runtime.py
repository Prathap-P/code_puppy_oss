"""Tests for logging behavior in code_puppy.agents._runtime."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from code_puppy.agents import _runtime
from code_puppy.callbacks import _callbacks, clear_callbacks


class ScriptedPydanticAgent:
    """Pydantic-agent stand-in that raises a scripted, unexpected exception."""

    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def run(self, prompt: Any, **kwargs: Any) -> Any:
        history = kwargs.get("message_history")
        self.calls.append(
            {
                "prompt": prompt,
                "message_history": list(history) if isinstance(history, list) else history,
            }
        )
        if not self._outcomes:
            raise AssertionError("Unexpected extra pydantic_agent.run() call")

        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class DummyAgent:
    """Runtime-compatible agent shell; no actual model/provider involved."""

    name = "dummy-agent"

    def __init__(self, pydantic_agent: ScriptedPydanticAgent) -> None:
        self._code_generation_agent = pydantic_agent
        self._message_history = ["already-started"]
        self._mcp_servers: list[Any] = []

    def get_model_name(self) -> str:
        return "dummy-model"

    def get_full_system_prompt(self) -> str:
        return "unused because message history is non-empty"


@pytest.fixture(autouse=True)
def isolated_runtime_callbacks(monkeypatch: pytest.MonkeyPatch):
    """Keep global callback state from leaking into or out of these tests."""
    snapshot = {phase: list(callbacks) for phase, callbacks in _callbacks.items()}
    clear_callbacks()
    monkeypatch.setattr(_runtime, "sigint_fallback_cancels", lambda: True)
    monkeypatch.setattr(_runtime, "get_enable_streaming", lambda: False)
    monkeypatch.setattr(_runtime, "should_render_fallback", lambda *_, **__: False)

    yield

    clear_callbacks()
    for phase, callbacks in snapshot.items():
        _callbacks[phase].extend(callbacks)


def test_module_level_logger_uses_module_name() -> None:
    """The module must define a module-level logger, not an inline import."""
    assert isinstance(_runtime.logger, logging.Logger)
    assert _runtime.logger.name == _runtime.__name__


async def test_unexpected_exception_logs_error_via_logger(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unexpected exception during a run must produce a logger.error record."""
    original = RuntimeError("kaboom")
    pydantic_agent = ScriptedPydanticAgent(original)
    agent = DummyAgent(pydantic_agent)

    # Silence the terminal-facing diagnostics call so the test only asserts
    # on the logging side-effect, not on emit_info's rich-rendering machinery.
    monkeypatch.setattr(_runtime, "emit_exception_diagnostics", lambda *a, **k: None)

    with caplog.at_level(logging.ERROR, logger=_runtime.__name__):
        with pytest.raises(RuntimeError, match="kaboom"):
            await _runtime.run_with_mcp(agent, "hello")

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(r.name == _runtime.__name__ for r in error_records)
    assert any("kaboom" in r.getMessage() or r.exc_info for r in error_records)
