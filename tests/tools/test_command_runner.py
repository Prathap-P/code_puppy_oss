"""Tests for module-level logging in command_runner's process-kill fallbacks."""

import logging
import sys
from unittest.mock import MagicMock

import pytest

from code_puppy.tools import command_runner


def test_module_logger_defined():
    """The module must define a standard logging.Logger at module scope."""
    assert isinstance(command_runner.logger, logging.Logger)
    assert command_runner.logger.name == "code_puppy.tools.command_runner"


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="exercises the POSIX killpg fallback path"
)
def test_kill_process_group_logs_warning_on_killpg_fallback(monkeypatch, caplog):
    """When os.getpgid/killpg fails and the direct proc.kill() fallback also
    raises, a logger.warning record must be emitted for each failure, and
    _kill_process_group must still return normally (no crash)."""
    proc = MagicMock()
    proc.pid = 424242
    proc.poll.return_value = None  # process still "running" throughout

    def fake_getpgid(pid):
        raise OSError("no such process group")

    proc.kill.side_effect = OSError("kill failed too")

    monkeypatch.setattr(command_runner.os, "getpgid", fake_getpgid)

    with caplog.at_level(logging.WARNING, logger=command_runner.logger.name):
        command_runner._kill_process_group(proc)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected at least one logger.warning record"
    joined = " ".join(r.getMessage() for r in warning_records)
    assert "424242" in joined


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="exercises the Windows kill fallback path"
)
def test_kill_process_group_logs_warning_on_windows_fallback(monkeypatch, caplog):
    """On Windows, if proc.kill() raises in the fallback path, a warning must
    be logged and _kill_process_group must not raise."""
    proc = MagicMock()
    proc.pid = 13131313
    proc.poll.return_value = None
    proc.kill.side_effect = OSError("kill failed")

    with caplog.at_level(logging.WARNING, logger=command_runner.logger.name):
        command_runner._kill_process_group(proc)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected at least one logger.warning record"
