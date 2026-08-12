"""Tests for module-level logging in agent_tools' session-history I/O helpers."""

import logging
import pickle

from code_puppy.tools import agent_tools


def test_module_logger_defined():
    """The module must define a standard logging.Logger at module scope."""
    assert isinstance(agent_tools.logger, logging.Logger)
    assert agent_tools.logger.name == "code_puppy.tools.agent_tools"


def test_load_session_history_logs_warning_on_corrupted_pickle(
    tmp_path, monkeypatch, caplog
):
    """When the session pickle file is corrupted, _load_session_history must
    log a warning record containing the file path and return an empty list
    instead of raising."""
    session_id = "corrupted-session"

    monkeypatch.setattr(
        agent_tools, "_get_subagent_sessions_dir", lambda: tmp_path
    )

    pkl_path = tmp_path / f"{session_id}.pkl"
    pkl_path.write_bytes(b"not a valid pickle stream")

    with caplog.at_level(logging.WARNING, logger=agent_tools.logger.name):
        result = agent_tools._load_session_history(session_id)

    assert result == []

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected at least one logger.warning record"
    joined = " ".join(r.getMessage() for r in warning_records)
    assert str(pkl_path) in joined


def test_save_session_history_logs_warning_on_metadata_update_failure(
    tmp_path, monkeypatch, caplog
):
    """When updating the existing .txt metadata file fails (e.g. invalid
    JSON already on disk), _save_session_history must log a warning record
    containing the file path and caught exception, then continue without
    raising."""
    session_id = "metadata-failure-session"

    monkeypatch.setattr(
        agent_tools, "_get_subagent_sessions_dir", lambda: tmp_path
    )

    # Pre-create a corrupted metadata file so json.load() raises inside the
    # except block we're targeting.
    txt_path = tmp_path / f"{session_id}.txt"
    txt_path.write_text("not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=agent_tools.logger.name):
        agent_tools._save_session_history(
            session_id=session_id,
            message_history=[],
            agent_name="test-agent",
            initial_prompt=None,
        )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected at least one logger.warning record"
    joined = " ".join(r.getMessage() for r in warning_records)
    assert str(txt_path) in joined

    # Fallback behavior preserved: pickle file is still written successfully.
    pkl_path = tmp_path / f"{session_id}.pkl"
    assert pkl_path.exists()
    with open(pkl_path, "rb") as f:
        assert pickle.load(f) == []
