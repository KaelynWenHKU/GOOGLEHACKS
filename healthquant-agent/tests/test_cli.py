"""Terminal behavior tests; no live API calls."""
from io import StringIO
from unittest.mock import MagicMock
import pytest
from agent.cli import TerminalAgent


def terminal(sample=True, answers=()):
    output = StringIO()
    iterator = iter(answers)
    app = TerminalAgent(sample=sample, output=output, read=lambda prompt: next(iterator))
    return app, output


def test_sample_questions_and_commands_never_call_live_services(monkeypatch):
    app, out = terminal()
    monkeypatch.setattr(app, "collect", lambda: pytest.fail("live collection"))
    monkeypatch.setattr(app, "generate", lambda *a: pytest.fail("live generation"))
    for line in ["/help", "/status", "/refresh", "/regime", "/catalysts", "What is the regime?"]:
        app.dispatch(line)
    assert "FICTIONAL" in out.getvalue()
    assert "not measured performance" in out.getvalue()


def test_unknown_and_empty_commands_do_not_prompt_or_call():
    app, out = terminal(sample=False)
    app.collect = MagicMock()
    app.dispatch("")
    app.dispatch("/unknown")
    app.dispatch("!rm file")
    app.collect.assert_not_called()
    assert "Unknown" in out.getvalue()


def test_live_consent_decline_prevents_calls():
    app, _ = terminal(sample=False, answers=["n"])
    app.collect = MagicMock()
    app.dispatch("hello")
    app.collect.assert_not_called()


def test_live_consent_runs_once_and_shows_actual_trace():
    app, out = terminal(sample=False, answers=["y"])
    app.collect = MagicMock(return_value={"mode": "Live data", "loaded_at": "now"})
    app.generate = MagicMock(return_value={"text": "Evidence-based result", "tool_calls": ["get_current_regime"]})
    app.dispatch("hello")
    app.collect.assert_called_once()
    app.generate.assert_called_once()
    assert "get_current_regime" in out.getvalue()


def test_export_never_overwrites_and_clear_removes_last_result(tmp_path):
    app, out = terminal()
    app.dispatch("hello")
    path = tmp_path / "brief.md"
    app.dispatch(f'/export "{path}"')
    original = path.read_text()
    app.dispatch(f'/export "{path}"')
    assert path.read_text() == original
    assert "exists" in out.getvalue()
    app.dispatch("/clear")
    assert app.last_brief is None and app.evidence is None


def test_provider_error_and_terminal_escape_are_not_emitted():
    app, out = terminal(sample=False, answers=["y"])
    app.collect = MagicMock(side_effect=RuntimeError("SECRET_TOKEN"))
    app.dispatch("hello")
    assert "SECRET_TOKEN" not in out.getvalue()
    app.say("text\x1b[2J\x00")
    assert "\x1b" not in out.getvalue() and "\x00" not in out.getvalue()


def test_eof_exits_cleanly_without_network():
    def eof(_prompt):
        raise EOFError
    out = StringIO()
    TerminalAgent(output=out, read=eof).chat()
    assert "Goodbye" in out.getvalue()


def test_failed_refresh_does_not_keep_old_evidence():
    app, _ = terminal(sample=False, answers=["yes"])
    app.evidence = {"loaded_at": "old"}
    app.last_brief = "old brief"
    app.collect = MagicMock(side_effect=RuntimeError("do not display"))
    app.dispatch("/refresh")
    assert app.evidence is None and app.last_brief is None and app.failed


def test_question_limit_checked_before_any_call():
    app, out = terminal(sample=False)
    app.collect = MagicMock()
    app.dispatch("x" * 4001)
    app.collect.assert_not_called()
    assert "4000" in out.getvalue()


def test_launcher_sample_subprocess_from_another_directory(tmp_path):
    import subprocess
    from pathlib import Path
    launcher = Path(__file__).resolve().parents[2] / "healthquant"
    result = subprocess.run([str(launcher), "chat", "--sample"],
        input="/refresh\n/catalysts\nhello\n/quit\n", text=True,
        capture_output=True, cwd=tmp_path, timeout=20)
    assert result.returncode == 0
    assert "FICTIONAL" in result.stdout and "Example Therapeutics" in result.stdout


def test_export_refuses_symlink(tmp_path):
    app, _ = terminal()
    app.dispatch("hello")
    target = tmp_path / "existing.md"
    target.write_text("original")
    link = tmp_path / "alias.md"
    link.symlink_to(target)
    app.dispatch(f'/export "{link}"')
    assert target.read_text() == "original"
