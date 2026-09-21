"""Offline UI and agent tests: real Streamlit/ADK with fake external services."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from agent import agent as runtime
from agent import tools
from agent.prompts import REQUIRED_SECTIONS, validate_brief
from ui.sample_data import sample_snapshot

APP = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"


def test_sample_dashboard_renders_all_panels_without_services(monkeypatch):
    monkeypatch.setattr(tools, "get_db", lambda: pytest.fail("Sample mode accessed MongoDB"))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.subheader) == 4
    assert "SAMPLE DATA" in app.warning[0].value
    assert "neutral" == app.session_state["snapshot"]["regime"]["regime_label"]


def test_live_switch_clears_sample_and_shows_missing_credentials(monkeypatch):
    for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "MONGODB_URI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_GENAI_USE_VERTEXAI"]:
        monkeypatch.delenv(key, raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.radio[0].set_value("Live data").run()
    assert not app.exception
    assert "snapshot" not in app.session_state
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["snapshot"]["regime"]["status"] == "unavailable"
    assert next(b for b in app.button if b.label == "Generate Gemini brief").disabled
    assert not any("SAMPLE DATA" in item.value for item in app.warning)


def test_provider_errors_do_not_leak_secrets(monkeypatch):
    def broken():
        raise RuntimeError("mongodb://secret-password@host")
    monkeypatch.setattr(tools, "get_db", broken)
    assert "secret-password" not in str(tools.collect_evidence())


def test_current_regime_preserves_date_and_state_order(monkeypatch):
    collection = MagicMock()
    doc = sample_snapshot()["regime"]
    doc["date"] = datetime(2024, 1, 1)
    doc["state_label_map"] = {"0": "neutral", "1": "risk-on", "2": "catalyst-fear"}
    collection.find_one.return_value = doc
    monkeypatch.setattr(tools, "get_db", lambda: {"regime_states": collection})
    result = tools.get_current_regime()
    assert result["stale"]
    assert result["date"].startswith("2024-01-01")
    assert result["state_label_map"]["0"] == "neutral"


def test_snapshot_tools_enforce_matching_evidence():
    evidence = sample_snapshot()
    trace = []
    regime, analogues, catalysts = runtime.snapshot_tools(evidence, trace)
    assert analogues([0] * 8)["status"] == "unavailable"
    observed = regime()
    assert analogues(observed["feature_vector"])["status"] == "ok"
    observed["feature_vector"][0] = 999
    assert evidence["regime"]["feature_vector"][0] != 999
    assert catalysts()["status"] == "ok"
    assert trace == ["get_current_regime", "find_historical_analogues", "get_upcoming_catalysts"]


def test_adk_runner_uses_real_tools_and_validates_final_response(monkeypatch):
    """Run the installed ADK's session, event loop and function dispatch offline."""
    from google.adk.agents import Agent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class ScriptedModel(BaseLlm):
        model: str = "offline-test"
        step: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            sequence = [
                types.Part(function_call=types.FunctionCall(name="get_current_regime", args={})),
                types.Part(function_call=types.FunctionCall(name="find_historical_analogues", args={"feature_vector": sample_snapshot()["regime"]["feature_vector"]})),
                types.Part(function_call=types.FunctionCall(name="get_upcoming_catalysts", args={})),
                types.Part(text="\n".join(header + "\nTest evidence." for header in REQUIRED_SECTIONS)),
            ]
            part = sequence[self.step]
            self.step += 1
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    def factory(evidence, trace):
        return Agent(name="healthquant", model=ScriptedModel(),
                     tools=runtime.snapshot_tools(evidence, trace))
    monkeypatch.setattr(runtime, "create_agent", factory)
    evidence = sample_snapshot()
    evidence["mode"] = "Live data"  # Test fixture only, never the production UI path.
    result = asyncio.run(runtime.run_agent_async("Explain the regime", evidence))
    assert not validate_brief(result["text"])
    assert len(result["tool_calls"]) == 3


def test_agent_rejects_sample_evidence(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-dummy")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    with pytest.raises(ValueError, match="Sample data"):
        runtime.create_agent(sample_snapshot())


def test_generate_button_is_only_trigger_and_refresh_invalidates_brief(monkeypatch):
    snapshot = sample_snapshot()
    snapshot["mode"] = "Live data"
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-dummy")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setattr(tools, "collect_evidence", lambda: deepcopy(snapshot))
    generate = MagicMock(return_value={"text": "Test Gemini response", "model": "offline-test", "tool_calls": []})
    monkeypatch.setattr(runtime, "run_agent_result", generate)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.radio[0].set_value("Live data").run()
    next(b for b in app.button if b.label == "Refresh evidence").click().run()
    generate.assert_not_called()
    next(b for b in app.button if b.label == "Generate Gemini brief").click().run()
    generate.assert_called_once()
    assert app.session_state["brief"]["text"] == "Test Gemini response"
    app.run()
    generate.assert_called_once()
    next(b for b in app.button if b.label == "Refresh evidence").click().run()
    assert "brief" not in app.session_state
    assert not app.exception


def test_agent_provider_error_is_safe(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("secret-api-key")
    monkeypatch.setattr(runtime, "run_agent_async", fail)
    response = runtime.run_agent_result("Question", {})
    assert response["status"] == "unavailable"
    assert "secret-api-key" not in str(response)
