"""Gemini through Google ADK, with isolated sessions and bounded inference.

Only explicit user actions invoke this module. Evidence is captured once and
exposed to three read-only ADK tools so the UI and brief use the same snapshot.
"""
import asyncio
from copy import deepcopy
import logging
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from agent.prompts import SYSTEM_PROMPT, DISCLAIMER, build_user_query, validate_brief

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def gemini_configuration() -> dict:
    """Return public configuration status, never the key or credential path."""
    vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    ready = bool(project and "<" not in project) if vertex else bool(key and "<" not in key)
    return {"configured": ready, "provider": "Vertex AI" if vertex else "Gemini Developer API",
            "model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            "setup": ("Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION and configure Application Default Credentials."
                      if vertex else "Set GOOGLE_API_KEY in healthquant-agent/.env using a Google AI Studio key.")}


def snapshot_tools(evidence: dict, trace: list):
    """Expose captured evidence as ADK tools without re-querying paid services."""
    def get_current_regime() -> dict:
        """Read the latest available HMM prediction, timestamp and probabilities."""
        trace.append("get_current_regime")
        return deepcopy(evidence["regime"])

    def find_historical_analogues(feature_vector: list[float], top_k: int = 3) -> dict:
        """Read captured Atlas analogue matches for the regime feature vector."""
        if "get_current_regime" not in trace:
            return {"status": "unavailable", "error": "Call get_current_regime first."}
        if feature_vector != evidence["regime"].get("feature_vector") or not 1 <= top_k <= 3:
            return {"status": "unavailable", "error": "Use the returned feature vector and top_k from 1 to 3."}
        trace.append("find_historical_analogues")
        result = deepcopy(evidence["analogues"])
        if "analogues" in result:
            result["analogues"] = result["analogues"][:top_k]
        return result

    def get_upcoming_catalysts(days: int = 30) -> dict:
        """Read the next 30 days of stored FDA and Phase 3 events."""
        if "get_current_regime" not in trace:
            return {"status": "unavailable", "error": "Call get_current_regime first."}
        if days != 30:
            return {"status": "unavailable", "error": "This snapshot covers 30 days."}
        trace.append("get_upcoming_catalysts")
        return deepcopy(evidence["catalysts"])
    return [get_current_regime, find_historical_analogues, get_upcoming_catalysts]


def create_agent(evidence=None, trace=None):
    """Build the ADK agent with configurable Gemini Developer API or Vertex."""
    config = gemini_configuration()
    if not config["configured"]:
        raise ValueError(config["setup"])
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_REGION", "us-central1"))
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.genai import types
    from agent.tools import collect_evidence
    evidence = collect_evidence() if evidence is None else evidence
    if evidence.get("mode") != "Live data":
        raise ValueError("Sample data cannot be sent to the live Gemini agent")
    return Agent(
        name="healthquant", model=Gemini(model=config["model"],
            retry_options=types.HttpRetryOptions(attempts=1)),
        instruction=SYSTEM_PROMPT,
        tools=snapshot_tools(evidence, trace if trace is not None else []),
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2, max_output_tokens=1800,
            http_options=types.HttpOptions(timeout=30000)))


async def run_agent_async(user_message: str, evidence=None) -> dict:
    """Run an isolated ADK session, requiring tool evidence and final sections."""
    query = build_user_query(user_message)
    if evidence is None:
        if not gemini_configuration()["configured"]:
            raise ValueError(gemini_configuration()["setup"])
        from agent.tools import collect_evidence
        evidence = collect_evidence()
    trace = []
    agent = create_agent(evidence, trace)
    from google.adk.runners import InMemoryRunner
    from google.adk.agents.run_config import RunConfig
    from google.genai import types
    runner = InMemoryRunner(agent=agent, app_name="healthquant")
    session = await runner.session_service.create_session(app_name="healthquant", user_id=uuid4().hex)
    answer = ""
    try:
        async with asyncio.timeout(90):
            async for event in runner.run_async(
                user_id=session.user_id, session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=query)]),
                run_config=RunConfig(max_llm_calls=6),
            ):
                if event.is_final_response() and event.content:
                    answer = "\n".join(part.text for part in event.content.parts or []
                                       if part.text and not part.thought)
    finally:
        await runner.close()
    required = {"get_current_regime", "get_upcoming_catalysts"}
    if evidence and evidence["regime"].get("status") == "ok":
        required.add("find_historical_analogues")
    if not required.issubset(trace):
        raise ValueError("Gemini did not consult the required evidence tools. Try again.")
    missing = validate_brief(answer)
    if not answer or missing:
        raise ValueError("Gemini returned an incomplete brief. Try again.")
    return {"text": answer + DISCLAIMER, "tool_calls": trace,
            "model": gemini_configuration()["model"], "generated_by": "Gemini / Google ADK"}


def run_agent_result(user_message: str, evidence: dict) -> dict:
    """Synchronous UI entry point; sanitize any provider exception."""
    try:
        return asyncio.run(run_agent_async(user_message, evidence))
    except Exception as exc:
        logger.warning("Gemini request failed (%s)", type(exc).__name__)
        return {"error": "Gemini could not complete the brief. Check credentials, model access and quota, then retry.",
                "status": "unavailable"}


def run_agent(user_message: str, verbose: bool = True) -> str:
    """Compatibility entry point used by command-line consumers."""
    from agent.tools import collect_evidence
    if not gemini_configuration()["configured"]:
        raise ValueError(gemini_configuration()["setup"])
    result = run_agent_result(user_message, collect_evidence())
    if "error" in result:
        raise RuntimeError(result["error"])
    if verbose:
        logger.info("Tools consulted: %s", ", ".join(result["tool_calls"]))
    return result["text"]


def run_interactive() -> None:
    """Run the real Gemini agent from the terminal."""
    while True:
        try:
            query = input("HealthQuant question (quit to exit): ").strip()
            if query.lower() in {"quit", "exit", "q"}:
                break
            if query:
                print(run_agent(query))
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    run_interactive()
