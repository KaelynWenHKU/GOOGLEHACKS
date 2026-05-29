"""
agent/agent.py
===============
Google ADK agent setup and run loop for the HealthQuant investment agent.

Configures the Gemini-powered agent with:
  - The system prompt from agent/prompts.py
  - Three tools from agent/tools.py
  - MongoDB Atlas as the persistent memory layer (via MCP server)

Usage:
    from agent.agent import run_agent
    response = run_agent("What is today's healthcare sector regime?")

Or run interactively:
    python agent/agent.py

Prerequisites:
  - GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS env vars set
  - MongoDB Atlas running and populated (run seed_historical.py first)
  - HMM model trained (run hmm/train.py first)
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from agent.prompts import SYSTEM_PROMPT, DISCLAIMER, build_user_query, validate_brief
from agent.tools import get_current_regime, find_historical_analogues, get_upcoming_catalysts

load_dotenv()
logger = logging.getLogger(__name__)

# Google Cloud project config — required for Gemini via Vertex AI
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

# Gemini model to use — gemini-2.0-flash is recommended for ADK (fast + capable)
GEMINI_MODEL = "gemini-2.0-flash"


def create_agent():
    """
    Instantiate and return the Google ADK HealthQuant agent.

    Registers the three tools and configures the agent with the system prompt.
    The agent uses Gemini via Vertex AI (configured by GOOGLE_CLOUD_PROJECT).

    Returns:
        Configured ADK Agent object ready to receive user messages.

    Raises:
        ValueError: If required Google Cloud environment variables are not set.
        ImportError: If google-adk is not installed.
    """
    # TODO: from google.adk.agents import Agent  (or equivalent ADK import path)
    # TODO: validate GOOGLE_CLOUD_PROJECT is set
    # TODO: create Agent(
    #     model=GEMINI_MODEL,
    #     system_prompt=SYSTEM_PROMPT,
    #     tools=[get_current_regime, find_historical_analogues, get_upcoming_catalysts],
    #     project=GOOGLE_CLOUD_PROJECT,
    #     location=GOOGLE_CLOUD_REGION,
    # )
    # TODO: return agent
    raise NotImplementedError


def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    Run the HealthQuant agent for a single user query.

    Wraps the ADK agent invocation with:
      - Query formatting (build_user_query)
      - Tool call logging (when verbose=True)
      - Output validation (validate_brief)
      - Disclaimer appending

    Args:
        user_message: The user's natural language query.
        verbose: If True, print tool calls to stdout as they happen.
                 Useful for the demo video (shows agent reasoning in real time).

    Returns:
        The full investment brief text, including the disclaimer.
    """
    # TODO: formatted_query = build_user_query(user_message)
    # TODO: agent = create_agent()
    # TODO: response = agent.run(formatted_query)  (ADK run method)
    # TODO: if verbose, log each tool call and its result
    # TODO: missing_sections = validate_brief(response.text)
    # TODO: if missing_sections, log a warning
    # TODO: return response.text + DISCLAIMER
    raise NotImplementedError


def run_interactive() -> None:
    """
    Launch an interactive REPL loop for testing the agent in the terminal.

    Type a message and press Enter to query the agent.
    Type "quit" or "exit" to stop.
    """
    print("HealthQuant Agent — Interactive Mode")
    print("Type your query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        print("\nHealthQuant: Analysing...\n")
        # TODO: response = run_agent(user_input, verbose=True)
        # TODO: print(response)
        print("[Agent not yet implemented — run after completing agent/tools.py]\n")


if __name__ == "__main__":
    run_interactive()
