# Terminal Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver the user-approved interactive HealthQuant research terminal, usable inside VS Code.

**Architecture:** A standard-library command loop wraps existing read-only evidence tools and isolated Gemini sessions. Explicit confirmation gates possibly paid refresh/generation. Offline sample fixtures never call providers. Each question is independent; current evidence persists only until refresh/clear/exit. A local launcher selects the repository virtual environment.

**Tech Stack:** Python argparse, existing Google ADK/Gemini backend, pytest.

### Tasks

1. Add failing tests in `healthquant-agent/tests/test_cli.py` for offline chat, commands, cost consent, sanitized errors and exclusive Markdown export.
2. Create `healthquant-agent/agent/cli.py` with `chat`, `ask`, `status` and slash commands `/help`, `/status`, `/refresh`, `/regime`, `/catalysts`, `/analogues`, `/export PATH`, `/clear`, `/quit`. Reject unknown commands and empty paid input. Do not execute shell commands. `/backtest` supplies the existing CLI instruction, not automatic training/writes.
3. Create root `healthquant` launcher; preserve existing dashboard and backend contract. Document launch, independent-query semantics, sample/live differences and remaining connectivity constraints in README.
4. Run focused tests, full pytest, sample subprocess and compile/whitespace checks. Scan secrets; commit named files and push verified changes.
5. Inspect Atlas's exact project and list. Confirm at action time before granting a new /32 address; never weaken TLS or allow all addresses. Recheck MongoDB after any approved change.

Research/decision notes: Python cmd documentation warns blank lines repeat the last command, so use a small explicit loop without that behavior (https://docs.python.org/3/library/cmd.html). argparse supplies command parsing (https://docs.python.org/3/library/argparse.html). ADK session semantics are documented at https://adk.dev/sessions/; v1 reuses isolated existing backend sessions, not persistent chat memory.

Alternative frameworks and remote APIs deferred to avoid extra dependencies/deployment. User approved this terminal direction; unavailable superpowers execution skill is replaced by direct test-first execution. Local shaping files are temporary and will be removed after completion.

## Verification

- Initial CLI tests failed because the module did not exist.
- All 11 CLI tests and all 182 repository tests passed; one existing joblib physical-core warning remains.
- Repository launcher tested in an actual subprocess from a different working directory, plus local status and fictional one-shot smoke tests.
- Compile and whitespace checks passed. No live Gemini generation was run.
- Atlas browser inspection reached the correct organization/project list; UI operation paused when the user switched browser tabs. No IP permissions were modified. TLS connectivity remains unresolved pending allowed-source verification.
