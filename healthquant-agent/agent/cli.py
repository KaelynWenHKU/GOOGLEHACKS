"""Interactive research terminal: read-only tools, explicit API consent, no shell."""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

HELP = """/help                  Show commands
/status                Local configuration and snapshot status (no API calls)
/refresh               Refresh evidence (Live may call Voyage)
/regime /catalysts /analogues  Inspect current evidence snapshot
/export PATH.md        Save last brief; never overwrite existing files
/clear                 Forget snapshot and last brief
/backtest              Show the existing validation command
/quit                  Exit
Or type a research question. Each question is independent (no chat memory).
Live questions call Gemini; paid operations ask for confirmation.
No shell execution, trading, training or database writes are exposed."""


class TerminalAgent:
    """One in-memory research workspace, with injectable IO for offline tests."""
    def __init__(self, sample=False, output=None, read=input, yes=False):
        self.sample = sample
        self.output = output if output is not None else sys.stdout
        self.read = read
        self.yes = yes
        self.evidence = None
        self.last_brief = None
        self.failed = False

    def say(self, text):
        """Do not let database/model text inject terminal control sequences."""
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(text))
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", "", text)
        print(text, file=self.output, flush=True)

    def collect(self):
        from agent.tools import collect_evidence
        return collect_evidence()

    def generate(self, question, evidence):
        from agent.agent import run_agent_result
        return run_agent_result(question, evidence)

    def consent(self, operation):
        return self.sample or self.yes or self.read(
            f"{operation}; provider charges may apply. Continue? [y/N] "
        ).strip().lower() in {"y", "yes"}

    def refresh(self):
        # Discard the old snapshot first so a failed refresh cannot look fresh.
        self.evidence = None
        self.last_brief = None
        self.say("Loading evidence...")
        if self.sample:
            from ui.sample_data import sample_snapshot
            self.evidence = sample_snapshot()
        else:
            self.evidence = self.collect()
        self.say(f"Snapshot: {self.evidence['loaded_at']} ({self.evidence['mode']})")

    def dispatch(self, line):
        """Handle an explicit user action; never evaluate it as shell/code."""
        line = line.strip()
        self.failed = False
        if not line:
            return False
        try:
            if line in {"/quit", "quit", "exit"}:
                return True
            if line == "/help":
                self.say(HELP)
            elif line == "/status":
                from agent.agent import gemini_configuration
                config = gemini_configuration()
                self.say("FICTIONAL SAMPLE — no live calls" if self.sample else "LIVE — database connectivity not checked by /status")
                self.say(f"Gemini configured: {config['configured']} | {config['model']}")
                self.say(f"Snapshot: {self.evidence['loaded_at']}" if self.evidence else "No evidence loaded. Use /refresh.")
            elif line == "/clear":
                self.evidence = self.last_brief = None
                self.say("Local evidence and brief cleared.")
            elif line == "/refresh":
                if self.consent("Read MongoDB and, when available, query Voyage analogues"):
                    self.refresh()
            elif line in {"/regime", "/catalysts", "/analogues"}:
                if self.evidence is None:
                    self.say("No evidence loaded. Use /refresh first.")
                else:
                    if self.sample:
                        self.say("FICTIONAL SAMPLE")
                    self.say(f"Evidence as of: {self.evidence['loaded_at']}")
                    self.say(json.dumps(self.evidence[line[1:]], ensure_ascii=False, indent=2, default=str))
            elif line.startswith("/export"):
                parts = shlex.split(line)
                if len(parts) != 2 or parts[0] != "/export" or Path(parts[1]).suffix.lower() != ".md":
                    self.say('Usage: /export "path/to/brief.md"')
                elif not self.last_brief:
                    self.say("No successful brief to export.")
                else:
                    # Exclusive creation refuses existing files and symlinks.
                    with Path(parts[1]).open("x", encoding="utf-8") as handle:
                        handle.write(self.last_brief)
                    self.say("Brief exported.")
            elif line == "/backtest":
                self.say("From healthquant-agent/: python -m scripts.validate_backtest --start 2020 --end 2024\nRequires audited stored predictions; this command has not been run.")
            elif line.startswith(("/", "!")):
                self.say("Unknown command. Use /help. Shell execution is disabled.")
            elif len(line) > 4000:
                self.say("Question must contain at most 4000 characters.")
            elif self.consent("Generate a Gemini brief; missing evidence will first be refreshed"):
                self.last_brief = None
                if self.evidence is None:
                    self.refresh()
                self.say(f"Using evidence snapshot: {self.evidence['loaded_at']}; /refresh updates it.")
                if self.sample:
                    from ui.sample_data import SAMPLE_BRIEF
                    result = {"text": "FICTIONAL SAMPLE — static example, not a Gemini answer.\n\n" + SAMPLE_BRIEF}
                else:
                    self.say("Generating Gemini brief (up to 6 model calls; 90-second inference timeout)...")
                    result = self.generate(line, self.evidence)
                if "error" in result:
                    self.failed = True
                    self.say("Gemini unavailable. Check credentials, model access and quota; no brief saved.")
                else:
                    self.last_brief = (f"# HealthQuant research brief\n\nMode: {self.evidence['mode']}\n"
                        f"Evidence snapshot: {self.evidence['loaded_at']}\n\n" + result["text"])
                    self.say(self.last_brief)
                    if result.get("tool_calls"):
                        self.say("Tools actually consulted: " + ", ".join(result["tool_calls"]))
        except FileExistsError:
            self.failed = True
            self.say("File already exists; choose a new filename. Nothing overwritten.")
        except (EOFError, KeyboardInterrupt, StopIteration):
            self.say("Cancelled.")
        except Exception as exc:
            self.failed = True
            self.say(f"Operation failed ({type(exc).__name__}). Check configuration/connectivity; credentials are not displayed.")
        return False

    def chat(self):
        self.say("HealthQuant terminal | " + ("FICTIONAL SAMPLE" if self.sample else "LIVE"))
        self.say(HELP)
        while True:
            try:
                line = self.read("HealthQuant > ")
            except (EOFError, KeyboardInterrupt):
                self.say("Goodbye.")
                return
            if self.dispatch(line):
                return


def main(argv=None):
    """Entry point usable from Terminal or the VS Code integrated terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["chat", "ask", "status"], nargs="?", default="chat")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--sample", action="store_true", help="Offline fictional fixtures; no paid calls")
    parser.add_argument("--yes", action="store_true", help="Explicitly approve API calls for this invocation")
    args = parser.parse_args(argv)
    if args.command == "ask" and not args.question:
        parser.error("ask requires a quoted question")
    app = TerminalAgent(sample=args.sample, yes=args.yes)
    if args.command == "chat":
        app.chat()
    else:
        app.dispatch("/status" if args.command == "status" else args.question)
    return 1 if app.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
