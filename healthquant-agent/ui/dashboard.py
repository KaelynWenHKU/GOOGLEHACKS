"""HealthQuant research console. Run from any directory with Streamlit."""
from pathlib import Path
import sys

# Streamlit executes a file, so make sibling packages importable from either cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from agent.agent import gemini_configuration, run_agent_result
from agent.tools import collect_evidence
from ui.sample_data import sample_snapshot, SAMPLE_BRIEF

LABELS = {"risk-on": "Risk-on accumulation", "neutral": "Neutral / macro-driven",
          "catalyst-fear": "Catalyst-driven fear"}
CSS = """
<style>
.stApp { background: #f7f5ef; color: #172c39; }
.block-container { max-width: 1240px; padding-top: 2.5rem; }
h1 { font-family: Georgia, serif !important; letter-spacing: -0.04em; }
h2,h3 { font-family: Georgia, serif !important; }
[data-testid="stSidebar"] { background: #eae7de; }
[data-testid="stMetricValue"] { color: #172c39; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
button[kind="primary"] { background: #185c51; border-color: #185c51; }
[data-testid="stCaptionContainer"] { color: #566571; }
</style>
"""


def unavailable(data: dict) -> bool:
    """Render a panel's unavailable state without substituting sample values."""
    if data.get("status") != "ok":
        st.info(data.get("error", "Evidence unavailable. Refresh live data to try again."))
        return True
    return False


def render_regime_gauge(regime_data: dict) -> None:
    """Display state probabilities in the persisted model's actual label order."""
    st.subheader("01 / Market regime")
    if unavailable(regime_data):
        return
    st.caption("AS OF " + regime_data["date"][:10])
    st.caption("LATEST CLASSIFICATION")
    st.markdown("### " + LABELS.get(regime_data["regime_label"], "Unknown"))
    if regime_data.get("stale"):
        st.warning("This prediction is more than four days old. It is not a current market signal.")
    mapping = regime_data.get("state_label_map", {})
    valid_mapping = set(mapping) == {"0", "1", "2"} and set(mapping.values()) == set(LABELS)
    for i, probability in enumerate(regime_data["state_probs"]):
        label = LABELS[mapping[str(i)]] if valid_mapping else f"Model state {i}"
        st.progress(float(probability), text=f"{label} · {probability:.0%}")
    if not valid_mapping:
        st.caption("Semantic state mapping was not stored; raw state IDs are shown.")
    fear = regime_data.get("transition_10d", {}).get("to_catalyst_fear")
    if isinstance(fear, (int, float)) and 0 <= fear <= 1:
        st.metric("Fear-state probability in 10 trading days", f"{fear:.0%}")
        if fear > 0.30:
            st.warning("Risk flag: ten-day fear probability exceeds 30%.")
    else:
        st.caption("Ten-day transition forecast unavailable.")
    with st.expander("Inspect model inputs"):
        st.json(regime_data.get("feature_summary", {}))
    st.caption(regime_data.get("source", ""))


def render_catalyst_calendar(catalysts: dict) -> None:
    """Render a sortable, dated calendar using text-only dataframe cells."""
    st.subheader("02 / Catalyst calendar")
    st.caption("NEXT 30 CALENDAR DAYS")
    if unavailable(catalysts):
        return
    rows = []
    for event in catalysts.get("pdufa_events", []):
        rows.append({"Date": event["date"][:10], "Company": event.get("company"),
                     "Ticker": event.get("ticker"), "Event": event.get("drug"),
                     "Type": "PDUFA · " + event.get("review_type", "standard")})
    for event in catalysts.get("phase3_completions", []):
        rows.append({"Date": event["expected_date"][:10], "Company": event.get("company"),
                     "Ticker": event.get("ticker"), "Event": event.get("trial_id"),
                     "Type": "Phase 3 completion"})
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Date"), hide_index=True, width="stretch")
    else:
        st.info("No matching events in the stored calendar. This does not establish that there are no upcoming events.")
    st.caption(catalysts.get("coverage_note", "Stored records only."))
    st.caption("Phase 3 completion dates are estimates, not guaranteed result-release dates.")


def render_analogue_cards(analogues_data: dict) -> None:
    """Show retrieval similarity and only available measured returns."""
    st.subheader("03 / Historical analogues")
    if unavailable(analogues_data):
        return
    matches = analogues_data.get("analogues", [])
    if not matches:
        st.info("No historical analogue matches are stored.")
    for match in matches:
        with st.container(border=True):
            left, right = st.columns([2, 1])
            with left:
                st.markdown("**" + str(match["date"])[:10] + "**")
                st.caption(LABELS.get(match.get("regime_label"), "Unclassified"))
            with right:
                score = match.get("similarity_score")
                st.metric("Similarity", f"{score:.0%}" if isinstance(score, (int, float)) else "—")
            actual = match.get("xlv_ret_10d_actual")
            st.caption(f"Observed XLV return, next 10 sessions: {actual:+.2%}" if isinstance(actual, (int, float))
                       else "Observed 10-day return unavailable / not yet verified.")
            st.write(match.get("description", ""))
    st.caption("Similarity measures retrieval relevance, not forecast confidence.")


def render_investment_brief(brief_text: str, last_updated: str) -> None:
    """Render model text as Markdown without permitting raw HTML."""
    st.markdown(brief_text)
    st.caption("Generated / example timestamp: " + last_updated)


def load_dashboard_data() -> dict:
    """Fetch real evidence only when the user explicitly requests a refresh."""
    return collect_evidence()


def setup_panel() -> dict:
    """Explain setup using presence checks, without exposing secret values."""
    config = gemini_configuration()
    with st.sidebar:
        st.caption("RESEARCH WORKSPACE")
        st.header("HealthQuant")
        st.write("Clinical catalysts.\nMarket context.\nOne evidence trail.")
        mode = st.radio("Data source", ["Sample data", "Live data"], key="mode")
        st.divider()
        st.caption("GEMINI CONNECTION")
        st.write(config["provider"])
        st.code(config["model"], language=None)
        st.caption("Configured (not verified)" if config["configured"] else "Not configured")
        with st.expander("Connection setup"):
            st.write(config["setup"])
            st.write("Live evidence also needs MONGODB_URI and populated collections. Analogue search needs VOYAGE_API_KEY and a compatible Atlas vector index.")
            for name in ["MONGODB_URI", "VOYAGE_API_KEY"]:
                value = os.getenv(name, "")
                st.caption(f"{name}: " + ("configured" if value and "<" not in value else "missing"))
            st.caption("Store credentials in healthquant-agent/.env. Never paste keys into a question.")
        st.divider()
        st.caption("Google Gemini · Google ADK\nMongoDB Atlas · Gaussian HMM")
    return {**config, "mode": mode}


def main() -> None:
    """Render the app with per-browser snapshots and explicit paid-call actions."""
    st.set_page_config(page_title="HealthQuant | Research desk", page_icon="◈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)  # Static app CSS only.
    config = setup_panel()
    mode = config["mode"]
    if st.session_state.get("active_mode") != mode:
        st.session_state.active_mode = mode
        st.session_state.pop("snapshot", None)
        st.session_state.pop("brief", None)
        if mode == "Sample data":
            st.session_state.snapshot = sample_snapshot()
    st.caption("HEALTHCARE INTELLIGENCE / RESEARCH DESK")
    st.title("Read the regime.\nUnderstand the catalysts.")
    st.write("A clinical and market evidence workspace for healthcare research.")
    if mode == "Sample data":
        st.warning("SAMPLE DATA · Fictional values, companies and returns. No live inference or Gemini call has run.")
    else:
        st.info("LIVE DATA · Reads your connected database. Missing services are shown explicitly.")
    left, right = st.columns([3, 1])
    with right:
        if st.button("Refresh evidence", width="stretch"):
            with st.spinner("Loading evidence…"):
                st.session_state.snapshot = sample_snapshot() if mode == "Sample data" else load_dashboard_data()
                st.session_state.pop("brief", None)
    snapshot = st.session_state.get("snapshot")
    with left:
        st.caption("Evidence snapshot: " + snapshot["loaded_at"] if snapshot else "Refresh evidence to load your connected data.")
    if snapshot is None:
        st.info("Connect MongoDB and refresh evidence. You can explore all four panels in Sample data.")
        return

    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1, st.container(border=True):
        render_regime_gauge(snapshot["regime"])
    with col2, st.container(border=True):
        render_catalyst_calendar(snapshot["catalysts"])
    st.divider()
    col3, col4 = st.columns([1, 1.2], gap="large")
    with col3:
        render_analogue_cards(snapshot["analogues"])
    with col4, st.container(border=True):
        st.subheader("04 / Research brief")
        if mode == "Sample data":
            st.caption("ILLUSTRATIVE TEXT · NOT GENERATED BY GEMINI")
            render_investment_brief(SAMPLE_BRIEF, snapshot["loaded_at"])
        else:
            with st.form("brief_form"):
                query = st.text_area("What would you like to understand?",
                    value="What does the latest regime imply, and which catalysts should I watch?",
                    max_chars=4000)
                submitted = st.form_submit_button("Generate Gemini brief", type="primary",
                    disabled=not config["configured"])
            st.caption("Explicit generation only · up to 6 model calls · 90-second timeout. Provider charges may apply.")
            if not config["configured"]:
                st.info(config["setup"])
            if submitted:
                st.session_state.pop("brief", None)
                if not query.strip():
                    st.warning("Enter a question first.")
                else:
                    with st.spinner("Gemini is consulting the evidence tools…"):
                        result = run_agent_result(query, snapshot)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state.brief = {**result, "generated_at": datetime.now(timezone.utc).isoformat()}
            brief = st.session_state.get("brief")
            if brief:
                render_investment_brief(brief["text"], brief["generated_at"])
                st.caption("Generated by " + brief["model"] + " through Google ADK")
                with st.expander("Evidence tool calls"):
                    for call in brief["tool_calls"]:
                        st.text(call)
                st.download_button("Download brief", brief["text"], "healthquant-brief.md", mime="text/markdown")
            else:
                st.caption("No Gemini brief has been generated for this evidence snapshot.")
    st.divider()
    st.download_button("Download evidence snapshot", json.dumps(snapshot, indent=2, allow_nan=False),
                       "healthquant-evidence.json", mime="application/json")
    st.caption("Educational research only. No validated historical performance is claimed. Past performance does not guarantee future results.")


if __name__ == "__main__":
    main()
