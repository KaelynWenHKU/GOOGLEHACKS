"""
ui/dashboard.py
================
Streamlit demo dashboard for the HealthQuant Agent.

Four-panel layout (Section 13):
  Panel 1 (top-left):    Regime Gauge — colour-coded indicator + state probability bars
  Panel 2 (top-right):   Upcoming Catalysts Calendar — next 30 days of PDUFA/Phase 3 events
  Panel 3 (bottom-left): Historical Analogue Cards — 3 most similar past market periods
  Panel 4 (bottom-right): Investment Brief — Gemini-generated analysis with Regenerate button

Run with:
    streamlit run ui/dashboard.py
"""

import logging
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# Colour mapping for each regime state
REGIME_COLOURS = {
    "risk-on": "#22c55e",        # green
    "neutral": "#f59e0b",         # amber
    "catalyst-fear": "#ef4444",  # red
}

# Human-readable labels for the dashboard
REGIME_DISPLAY_NAMES = {
    "risk-on": "Risk-On Accumulation",
    "neutral": "Neutral / Macro-Driven",
    "catalyst-fear": "Catalyst-Driven Fear",
}


def render_regime_gauge(regime_data: dict) -> None:
    """
    Render Panel 1: the large regime indicator with state probability bars.

    Args:
        regime_data: Output dict from agent tool get_current_regime().
                     Keys: regime_label, state_probs, transition_10d, date.
    """
    # TODO: display large coloured metric using st.metric or custom HTML
    # TODO: show "RISK-ON" / "NEUTRAL" / "CATALYST-FEAR" in large bold text
    # TODO: add three horizontal bars for state probabilities using st.progress
    # TODO: display transition_10d["to_catalyst_fear"] as a fear probability gauge
    # TODO: show "Regime since: [date]" if regime unchanged from yesterday
    raise NotImplementedError


def render_catalyst_calendar(catalysts: dict) -> None:
    """
    Render Panel 2: upcoming PDUFA events and Phase 3 completions calendar.

    Args:
        catalysts: Output dict from agent tool get_upcoming_catalysts().
                   Keys: pdufa_events, phase3_completions, total_catalyst_density_score.
    """
    # TODO: combine pdufa_events and phase3_completions, sort by date
    # TODO: for each event, render a row with:
    #   - date badge
    #   - company ticker in bold
    #   - drug name
    #   - type badge: "PDUFA Priority" (amber) | "PDUFA Standard" (grey) | "Phase 3" (blue)
    # TODO: display total_catalyst_density_score as a pressure meter
    raise NotImplementedError


def render_analogue_cards(analogues_data: dict) -> None:
    """
    Render Panel 3: three historical analogue comparison cards.

    Args:
        analogues_data: Output dict from agent tool find_historical_analogues().
                        Keys: analogues (list of 3 dicts).
    """
    # TODO: use st.columns(3) to create three side-by-side cards
    # TODO: for each analogue card show:
    #   - Date range (period name)
    #   - Similarity score as a percentage
    #   - Regime label at that time
    #   - Actual XLV 10d return with green/red background
    # TODO: colour the card background: green if xlv_ret_10d > 0, red if < 0
    raise NotImplementedError


def render_investment_brief(brief_text: str, last_updated: datetime) -> None:
    """
    Render Panel 4: the full Gemini-generated investment brief.

    Args:
        brief_text: Markdown-formatted brief from run_agent().
        last_updated: Timestamp of the last agent run.
    """
    # TODO: st.markdown(brief_text) to render the formatted brief
    # TODO: show "Last updated: [timestamp]" in grey text
    # TODO: add a "Regenerate" button: st.button("🔄 Regenerate Brief")
    #   on click, call run_agent() and update st.session_state
    raise NotImplementedError


def load_dashboard_data() -> dict:
    """
    Fetch all data needed for the dashboard by calling the three ADK tools.

    Data is cached in st.session_state to avoid redundant API calls on
    each Streamlit rerender. The cache is refreshed on page load or
    when the user clicks "Regenerate".

    Returns:
        Dict with keys: regime, analogues, catalysts, brief_text, loaded_at.
    """
    # TODO: from agent.tools import get_current_regime, find_historical_analogues, get_upcoming_catalysts
    # TODO: regime = get_current_regime()
    # TODO: analogues = find_historical_analogues(regime["feature_vector"])
    # TODO: catalysts = get_upcoming_catalysts(days=30)
    # TODO: from agent.agent import run_agent
    # TODO: brief = run_agent("Provide today's full healthcare sector investment brief.")
    # TODO: return all results bundled as a dict
    raise NotImplementedError


def main() -> None:
    """
    Main Streamlit app entry point.

    Configures the page layout, loads data, and renders the four panels.
    """
    st.set_page_config(
        page_title="HealthQuant Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("HealthQuant Agent — Healthcare Sector Intelligence")
    st.caption("Powered by Gemini · Hidden Markov Models · MongoDB Atlas Vector Search")

    # TODO: load data into session state (call load_dashboard_data if not cached)
    # TODO: if "dashboard_data" not in st.session_state, call load_dashboard_data()

    # Two-row, two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regime Gauge")
        # TODO: render_regime_gauge(st.session_state.dashboard_data["regime"])

    with col2:
        st.subheader("Upcoming Catalysts (Next 30 Days)")
        # TODO: render_catalyst_calendar(st.session_state.dashboard_data["catalysts"])

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Historical Analogues")
        # TODO: render_analogue_cards(st.session_state.dashboard_data["analogues"])

    with col4:
        st.subheader("Investment Brief")
        # TODO: render_investment_brief(
        #     st.session_state.dashboard_data["brief_text"],
        #     st.session_state.dashboard_data["loaded_at"],
        # )

    st.info(
        "⚠️ This dashboard is for educational and research purposes only. "
        "It does not constitute financial advice."
    )


if __name__ == "__main__":
    main()
