"""
scripts/daily_update.py
========================
Daily data refresh script — run every trading day before market open.

What it does:
  1. Fetches latest clinical trial updates from ClinicalTrials.gov
  2. Checks for new SEC 8-K PDUFA filings from the past week
  3. Computes today's 8-dim feature vector
  4. Classifies today's regime using the trained HMM
  5. Generates today's Voyage AI embedding
  6. Upserts the regime_states document for today
  7. Optionally re-runs the Gemini agent to generate today's brief

Schedule: Run at 9:00 AM ET on US trading days (M–F, non-holidays).
Suggested cron: 0 9 * * 1-5 (adjust for your timezone)

Usage:
    python scripts/daily_update.py
    python scripts/daily_update.py --skip-brief  # skip Gemini call (saves cost)
"""

import argparse
import logging
import sys
from datetime import datetime, date

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def is_trading_day(target_date: date) -> bool:
    """
    Check if the given date is a US stock market trading day.

    Skips weekends. Does not check for market holidays (good enough for hackathon).
    For production, use pandas_market_calendars or a holidays library.

    Args:
        target_date: The date to check.

    Returns:
        True if it's a weekday (Mon–Fri), False for weekends.
    """
    # TODO: return target_date.weekday() < 5
    raise NotImplementedError


def refresh_trial_data(db) -> int:
    """
    Fetch recent ClinicalTrials.gov updates for tracked companies.

    Args:
        db: pymongo Database object.

    Returns:
        Number of trial events updated/inserted.
    """
    # TODO: fetch active Phase 2/3 trials for all tickers in company_ticker_map
    # TODO: upsert each trial to trial_events (filter: nct_id)
    # TODO: log count of new vs updated records
    raise NotImplementedError


def refresh_pdufa_data(db) -> int:
    """
    Check for new PDUFA-related 8-K filings from the past 7 days.

    Args:
        db: pymongo Database object.

    Returns:
        Number of PDUFA events inserted.
    """
    # TODO: compute start_date = today - 7 days
    # TODO: call pdufa_events.run_ingestion(start_date, today, db)
    raise NotImplementedError


def compute_and_store_todays_regime(db, generate_brief: bool = True) -> dict:
    """
    Compute today's feature vector, classify the regime, embed it, and store to MongoDB.

    Args:
        db: pymongo Database object.
        generate_brief: If True, call the Gemini agent to generate today's brief.
                        Set to False to save on API costs during development.

    Returns:
        Today's regime_state document as a dict.
    """
    # TODO: today = datetime.utcnow().strftime("%Y-%m-%d")
    # TODO: raw_features = features.build_feature_vector(today, db)
    # TODO: regime_result = predict.classify_current_regime(db)
    # TODO: text = vector_search.build_regime_text(regime_result)
    # TODO: embedding = vector_search.embed_regime_document(text)
    # TODO: if generate_brief: brief = agent.run_agent("Today's regime brief")
    # TODO: assemble document and upsert to regime_states
    raise NotImplementedError


def main(skip_brief: bool = False) -> None:
    """
    Run the full daily update pipeline.

    Args:
        skip_brief: If True, skip the Gemini brief generation step.
    """
    from database.mongo_client import get_db

    today = date.today()

    if not is_trading_day(today):
        logger.info(f"{today} is not a trading day. Skipping update.")
        sys.exit(0)

    logger.info(f"Starting daily update for {today}...")
    db = get_db()

    logger.info("[1/4] Refreshing trial data...")
    n_trials = refresh_trial_data(db)
    logger.info(f"      {n_trials} trial records updated")

    logger.info("[2/4] Refreshing PDUFA data...")
    n_pdufa = refresh_pdufa_data(db)
    logger.info(f"      {n_pdufa} PDUFA events added")

    logger.info("[3/4] Computing today's regime...")
    regime_doc = compute_and_store_todays_regime(db, generate_brief=not skip_brief)
    logger.info(f"      Regime: {regime_doc.get('regime_label', 'unknown')}")

    logger.info("[4/4] Daily update complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthQuant daily data update")
    parser.add_argument(
        "--skip-brief",
        action="store_true",
        help="Skip Gemini brief generation (saves API cost)",
    )
    args = parser.parse_args()
    main(skip_brief=args.skip_brief)
