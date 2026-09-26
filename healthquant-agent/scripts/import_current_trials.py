"""Import a bounded real registry snapshot; dry run unless --write is supplied.

From healthquant-agent/: python -m scripts.import_current_trials --write
No historical HMM predictions, returns, PDUFA dates or ticker matches are invented.
"""
import argparse
import json

from data.ingestion.clinical_trials import fetch_current_snapshot


def import_snapshot(db, snapshot) -> int:
    """Upsert current administrative fields, preserving independent annotations.

    Coverage metadata is written only after all records succeed. A failed run
    may leave some records imported; rerunning uses stable NCT identity keys.
    """
    for doc in snapshot["documents"]:
        db["trial_events"].update_one({"nct_id": doc["nct_id"]},
            {"$set": {**doc, "ingestion_run_id": snapshot["run_id"]},
             "$setOnInsert": {"created_at": doc["known_as_of"]}}, upsert=True)
    db["ingestion_runs"].update_one({"_id": snapshot["run_id"]},
        {"$set": {"collection": "trial_events", **snapshot["coverage"]}}, upsert=True)
    return len(snapshot["documents"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1)
    args = parser.parse_args()
    try:
        snapshot = fetch_current_snapshot(args.page_size, args.max_pages)
        print(json.dumps(snapshot["coverage"], default=str), flush=True)
        if args.write:
            from database.mongo_client import get_db
            print(f"Imported {import_snapshot(get_db(), snapshot)} current trial records.")
        else:
            print("Dry run: no database writes. Use --write to import.")
    except Exception as exc:
        print(f"Import incomplete ({type(exc).__name__}); check service connectivity/configuration.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
