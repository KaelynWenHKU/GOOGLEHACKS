"""Bounded embedding of existing walk-forward predictions; dry run by default.

Run from healthquant-agent with python -m scripts.embed_historical_regimes.
--write allows Voyage calls (charges may apply) and guarded MongoDB updates.
This does not create missing history, train a model or establish profitability.
"""
import argparse
import json


def main(argv=None):
    """Report counts without exposing provider errors or connection secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-documents", type=int, default=25)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        from database.mongo_client import get_db
        from database.seed_historical import embed_stored_predictions
        result = embed_stored_predictions(get_db(), args.start, args.end,
                                           args.max_documents, args.write)
        print(json.dumps(result))
    except Exception as exc:
        print(f"Embedding run incomplete ({type(exc).__name__}). Check source metadata and service connectivity. "
              "A write run may have partially completed; existing embeddings are preserved on retry.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
