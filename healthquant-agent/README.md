# HealthQuant Agent

HealthQuant is an early-stage investment-research agent for healthcare and biotechnology markets. It combines clinical-trial activity, FDA catalyst calendars, and ETF market signals in an eight-feature Hidden Markov Model (HMM), then uses Gemini and MongoDB Atlas Vector Search to produce a structured research brief.

> [!IMPORTANT]
> This is a hackathon prototype under active development. The market-data module and project scaffolding are present, but several core functions are still marked `TODO` and raise `NotImplementedError`.

## Components

- `data/` — market, clinical-trial, and regulatory-data ingestion
- `hmm/` — feature assembly, training, prediction, and walk-forward backtesting
- `database/` — MongoDB persistence, schemas, and Atlas Vector Search
- `agent/` — Gemini/Google ADK tools and investment-brief prompts
- `ui/` — the Streamlit dashboard scaffold
- `scripts/` — index setup, daily updates, and validation entry points

See the repository's [project specification](../PROJECT_SPEC.md) for the architecture, data model, methodology, and implementation roadmap.

## Setup

Requirements: Python 3.11+, MongoDB Atlas, Google Cloud credentials for Gemini, and a Voyage AI API key.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your own credentials. Never commit `.env`, service-account files, cached datasets, or trained models.

Once the remaining modules are implemented, the intended workflow is:

```bash
python scripts/setup_atlas_indexes.py
python database/seed_historical.py
python scripts/validate_backtest.py
streamlit run ui/dashboard.py
```

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, and past performance does not guarantee future results.

## License

Licensed under the [MIT License](LICENSE).
