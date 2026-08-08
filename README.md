# HealthQuant Agent

HealthQuant is an early-stage investment-research agent for healthcare and biotechnology markets. It combines clinical-trial activity, FDA catalyst calendars, and ETF market signals in an eight-feature Hidden Markov Model (HMM), then uses Gemini and MongoDB Atlas Vector Search to turn the model output into a structured research brief.

> [!IMPORTANT]
> This repository is a hackathon prototype under active development. The market-data module and project scaffolding are present, but several database, model, agent, and dashboard functions are still marked `TODO` and raise `NotImplementedError`. It is not yet an end-to-end runnable product.

## Why HealthQuant

Generic regime models typically rely on price and volatility alone. HealthQuant is designed to add sector-specific context:

- upcoming FDA PDUFA decisions;
- Phase 2 and Phase 3 clinical-trial activity;
- healthcare and biotech ETF momentum and volatility; and
- historical market analogues retrieved from MongoDB Atlas.

The intended output is a concise, evidence-grounded healthcare market brief—not an automated trading signal.

## Architecture

```text
ClinicalTrials.gov ─┐
SEC EDGAR ──────────┼──> ingestion ──> MongoDB Atlas
Yahoo Finance ──────┘                       │
                                           ├──> 3-state Gaussian HMM
                                           ├──> Atlas Vector Search
                                           └──> Gemini agent ──> Streamlit dashboard
```

## Current status

| Area | Status |
| --- | --- |
| Market-data feature calculation | Implemented |
| Clinical-trial and PDUFA ingestion | Scaffolded; core functions incomplete |
| HMM training and prediction | Scaffolded; core functions incomplete |
| MongoDB persistence and vector search | Scaffolded; core functions incomplete |
| Gemini/Google ADK agent | Scaffolded; core functions incomplete |
| Streamlit dashboard | Layout scaffolded; render functions incomplete |

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full design, data model, validation protocol, and implementation roadmap.

## Repository layout

```text
healthquant-agent/
├── agent/          Gemini/Google ADK agent, tools, and prompts
├── data/           Market, clinical-trial, and regulatory-data ingestion
├── database/       MongoDB client, schemas, seed workflow, and vector search
├── hmm/            Feature assembly, training, prediction, and backtesting
├── scripts/        Index setup, daily updates, and validation entry points
└── ui/             Streamlit dashboard
```

## Development setup

Requirements: Python 3.11+, a MongoDB Atlas deployment, Google Cloud credentials for Gemini, and a Voyage AI API key.

```bash
git clone https://github.com/KaelynWenHKU/GOOGLEHACKS.git
cd GOOGLEHACKS/healthquant-agent

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Fill in `.env` with your own credentials. Never commit `.env`, service-account files, cached datasets, or trained model artifacts.

To check which paths are still incomplete:

```bash
rg 'TODO|NotImplementedError' .
```

## Intended workflow

Once the remaining modules are implemented:

```bash
python scripts/setup_atlas_indexes.py
python database/seed_historical.py
python scripts/validate_backtest.py
streamlit run ui/dashboard.py
```

## Methodology note

Backtests must use walk-forward validation and strictly point-in-time features. Do not train on the full dataset and report in-sample predictions. Any performance figures should be treated as experimental until reproduced by the validation script.

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, and past performance does not guarantee future results.

## License

The project is licensed under the [MIT License](healthquant-agent/LICENSE).
