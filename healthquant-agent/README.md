# HealthQuant Agent

HealthQuant is a Gemini-powered investment intelligence agent that detects market regime shifts in the healthcare and biotechnology sectors by combining clinical trial pipeline data with market microstructure signals in a single Hidden Markov Model feature space. Unlike existing regime-detection tools that operate on generic market indices, HealthQuant ingests live Phase 2/3 trial enrollment data and upcoming FDA PDUFA decision dates from ClinicalTrials.gov and SEC EDGAR, fusing them with XLV/IBB price signals into an 8-dimensional feature vector that drives a 3-state GaussianHMM classifying the market as *risk-on*, *neutral*, or *catalyst-driven fear*. When a user queries the agent, it calls three Google ADK tools — regime classification, MongoDB Atlas Vector Search for historical analogues, and upcoming catalyst retrieval — then uses Gemini to synthesise everything into a structured investment brief with concrete ETF positioning recommendations. The data layer runs entirely on MongoDB Atlas (document store, vector search, and persistent agent memory in one platform), with Voyage AI embeddings enabling millisecond retrieval of the most historically similar market periods from a decade of daily regime documents.

## Architecture

```
ClinicalTrials.gov API  ──┐
SEC EDGAR (8-K filings) ──┤  data/ingestion/   →  MongoDB Atlas
Yahoo Finance (yfinance) ──┘                       ├── regime_states (+ Vector Search)
                                                   ├── trial_events
                           hmm/                    ├── pdufa_events
                           GaussianHMM (3 states)  └── company_ticker_map
                                │
                           agent/                  Voyage AI voyage-3-large
                           Google ADK + Gemini     (1024-dim embeddings)
                                │
                           ui/dashboard.py
                           Streamlit (4-panel demo)
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd healthquant-agent
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in MONGODB_URI, VOYAGE_API_KEY, GOOGLE_CLOUD_PROJECT

# 3. Set up Atlas indexes (run once)
python scripts/setup_atlas_indexes.py

# 4. Seed historical data 2015–2024 (takes ~2–4 hours)
python database/seed_historical.py

# 5. Train the HMM
python -c "from hmm.train import run_walk_forward_training; from database.mongo_client import get_db; run_walk_forward_training(get_db())"

# 6. Run the Streamlit dashboard
streamlit run ui/dashboard.py
```

## Implementation Order

Follow Section 18 of PROJECT_SPEC.md for the recommended build order.

## Disclaimer

*This tool is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results.*

## License

MIT — see LICENSE file.
