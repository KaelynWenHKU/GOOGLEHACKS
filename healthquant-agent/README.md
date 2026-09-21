# HealthQuant Agent

A healthcare research dashboard with a three-state Gaussian HMM, a MongoDB evidence layer, and a Gemini agent built with Google ADK.

## Run the dashboard

From the repository root (`GOOGLEHACKS`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r healthquant-agent/requirements.txt
python -m streamlit run healthquant-agent/ui/dashboard.py
```

Open the local URL printed by Streamlit. **Sample data** works without accounts or API keys. Its companies, probabilities, dates and returns are fictional; its example brief is static text, not a Gemini response.

The four panels show the market regime, upcoming catalysts, historical analogues and a research brief. Download evidence as JSON and generated briefs as Markdown. The sidebar switches between sample and live modes.

## Connect Gemini and live evidence

Copy `.env.example` to `healthquant-agent/.env` and replace placeholders locally. Never commit credentials or enter them into a research question.

- **Gemini Developer API:** set `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey). Leave `GOOGLE_GENAI_USE_VERTEXAI=false`.
- **Vertex AI alternative:** set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`. Configure Application Default Credentials, for example with `gcloud auth application-default login`, or use your existing service identity. A project name alone does not verify authorization, billing or model availability.
- **MongoDB:** set `MONGODB_URI` and `MONGODB_DB_NAME`. The dashboard reads existing `regime_states`, `pdufa_events`, and `trial_events`.
- **Analogues:** set `VOYAGE_API_KEY`. Historical documents need 1024-dimensional Voyage embeddings and the Atlas `regime_vector_index` specified in PROJECT_SPEC.md. The index needs a date filter field. Its embedding model must match `VOYAGE_MODEL`.

Restart Streamlit after changing environment configuration. Select **Live data → Refresh evidence**, then enter a question and press **Generate Gemini brief**. Refreshing reads the database and may call Voyage; generating a brief calls Gemini. Normal widget reruns do not repeat these calls. Each brief gets an isolated ADK session, up to six model calls, 1,800 output tokens per call and a 90-second run timeout. Tool outputs are snapshots of the evidence shown on screen. Missing services never silently fall back to fictional values.

The latest stored HMM prediction is displayed with its original timestamp; Refresh does **not** train a new model. Predictions older than four days are marked stale. Probability bars use the stored `state_label_map`, or raw state IDs when no map exists. Ten-day analogue returns are displayed only with `return_observed_at` provenance available by the snapshot cutoff.

## Cost and competition resources

Default: `GEMINI_MODEL=gemini-2.5-flash-lite`. Google's [pricing page](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash-lite), checked September 21, 2026, lists standard text rates of **$0.10 per million input tokens** and **$0.40 per million output tokens**, with a limited free tier. Free-tier content may be used to improve Google's products; review the terms before submitting private data.

For illustration, 5,000 total input tokens plus 1,000 total output tokens across a complete run would cost $0.0009 in Gemini token charges. This is not a measured per-brief cost: tool loops increase tokens, and Atlas, Voyage, hosting and any other services are separate. Change `GEMINI_MODEL` to a model available in your account if needed.

The [competition rules](https://rapid-agent.devpost.com/rules) offered a $100 Google Cloud credit request with a June 4, 2026 deadline and no guarantee of approval. Do not assume these credits are currently available. The [MongoDB track resources](https://rapid-agent.devpost.com/details/mongodb-resources) remain a technical reference. Developer API keys and Google Cloud credits are different billing routes.

## Status and verification

Implemented: four-panel UI, explicit sample/live modes, three read-only evidence tools, ADK/Gemini runner, MongoDB connection helpers, Voyage embedding and Atlas search wrappers, HMM training and backtest modules.

Still required for a real end-to-end run: authorized credentials, populated and audited historical data, a trained model's persisted predictions, and an Atlas vector index/embeddings. The ingestion, historical-seeding and automatic-update scripts elsewhere in the scaffold remain incomplete. There is no hosted deployment or verified live Gemini response yet. The repository does not claim a profitable or fully validated historical strategy.

Offline verification:

```bash
# From the repository root, with the virtual environment activated:
python -m pytest -q
```

Tests exercise Streamlit mode switching, unavailable services, explicit generation and cache invalidation; the installed ADK runner's real tool dispatch/session flow uses a scripted model so no external fees or credentials are needed. Provider wrappers are tested with mocked service responses. These tests do not establish live authentication or model quality.

## Quantitative validation boundaries

Monthly expanding-window fits use only prior data and a training-only scaler. Daily inference uses sequence prefixes ending on each prediction date. The simulator uses close-to-close returns with a one-session signal lag, zero trading costs and zero cash yield; this is an idealized execution assumption.

The validation loader requires `predicted_regime` and a `train_end_date` before each prediction date. It rejects duplicate dates and missing sessions inside the downloaded interval. Historical clinical features still need archived point-in-time source data: recorded model cutoffs alone do not establish absence of lookahead.

From `healthquant-agent/`, run `python -m scripts.validate_backtest --start 2020 --end 2024` after populating validated predictions. It writes timeline/report HTML and a metrics JSON file. Targets in PROJECT_SPEC.md are **not actual results**.

For educational research only; not financial advice. Past performance does not guarantee future results.

MIT license — see LICENSE.
