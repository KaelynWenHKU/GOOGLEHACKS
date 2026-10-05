# HealthQuant Agent

## Interactive terminal (also works in VS Code's terminal)

From the repository root, with the existing `.venv` and requirements installed:

```bash
./healthquant chat --sample    # fictional offline demo, no API calls
./healthquant chat             # live research; prompts before API calls
./healthquant status           # local configuration only; not a connectivity test
./healthquant ask "Which catalysts should I watch?"
```

Alternatively, from `healthquant-agent/`, use `../.venv/bin/python -m agent.cli chat`.
Use `/help`, `/refresh`, `/regime`, `/catalysts`, `/analogues`, `/status`, `/clear`, `/export "new-file.md"`, and `/quit`. `/backtest` shows the existing validation command without running it. Markdown export refuses existing files and symlinks. The repository launcher runs inside `healthquant-agent/`, so relative export paths are relative to that directory; use an absolute path if needed.

Natural-language questions invoke the existing bounded Gemini/ADK research workflow. Each question starts an independent model session (v1 has **no multi-turn conversation memory**). Evidence stays in memory until `/refresh`, `/clear` or exit, and every brief displays its evidence timestamp. Actual tool-call names are shown after successful generation, not simulated as live progress. Sample answers are static fictional demonstrations, not generated answers to your question.

Live refresh can call Voyage; live questions can call Gemini (up to six model calls per question). Both require confirmation, or explicit `--yes` for this invocation. Provider charges may apply. No API calls occur merely on startup or `/status`; no shell execution, trades, model training, automatic IP changes or database writes are exposed. Do not paste credentials into questions. This is a research terminal, not a general-purpose coding agent.

The terminal reuses the dashboard's services and does not fix network access: Atlas still needs the Python process's outbound IP allowed. A changing VPN can break access again; a fixed outbound server or dedicated VPN exit is the durable deployment option. Missing HMM history remains unavailable, never replaced with sample data in live mode.

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

### Bootstrap current live data

From `healthquant-agent/`, with the virtual environment active:

```bash
python -m scripts.setup_atlas_indexes
python -m scripts.import_current_trials                 # fetch/validate only
python -m scripts.import_current_trials --write         # explicitly write to Atlas
```

Index setup creates standard identity/calendar indexes plus the 1024-dimensional
`regime_vector_index` and `trial_text_index`. It reuses matching search definitions,
rejects conflicts without replacing them, and waits for READY/queryable status.
If a build times out, check Atlas and rerun; do not delete data to resolve it.

The initial import is limited to **100 recently updated active Phase 3 records**
from ClinicalTrials.gov. This is partial registry coverage, not a public-company
universe. `--page-size` and `--max-pages` control the bounded fetch. Completed
imports record the source, fetch time, count and truncation in `ingestion_runs`.
Reruns upsert by NCT ID, preserving independent ticker/outcome annotations.
Month/year-only dates retain their raw precision and are not placed on a specific
calendar day. Missing enrollment remains null. Current revisions have a current
`known_as_of`; old first-posted dates do not make those revisions historical evidence.

This importer does **not** populate PDUFA events, company mappings or historical
regime states; it does not train a model, generate returns or run paid embeddings.
The older ticker-specific/historical ingestion scaffold is still incomplete.
An empty regime collection therefore correctly remains unavailable in Live mode.

Live bootstrap verified September 26, 2026: 100 current trial records were written
and counted in Atlas; both search indexes reported READY and queryable. The live
catalyst tool returned one exact-date Phase 3 completion in its next-30-day window.
This is a dated partial-snapshot check, not a comprehensive calendar. PDUFA and
regime collections remained empty. Atlas SRV connections use the certifi trusted
CA bundle with certificate verification enabled; credentials remain local.

Implemented: four-panel UI, explicit sample/live modes, three read-only evidence tools, ADK/Gemini runner, MongoDB connection helpers, Voyage embedding and Atlas search wrappers, HMM training and backtest modules.

Still required for a real end-to-end run: authorized credentials, populated and audited historical data, a trained model's persisted predictions, and an Atlas vector index/embeddings. The ingestion, historical-seeding and automatic-update scripts elsewhere in the scaffold remain incomplete. There is no hosted deployment or verified live Gemini response yet. The repository does not claim a profitable or fully validated historical strategy.

Offline verification:

```bash
# From the repository root, with the virtual environment activated:
python -m pytest -q
```

Tests exercise Streamlit mode switching, unavailable services, explicit generation and cache invalidation; the installed ADK runner's real tool dispatch/session flow uses a scripted model so no external fees or credentials are needed. Provider wrappers are tested with mocked service responses. These tests do not establish live authentication or model quality.

## Quantitative validation boundaries

Clinical feature queries prioritize `known_as_of` over `announced_at`, then
`first_posted_date`. A newer knowledge timestamp cannot be bypassed by an older
publication date; records without any provenance are excluded. If no eligible
enrollment observations remain, feature construction raises an error rather
than inventing a value of 5000. These guards do not establish source completeness
or reconstruct overwritten historical revisions; audited snapshots remain required.

Monthly expanding-window fits use only prior data and a training-only scaler. Daily inference uses sequence prefixes ending on each prediction date. The simulator uses close-to-close returns with a one-session signal lag and zero cash yield; this remains an idealized execution assumption, not an executable fill model.

Trading costs are configurable with `--transaction-cost-bps` (one-way basis points per traded portfolio weight). The default `0` is explicitly frictionless. Costs apply to lagged strategy rebalancing, including drift in a half-invested portfolio, and to the benchmark's initial purchase. Net returns use `(1 - turnover * bps / 10000) * (1 + weight * market_return) - 1`, assuming target allocation after cost deduction. This is a proportional cost approximation, not a broker fee or market-impact estimate. There is no terminal liquidation, tax model or separate slippage model. Reported Sharpe uses net returns and a 4% annual risk-free reference; directional hit rate still uses raw XLV returns (non-neutral signals, overlapping 10-session windows).

The validation loader requires `predicted_regime` and a `train_end_date` before each prediction date. It rejects duplicate dates and missing sessions inside the downloaded interval. Historical clinical features still need archived point-in-time source data: recorded model cutoffs alone do not establish absence of lookahead.

From `healthquant-agent/`, run `python -m scripts.validate_backtest --start 2020 --end 2024` after populating validated predictions. It writes timeline/report HTML and a metrics JSON file. Targets in PROJECT_SPEC.md are **not actual results**.

For cost sensitivity, rerun with, for example, `--transaction-cost-bps 10`. The rate is a user-selected assumption, not a measured cost. The HTML, console and JSON disclose the chosen rate; JSON also reports total strategy turnover. Each run replaces the generated output files, so preserve them before comparing scenarios.

The validation loader rejects nonfinite, nonnumeric, zero or negative XLV closes, and missing or duplicate price timestamps, before computing returns. Valid prices are sorted chronologically. Missing prediction sessions or missing preceding closes remain errors rather than silently discarded observations.

Prediction and transition helpers require the explicit `state_label_map` from the same fitted checkpoint as the model. They never assume state 0 is risk-on or state 2 is fear: HMM state IDs can permute after retraining. Missing or malformed maps raise an error instead of returning potentially mislabeled probabilities.

Checkpoints must explicitly record the current ordered `feature_names` and a valid `train_end_date` in `YYYY-MM-DD` format. Older files missing this metadata are rejected; regenerate them from verified training inputs rather than guessing their feature order or cutoff. Only load trusted local pickle files: these metadata checks occur after deserialization and do not make untrusted pickle files safe.

### Prepare stored predictions for historical analogue search

After auditing historical point-in-time features and persisting monthly walk-forward predictions, run from `healthquant-agent/`:

```bash
python -m scripts.embed_historical_regimes --start 2020-01-01 --end 2024-12-31 --max-documents 25
```

This reads at most 25 rows missing `feature_embedding` and validates them without calling Voyage or writing MongoDB. Add `--write` to explicitly allow Voyage document embeddings (charges may apply) and guarded updates; the maximum batch size is 100. It checks training cutoff precedes prediction date, feature order/values, labels and probability distributions, but these checks do not establish upstream point-in-time data quality.

Only date, predicted label and raw features enter embedding text—not realized returns or later summaries. Existing embeddings are skipped; source changes during a run are reported as conflicts. Provider/model metadata and creation time are stored with each vector. Partial writes can be resumed by rerunning; conflicts may still consume embedding tokens. This is not a migration tool for existing embeddings or model changes. The query and document embedding models must agree.

The full historical ingestion pipeline remains incomplete. This command cannot create missing predictions or resolve Atlas network access. No live embeddings were generated to validate this implementation; tests use offline fixtures.

Persisting a walk-forward rerun synchronizes the stored regime aliases and atomically removes previous generated embeddings and summaries. It also clears unavailable evaluation returns and their observation timestamps, including the legacy forward-return alias. Unrelated annotations remain untouched. Reruns therefore require an explicit re-embedding pass before those dates become searchable again—even when their predictions are unchanged; re-embedding may incur provider charges.

For educational research only; not financial advice. Past performance does not guarantee future results.

MIT license — see LICENSE.
