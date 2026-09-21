# HealthQuant Agent

A healthcare research dashboard combining Gaussian HMM regimes, clinical catalysts, MongoDB evidence, and a Gemini agent built with Google ADK.

See [setup, dashboard instructions and implementation status](healthquant-agent/README.md).

After installing the documented dependencies, launch from this repository root:

```bash
python -m streamlit run healthquant-agent/ui/dashboard.py
```

Sample mode works without credentials and uses clearly labeled fictional data. Live mode requires Gemini, MongoDB and Voyage configuration plus populated evidence. No verified investment-performance claims are made.
