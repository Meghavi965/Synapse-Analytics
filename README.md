# Synapse-Analytics
## Self-Healing BI Agent

A Streamlit business intelligence demo that explores data quality, schema drift, natural-language questions, forecasting, and executive reporting in one dashboard. It includes generated demo data so you can try the app without supplying a dataset.

### Highlights

- **Data quality and repair:** inspect data health, map familiar column names, and demonstrate recovery from common schema changes or missing values.
- **Ask questions about data:** use the natural-language query interface to request summaries and insights from the dashboard data.
- **Forecasting and decision support:** explore time-series forecasts and evidence-backed business recommendations where the selected data supports them.
- **Market context:** retrieve metals pricing data from Yahoo Finance; optional market research can use Firecrawl when configured.
- **Executive reports:** generate downloadable report artifacts from the executive dashboard.
- **Container-ready:** run the app locally or build it with the included Dockerfile.

### Requirements

- Python 3.10 or newer (the included Dockerfile uses Python 3.12)
- pip

### Run locally

Open a terminal in this directory (`self_healing_bi_agent`) and run:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch the dashboard:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the app starts.

### Run with Docker

From this directory:

```bash
docker build -t self-healing-bi-agent .
docker run --rm -p 8501:8501 self-healing-bi-agent
```

Then open `http://localhost:8501` in a browser.

### Optional market research

Core demo features work without API keys. To enable the optional Firecrawl-backed market research integration, set `FIRECRAWL_API_KEY` in your environment and install the optional `firecrawl-py` package:

```bash
pip install firecrawl-py
```

Keep API keys in environment variables or a local, untracked `.env` file. Never commit credentials.

### Project layout

```text
self_healing_bi_agent/
├── main.py                    # Streamlit entry point
├── executive_suite_app.py     # Main executive dashboard
├── dashboard_app.py           # Predictive dashboard/demo
├── ui_theme.py                # Shared UI styling
├── core/                      # Data health, repair, forecasting, insights
├── utils/                     # NLQ, market data, safeguards, report utilities
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container setup
└── .dockerignore              # Files excluded from the Docker build context
```

### Notes

- `main.py` launches the executive suite. `dashboard_app.py` also contains a standalone predictive dashboard and can be run directly with `streamlit run dashboard_app.py`.
- The dashboard's sample data is synthetic and intended for demonstration. Do not treat its results as real business or investment advice.
- Live market data requires an internet connection and depends on the upstream Yahoo Finance chart endpoint being available.
- Generated reports and temporary runtime files may be written under the parent workspace's `output/`, `reports/`, or related folders, depending on the feature and workspace layout.
