# Fortune 500 Growth & Performance Analysis

**Excel · Python · SQL · Plotly · Streamlit**

An end-to-end analysis of Fortune-500-style company financials (2014–2023): 500 companies
across 10 sectors, tracked year-over-year on revenue, margin, headcount, R&D spend, and
market cap. The project moves from raw data → SQL exploration → statistical testing →
machine-learning segmentation → forecasting → an interactive Streamlit dashboard, mirroring
a real analyst workflow (Excel-style KPI rollups → decision-ready visuals → a presentable
tool stakeholders can use themselves).

**Live methodology, not just charts.** Every number in the dashboard is backed by a
statistical test, a documented model, or a reproducible SQL query — not a static screenshot.

---

## What's inside

| Layer | What it does | Where |
|---|---|---|
| **Data generation** | Builds a realistic 10-year, 500-company panel with sector-calibrated growth/margin assumptions and macro shocks (incl. a 2020 downturn) | `data/generate_data.py` |
| **SQL analysis** | CAGR ranking, sector rollups, rolling averages, rank-volatility, R&D-intensity buckets — all via CTEs and window functions | `sql/queries.sql` |
| **Notebook** | Full EDA, OLS trend-significance testing, correlation analysis, ARIMA + Prophet forecasting, KMeans clustering, Isolation Forest anomaly detection (executed, with outputs saved) | `notebooks/eda_and_modeling.ipynb` |
| **Shared analytics module** | The same functions the notebook uses, imported directly by the dashboard so results are always consistent | `utils/analytics.py`, `utils/data_loader.py` |
| **Streamlit dashboard** | 5-page interactive app: Overview, Sector Deep Dive, Growth Forecasting, Clustering & Anomalies, Company Explorer | `app.py`, `pages/` |

---

## Why synthetic data?

Real Fortune 500 financials are licensed (Fortune / data.ai) or scattered across individual
10-K filings. To keep this project **fully open-source, reproducible, and runnable by
anyone without a paid data license**, `data/generate_data.py` generates a statistically
realistic panel calibrated to public, well-known industry benchmarks — sector CAGR ranges,
typical net margins, employee-to-revenue ratios — with a shared macro shock per year (e.g.
a -9% average shock in 2020) so trends behave the way real Fortune 500 data does.

Want to run this on **real** data instead? Swap the source in `utils/data_loader.py`:
point `load_data()` at a CSV/SQL table pulled from SEC EDGAR filings, Kaggle's Fortune 500
datasets, or a licensed Fortune/data.ai export. Every downstream function (`utils/analytics.py`,
all five dashboard pages) expects the same schema (see **Data dictionary** below), so nothing
else needs to change.

---

## Data dictionary

| Column | Description |
|---|---|
| `company_id`, `company_name` | Unique identifier and display name |
| `sector` | One of 10 GICS-style sectors |
| `hq_state`, `founded_year` | Headquarters US state, year founded |
| `fiscal_year` | 2014–2023 |
| `revenue_musd`, `net_income_musd` | Revenue / net income, $ millions |
| `net_margin_pct` | Net income ÷ revenue |
| `employees` | Headcount |
| `rd_spend_musd` | R&D spend, $ millions |
| `market_cap_musd`, `total_assets_musd` | Balance-sheet-style figures, $ millions |
| `revenue_growth_pct` | YoY revenue growth |
| `fortune500_rank` | Revenue rank within that fiscal year |
| `roa_pct` | Return on assets |
| `revenue_per_employee_k` | Revenue per employee, $ thousands |

---

## Project structure

```
fortune500-growth-analysis/
├── app.py                          # Streamlit entry point (Overview page)
├── pages/
│   ├── 1_Sector_Deep_Dive.py       # Sector rollups + OLS trend testing + correlation
│   ├── 2_Growth_Forecasting.py     # ARIMA + Prophet forecasting
│   ├── 3_Clustering_Anomalies.py   # KMeans + Isolation Forest
│   └── 4_Company_Explorer.py       # Single-company deep dive
├── utils/
│   ├── data_loader.py              # Cached data access (SQLite -> CSV fallback)
│   └── analytics.py                # CAGR, forecasting, clustering, anomaly detection
├── data/
│   ├── generate_data.py            # Reproducible synthetic data generator
│   ├── fortune500_panel.csv        # Generated dataset (long/panel format)
│   └── fortune500_panel.db         # Same data as SQLite (for SQL analysis / BI tools)
├── sql/
│   └── queries.sql                 # 8 advanced analytical queries (CTEs, window functions)
├── notebooks/
│   └── eda_and_modeling.ipynb      # Full analysis notebook, pre-executed
├── build_notebook.py               # Rebuilds the notebook programmatically (dev tool)
├── requirements.txt                # Runtime dependencies (for Streamlit Cloud)
├── requirements-dev.txt            # Notebook-only dependencies (not needed to deploy)
├── .streamlit/config.toml          # Custom theme
└── .gitignore
```

---

## Run it locally

```bash
git clone https://github.com/<your-username>/fortune500-growth-analysis.git
cd fortune500-growth-analysis
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# Data is already generated and committed (data/*.csv, data/*.db).
# To regenerate it (e.g. after changing assumptions):
python data/generate_data.py

streamlit run app.py
```

Open the notebook (optional, only if you want to re-run the analysis / regenerate outputs):

```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/eda_and_modeling.ipynb
```

Run the SQL queries directly against the SQLite file:

```bash
sqlite3 data/fortune500_panel.db < sql/queries.sql
```

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a **new GitHub repository** (public or private).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"** → select your repository and branch → set:
   - **Main file path:** `app.py`
   - **Python version:** 3.11 (Advanced settings, if prompted)
4. Click **Deploy**. Streamlit Cloud installs everything in `requirements.txt` automatically.
5. First deploy can take **5–10 minutes** — Prophet's dependency (`cmdstanpy`) compiles a
   Stan model binary on first install. This is normal; subsequent redeploys are much faster
   because the build is cached.
6. If Prophet fails to build on your plan/tier, the app still works — the **Growth
   Forecasting** page catches that specific failure and falls back to showing the ARIMA
   forecast only (with an on-screen note). You can also remove `prophet` and `cmdstanpy`
   from `requirements.txt` entirely if you'd rather skip that model.

No secrets, API keys, or external services are required — everything runs on the committed
data files.

---

## Methodology notes

- **Trend significance testing:** rather than eyeballing a line chart, each sector's
  revenue-vs-year relationship is fit with OLS (`scipy.stats.linregress`) and reported with
  R² and a p-value, so "sector X is growing" is a statistically backed claim, not a visual
  impression.
- **Two-model forecasting:** ARIMA (statsmodels) and Prophet (Meta) are fit independently.
  Where they agree, the projection is more trustworthy; where they diverge, that's flagged
  as a signal the forecast is model-sensitive.
- **Clustering:** KMeans on standardized growth/margin/ROA/efficiency features groups
  companies into interpretable archetypes (e.g. "High growth, high margin") rather than
  relying on sector labels alone.
- **Anomaly detection:** Isolation Forest flags companies whose combination of metrics is
  statistically unusual — a lightweight way to triage 500 companies down to a shortlist
  worth a closer look.

---

## Tech stack

`Python` · `Pandas` · `NumPy` · `SciPy` · `Statsmodels` (ARIMA) · `Prophet` · `scikit-learn`
(KMeans, Isolation Forest) · `SQLite` / `SQL` · `Plotly` · `Streamlit` · `Matplotlib` /
`Seaborn` (notebook) — designed to also plug into **Tableau** or **Power BI** by pointing
either tool directly at `data/fortune500_panel.db` (SQLite) or `data/fortune500_panel.csv`.

## License

MIT — use freely for learning, portfolio, or as a template for a real analysis.
