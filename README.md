# Fortune 500 Growth & Performance Analysis

**Which sectors and companies actually compounded revenue over the last decade — and is that
growth statistically real or just noise?**

A full-stack analysis of 500 large companies across 10 sectors, 2014–2023: revenue, margin,
headcount, R&D spend, and market cap tracked year over year. The project answers four
questions an investor, strategy team, or analyst would actually ask — and backs every answer
with a statistical test or a model, not just a chart.

**[Live dashboard →](#)** *(https://fortune500-growth-analysis.streamlit.app/)*

---

## The questions this project answers

1. **Which sectors grew fastest, and is that growth statistically significant** or within
   the range of normal year-to-year noise?
2. **Which individual companies were consistent compounders** vs. one-year flukes?
3. **Where is next year's revenue likely to land**, and how much should you trust that number?
4. **Which companies don't fit their peer group** — either breakout outperformers or
   early warning signs?

---

## Key findings

- **Technology compounded fastest**, growing revenue at a **14.8% CAGR** over the decade —
  roughly double the next-fastest sector (Health Care, 8.4%) — and this trend is
  statistically significant (OLS p < 0.05), not just a couple of standout years.
- **Telecom and Energy grew slowest** (2.8% and 3.1% CAGR), and Energy's trend is also the
  most volatile year-to-year of any sector, consistent with its commodity-price exposure.
- The panel's **total revenue grew from ~$65B to ~$116B** (2014→2023), with a clear,
  synchronized dip in 2020 across every sector — visible in the Overview page's trend chart.
- **KMeans clustering** on growth, margin, ROA, and revenue-per-employee separates companies
  into four repeatable archetypes (e.g. "high growth, high margin" vs. "low growth, low
  margin") that cut across sector labels — two companies in the same sector often behave
  completely differently.
- **Isolation Forest** flags roughly 5% of companies each year as statistical outliers —
  useful as a first-pass screen before a deeper manual review.
- **ARIMA and Prophet forecasts agree** on overall market direction but diverge more at the
  individual-company level, which is itself a useful signal: agreement between two
  independent models means a projection is robust, not just an artifact of one method.

---

## How it's built

| Stage | What happens | Where |
|---|---|---|
| **Data** | A realistic 10-year panel, sector-calibrated to public growth/margin/volatility benchmarks, with a shared macro shock (e.g. the 2020 downturn) applied across all companies | `data/generate_data.py` |
| **SQL** | CAGR ranking, sector rollups, rolling 3-year growth averages, Fortune-500-rank volatility, R&D-intensity buckets — via CTEs and window functions | `sql/queries.sql` |
| **Statistical testing & modeling** | OLS trend significance testing, correlation analysis, ARIMA + Prophet forecasting, KMeans clustering, Isolation Forest anomaly detection — fully executed, outputs saved | `notebooks/eda_and_modeling.ipynb` |
| **Dashboard** | 5-page interactive Streamlit app so the analysis is explorable, not just static | `app.py`, `pages/` |

The notebook and the dashboard call the **same underlying functions** (`utils/analytics.py`),
so the numbers you see in the app always match the
