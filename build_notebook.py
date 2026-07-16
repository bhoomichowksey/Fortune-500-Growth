"""
build_notebook.py — programmatically assembles notebooks/eda_and_modeling.ipynb
using nbformat, then it is executed via nbconvert so committed outputs are real.
Not part of the deliverable itself — a build tool kept here for transparency
and reproducibility (re-run if you edit the analysis).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Fortune 500 Growth & Performance Analysis
### Exploratory Data Analysis, Statistical Testing, Forecasting & Segmentation

**Stack:** Python (Pandas, NumPy, SciPy, Statsmodels, scikit-learn, Prophet), SQL (SQLite), Plotly

This notebook is the analytical backbone behind the Streamlit dashboard (`app.py` + `pages/`).
It walks through the same methodology an analyst would use in Excel — trend lines, YoY growth,
correlation — but scaled across 500 companies x 10 fiscal years x 10 sectors, with statistical
significance testing and machine-learning-based segmentation layered on top.

**Sections**
1. Data loading & sanity checks
2. Descriptive statistics & Excel-style KPI rollups
3. Sector-level growth & margin trends
4. Statistical trend significance testing (OLS)
5. Correlation analysis
6. Time-series forecasting (ARIMA vs. Prophet)
7. KMeans performance clustering
8. Isolation Forest anomaly detection
9. Key findings summary
""")

code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils.analytics import (
    compute_cagr, sector_summary, linear_trend_stats,
    forecast_arima, forecast_prophet, cluster_companies,
    detect_anomalies, correlation_matrix,
)

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_style('whitegrid')
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
""")

md("## 1. Data Loading & Sanity Checks\n\nLoad from the SQLite panel (built by `data/generate_data.py`).")

code("""conn = sqlite3.connect('../data/fortune500_panel.db')
df = pd.read_sql('SELECT * FROM fortune500', conn)
conn.close()

print(f"Shape: {df.shape}")
print(f"Years: {df.fiscal_year.min()}-{df.fiscal_year.max()}")
print(f"Companies: {df.company_id.nunique()}, Sectors: {df.sector.nunique()}")
df.head()
""")

code("""# Missing value / integrity check
print(df.isna().sum().sum(), "missing values")
assert df.groupby('company_id').fiscal_year.nunique().eq(10).all(), "every company should have 10 years"
print("Integrity check passed: every company has a full 10-year history.")
""")

md("## 2. Descriptive Statistics & Excel-Style KPI Rollups\n\n"
   "These are the numbers that would normally live in a pivot table: total revenue, "
   "net income, average margin, and headcount, for the most recent fiscal year.")

code("""latest_year = df.fiscal_year.max()
latest = df[df.fiscal_year == latest_year]

kpis = {
    'Total Revenue ($B)': latest.revenue_musd.sum() / 1000,
    'Total Net Income ($B)': latest.net_income_musd.sum() / 1000,
    'Revenue-Weighted Avg Margin (%)': latest.net_income_musd.sum() / latest.revenue_musd.sum() * 100,
    'Total Employees (M)': latest.employees.sum() / 1_000_000,
    'Median Company Revenue ($M)': latest.revenue_musd.median(),
}
pd.Series(kpis).to_frame('Value')
""")

code("""df.groupby('fiscal_year').agg(
    total_revenue_musd=('revenue_musd', 'sum'),
    avg_margin_pct=('net_margin_pct', 'mean'),
    avg_growth_pct=('revenue_growth_pct', 'mean'),
).round(2)
""")

md("## 3. Sector-Level Growth & Margin Trends")

code("""sec_summary = sector_summary(df, latest_year)
sec_summary
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

trend = df.groupby(['fiscal_year', 'sector'])['revenue_musd'].sum().unstack()
trend.plot(ax=axes[0], linewidth=1.5)
axes[0].set_title('Revenue by Sector, 2014-2023')
axes[0].set_ylabel('Revenue ($M)')
axes[0].legend(fontsize=7, ncol=2)

margin_trend = df.groupby(['fiscal_year', 'sector'])['net_margin_pct'].mean().unstack()
margin_trend.plot(ax=axes[1], linewidth=1.5)
axes[1].set_title('Avg Net Margin by Sector, 2014-2023')
axes[1].set_ylabel('Net Margin (%)')
axes[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.show()
""")

md("## 4. Statistical Trend Significance Testing (OLS)\n\n"
   "For each sector, fit `revenue ~ year` and test whether the slope is significantly "
   "different from zero (p < 0.05). This distinguishes a genuine structural trend from "
   "year-to-year noise — the same question `=SLOPE()` / `=LINEST()` answers in Excel, "
   "run here programmatically across every sector at once.")

code("""rows = []
for s in df.sector.unique():
    series = df[df.sector == s].groupby('fiscal_year')['revenue_musd'].sum().sort_index()
    res = linear_trend_stats(series)
    rows.append({'sector': s, **res, 'significant': res['p_value'] < 0.05})

trend_df = pd.DataFrame(rows).sort_values('slope', ascending=False)
trend_df
""")

md("**Interpretation:** sectors with `significant = True` and a positive slope have a "
   "statistically defensible growth trend over the decade (not just a lucky year); "
   "sectors with high p-values have revenue trajectories that are statistically "
   "indistinguishable from flat, even if the raw numbers wiggle.")

md("## 5. Correlation Analysis")

code("""corr = correlation_matrix(df, latest_year)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0, vmin=-1, vmax=1)
plt.title(f'Cross-Metric Correlation, FY{latest_year}')
plt.tight_layout()
plt.show()
""")

md("## 6. Time-Series Forecasting: ARIMA vs. Prophet\n\n"
   "Two independent models on total panel revenue, cross-checked against each other. "
   "Divergence between the two flags where the forecast is model-sensitive rather than robust.")

code("""market_series = df.groupby('fiscal_year')['revenue_musd'].sum().sort_index()

arima_fc = forecast_arima(market_series, periods=3)
prophet_fc = forecast_prophet(market_series, periods=3)

print("ARIMA forecast:")
display(arima_fc)
print("\\nProphet forecast:")
display(prophet_fc)
""")

code("""plt.figure(figsize=(10, 5))
plt.plot(market_series.index, market_series.values, marker='o', label='Historical', color='black')
plt.plot(arima_fc.fiscal_year, arima_fc.forecast_musd, marker='o', linestyle='--', label='ARIMA', color='#2451B7')
plt.fill_between(arima_fc.fiscal_year, arima_fc.lower_95_musd, arima_fc.upper_95_musd, alpha=0.15, color='#2451B7')
plt.plot(prophet_fc.fiscal_year, prophet_fc.forecast_musd, marker='o', linestyle=':', label='Prophet', color='#C0392B')
plt.legend()
plt.title('Total Panel Revenue: Historical + 3-Year Forecast')
plt.ylabel('Revenue ($M)')
plt.show()
""")

md("## 7. KMeans Performance Clustering\n\n"
   "Segment companies into performance archetypes using growth, margin, ROA, and "
   "revenue-per-employee — features standardized before clustering so no single metric "
   "dominates purely due to scale.")

code("""clustered, centroids, labels = cluster_companies(df, latest_year, k=4)
print("Cluster labels:", labels)
centroids
""")

code("""plt.figure(figsize=(9, 6))
sns.scatterplot(data=clustered, x='revenue_growth_pct', y='net_margin_pct',
                 hue='cluster_label', size='revenue_musd', sizes=(20, 300), alpha=0.7)
plt.title(f'Performance Clusters, FY{latest_year}')
plt.xlabel('YoY Revenue Growth (%)')
plt.ylabel('Net Margin (%)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
""")

code("""pd.crosstab(clustered.cluster_label, clustered.sector)
""")

md("## 8. Isolation Forest Anomaly Detection\n\n"
   "Flags companies whose combination of growth, margin, ROA, and efficiency is "
   "statistically unusual relative to the full population — useful both for spotting "
   "breakout performers and for early red-flag detection.")

code("""anomalies = detect_anomalies(df, latest_year, contamination=0.05)
flagged = anomalies[anomalies.anomaly_flag]
print(f"{len(flagged)} companies flagged out of {len(anomalies)} evaluated")
flagged[['company_name', 'sector', 'revenue_growth_pct', 'net_margin_pct', 'anomaly_severity']].head(15)
""")

md("## 9. Key Findings Summary\n\n"
   "- Sector rollups (Section 2-3) show which sectors compounded revenue fastest over the decade, "
   "and which margins expanded vs. compressed.\n"
   "- OLS trend testing (Section 4) separates statistically real growth trends from noisy ones.\n"
   "- Correlation analysis (Section 5) quantifies how growth, margin, R&D intensity, and efficiency "
   "move together.\n"
   "- ARIMA/Prophet forecasts (Section 6) project the next 3 fiscal years with uncertainty bands.\n"
   "- Clustering (Section 7) surfaces four performance archetypes analysts can use to triage "
   "500 companies quickly.\n"
   "- Anomaly detection (Section 8) is a lightweight way to flag companies worth a closer look, "
   "either as standouts or as risks.\n\n"
   "**Next steps for a production version:** swap the synthetic panel for real filings "
   "(SEC EDGAR / Fortune's own dataset), add sentiment features from earnings-call transcripts, "
   "and retrain the forecasting models quarterly.")

nb['cells'] = cells
with open('notebooks/eda_and_modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/eda_and_modeling.ipynb")
