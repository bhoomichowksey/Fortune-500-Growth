"""pages/1_Sector_Deep_Dive.py — sector rollups, trend significance testing, correlations."""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from utils.data_loader import load_data
from utils.analytics import sector_summary, linear_trend_stats, correlation_matrix

st.set_page_config(page_title="Sector Deep Dive", page_icon="🏭", layout="wide")

df = load_data()
st.title("🏭 Sector Deep Dive")
st.caption("Revenue-weighted margins, growth-trend significance testing (OLS), and cross-metric correlation.")

sectors = sorted(df.sector.unique())
sel = st.multiselect("Sectors to compare", sectors, default=sectors)
d = df[df.sector.isin(sel)]

year = st.slider("Fiscal year for point-in-time comparisons", int(df.fiscal_year.min()),
                  int(df.fiscal_year.max()), int(df.fiscal_year.max()))

st.subheader(f"Sector Scorecard — FY{year}")
summary = sector_summary(d, year)
st.dataframe(
    summary.style.format({
        "total_revenue_musd": "{:,.0f}", "avg_net_margin_pct": "{:.2f}",
        "revenue_weighted_margin_pct": "{:.2f}", "avg_yoy_growth_pct": "{:+.2f}",
        "total_employees": "{:,.0f}",
    }),
    use_container_width=True, hide_index=True,
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Trend by Sector")
    trend = d.groupby(["fiscal_year", "sector"], as_index=False)["revenue_musd"].sum()
    fig = px.line(trend, x="fiscal_year", y="revenue_musd", color="sector",
                  labels={"revenue_musd": "Revenue ($M)", "fiscal_year": "Fiscal Year"})
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Net Margin Trend by Sector")
    trend_m = d.groupby(["fiscal_year", "sector"], as_index=False)["net_margin_pct"].mean()
    fig2 = px.line(trend_m, x="fiscal_year", y="net_margin_pct", color="sector",
                   labels={"net_margin_pct": "Avg Net Margin (%)", "fiscal_year": "Fiscal Year"})
    fig2.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Statistical trend significance test (OLS slope + p-value) per sector
# ---------------------------------------------------------------------------
st.subheader("Growth Trend Significance Testing")
st.caption(
    "Fits an OLS regression of total sector revenue against fiscal year. "
    "A low p-value (< 0.05) means the growth (or decline) trend is statistically "
    "significant rather than noise — this is the same test an analyst would run "
    "in Excel with `LINEST`/`TREND`, done here at scale across every sector."
)

rows = []
for s in sel:
    series = d[d.sector == s].groupby("fiscal_year")["revenue_musd"].sum().sort_index()
    stats_res = linear_trend_stats(series)
    rows.append({
        "Sector": s,
        "Slope ($M/yr)": stats_res["slope"],
        "R²": stats_res["r2"],
        "p-value": stats_res["p_value"],
        "Significant (p<0.05)": "Yes" if pd.notna(stats_res["p_value"]) and stats_res["p_value"] < 0.05 else "No",
    })
trend_stats_df = pd.DataFrame(rows).sort_values("Slope ($M/yr)", ascending=False)
st.dataframe(
    trend_stats_df.style.format({"Slope ($M/yr)": "{:,.1f}", "R²": "{:.3f}", "p-value": "{:.4f}"}),
    use_container_width=True, hide_index=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
st.subheader(f"Cross-Metric Correlation — FY{year}")
st.caption("Pearson correlation across financial and operational metrics for the selected sectors.")
corr = correlation_matrix(d, year)
fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                  aspect="auto")
fig3.update_layout(height=500)
st.plotly_chart(fig3, use_container_width=True)
