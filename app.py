"""
app.py — Fortune 500 Growth & Performance Analysis
====================================================
Main entry point / Overview page. Additional analysis lives in `pages/`
(Streamlit's native multi-page app routing).

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then on share.streamlit.io point to app.py.
    (See README.md for full step-by-step deployment instructions.)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data, year_range

st.set_page_config(
    page_title="Fortune 500 Growth & Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

min_year, max_year = year_range(df)

# ---------------------------------------------------------------------------
# Sidebar filters (shared mental model across pages via session_state)
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Fortune 500 Analysis")
st.sidebar.caption("Growth & Performance Dashboard · 2014–2023")

selected_year = st.sidebar.slider("Fiscal year", min_year, max_year, max_year, key="year")
selected_sectors = st.sidebar.multiselect(
    "Sectors", options=sorted(df.sector.unique()), default=sorted(df.sector.unique()), key="sectors"
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Pages**\n\n"
    "- Overview *(this page)*\n"
    "- Sector Deep Dive\n"
    "- Growth Forecasting\n"
    "- Clustering & Anomalies\n"
    "- Company Explorer"
)
st.sidebar.divider()
st.sidebar.caption("Data: synthetic panel calibrated to public sector benchmarks. "
                   "See README for methodology.")

d_year = df[(df.fiscal_year == selected_year) & (df.sector.isin(selected_sectors))]
d_all = df[df.sector.isin(selected_sectors)]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Fortune 500 Growth & Performance Analysis")
st.caption(
    f"Portfolio-scale analysis of {df.company_id.nunique()} companies across "
    f"{df.sector.nunique()} sectors, {min_year}–{max_year} · Excel-style KPI rollups, "
    "statistical trend testing, and forecasting built with Python + SQL."
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
prev_year_df = df[(df.fiscal_year == selected_year - 1) & (df.sector.isin(selected_sectors))]

total_rev = d_year.revenue_musd.sum()
prev_rev = prev_year_df.revenue_musd.sum() if len(prev_year_df) else None
rev_delta = f"{(total_rev / prev_rev - 1) * 100:+.1f}% YoY" if prev_rev else None

total_ni = d_year.net_income_musd.sum()
avg_margin = (total_ni / total_rev * 100) if total_rev else 0
prev_margin = (prev_year_df.net_income_musd.sum() / prev_rev * 100) if prev_rev else None
margin_delta = f"{avg_margin - prev_margin:+.1f} pts YoY" if prev_margin is not None else None

total_employees = d_year.employees.sum()
avg_growth = d_year.revenue_growth_pct.mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", f"${total_rev/1000:,.1f}B", rev_delta)
c2.metric("Net Income", f"${total_ni/1000:,.1f}B")
c3.metric("Avg Net Margin", f"{avg_margin:.1f}%", margin_delta)
c4.metric("Avg YoY Growth", f"{avg_growth:.1f}%")
c5.metric("Total Employees", f"{total_employees/1_000_000:,.2f}M")

st.divider()

# ---------------------------------------------------------------------------
# Revenue trend (decade view) + sector mix
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("Aggregate Revenue Trend, 2014–2023")
    trend = d_all.groupby("fiscal_year", as_index=False)["revenue_musd"].sum()
    trend["revenue_bn"] = trend.revenue_musd / 1000
    fig = px.area(trend, x="fiscal_year", y="revenue_bn",
                  labels={"fiscal_year": "Fiscal Year", "revenue_bn": "Revenue ($B)"})
    fig.update_traces(line_color="#2451B7", fillcolor="rgba(36,81,183,0.15)")
    fig.add_vline(x=2020, line_dash="dot", line_color="#C0392B",
                  annotation_text="2020 downturn", annotation_position="top")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"Sector Mix, FY{selected_year}")
    mix = d_year.groupby("sector", as_index=False)["revenue_musd"].sum().sort_values("revenue_musd")
    fig2 = px.bar(mix, x="revenue_musd", y="sector", orientation="h",
                  labels={"revenue_musd": "Revenue ($M)", "sector": ""},
                  color="revenue_musd", color_continuous_scale="Blues")
    fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Top movers table
# ---------------------------------------------------------------------------
st.subheader(f"Top 10 Fastest-Growing Companies, FY{selected_year}")
top_growth = (
    d_year.sort_values("revenue_growth_pct", ascending=False)
    .head(10)[["company_name", "sector", "revenue_musd", "revenue_growth_pct",
               "net_margin_pct", "fortune500_rank"]]
    .rename(columns={
        "company_name": "Company", "sector": "Sector", "revenue_musd": "Revenue ($M)",
        "revenue_growth_pct": "YoY Growth (%)", "net_margin_pct": "Net Margin (%)",
        "fortune500_rank": "F500 Rank",
    })
)
st.dataframe(
    top_growth.style.format({"Revenue ($M)": "{:,.0f}", "YoY Growth (%)": "{:+.1f}",
                              "Net Margin (%)": "{:.1f}"}),
    use_container_width=True, hide_index=True,
)

st.caption(
    "Navigate to **Sector Deep Dive**, **Growth Forecasting**, **Clustering & Anomalies**, "
    "or **Company Explorer** in the sidebar for the full statistical analysis."
)
