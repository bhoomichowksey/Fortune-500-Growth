"""pages/4_Company_Explorer.py — single-company deep dive across the full decade."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_data, company_options
from utils.analytics import compute_cagr

st.set_page_config(page_title="Company Explorer", page_icon="🔎", layout="wide")

df = load_data()
st.title("🔎 Company Explorer")
st.caption("Full financial history, CAGR, and peer benchmarking for any single company in the panel.")

opts = company_options(df)
choice = st.selectbox("Search / select a company", opts)
cid = choice.split("(")[-1].rstrip(")")
c = df[df.company_id == cid].sort_values("fiscal_year")
sector = c.sector.iloc[0]

cagr_df = compute_cagr(df[df.company_id == cid])
cagr_val = cagr_df.cagr_pct.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sector", sector)
c2.metric("Revenue CAGR (2014–2023)", f"{cagr_val:.1f}%" if pd.notna(cagr_val) else "n/a")
c3.metric(f"FY{int(c.fiscal_year.max())} Revenue", f"${c.revenue_musd.iloc[-1]:,.0f}M")
c4.metric(f"FY{int(c.fiscal_year.max())} F500 Rank", f"#{int(c.fortune500_rank.iloc[-1])}")

st.divider()

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=c.fiscal_year, y=c.revenue_musd, name="Revenue ($M)", marker_color="#2451B7"),
              secondary_y=False)
fig.add_trace(go.Scatter(x=c.fiscal_year, y=c.net_margin_pct, name="Net Margin (%)",
                          mode="lines+markers", line=dict(color="#C0392B")), secondary_y=True)
fig.update_layout(title=f"{c.company_name.iloc[0]} — Revenue & Margin, 2014–2023", height=460,
                   legend=dict(orientation="h", yanchor="bottom", y=-0.3))
fig.update_yaxes(title_text="Revenue ($M)", secondary_y=False)
fig.update_yaxes(title_text="Net Margin (%)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Fortune 500 Rank Over Time")
    fig2 = go.Figure(go.Scatter(x=c.fiscal_year, y=c.fortune500_rank, mode="lines+markers",
                                 line=dict(color="#1B1F27")))
    fig2.update_yaxes(autorange="reversed", title="Rank (lower is better)")
    fig2.update_layout(height=360, xaxis_title="Fiscal Year")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader(f"vs. Sector Peers ({sector}), FY{int(c.fiscal_year.max())}")
    peer_year = df[(df.sector == sector) & (df.fiscal_year == c.fiscal_year.max())]
    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=peer_year.revenue_growth_pct, name="Sector peers", marker_color="#B0B7C3"))
    fig3.add_trace(go.Scatter(
        y=[c.revenue_growth_pct.iloc[-1]], x=["Sector peers"], mode="markers",
        marker=dict(color="#C0392B", size=14, symbol="diamond"), name=c.company_name.iloc[0],
    ))
    fig3.update_layout(height=360, yaxis_title="YoY Revenue Growth (%)")
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Full Financial History")
show = c[["fiscal_year", "revenue_musd", "net_income_musd", "net_margin_pct", "revenue_growth_pct",
          "employees", "rd_spend_musd", "market_cap_musd", "roa_pct", "fortune500_rank"]]
st.dataframe(
    show.rename(columns={
        "fiscal_year": "Year", "revenue_musd": "Revenue ($M)", "net_income_musd": "Net Income ($M)",
        "net_margin_pct": "Net Margin (%)", "revenue_growth_pct": "YoY Growth (%)",
        "employees": "Employees", "rd_spend_musd": "R&D Spend ($M)", "market_cap_musd": "Market Cap ($M)",
        "roa_pct": "ROA (%)", "fortune500_rank": "F500 Rank",
    }).style.format({
        "Revenue ($M)": "{:,.0f}", "Net Income ($M)": "{:,.0f}", "Net Margin (%)": "{:.1f}",
        "YoY Growth (%)": "{:+.1f}", "Employees": "{:,.0f}", "R&D Spend ($M)": "{:,.0f}",
        "Market Cap ($M)": "{:,.0f}", "ROA (%)": "{:.1f}",
    }),
    use_container_width=True, hide_index=True,
)

csv = show.to_csv(index=False).encode("utf-8")
st.download_button("Download this company's history as CSV", csv,
                    file_name=f"{c.company_name.iloc[0].replace(' ', '_')}_history.csv", mime="text/csv")
