"""views/growth_forecasting.py — ARIMA and Prophet forecasting for revenue projections."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_data, company_options
from utils.analytics import forecast_arima, forecast_prophet


df = load_data()
st.title("📈 Growth Forecasting")
st.caption(
    "Projects future revenue using two independent time-series models — ARIMA (statsmodels) "
    "and Prophet (Meta's additive trend model) — so forecasts can be cross-checked rather "
    "than relying on a single method."
)

level = st.radio("Forecast level", ["Overall market", "Single sector", "Single company"], horizontal=True)

if level == "Overall market":
    series = df.groupby("fiscal_year")["revenue_musd"].sum().sort_index()
    label = "Total Fortune 500 Panel Revenue"
elif level == "Single sector":
    sector = st.selectbox("Sector", sorted(df.sector.unique()))
    series = df[df.sector == sector].groupby("fiscal_year")["revenue_musd"].sum().sort_index()
    label = f"{sector} Sector Revenue"
else:
    opts = company_options(df)
    choice = st.selectbox("Company", opts)
    cid = choice.split("(")[-1].rstrip(")")
    series = df[df.company_id == cid].set_index("fiscal_year")["revenue_musd"].sort_index()
    label = choice

periods = st.slider("Years to forecast", 1, 5, 3)

col1, col2 = st.columns(2)
with col1:
    arima_order_p = st.number_input("ARIMA p (AR order)", 0, 3, 1)
with col2:
    arima_order_q = st.number_input("ARIMA q (MA order)", 0, 3, 1)

if st.button("Run forecast", type="primary"):
    with st.spinner("Fitting ARIMA and Prophet models..."):
        arima_fc = forecast_arima(series, periods=periods, order=(arima_order_p, 1, arima_order_q))
        try:
            prophet_fc = forecast_prophet(series, periods=periods)
            prophet_ok = True
        except Exception as e:
            prophet_ok = False
            st.warning(f"Prophet could not fit this series (falling back to ARIMA only): {e}")

    hist_df = series.reset_index()
    hist_df.columns = ["fiscal_year", "revenue_musd"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df.fiscal_year, y=hist_df.revenue_musd,
                              mode="lines+markers", name="Historical", line=dict(color="#1B1F27")))

    fig.add_trace(go.Scatter(x=arima_fc.fiscal_year, y=arima_fc.forecast_musd,
                              mode="lines+markers", name="ARIMA forecast", line=dict(color="#2451B7", dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(arima_fc.fiscal_year) + list(arima_fc.fiscal_year[::-1]),
        y=list(arima_fc.upper_95_musd) + list(arima_fc.lower_95_musd[::-1]),
        fill="toself", fillcolor="rgba(36,81,183,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="ARIMA 95% CI", showlegend=True,
    ))

    if prophet_ok:
        fig.add_trace(go.Scatter(x=prophet_fc.fiscal_year, y=prophet_fc.forecast_musd,
                                  mode="lines+markers", name="Prophet forecast",
                                  line=dict(color="#C0392B", dash="dot")))

    fig.update_layout(
        title=f"{label} — Historical + Forecast",
        xaxis_title="Fiscal Year", yaxis_title="Revenue ($M)",
        height=520, legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ARIMA**")
        st.dataframe(arima_fc.style.format({"forecast_musd": "{:,.0f}", "lower_95_musd": "{:,.0f}",
                                             "upper_95_musd": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
    with c2:
        if prophet_ok:
            st.markdown("**Prophet**")
            st.dataframe(prophet_fc.style.format({"forecast_musd": "{:,.0f}", "lower_95_musd": "{:,.0f}",
                                                   "upper_95_musd": "{:,.0f}"}),
                         use_container_width=True, hide_index=True)

    st.caption(
        "Note: this dataset is synthetic, so forecasts illustrate methodology rather than "
        "real market predictions. Swap in real filings (e.g. via SEC EDGAR) to make this "
        "production-grade — see README for the data-source-swap instructions."
    )
else:
    st.info("Configure options above, then click **Run forecast**.")
