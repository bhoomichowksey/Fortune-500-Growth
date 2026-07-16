"""views/clustering_anomalies.py — KMeans performance segmentation + IsolationForest anomaly detection."""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.analytics import cluster_companies, detect_anomalies, CLUSTER_FEATURES


df = load_data()
st.title("🧭 Performance Clustering & Anomaly Detection")
st.caption(
    "Unsupervised segmentation (KMeans) groups companies into performance archetypes; "
    "IsolationForest flags statistical outliers — unusually strong or weak performers "
    "relative to their peers on the same set of features."
)

year = st.slider("Fiscal year", int(df.fiscal_year.min()), int(df.fiscal_year.max()), int(df.fiscal_year.max()))
k = st.slider("Number of clusters (k)", 2, 8, 4)

st.subheader("Performance Clusters")
clustered, centroids, labels = cluster_companies(df, year, k=k)

fig = px.scatter(
    clustered, x="revenue_growth_pct", y="net_margin_pct", color="cluster_label",
    size="revenue_musd", hover_name="company_name", hover_data={"sector": True, "revenue_musd": ":,.0f"},
    labels={"revenue_growth_pct": "YoY Revenue Growth (%)", "net_margin_pct": "Net Margin (%)"},
    title=f"Growth vs. Margin Clusters — FY{year}",
)
fig.update_layout(height=550, legend=dict(orientation="h", yanchor="bottom", y=-0.35))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Cluster centroids (feature averages)"):
    centroids_display = centroids.copy()
    centroids_display.insert(0, "cluster_label", [labels[i] for i in centroids_display.index])
    st.dataframe(centroids_display.style.format({c: "{:.2f}" for c in CLUSTER_FEATURES}),
                 use_container_width=True, hide_index=True)

sector_mix = pd.crosstab(clustered.cluster_label, clustered.sector)
st.subheader("Sector Composition per Cluster")
st.dataframe(sector_mix, use_container_width=True)

st.divider()

st.subheader("Anomaly Detection")
contamination = st.slider("Expected anomaly rate", 0.01, 0.15, 0.05, step=0.01,
                           help="IsolationForest contamination parameter — the assumed fraction of outliers.")
anomalies = detect_anomalies(df, year, contamination=contamination)
flagged = anomalies[anomalies.anomaly_flag].head(30)

st.caption(f"{len(flagged)} companies flagged as statistical outliers out of "
           f"{len(anomalies)} evaluated for FY{year}.")

fig2 = px.scatter(
    anomalies, x="revenue_growth_pct", y="net_margin_pct", color="anomaly_flag",
    color_discrete_map={True: "#C0392B", False: "#B0B7C3"},
    hover_name="company_name", hover_data={"sector": True},
    labels={"revenue_growth_pct": "YoY Revenue Growth (%)", "net_margin_pct": "Net Margin (%)",
            "anomaly_flag": "Flagged"},
    title=f"Anomaly Map — FY{year}",
)
fig2.update_layout(height=480)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Flagged Companies (by anomaly severity)")
show_cols = ["company_name", "sector", "revenue_growth_pct", "net_margin_pct", "roa_pct",
             "revenue_per_employee_k", "anomaly_severity"]
st.dataframe(
    flagged[show_cols].rename(columns={
        "company_name": "Company", "sector": "Sector", "revenue_growth_pct": "YoY Growth (%)",
        "net_margin_pct": "Net Margin (%)", "roa_pct": "ROA (%)",
        "revenue_per_employee_k": "Revenue/Employee ($k)", "anomaly_severity": "Anomaly Severity",
    }).style.format({"YoY Growth (%)": "{:+.1f}", "Net Margin (%)": "{:.1f}", "ROA (%)": "{:.1f}",
                     "Revenue/Employee ($k)": "{:,.0f}", "Anomaly Severity": "{:.3f}"}),
    use_container_width=True, hide_index=True,
)
