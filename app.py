"""
app.py — Fortune 500 Growth & Performance Analysis
====================================================
Entry point / router. Defines the page set and the "Fortune 500" sidebar
section header via st.navigation; each page's actual content lives in
`views/`.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then on share.streamlit.io point to app.py.
    (See README.md for full step-by-step deployment instructions.)
"""

import streamlit as st

st.set_page_config(
    page_title="Fortune 500 Growth & Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page("views/overview.py", title="Overview", icon="📊", default=True)
sector_deep_dive = st.Page("views/sector_deep_dive.py", title="Sector Deep Dive", icon="🏭")
growth_forecasting = st.Page("views/growth_forecasting.py", title="Growth Forecasting", icon="📈")
clustering_anomalies = st.Page("views/clustering_anomalies.py", title="Clustering & Anomalies", icon="🧭")
company_explorer = st.Page("views/company_explorer.py", title="Company Explorer", icon="🔎")

pg = st.navigation({
    "Fortune 500": [overview, sector_deep_dive, growth_forecasting, clustering_anomalies, company_explorer],
})

st.sidebar.caption("Growth & Performance Dashboard · 2014–2023")

pg.run()
