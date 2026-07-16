"""
utils/analytics.py
===================
Reusable analytics functions shared by the Jupyter notebook and the Streamlit
app: growth metrics, time-series forecasting, clustering, and anomaly detection.
Kept dependency-light and cached-friendly so Streamlit can call these directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Growth metrics
# ---------------------------------------------------------------------------

def compute_cagr(df: pd.DataFrame, value_col: str = "revenue_musd",
                  group_col: str = "company_id", year_col: str = "fiscal_year") -> pd.DataFrame:
    """CAGR per group between first and last year present in df."""
    out = []
    for gid, g in df.groupby(group_col):
        g = g.sort_values(year_col)
        first, last = g.iloc[0], g.iloc[-1]
        n_years = last[year_col] - first[year_col]
        if n_years <= 0 or first[value_col] <= 0:
            cagr = np.nan
        else:
            cagr = (last[value_col] / first[value_col]) ** (1 / n_years) - 1
        out.append({group_col: gid, "cagr_pct": cagr * 100 if pd.notna(cagr) else np.nan})
    return pd.DataFrame(out)


def sector_summary(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Sector-level rollup for a given fiscal year (revenue-weighted margin included)."""
    d = df[df.fiscal_year == year]
    agg = d.groupby("sector").apply(
        lambda x: pd.Series({
            "num_companies": x.company_id.nunique(),
            "total_revenue_musd": x.revenue_musd.sum(),
            "avg_net_margin_pct": x.net_margin_pct.mean(),
            "revenue_weighted_margin_pct": (x.net_income_musd.sum() / x.revenue_musd.sum()) * 100,
            "avg_yoy_growth_pct": x.revenue_growth_pct.mean(),
            "total_employees": x.employees.sum(),
        })
    ).reset_index()
    return agg.sort_values("total_revenue_musd", ascending=False)


def linear_trend_stats(series: pd.Series) -> dict:
    """Fit revenue-vs-year OLS trend; return slope, r^2, p-value for significance testing."""
    x = np.arange(len(series))
    if len(series) < 3 or series.isna().any():
        return {"slope": np.nan, "r2": np.nan, "p_value": np.nan}
    slope, intercept, r, p, se = stats.linregress(x, series.values)
    return {"slope": slope, "r2": r ** 2, "p_value": p}


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def forecast_arima(history: pd.Series, periods: int = 3, order=(1, 1, 1)) -> pd.DataFrame:
    """
    Fit an ARIMA(p,d,q) model on an annual revenue series and forecast `periods`
    years ahead, returning point forecast + 95% CI.
    `history` must be indexed by fiscal_year (int) and sorted ascending.
    """
    series = history.astype(float).reset_index(drop=True)
    try:
        model = ARIMA(series, order=order)
        fit = model.fit()
        fc = fit.get_forecast(steps=periods)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        last_year = int(history.index.max())
        years = [last_year + i + 1 for i in range(periods)]
        return pd.DataFrame({
            "fiscal_year": years,
            "forecast_musd": mean.values,
            "lower_95_musd": ci.iloc[:, 0].values,
            "upper_95_musd": ci.iloc[:, 1].values,
        })
    except Exception:
        # Fallback: naive CAGR-based projection if ARIMA fails to converge (e.g. too few points)
        growth = (series.iloc[-1] / series.iloc[0]) ** (1 / max(len(series) - 1, 1)) - 1
        last_year = int(history.index.max())
        vals, cur = [], series.iloc[-1]
        for i in range(periods):
            cur = cur * (1 + growth)
            vals.append(cur)
        return pd.DataFrame({
            "fiscal_year": [last_year + i + 1 for i in range(periods)],
            "forecast_musd": vals,
            "lower_95_musd": [v * 0.85 for v in vals],
            "upper_95_musd": [v * 1.15 for v in vals],
        })


def forecast_prophet(history: pd.Series, periods: int = 3) -> pd.DataFrame:
    """Prophet-based alternative forecast (captures trend + uncertainty differently than ARIMA)."""
    from prophet import Prophet

    dfp = pd.DataFrame({
        "ds": pd.to_datetime(history.index.astype(int).astype(str) + "-12-31"),
        "y": history.values,
    })
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                interval_width=0.95)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq="YE")
    fcst = m.predict(future)
    tail = fcst.tail(periods)
    return pd.DataFrame({
        "fiscal_year": tail["ds"].dt.year.values,
        "forecast_musd": tail["yhat"].values,
        "lower_95_musd": tail["yhat_lower"].values,
        "upper_95_musd": tail["yhat_upper"].values,
    })


# ---------------------------------------------------------------------------
# Clustering & anomaly detection
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = ["revenue_growth_pct", "net_margin_pct", "roa_pct", "revenue_per_employee_k"]


def cluster_companies(df: pd.DataFrame, year: int, k: int = 4, features=None, random_state=42):
    """
    KMeans cluster companies for a given year on growth/profitability/efficiency
    features. Returns the dataframe with a `cluster` column plus the fitted
    scaler/model for reuse, and a human-readable label per cluster based on
    centroid characteristics (e.g. "High growth, high margin").
    """
    features = features or CLUSTER_FEATURES
    d = df[df.fiscal_year == year].dropna(subset=features).copy()
    X = d[features].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    d["cluster"] = km.fit_predict(Xs)

    # Label clusters by relative centroid position on growth & margin
    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features)
    growth_median = centroids["revenue_growth_pct"].median()
    margin_median = centroids["net_margin_pct"].median()

    labels = {}
    for cid, row in centroids.iterrows():
        g = "High growth" if row["revenue_growth_pct"] >= growth_median else "Low growth"
        m = "high margin" if row["net_margin_pct"] >= margin_median else "low margin"
        labels[cid] = f"{g}, {m}"
    d["cluster_label"] = d["cluster"].map(labels)
    return d, centroids, labels


def detect_anomalies(df: pd.DataFrame, year: int, contamination: float = 0.05,
                      features=None, random_state=42) -> pd.DataFrame:
    """
    IsolationForest anomaly detection across financial metrics for a given year
    — flags companies whose combination of growth/margin/efficiency is unusual
    (either exceptional outperformers or red-flag underperformers).
    """
    features = features or CLUSTER_FEATURES
    d = df[df.fiscal_year == year].dropna(subset=features).copy()
    X = StandardScaler().fit_transform(d[features].values)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    d["anomaly_score"] = iso.fit_predict(X)          # -1 = anomaly, 1 = normal
    d["anomaly_flag"] = d["anomaly_score"] == -1
    d["anomaly_severity"] = -iso.decision_function(X)  # higher = more anomalous
    return d.sort_values("anomaly_severity", ascending=False)


def correlation_matrix(df: pd.DataFrame, year: int, features=None) -> pd.DataFrame:
    features = features or ["revenue_musd", "revenue_growth_pct", "net_margin_pct", "roa_pct",
                             "rd_spend_musd", "employees", "revenue_per_employee_k", "market_cap_musd"]
    d = df[df.fiscal_year == year]
    return d[features].corr()
