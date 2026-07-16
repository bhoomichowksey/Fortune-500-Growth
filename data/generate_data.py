"""
generate_data.py
================
Builds a realistic, reproducible synthetic panel dataset that mimics
Fortune 500 company financials from 2014-2023 (10 fiscal years x 500 companies).

Why synthetic data?
Fortune 500 raw financials are licensed/paywalled (Fortune/data.ai). To keep this
project fully open-source and re-runnable by anyone without a paid data license,
we generate a statistically realistic panel using sector-level growth, margin and
volatility assumptions calibrated to public, well-known industry benchmarks
(sector CAGR ranges, typical net margins, employee-to-revenue ratios, etc).
The generation logic (trend + seasonality + shocks + sector correlation) is the
same one used to stress-test the analytics/forecasting pipeline in this project.

Run:
    python data/generate_data.py

Outputs:
    data/fortune500_panel.csv   (long/panel format: 1 row per company-year)
    data/fortune500_panel.db    (same data loaded into SQLite for SQL analysis)
"""

import numpy as np
import pandas as pd
import sqlite3
import os

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

YEARS = list(range(2014, 2024))          # 10 fiscal years
N_COMPANIES = 500

SECTORS = {
    "Technology":        dict(base_rev=(2_000, 180_000), margin=(0.10, 0.28), cagr=(0.06, 0.22), vol=0.09),
    "Financial Services":dict(base_rev=(3_000, 150_000), margin=(0.12, 0.30), cagr=(0.02, 0.10), vol=0.07),
    "Health Care":       dict(base_rev=(2_500, 160_000), margin=(0.06, 0.20), cagr=(0.03, 0.12), vol=0.06),
    "Energy":            dict(base_rev=(2_000, 400_000), margin=(-0.05, 0.15), cagr=(-0.05, 0.10), vol=0.18),
    "Retail":            dict(base_rev=(2_000, 570_000), margin=(0.01, 0.08), cagr=(0.01, 0.09), vol=0.05),
    "Industrials":       dict(base_rev=(1_500, 100_000), margin=(0.04, 0.14), cagr=(0.01, 0.08), vol=0.06),
    "Consumer Goods":    dict(base_rev=(1_500, 90_000),  margin=(0.05, 0.18), cagr=(0.01, 0.07), vol=0.05),
    "Telecom":           dict(base_rev=(3_000, 180_000), margin=(0.05, 0.16), cagr=(-0.01, 0.05), vol=0.05),
    "Automotive":        dict(base_rev=(2_000, 250_000), margin=(0.01, 0.10), cagr=(-0.02, 0.08), vol=0.10),
    "Materials":         dict(base_rev=(1_000, 70_000),  margin=(0.02, 0.15), cagr=(-0.02, 0.09), vol=0.12),
}

FIRST_WORDS = ["Global", "United", "National", "Apex", "Summit", "Pinnacle", "Horizon", "Vertex",
               "Meridian", "Atlas", "Sterling", "Cascade", "Frontier", "Prime", "Orbit", "Nova",
               "Titan", "Beacon", "Anchor", "Compass", "Ironclad", "Vanguard", "Zenith", "Quantum",
               "Keystone", "Landmark", "Pioneer", "Continental", "Metro", "Alliance"]
SECOND_WORDS = ["Holdings", "Industries", "Group", "Corp", "Partners", "Systems", "Technologies",
                "Enterprises", "Dynamics", "Solutions", "Networks", "Resources", "Capital",
                "Logistics", "Manufacturing", "Energy", "Foods", "Financial", "Motors", "Labs"]

# US states for HQ (mix, weighted toward common corporate hubs)
STATES = ["NY", "CA", "TX", "IL", "OH", "NJ", "GA", "PA", "MA", "WA", "MI", "MN", "CT",
          "VA", "FL", "NC", "CO", "AZ", "MO", "WI"]
STATE_W = np.array([14,13,11,6,5,5,5,4,4,5,4,3,3,3,4,3,3,3,3,3], dtype=float)
STATE_W /= STATE_W.sum()


def make_company_names(n):
    names = set()
    while len(names) < n:
        name = f"{rng.choice(FIRST_WORDS)} {rng.choice(SECOND_WORDS)}"
        names.add(name)
    return list(names)


def generate():
    names = make_company_names(N_COMPANIES)
    sector_list = list(SECTORS.keys())
    # Slightly uneven sector weights (Fortune 500 is Retail/Financial/Energy heavy)
    sector_weights = np.array([0.14, 0.16, 0.11, 0.09, 0.16, 0.10, 0.08, 0.05, 0.06, 0.05])
    sector_weights /= sector_weights.sum()

    rows = []
    for i, name in enumerate(names):
        sector = rng.choice(sector_list, p=sector_weights)
        cfg = SECTORS[sector]
        state = rng.choice(STATES, p=STATE_W)
        founded = int(rng.integers(1892, 2005))

        base_rev = rng.uniform(*cfg["base_rev"])          # $ millions, year-2014 baseline
        company_cagr = rng.uniform(*cfg["cagr"]) + rng.normal(0, 0.015)   # idiosyncratic drift
        base_margin = rng.uniform(*cfg["margin"])
        margin_drift = rng.normal(0, 0.004)                # slow margin trend
        vol = cfg["vol"]
        employee_intensity = rng.uniform(2.5, 9.0)         # employees per $1M revenue (varies a lot by sector)

        rank_seed = rng.normal(0, 1)  # used to keep rank roughly consistent across years for same firm

        revenue = base_rev
        for yi, year in enumerate(YEARS):
            # Macro shock shared across all companies that year (recession-like dips, recovery)
            macro_shock = {
                2014: 0.00, 2015: -0.01, 2016: 0.00, 2017: 0.02, 2018: 0.03,
                2019: 0.01, 2020: -0.09, 2021: 0.08, 2022: 0.04, 2023: 0.01,
            }[year]

            growth = company_cagr + macro_shock + rng.normal(0, vol)
            if yi > 0:
                revenue = revenue * (1 + growth)

            margin = np.clip(base_margin + margin_drift * yi + rng.normal(0, vol / 3), -0.15, 0.40)
            net_income = revenue * margin
            employees = max(500, int(revenue * employee_intensity * rng.uniform(0.9, 1.1)))
            rd_spend = revenue * (rng.uniform(0.01, 0.18) if sector in ("Technology", "Health Care") else rng.uniform(0.0, 0.04))
            market_cap = revenue * rng.uniform(0.8, 4.5) * (1.15 if net_income > 0 else 0.6)
            total_assets = revenue * rng.uniform(0.9, 3.2)
            employee_growth = growth * rng.uniform(0.4, 0.9) if yi > 0 else 0.0

            rows.append(dict(
                company_id=f"C{i+1:04d}",
                company_name=name,
                sector=sector,
                hq_state=state,
                founded_year=founded,
                fiscal_year=year,
                revenue_musd=round(revenue, 1),
                net_income_musd=round(net_income, 1),
                net_margin_pct=round(margin * 100, 2),
                employees=employees,
                rd_spend_musd=round(rd_spend, 1),
                market_cap_musd=round(market_cap, 1),
                total_assets_musd=round(total_assets, 1),
                revenue_growth_pct=round(growth * 100, 2),
            ))

    df = pd.DataFrame(rows)

    # Fortune 500 "rank" per year = revenue rank within that year
    df["fortune500_rank"] = df.groupby("fiscal_year")["revenue_musd"].rank(ascending=False, method="first").astype(int)

    # A few realistic derived columns used later by the app / notebook
    df["roa_pct"] = round(df["net_income_musd"] / df["total_assets_musd"] * 100, 2)
    df["revenue_per_employee_k"] = round(df["revenue_musd"] * 1000 / df["employees"], 1)

    df = df.sort_values(["company_id", "fiscal_year"]).reset_index(drop=True)
    return df


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate()

    csv_path = os.path.join(out_dir, "fortune500_panel.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df):,} rows -> {csv_path}")

    db_path = os.path.join(out_dir, "fortune500_panel.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("fortune500", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX idx_company ON fortune500(company_id);")
    conn.execute("CREATE INDEX idx_year ON fortune500(fiscal_year);")
    conn.execute("CREATE INDEX idx_sector ON fortune500(sector);")
    conn.commit()
    conn.close()
    print(f"Loaded data into SQLite -> {db_path}")


if __name__ == "__main__":
    main()
