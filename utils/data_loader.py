"""
utils/data_loader.py
=====================
Central data-access layer. Loads the panel dataset from SQLite (falls back to
CSV) and exposes small typed helper accessors so the Streamlit pages don't
duplicate I/O logic.
"""

import os
import sqlite3
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "fortune500_panel.db")
CSV_PATH = os.path.join(DATA_DIR, "fortune500_panel.csv")


@st.cache_data(show_spinner="Loading Fortune 500 panel dataset...")
def load_data() -> pd.DataFrame:
    """Load the panel dataset, preferring SQLite (matches the SQL layer used
    in the notebook / Tableau / Power BI), falling back to CSV if the .db
    file isn't present (e.g. fresh clone before running generate_data.py)."""
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM fortune500", conn)
        conn.close()
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first to "
            "generate data/fortune500_panel.csv and .db."
        )
    return df


def year_range(df: pd.DataFrame):
    return int(df.fiscal_year.min()), int(df.fiscal_year.max())


def sector_list(df: pd.DataFrame):
    return sorted(df.sector.unique().tolist())


def company_options(df: pd.DataFrame):
    return sorted(df[["company_id", "company_name"]].drop_duplicates()
                  .apply(lambda r: f"{r.company_name} ({r.company_id})", axis=1).tolist())
