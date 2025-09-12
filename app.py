# /// script
# dependencies = ["streamlit", "pandas", "plotly.express"]
# ///
import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Ironman Splits Viewer", layout="wide")

# Optional auto-refresh every 60 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="auto-refresh")
except Exception:
    pass

DATA_FILE = "long.csv"

@st.cache_data(ttl=60)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Keep expected columns
    keep = [c for c in ["name", "split", "netTime"] if c in df.columns]
    df = df[keep]

    # Parse netTime into timedelta
    def parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        # Normalize 1:23:45 -> 01:23:45 for to_timedelta
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
            s = "0" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            return pd.NaT

    df["net_td"] = df["netTime"].apply(parse_td)

    # Normalize text columns
    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    return df

That's not quite right. They are still displaying with the distance behind the leader going up the X axis. Try calculating the difference the other way, so leadertime - notleadertimer, so all figures will be negative

with st.expander("Show data"):
    st.dataframe(
        sel.sort_values(["name", "split"])[["name", "split", "netTime"]].reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
