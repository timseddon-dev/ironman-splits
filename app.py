import os
import sys
import time
import math
import pandas as pd
import streamlit as st

# Optional charting (installed via requirements)
import plotly.express as px

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="IRONMAN Splits Viewer", layout="wide")
st.title("IRONMAN Splits Viewer")

# Small helper to show environment versions
with st.expander("Environment info", expanded=False):
    st.write("Working directory:", os.getcwd())
    try:
        import numpy as np
        st.write("Python:", sys.version)
        st.write("Streamlit:", st.__version__)
        st.write("Pandas:", pd.__version__)
    except Exception:
        pass

CSV_PATH = "long.csv"

# ------------------------------
# Load data
# ------------------------------
if not os.path.exists(CSV_PATH):
    st.error("long.csv not found in the repository root.")
    st.info("To create it: go to GitHub → Actions → 'Run updater (manual)' → Run workflow. Then refresh this app.")
    st.stop()

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error("Could not read long.csv.")
    st.exception(e)
    st.stop()

if df.empty:
    st.warning("long.csv is empty. Run the GitHub Action to fetch data, then refresh.")
    st.stop()

# Normalize column names we expect
for col in ["name", "bib", "split", "netTime", "label", "placeChange", "pace", "speed", "timestamp"]:
    if col not in df.columns:
        df[col] = None

# Clean types
df["name"] = df["name"].astype(str).str.strip().replace({"None": None, "nan": None})
df["split"] = df["split"].astype(str).str.strip().replace({"None": None, "nan": None})
df["label"] = df["label"].astype(str).str.strip().replace({"None": None, "nan": None})
df["placeChange"] = df["placeChange"].astype(str).str.strip().replace({"None": None, "nan": None})
df["pace"] = df["pace"].astype(str).str.strip().replace({"None": None, "nan": None})
df["speed"] = df["speed"].astype(str).str.strip().replace({"None": None, "nan": None})

# bib could be numeric or string; keep as string for easy matching
df["bib"] = df["bib"].astype(str).str.strip().replace({"None": None, "nan": None})

# ------------------------------
# Helpers
# ------------------------------
def parse_net_time_to_seconds(t):
    """
    Parse netTime variants like:
    - "00:45:10.609"
    - "5:12:34"
    - "45:10"
    Returns float seconds or None.
    """
    if pd.isna(t):
        return None
    s = str(t).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    # Split by ':'
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except Exception:
        return None

    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    elif len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    elif len(parts) == 1:
        return parts[0]  # already seconds
    return None

df["netTime_sec"] = df["netTime"].apply(parse_net_time_to_seconds)

def split_sort_key(s):
    """Sort splits in a sensible triathlon order."""
    s = str(s or "")
    # Map common exact points
    if s == "SWIM": return (0, 0)
    if s == "T1": return (1, 0)
    if s.startswith("BIKE"):
        # Handle "BIKE1"..."BIKE25" and "BIKE"
        if s == "BIKE": return (2, 999)
        try:
            n = int(s.replace("BIKE", ""))
            return (2, n)
        except Exception:
            return (2, 998)
    if s == "T2": return (3, 0)
    if s.startswith("RUN"):
        # Handle "RUN1"..."RUN22" and "RUN"
        if s == "RUN": return (4, 999)
        try:
            n = int(s.replace("RUN", ""))
            return (4, n)
        except Exception:
            return (4, 998)
    if s == "FINISH": return (5, 0)
    return (9, s)

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Filters")

# Split filter (multiselect shows all available)
all_splits = sorted(df["split"].dropna().unique().tolist(), key=split_sort_key)
selected_splits = st.sidebar.multiselect("Splits", options=all_splits, default=all_splits)

# Name search
name_query = st.sidebar.text_input("Search name (substring)")

# Bib search
bib_query = st.sidebar.text_input("Search bib (exact or substring)")

# Place change quick filter
place_change_choice = st.sidebar.selectbox(
    "Place change filter",
    options=["All", "Positive only", "Negative only", "Zero only", "Missing only"],
    index=0,
)

# Apply filters
df_view = df.copy()

if selected_splits:
    df_view = df_view[df_view["split"].isin(selected_splits)]

if name_query:
    nq = name_query.strip().lower()
    df_view = df_view[df_view["name"].fillna("").str.lower().str.contains(nq)]

if bib_query:
    bq = bib_query.strip().lower()
    df_view = df_view[df_view["bib"].fillna("").str.lower().str.contains(bq)]

if place_change_choice != "All":
    def pc_bucket(x):
        if x in (None, "", "None", "nan"):
            return "Missing"
        try:
            val = float(x)
        except Exception:
            return "Missing"
        if val > 0:
            return "Positive"
        if val < 0:
            return "Negative"
        return "Zero"
    bucket = {
        "Positive only": "Positive",
        "Negative only": "Negative",
        "Zero only": "Zero",
        "Missing only": "Missing",
    }[place_change_choice]
    df_view = df_view[df_view["placeChange"].apply(pc_bucket) == bucket]

# Order for readability
df_view = df_view.sort_values(["name", "split"], key=lambda c: c.map(split_sort_key) if c.name == "split" else c)

# ------------------------------
# Top metrics
# ------------------------------
st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df_view):,}")
c2.metric("Athletes", f"{df_view['name'].nunique():,}")
c3.metric("Splits", f"{df_view['split'].nunique():,}")
# Show how many have placeChange
has_pc = df_view["placeChange"].notna().sum()
c4.metric("Rows with placeChange", f"{has_pc:,}")

st.caption("Need newer data? Run the GitHub Action:")
st.link_button("Run updater (manual) on GitHub", url="https://github.com/timseddon-dev/ironman-splits/actions")

# ------------------------------
# Data preview
# ------------------------------
st.subheader("Data preview")
preview_cols = ["name", "bib", "split", "netTime", "label", "placeChange", "pace", "speed", "timestamp"]
existing_preview_cols = [c for c in preview_cols if c in df_view.columns]
st.dataframe(df_view[existing_preview_cols].head(250), use_container_width=True)

# ------------------------------
# Charts
# ------------------------------
st.subheader("Charts")

tab1, tab2 = st.tabs(["Distribution by split (box plot)", "Per-athlete split trend"])

with tab1:
    df_box = df_view.dropna(subset=["netTime_sec", "split"]).copy()
    if df_box.empty:
        st.info("No parsable netTime values yet for charting.")
    else:
        df_box["split_sorted"] = df_box["split"].apply(split_sort_key)
        df_box = df_box.sort_values(["split_sorted"])
        fig = px.box(
            df_box,
            x="split",
            y="netTime_sec",
            points=False,
            title="Net time (seconds) by split",
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Let user choose athletes for the trend
    athletes = df_view["name"].dropna().unique().tolist()
    default_sel = athletes[:5] if len(athletes) > 0 else []
    selected_athletes = st.multiselect("Select athletes", options=athletes, default=default_sel, key="ath_trend")

    df_trend = df_view[df_view["name"].isin(selected_athletes)].dropna(subset=["netTime_sec", "split"]).copy()
    if df_trend.empty:
        st.info("Select athletes that have netTime values to see trends.")
    else:
        # Keep split order consistent
        df_trend["split_sorted"] = df_trend["split"].apply(split_sort_key)
        df_trend = df_trend.sort_values(["name", "split_sorted"])
        fig2 = px.line(
            df_trend,
            x="split",
            y="netTime_sec",
            color="name",
            markers=True,
            title="Per-athlete net time trend across splits",
        )
        fig2.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# Place change highlight table
# ------------------------------
st.subheader("Place change highlights")
df_pc = df_view.copy()
def pc_numeric(x):
    try:
        return float(x)
    except Exception:
        return math.nan
df_pc["placeChange_num"] = df_pc["placeChange"].apply(pc_numeric)

if df_pc["placeChange_num"].notna().any():
    # Sort by magnitude of change for visibility
    df_pc = df_pc.sort_values("placeChange_num", key=lambda s: s.abs(), ascending=False)
    show_cols = ["name", "bib", "split", "netTime", "label", "placeChange", "pace", "speed"]
    show_cols = [c for c in show_cols if c in df_pc.columns]
    st.dataframe(df_pc[show_cols].head(100), use_container_width=True)
else:
    st.info("No numeric placeChange values in the current view.")

# ------------------------------
# Footer info
# ------------------------------
st.divider()
st.caption("Tip: Use the sidebar to filter by split, name, and bib. Charts update automatically.")
