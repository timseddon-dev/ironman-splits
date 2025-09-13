# app.py
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")
st.title("Race Gaps vs Leader")


# ======================================
# 0) Imports, Config, and Constants
# ======================================
# /// script
# dependencies = ["streamlit", "pandas", "plotly-express"]
# ///
import os
import re
import math
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

# ======================================
# 1) Data Loading and Utilities — FIXED to always produce net_td
# ======================================
@st.cache_data(ttl=60)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Normalize column names (trim/lower for safety), but keep originals to reference
    cols_lower = {c: c.strip() for c in df.columns}
    df.rename(columns=cols_lower, inplace=True)

    # Accept either 'netTime' (original) or 'net_td' if already present
    has_netTime = "netTime" in df.columns
    has_net_td = "net_td" in df.columns

    # Keep only expected columns that exist
    keep = [c for c in ["name", "split", "netTime", "net_td"] if c in df.columns]
    df = df[keep].copy()

    # If net_td is missing but netTime exists, derive it
    if not has_net_td and has_netTime:
        def parse_td(x):
            if pd.isna(x):
                return pd.NaT
            s = str(x).strip()
            # Normalize 1:23:45 -> 01:23:45
            if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
                s = "0" + s
            try:
                return pd.to_timedelta(s)
            except Exception:
                return pd.NaT
        df["net_td"] = df["netTime"].apply(parse_td)

    # If both exist but net_td is empty, try to parse again from netTime
    if "net_td" in df.columns and df["net_td"].isna().all() and "netTime" in df.columns:
        try:
            df["net_td"] = pd.to_timedelta(df["netTime"])
        except Exception:
            def parse_td2(x):
                try:
                    return pd.to_timedelta(str(x).strip())
                except Exception:
                    return pd.NaT
            df["net_td"] = df["netTime"].apply(parse_td2)

    # Normalize text columns
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Final minimal columns check
    if "name" not in df.columns or "split" not in df.columns or "net_td" not in df.columns:
        # Give a precise message with available columns
        st.error(f"long.csv must include name/split plus either netTime (so we can derive net_td) or net_td directly. Found columns: {list(df.columns)}")
        return pd.DataFrame()

    return df.dropna(subset=["name", "split"]).copy()


def expected_order():
    return (
        ["SWIM", "T1"]
        + [f"BIKE{i}" for i in range(1, 26)]
        + ["BIKE", "T2"]
        + [f"RUN{i}" for i in range(1, 23)]
        + ["FINISH"]
    )


def available_splits_in_order(df: pd.DataFrame):
    order = expected_order()
    present = [s for s in order if s in df["split"].unique()]
    if present:
        return present
    return sorted(df["split"].dropna().unique().tolist())


def compute_leaders(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["net_td"]).copy()
    leaders = d.groupby("split", as_index=False)["net_td"].min().rename(columns={"net_td": "leader_td"})
    return leaders


def latest_common_split(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    order = available_splits_in_order(df)
    for s in reversed(order):
        if df.loc[(df["split"] == s) & (~df["net_td"].isna())].shape[0] > 0:
            return s
    return None


def compute_positions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["net_td"]).copy()
    d["pos"] = d.groupby("split")["net_td"].rank(method="first")
    return d


def minute_ticks(series: pd.Series, min_step: int = 1):
    if series.empty:
        return []
    lo, hi = float(series.min()), float(series.max())
    start = math.floor(lo)
    end = math.ceil(hi)
    step = max(1, min_step)
    return list(range(start, end + 1, step))


# ======================================
# 2.0) Load df and prepare options — unchanged
# ======================================
df = load_data(DATA_FILE)
if df.empty:
    st.error("No usable data found in long.csv. See message above for required columns.")
    st.stop()

all_athletes = sorted(df["name"].dropna().unique().tolist())
splits_ordered = available_splits_in_order(df)
default_selection = all_athletes[:6] if len(all_athletes) >= 6 else all_athletes


# ======================================
# 2.1) UI controls (Test Mode + Max hours) — unchanged
# ======================================
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 7.0, 0.5)

colA, colB = st.columns(2)
with colA:
    selected = st.multiselect("Athletes (ordered by current position)", options=all_athletes, default=default_selection)
with colB:
    from_split = st.selectbox("From split", options=splits_ordered, index=splits_ordered.index("SWIM") if "SWIM" in splits_ordered else 0)
    to_split = st.selectbox("To split", options=splits_ordered, index=splits_ordered.index("FINISH") if "FINISH" in splits_ordered else len(splits_ordered) - 1)


# ======================================
# 2.2) Apply test filter (per-athlete) — now guarded
# ======================================
if test_mode:
    if "net_td" not in df.columns:
        st.error("Test filter requires a 'net_td' Timedelta column. Check your long.csv (needs netTime or net_td).")
        st.stop()

    max_td = pd.to_timedelta(max_hours, unit="h")
    before_rows = len(df)
    df = df.dropna(subset=["net_td"]).copy()
    df = df[df["net_td"] < max_td].copy()
    st.caption(f"Test mode active: {len(df):,} rows (from {before_rows:,}) with athlete elapsed < {max_hours:.1f}h")


# =# ======================================
# 2.5) Race snapshot (Top 10)
# ======================================
import re

def friendly_split_label(split: str) -> str:
    s = str(split).upper()
    if s == "FINISH":
        return "Finish"
    if s == "SWIM":
        return "Swim 3.8 km"
    if s == "T1":
        return "T1"
    if s == "T2":
        return "T2"
    m = re.match(r"BIKE(\d+)$", s)
    if m:
        i = int(m.group(1))
        segments = 25
        total_km = 180.0
        km = total_km * min(max(i, 1), segments) / segments
        return f"Bike {km:.1f} km"
    m = re.match(r"RUN(\d+)$", s)
    if m:
        i = int(m.group(1))
        segments = 22
        total_km = 42.2
        km = total_km * min(max(i, 1), segments) / segments
        return f"Run {km:.1f} km"
    if s == "BIKE":
        return "Bike"
    return s

# Leaders (computed on current df after any Test mode filtering)
leaders_now = (
    df.dropna(subset=["net_td"])
      .sort_values(["split", "net_td"])
      .groupby("split", as_index=False)
      .agg(leader_td=("net_td", "min"))
)

df_now = df.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])

st.subheader("Race snapshot (Top 10)")
if df_now.empty:
    st.info("No data available for the current dataset.")
else:
    latest_now = (
        df_now.sort_values(["name", "net_td"])
              .groupby("name", as_index=False)
              .tail(1)
              .reset_index(drop=True)
    )
    latest_now["gap_min"] = (latest_now["net_td"] - latest_now["leader_td"]).dt.total_seconds() / 60.0
    latest_now["gap_min"] = latest_now["gap_min"].clip(lower=0)
    snapshot = latest_now.sort_values(["leader_td", "gap_min"], ascending=[False, True])

    top10 = (
        snapshot[["name", "split", "gap_min"]]
        .rename(columns={"name": "Athlete", "split": "Latest split", "gap_min": "Behind (min)"})
        .copy()
    )
    top10["Latest split"] = top10["Latest split"].map(friendly_split_label)
    top10["Behind (min)"] = top10["Behind (min)"].map(lambda x: f"{x:.1f}")

    top10_out = top10.head(10).reset_index(drop=True)
    top10_out.index = top10_out.index + 1  # number from 1..10
    st.dataframe(top10_out, use_container_width=True, height=320)
# ======================================
# 3) Plot prep
# ======================================

# Keep split categorical with master order
master_order = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=master_order, ordered=True)
except Exception:
    pass

def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1+1]
    else:
        return splits[i1:i0+1]

range_splits = split_range(master_order, from_split, to_split)

# Align the requested range to the splits that still exist in df
splits_in_df = set(df["split"].dropna().astype(str).unique().tolist())
range_splits = [s for s in range_splits if str(s) in splits_in_df]
if not range_splits:
    st.info("No splits remaining in the selected range after filtering. Try widening the range or turning Test mode off.")
    st.stop()

# Selection
sel = df[(df["name"].isin(selected)) & (df["split"].astype(str).isin(range_splits))].copy()
if sel.empty:
    st.info("No rows match the current athlete selection and split range.")
    st.stop()

# Leaders on the same filtered df scope
leaders_now = (
    df.dropna(subset=["net_td"])
      .sort_values(["split", "net_td"])
      .groupby("split", as_index=False)
      .agg(leader_td=("net_td", "min"))
)

# Merge and compute X/Y
xy_df = sel.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])
xy_df["leader_hr"] = xy_df["leader_td"].dt.total_seconds() / 3600.0
xy_df["y_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0

# Friendly labels used in hover
def friendly_split_label_for_plot(split: str) -> str:
    return friendly_split_label(split)

# Include START at 0 if in range
include_start = (len(range_splits) > 0 and str(range_splits[0]).upper() == "START")
if include_start:
    start_rows = pd.DataFrame({
        "name": list(dict.fromkeys(selected)),
        "split": "START",
        "leader_td": pd.to_timedelta(0, unit="s"),
        "net_td": pd.to_timedelta(0, unit="s"),
        "leader_hr": 0.0,
        "y_gap_min": 0.0,
    })
    xy_df = pd.concat([start_rows, xy_df], ignore_index=True, sort=False)

xy_df["split_label"] = xy_df["split"].astype(str).map(friendly_split_label_for_plot)
xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")
if xy_df.empty:
    st.info("No rows to plot for the current selection.")
    st.stop()
    
# ======================================
# 4) Plot
# ======================================
import plotly.graph_objects as go

fig = go.Figure()

for nm, g in xy_df.groupby("name", sort=False):
    g = g.sort_values("leader_hr")
    fig.add_trace(go.Scatter(
        x=g["leader_hr"],
        y=g["y_gap_min"],
        mode="lines",
        line=dict(width=1.8),
        name=nm,
        showlegend=False,
        hovertemplate="Athlete: %{text}<br>Split: %{meta}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{y:.1f} min",
        text=[nm]*len(g),
        meta=g["split_label"],
    ))

# End labels
end_annotations = []
stagger_cycle = [-0.15, +0.15, -0.25, +0.25, -0.35, +0.35, 0.0]
last_points = (
    xy_df.sort_values(["name", "leader_hr"])
         .groupby("name", as_index=False)
         .tail(1)
         .reset_index(drop=True)
)
for i, row in last_points.iterrows():
    dy = stagger_cycle[i % len(stagger_cycle)]
    end_annotations.append(dict(
        x=float(row["leader_hr"]) + 0.01,
        y=float(row["y_gap_min"]) + dy,
        xref="x",
        yref="y",
        text=str(row["name"]),
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(size=12, color="rgba(0,0,0,1)"),
    ))


# ======================================
# 5) Axes and Layout
# ======================================
def fmt_hmm(hours_float: float) -> str:
    total_minutes = int(round(hours_float * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh}:{mm:02d}"

def hour_ticks(lo_h: float, hi_h: float, step: float = 0.5) -> list:
    start = math.floor(lo_h / step) * step
    end = math.ceil(hi_h / step) * step
    vals, v = [], start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals

x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
x_right_raw = x_max_leader + 0.5
x_left = math.floor(x_min_data / 0.5) * 0.5
x_right = min(x_right_raw, float(max_hours)) if test_mode else x_right_raw
x_ticks_all = hour_ticks(x_left, x_right, step=0.5)

fig.update_xaxes(
    tickmode="array",
    tickvals=x_ticks_all,
    ticktext=[fmt_hmm(v) for v in x_ticks_all],
    title="Leader elapsed (h:mm)",
    range=[x_left, x_right],
    zeroline=True,
    zerolinecolor="#bbb",
    showline=True,
    mirror=True,
    ticks="outside",
    anchor="y",
)

y_min_val = float(xy_df["y_gap_min"].min())
y_max_val = float(xy_df["y_gap_min"].max())
y_start = math.floor(min(y_min_val, 0))
y_end = math.ceil(max(y_max_val, 0))
y_ticks = list(range(y_start, y_end + 1, 1))

fig.update_yaxes(
    tickmode="array",
    tickvals=y_ticks,
    ticktext=[str(abs(int(v))) for v in y_ticks],
    title="Time behind leader (minutes)",
    showline=True,
    mirror=True,
    ticks="outside",
    zeroline=True,
    zerolinecolor="#bbb",
)

fig.update_layout(
    annotations=end_annotations,
    showlegend=False,
    height=650,
    margin=dict(l=40, r=160, t=30, b=40),
)

st.plotly_chart(fig, use_container_width=True)


st.plotly_chart(fig, use_container_width=True)
