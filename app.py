# app.py
import os, re, math
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")

# Optional auto-refresh every 60 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="auto-refresh")
except Exception:
    pass

DATA_FILE = "long.csv"

# ======================================
# 1) Data Loading and Utilities
# ======================================
@st.cache_data(ttl=60, show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing {path}. Ensure it exists alongside the app and contains columns: name, split, and netTime or net_td.")
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Keep known columns if present (km is optional but supported)
    keep_cols = [c for c in ["name", "split", "netTime", "net_td", "km"] if c in df.columns]
    df = df[keep_cols].copy()

    # Derive net_td if needed
    has_net_td = "net_td" in df.columns
    has_netTime = "netTime" in df.columns

    def parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        # Normalize 1:23:45 -> 01:23:45
        import re as _re
        if _re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
            s = "0" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    m, sec = int(parts[0]), int(parts[1])
                    return pd.to_timedelta(m, unit="m") + pd.to_timedelta(sec, unit="s")
                elif len(parts) == 1 and parts[0].isdigit():
                    return pd.to_timedelta(int(parts[0]), unit="s")
            except Exception:
                return pd.NaT
            return pd.NaT

    if not has_net_td and has_netTime:
        df["net_td"] = df["netTime"].apply(parse_td)
    elif has_net_td:
        try:
            df["net_td"] = pd.to_timedelta(df["net_td"])
        except Exception:
            df["net_td"] = df["net_td"].apply(parse_td)
    else:
        st.error("long.csv must include name/split plus either netTime (so we can derive net_td) or net_td directly.")
        return pd.DataFrame()

    # Normalize text columns
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.upper()

    # km column to numeric if present
    if "km" in df.columns:
        df["km"] = pd.to_numeric(df["km"], errors="coerce")

    # Remove rows missing essentials
    df = df.dropna(subset=["name", "split"]).copy()
    return df

def expected_order():
    return (
        ["START", "SWIM", "T1"]
        + [f"BIKE{i}" for i in range(1, 26)] + ["BIKE", "T2"]
        + [f"RUN{i}" for i in range(1, 23)]
        + ["FINISH"]
    )

def available_splits_in_order(df: pd.DataFrame):
    order = expected_order()
    present = [s for s in order if s in df["split"].dropna().unique().tolist()]
    if present:
        return present
    d = df.dropna(subset=["net_td"]).copy()
    if d.empty:
        return sorted(df["split"].dropna().unique().tolist())
    tmp = (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(first_td=("net_td", "min"))
         .sort_values("first_td")
    )
    return tmp["split"].tolist()

def compute_leaders(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["net_td"]).copy()
    return (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(leader_td=("net_td", "min"))
    )

def friendly_split_label(split: str, km_lookup: dict | None = None) -> str:
    s = str(split).upper()
    if s == "FINISH":
        return "Finish"
    if s == "SWIM":
        # Prefer km from lookup
        if km_lookup and s in km_lookup and pd.notna(km_lookup[s]):
            return f"Swim {km_lookup[s]:.1f} km"
        return "Swim 3.8 km"
    if s in ("T1", "T2"):
        return s
    # If we have exact km in the CSV, use it
    if km_lookup and s in km_lookup and pd.notna(km_lookup[s]):
        if s.startswith("BIKE"):
            return f"Bike {km_lookup[s]:.1f} km"
        if s.startswith("RUN"):
            return f"Run {km_lookup[s]:.1f} km"
    # Fallback to generic formatting
    if s.startswith("BIKE"):
        return "Bike"
    if s.startswith("RUN"):
        return "Run"
    return s

def fmt_hmm(hours_float: float) -> str:
    # Format float hours as H:MM
    total_minutes = int(round(hours_float * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh}:{mm:02d}"

def hour_ticks(lo_h: float, hi_h: float, step: float = 0.5) -> list:
    import math as _math
    start = _math.floor(lo_h / step) * step
    end = _math.ceil(hi_h / step) * step
    vals, v = [], start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals

# ======================================
# 2) UI and options (Test mode, positions, athlete selection, range)
# ======================================

with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 5.5, 0.5)

if test_mode:
    # Guard that net_td exists
    if "net_td" not in df.columns:
        st.error("Test filter requires a 'net_td' Timedelta column. Check long.csv (needs netTime or net_td).")
        st.stop()

    max_td = pd.to_timedelta(max_hours, unit="h")
    # Count rows before filtering
    before_rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0

    # Keep only rows with valid elapsed and below threshold
    d = df.dropna(subset=["net_td"])
    df = d[d["net_td"] < max_td].copy()

    after_rows = len(df)
    st.caption(f"Test mode active: {after_rows:,} rows (from {before_rows:,}) with athlete elapsed < {max_hours:.1f} h")
# Recompute split categories after any filtering
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass

# 2.0) Positions + Athlete options (ordered by current position) and defaults
leaders_for_pos = compute_leaders(df)
latest_per_athlete = (
    df.merge(leaders_for_pos, on="split", how="left")
      .dropna(subset=["net_td"])
      .sort_values(["name", "net_td"])
      .groupby("name", as_index=False)
      .tail(1)
      .reset_index(drop=True)
)
# Position by latest net time
latest_per_athlete["pos"] = latest_per_athlete["net_td"].rank(method="first").astype(int)
athletes_ordered = latest_per_athlete.sort_values("pos")[["pos", "name"]].reset_index(drop=True)

def make_label(row):
    return f"P{row['pos']:02d}  {row['name']}"
athlete_labels = athletes_ordered.apply(make_label, axis=1).tolist()
athlete_names_in_order = athletes_ordered["name"].tolist()
default_selection_names = athlete_names_in_order[:20]
label_to_name = dict(zip(athlete_labels, athlete_names_in_order))
name_to_label = {v: k for k, v in label_to_name.items()}

# 2.1) UI controls (ordered by position; default top 20)
colA, colB = st.columns([3, 2])

with colA:
    st.caption("Athletes (ordered by current position)")
    selected_labels = st.multiselect(
        " ",
        options=athlete_labels,
        default=[name_to_label[n] for n in default_selection_names if n in name_to_label],
        label_visibility="collapsed",
    )
    selected = [label_to_name[lbl] for lbl in selected_labels]

with colB:
    st.caption("Split range")
    splits_ordered = available_splits_in_order(df)
    from_idx = splits_ordered.index("SWIM") if "SWIM" in splits_ordered else 0
    to_idx = splits_ordered.index("FINISH") if "FINISH" in splits_ordered else len(splits_ordered) - 1
    from_split = st.selectbox("From split", options=splits_ordered, index=from_idx)
    to_split = st.selectbox("To split", options=splits_ordered, index=to_idx)

if not selected:
    selected = default_selection_names

# ======================================
# 2.5) Race snapshot (Top 10)
# ======================================
leaders_now = compute_leaders(df)
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
    top10["Latest split"] = top10["Latest split"].map(lambda s: friendly_split_label(s, km_lookup))
    top10["Behind (min)"] = top10["Behind (min)"].map(lambda x: f"{x:.1f}")

    top10_out = top10.head(10).reset_index(drop=True)
    top10_out.index = top10_out.index + 1
    st.dataframe(top10_out, use_container_width=True, height=320)

# ======================================
# 3) Plot prep
# ======================================
def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1+1]
    else:
        return splits[i1:i0+1]

master_order = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=master_order, ordered=True)
except Exception:
    pass

range_splits = split_range(master_order, from_split, to_split)
splits_in_df = set(df["split"].dropna().astype(str).unique().tolist())
range_splits = [s for s in range_splits if str(s) in splits_in_df]
if not range_splits:
    st.info("No splits remaining in the selected range after filtering. Try widening the range or turning Test mode off.")
    st.stop()

sel = df[(df["name"].isin(selected)) & (df["split"].astype(str).isin(range_splits))].copy()
if sel.empty:
    st.info("No rows match the current athlete selection and split range.")
    st.stop()

leaders_now = compute_leaders(df)
xy_df = sel.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])
xy_df["leader_hr"] = xy_df["leader_td"].dt.total_seconds() / 3600.0
xy_df["y_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0

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

xy_df["split_label"] = xy_df["split"].astype(str).map(friendly_split_label)
xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")
if xy_df.empty:
    st.info("No rows to plot for the current selection.")
    st.stop()

# ======================================
# 4) Plot
# ======================================
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
