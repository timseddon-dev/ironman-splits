# app.py
import math
import re
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ======================================
# Configuration
# ======================================
RTRT_URL = "https://app.rtrt.me/IRM-WORLDCHAMPIONSHIP-MEN-2024/leaderboard/pro-men-ironman/"

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")
st.title("Race Gaps vs Leader")

# ======================================
# Helpers
# ======================================
def parse_hms_to_timedelta(s: str) -> pd.Timedelta | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # Accept formats like H:MM:SS or MM:SS; also tolerate missing hours
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        else:
            # try seconds
            sec = int(float(s))
            h, m = 0, 0
        return pd.to_timedelta(hours=h, minutes=m, seconds=sec)
    except Exception:
        return None

def available_splits_in_order(df: pd.DataFrame) -> list:
    tmp = (
        df.dropna(subset=["net_td"])
          .sort_values(["split", "net_td"])
          .groupby("split", as_index=False)
          .agg(first_td=("net_td", "min"))
          .sort_values("first_td")
    )
    return tmp["split"].tolist()

def compute_leaders(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["net_td"])
          .sort_values(["split", "net_td"])
          .groupby("split", as_index=False)
          .agg(leader_td=("net_td", "min"))
    )

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

def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1+1]
    else:
        return splits[i1:i0+1]

# ======================================
# Data fetch and parse from RTRT
# ======================================
@st.cache_data(show_spinner=True, ttl=300)
def load_rtrt_df(url: str) -> pd.DataFrame:
    # Fetch HTML
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BearlyFocus/1.0; +https://bearly.ai)"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    html = r.text

    # Parse
    soup = BeautifulSoup(html, "html.parser")

    # Heuristic: RTRT renders a table of athletes with splits. Markup may be nested.
    # We'll extract rows where athlete names appear, and per-split elapsed.
    # Fallback approach: look for rows with data-label or aria-label attributes.

    # Collect candidate rows
    rows = []
    # Attempt 1: data rows with role="row"
    for tr in soup.select("tr"):
        txt = tr.get_text(" ", strip=True)
        if not txt:
            continue
        # crude name detection: contains a space and no excessive punctuation
        tds = tr.find_all(["td", "th"])
        if len(tds) < 3:
            continue
        # Find a probable name cell
        name_cell = None
        for td in tds:
            t = td.get_text(" ", strip=True)
            if t and len(t.split()) >= 2 and not re.search(r"\d{2}:\d{2}", t):
                name_cell = t
                break
        if not name_cell:
            continue

        # Extract splits with time-like strings
        # We will capture pairs (split_label, elapsed_text)
        splits = []
        for td in tds:
            t = td.get_text(" ", strip=True)
            if re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\b", t):
                splits.append(t)

        if splits:
            rows.append({"name": name_cell, "splits": splits})

    # If nothing found, bail out with an informative error
    if not rows:
        raise RuntimeError("Could not parse RTRT page. The site structure may have changed.")

    # Normalize into (name, split, net_td)
    # We don't have explicit split keys from HTML reliably; we infer order:
    # START, SWIM, T1, BIKE..., T2, RUN..., FINISH — but we only have elapsed times sequence.
    # We'll generate generic labels: START (0:00:00), then S1, S2, ... in observed order.
    df_list = []
    generic_splits = ["START"] + [f"S{i}" for i in range(1, 200)]
    for row in rows:
        name = row["name"]
        elapsed_list = row["splits"]
        # Force START = 0
        df_list.append({"name": name, "split": "START", "net_td": pd.to_timedelta(0, unit="s")})
        for i, t in enumerate(elapsed_list, start=1):
            td = parse_hms_to_timedelta(t)
            if td is not None:
                df_list.append({"name": name, "split": generic_splits[i], "net_td": td})

    df = pd.DataFrame(df_list)
    # Order splits by earliest occurrence time
    if not df.empty:
        order = df.groupby("split")["net_td"].min().sort_values().index.tolist()
        df["split"] = pd.Categorical(df["split"], categories=order, ordered=True)
        df = df.sort_values(["split", "name"]).reset_index(drop=True)
    return df

# Load live data
try:
    df = load_rtrt_df(RTRT_URL)
    st.caption("Data source: RTRT leaderboard (parsed live).")
except Exception as e:
    st.error(f"Failed to fetch/parse RTRT data. Details: {e}")
    st.stop()

# ======================================
# 2) UI controls
# ======================================
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset to athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 7.0, 0.5)

all_athletes = sorted(df["name"].dropna().unique().tolist())
splits_ordered = available_splits_in_order(df)
default_athletes = all_athletes[:6] if len(all_athletes) >= 6 else all_athletes

colA, colB = st.columns(2)
with colA:
    selected = st.multiselect("Athletes", options=all_athletes, default=default_athletes)
with colB:
    default_from = 0
    default_to = len(splits_ordered) - 1 if splits_ordered else 0
    from_split = st.selectbox("From split", options=splits_ordered, index=default_from if splits_ordered else 0)
    to_split = st.selectbox("To split", options=splits_ordered, index=default_to if (splits_ordered and default_to is not None and default_to >= 0) else 0)

# Apply per-athlete elapsed filter if test mode is on
if test_mode:
    max_td = pd.to_timedelta(max_hours, unit="h")
    before = len(df)
    df = df.dropna(subset=["net_td"]).copy()
    df = df[df["net_td"] < max_td].copy()
    st.caption(f"Test mode active: {len(df):,} rows (from {before:,}) with athlete elapsed < {max_hours:.1f}h")

# Early guard
if df.empty or len(selected) == 0 or len(splits_ordered) == 0:
    st.info("Please ensure there is data and at least one athlete selected.")
    st.stop()

# Range of splits inclusive
range_splits = split_range(splits_ordered, from_split, to_split)

# ======================================
# 3) Summary table (Top 10 on current dataset)
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
    top10["Behind (min)"] = top10["Behind (min)"].map(lambda x: f"{x:.1f}")
    st.dataframe(top10.head(10).reset_index(drop=True), use_container_width=True, height=320)

# ======================================
# 4) Data prep for plot
# ======================================
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
xy_df = sel.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])
xy_df["leader_hr"] = xy_df["leader_td"].dt.total_seconds() / 3600.0
xy_df["y_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0

# START point only when From split == START (our generic first split)
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

xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")
if xy_df.empty:
    st.info("No rows to plot for the current selection.")
    st.stop()

# ======================================
# 5) Plot
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
        meta=g["split"],
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

# Reference lines: we don't have explicit SWIM/BIKE labels from RTRT generic splits,
# so we skip them here (we only have START, S1, S2, ... inferred).
# If later we map generic splits to disciplines, we can re-enable markers.

# X axis bounds: left = first visible point snapped to prev 30-min; right = latest leader time + 0.5h.
x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
x_right_raw = x_max_leader + 0.5
x_left = math.floor(x_min_data / 0.5) * 0.5

# If test mode is active, also cap right bound at Max hours
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
    margin=dict(l=40, r=160, t=30
