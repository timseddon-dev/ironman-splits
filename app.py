# app.py
import math
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ======================================
# 0) Utilities and data helpers
# ======================================
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
    leaders = (
        df.dropna(subset=["net_td"])
          .sort_values(["split", "net_td"])
          .groupby("split", as_index=False)
          .agg(leader_td=("net_td", "min"))
    )
    return leaders

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

# ======================================
# 1) Generate a synthetic dataset (so the app runs without any external data)
# ======================================
@st.cache_data(show_spinner=False)
def generate_synthetic_df(seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    athletes = [
        "Patrick Lange", "Magnus Ditlev", "Rudy Von Berg", "Menno Koolhaas",
        "Leon Chevalier", "Gregory Barnaby", "Kieran Lindars", "Cameron Wurf",
        "Kristian Hogenhaug", "Matt Hanson", "Bradley Weiss"
    ]

    # Define a sequence of splits resembling a race
    # START, SWIM, T1, BIKE checkpoints every ~20-25min, T2, RUN splits every ~10min, FINISH
    splits = ["START", "SWIM", "T1"] + [f"BIKE{i}" for i in range(1, 13)] + ["T2"] + [f"RUN{i}" for i in range(1, 21)] + ["FINISH"]

    rows = []
    for name in athletes:
        elapsed = timedelta(0)
        # Athlete-specific pace modifiers
        swim_bias = np.random.normal(0, 60)        # +/- 1 min
        bike_bias = np.random.normal(0, 120)       # +/- 2 min
        run_bias = np.random.normal(0, 90)         # +/- 1.5 min
        overall_variability = np.random.normal(1.0, 0.03)  # small pace scaling

        for sp in splits:
            if sp == "START":
                elapsed = timedelta(0)
            elif sp == "SWIM":
                base = 55 * 60  # 55 min swim
                elapsed += timedelta(seconds=max(40*60, base + swim_bias*np.random.uniform(0.7, 1.3)))
            elif sp == "T1":
                elapsed += timedelta(seconds=np.random.randint(90, 180))
            elif sp.startswith("BIKE"):
                base = 22 * 60  # ~22 min segment chunks (to total around ~4h30)
                seg = max(15*60, int(base*overall_variability + bike_bias*np.random.uniform(-0.1, 0.1)))
                elapsed += timedelta(seconds=seg)
            elif sp == "T2":
                elapsed += timedelta(seconds=np.random.randint(60, 150))
            elif sp.startswith("RUN"):
                base = 10 * 60  # ~10 min segments
                seg = max(6*60, int(base*overall_variability + run_bias*np.random.uniform(-0.15, 0.15)))
                elapsed += timedelta(seconds=seg)
            elif sp == "FINISH":
                # Add small extra to simulate final sprint variance
                elapsed += timedelta(seconds=np.random.randint(60, 160))
            rows.append({"name": name, "split": sp, "net_td": elapsed})

    df = pd.DataFrame(rows)
    # Ensure splits are categorical in intended order
    order = (
        df.groupby("split")["net_td"].min().sort_values().index.tolist()
    )
    df["split"] = pd.Categorical(df["split"], categories=order, ordered=True)
    df = df.sort_values(["split", "name"]).reset_index(drop=True)
    return df

df = generate_synthetic_df()

# ======================================
# 2.1) UI controls
# ======================================
st.title("Race Gaps vs Leader")

with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset to athlete elapsed < Max hours)", value=True)
    max_hours = st.slider("Max hours", min_value=1.0, max_value=12.0, value=7.0, step=0.5, help="When test mode is ON, keep only rows where athlete elapsed (net_td) is below this.")

# Prepare athlete and split options
all_athletes = sorted(df["name"].dropna().unique().tolist()) if not df.empty else []
splits_ordered = available_splits_in_order(df) if not df.empty else []
default_athletes = all_athletes[:6] if len(all_athletes) >= 6 else all_athletes

colA, colB = st.columns(2)
with colA:
    selected = st.multiselect("Athletes", options=all_athletes, default=default_athletes)
with colB:
    default_from = splits_ordered.index("SWIM") if "SWIM" in splits_ordered else 0
    default_to = splits_ordered.index("FINISH") if "FINISH" in splits_ordered else (len(splits_ordered) - 1 if splits_ordered else 0)
    from_split = st.selectbox("From split", options=splits_ordered, index=default_from if splits_ordered else 0)
    to_split = st.selectbox("To split", options=splits_ordered, index=default_to if (splits_ordered and default_to is not None and default_to >= 0) else 0)

# ======================================
# 2.2) Apply test filter (per-athlete) if enabled
# ======================================
if test_mode:
    max_td = pd.to_timedelta(max_hours, unit="h")
    before_rows = len(df)
    df = df.dropna(subset=["net_td"]).copy()
    df = df[df["net_td"] < max_td].copy()
    st.caption(f"Test mode active: {len(df):,} rows (from {before_rows:,}) with athlete elapsed < {max_hours:.1f}h")

if df.empty or len(selected) == 0 or len(splits_ordered) == 0:
    st.info("Please select athletes and ensure splits are available.")
    st.stop()

def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1+1]
    else:
        return splits[i1:i0+1]

range_splits = split_range(splits_ordered, from_split, to_split)

# ======================================
# 2.5) Summary Table (Top 10 on current dataset)
# ======================================
leaders_now = compute_leaders(df)

df_now = (
    df.merge(leaders_now, on="split", how="left")
      .dropna(subset=["net_td", "leader_td"])
)

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
# 3) Data Prep For Plot
# ======================================
leaders = leaders_now

sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
xy_df = sel.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])

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

xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")

if xy_df.empty:
    st.info("No rows to plot for the current selection.")
    st.stop()

# ======================================
# 4) Plot: Lines + End Labels + Reference Lines
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
        bgcolor="rgba(255,255,255,0.0)",
        bordercolor="rgba(0,0,0,0.0)",
    ))

swim_x = None
bike_x = None
if "SWIM" in leaders["split"].values:
    swim_td = leaders.loc[leaders["split"] == "SWIM", "leader_td"].min()
    if pd.notna(swim_td):
        swim_x = swim_td.total_seconds() / 3600.0
if "BIKE" in leaders["split"].values:
    bike_td = leaders.loc[leaders["split"] == "BIKE", "leader_td"].min()
    if pd.notna(bike_td):
        bike_x = bike_td.total_seconds() / 3600.0

# ======================================
# 5) Axes and Layout
# ======================================
x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())

x_right_raw = x_max_leader + 0.5
if test_mode:
    x_right = min(max_hours, x_right_raw)
else:
    x_right = x_right_raw

x_left = math.floor(x_min_data / 0.5) * 0.5
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

axis_floor = y_start
ref_shapes = []
if swim_x is not None and x_left <= swim_x <= x_right:
    ref_shapes.append(dict(type="line", x0=swim_x, x1=swim_x, y0=0, y1=axis_floor,
                           line=dict(color="#888", width=1, dash="dot")))
if bike_x is not None and x_left <= bike_x <= x_right:
    ref_shapes.append(dict(type="line", x0=bike_x, x1=bike_x, y0=0, y1=axis_floor,
                           line=dict(color="#888", width=1, dash="dot")))

fig.update_layout(
    shapes=ref_shapes,
    annotations=end_annotations,
    showlegend=False,
    height=650,
    margin=dict(l=40, r=160, t=30, b=40),
)

st.plotly_chart(fig, use_container_width=True)
