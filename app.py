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
# 1) Data Loading and Utilities
# ======================================
@st.cache_data(ttl=60)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Keep only expected columns if present
    keep = [c for c in ["name", "split", "netTime"] if c in df.columns]
    df = df[keep].copy()

    # Parse netTime -> timedelta
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

    # Normalize text
    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    return df


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
# 2.0) Test Mode Filter: keep only rows where athlete elapsed < 7h
# ======================================
# Preconditions: df is already loaded and has at least columns ["name", "split", "net_td"].
# net_td is a pandas Timedelta (elapsed for the athlete at that split).

if "net_td" not in df.columns:
    st.error("Test filter requires a 'net_td' Timedelta column on df. Make sure df is loaded before Section 2.0.")
    st.stop()

# Keep rows where the athlete's elapsed < 7 hours
seven_hours = pd.to_timedelta(7, unit="h")
df = df.dropna(subset=["net_td"]).copy()
df = df[df["net_td"] < seven_hours].copy()

# Optional: if you want to still compute leader_td later using your existing functions, do nothing else here.
# We do NOT add leader_hr here; later sections can compute what they need.

# ======================================
# 2.2) Apply test filter (per-athlete) if enabled
# ======================================
# Preconditions: df is already loaded and has column "net_td" as a pandas Timedelta.

if test_mode:
    if "net_td" not in df.columns:
        st.error("Test filter requires a 'net_td' Timedelta column on df. Ensure df is loaded before applying the filter.")
        st.stop()

    seven_hours = pd.to_timedelta(7, unit="h")

    # Keep only rows where the athlete's elapsed < 7 hours
    df = df.dropna(subset=["net_td"]).copy()
    df = df[df["net_td"] < seven_hours].copy()

    st.caption(f"Test mode active: {len(df):,} rows with athlete elapsed < 7h")


# ======================================
# 2.5) Summary Table (Top 10 on current dataset)
# ======================================
# Build leader times for the current (possibly filtered) df
leaders_now = (
    df.dropna(subset=["net_td"])
      .sort_values(["split", "net_td"])
      .groupby("split", as_index=False)
      .agg(leader_td=("net_td", "min"))
)

df_now = (
    df.merge(leaders_now, on="split", how="left")
      .dropna(subset=["net_td", "leader_td"])
)

st.subheader("Race snapshot (Top 10)")

if df_now.empty:
    st.info("No data available for the current dataset.")
else:
    # Take each athlete's latest available row in the current dataset
    latest_now = (
        df_now.sort_values(["name", "net_td"])
              .groupby("name", as_index=False)
              .tail(1)
              .reset_index(drop=True)
    )

    # Gap to leader at that split (non-negative for display)
    latest_now["gap_min"] = (latest_now["net_td"] - latest_now["leader_td"]).dt.total_seconds() / 60.0
    latest_now["gap_min"] = latest_now["gap_min"].clip(lower=0)

    # Order by progression (leader_td) then smallest gap
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
leaders = compute_leaders(df)

# Restrict to selected athletes and chosen range
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
xy_df = sel.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])

# X: leader elapsed hours at split
xy_df["leader_hr"] = xy_df["leader_td"].dt.total_seconds() / 3600.0

# Y: leader - athlete in minutes (leader = 0; others negative)
xy_df["y_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0

# Limit to mid‑race subset: up to 7.0 hours (for stress testing)
xy_df = xy_df[xy_df["leader_hr"] <= 7.0].copy()

# Inject synthetic START points (elapsed=0, gap=0) but include them ONLY
# when the chosen range explicitly starts at START.
include_start = (len(range_splits) > 0 and str(range_splits[0]).upper() == "START")
if include_start:
    start_rows = pd.DataFrame({
        "name": list(dict.fromkeys(selected)),  # preserve selection order, unique
        "split": "START",
        "leader_td": pd.to_timedelta(0, unit="s"),
        "net_td": pd.to_timedelta(0, unit="s"),
        "leader_hr": 0.0,
        "y_gap_min": 0.0,
    })
    # If we're subsetting to ≤7h, START is fine (0 ≤ 7)
    xy_df = pd.concat([start_rows, xy_df], ignore_index=True, sort=False)

# Sort for consistent line drawing
xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")

if xy_df.empty:
    st.info("No rows to plot for the current selection (≤ 7h). Try selecting more athletes or a different split range.")
    st.stop()

# ======================================
# 4) Plot: Lines + End Labels (annotations) + Reference Lines
# ======================================
# Lines only (no symbols from the base scatter)
fig = px.scatter(
    xy_df,
    x="leader_hr",
    y="y_gap_min",
    color="name",
    hover_data={"name": True, "split": True, "leader_hr": ":.2f", "y_gap_min": ":.2f"},
    labels={"leader_hr": "Leader elapsed (hours)", "y_gap_min": "Time behind leader (minutes)"},
)
fig.update_traces(mode="lines", selector=dict(mode="markers"))

# Explicit line traces per athlete
for nm, grp in xy_df.groupby("name"):
    fig.add_scatter(
        x=grp["leader_hr"],
        y=grp["y_gap_min"],
        mode="lines",
        line=dict(width=1),
        name=nm,
        showlegend=False,
    )

# Build end-of-series annotations (left-aligned, 8px right shift)
last_points = (
    xy_df.groupby("name", as_index=False)
         .apply(lambda g: g.sort_values("leader_hr").tail(1))
         .reset_index(drop=True)[["name", "leader_hr", "y_gap_min"]]
)

# vertical staggering in data units
stagger = [-0.15, +0.15, -0.25, +0.25, -0.35, +0.35, -0.45, +0.45]
end_annotations = []
for i, (_, row) in enumerate(last_points.iterrows()):
    dy = stagger[i % len(stagger)]
    end_annotations.append(dict(
        x=float(row["leader_hr"]),
        y=float(row["y_gap_min"] + dy),
        xref="x",
        yref="y",
        text=str(row["name"]),
        showarrow=False,
        xanchor="left",
        align="left",
        xshift=8,   # 8px to the right from the end point
        yshift=0,
        font=dict(size=12, color="rgba(0,0,0,1)"),
        bgcolor="rgba(255,255,255,0.0)",
        bordercolor="rgba(0,0,0,0.0)",
    ))

# Compute reference line positions (no labels)
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
# Axis starts at the first visible data point (snapped back to previous 30-min),
# and ends at min(7.0h, last leader time + 0.5h padding) so we don't imply
# everyone reaches 7h if the leader hasn't either.

def hour_ticks(lo_h, hi_h, step=0.5):
    start = math.floor(lo_h / step) * step
    end = math.ceil(hi_h / step) * step
    vals, v = [], start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals

if xy_df.empty:
    st.info("No rows to plot for the current selection.")
    st.stop()

# Compute left bound from visible data
x_min_data = float(xy_df["leader_hr"].min())

# Compute right bound:
# - Find the leader's latest time present in the plotting data (xy_df already ≤ 7h)
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
# - Add 0.5h for label space but cap at 7.0h (mid-race test cap)
x_right = min(7.0, x_max_leader + 0.5)

# Snap left bound to previous 0.5h tick
x_left = math.floor(x_min_data / 0.5) * 0.5

# Build ticks
x_ticks_all = hour_ticks(x_left, x_right, step=0.5)

def fmt_hmm(h):
    total_minutes = int(round(h * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh}:{mm:02d}"

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

# Y-axis: absolute values (no minus sign); orientation unchanged
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

# Reference lines down to the axis minimum tick
axis_floor = y_start
ref_shapes = []
if "SWIM" in leaders["split"].values:
    swim_td = leaders.loc[leaders["split"] == "SWIM", "leader_td"].min()
    if pd.notna(swim_td):
        swim_x = swim_td.total_seconds() / 3600.0
        if x_left <= swim_x <= x_right:
            ref_shapes.append(dict(type="line", x0=swim_x, x1=swim_x, y0=0, y1=axis_floor,
                                   line=dict(color="#888", width=1, dash="dot")))
if "BIKE" in leaders["split"].values:
    bike_td = leaders.loc[leaders["split"] == "BIKE", "leader_td"].min()
    if pd.notna(bike_td):
        bike_x = bike_td.total_seconds() / 3600.0
        if x_left <= bike_x <= x_right:
            ref_shapes.append(dict(type="line", x0=bike_x, x1=bike_x, y0=0, y1=axis_floor,
                                   line=dict(color="#888", width=1, dash="dot")))

fig.update_layout(
    xaxis=dict(
        range=[x_left, x_right],
        zeroline=True,
        zerolinecolor="#bbb",
        constrain="domain",
    ),
    yaxis=dict(
        anchor="x",
        zeroline=True,
        zerolinecolor="#bbb",
    ),
    shapes=ref_shapes,
    annotations=end_annotations,   # from Section 4
    showlegend=False,
    height=650,
    margin=dict(l=40, r=160, t=30, b=40),
)

st.plotly_chart(fig, use_container_width=True)



# ======================================
# 6) Data Table
# ======================================
with st.expander("Show data"):
    st.dataframe(
        df[df["name"].isin(selected) & df["split"].isin(range_splits)]
          .sort_values(["name", "split"])[["name", "split", "netTime"]]
          .reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
