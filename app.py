# /// script
# dependencies = ["streamlit", "pandas", "plotly.express"]
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

def compute_leader_by_split(df: pd.DataFrame) -> pd.DataFrame:
    # Returns per-split leader time and leader name
    d = df.dropna(subset=["net_td"]).copy()
    grp = d.groupby("split", as_index=False)["net_td"].min().rename(columns={"net_td": "leader_td"})
    return grp

def latest_common_split(df: pd.DataFrame) -> str | None:
    # Find the latest split that has at least one non-null time
    if df.empty:
        return None
    order = (
        ["SWIM", "T1"]
        + [f"BIKE{i}" for i in range(1, 26)]
        + ["BIKE", "T2"]
        + [f"RUN{i}" for i in range(1, 23)]
        + ["FINISH"]
    )
    splits_present = [s for s in order if s in df["split"].unique()]
    if not splits_present:
        splits_present = sorted(df["split"].dropna().unique().tolist())
    # Choose the last split where at least one athlete has a time
    for s in reversed(splits_present):
        if df.loc[(df["split"] == s) & (~df["net_td"].isna())].shape[0] > 0:
            return s
    return None

def compute_positions(df: pd.DataFrame) -> pd.DataFrame:
    # Compute position per split (1 = leader) using net_td ordering
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

st.title("Ironman Splits Viewer")
st.caption("XY view: X = leader elapsed (minutes). Y = minutes behind leader (0 at top).")

df = load_data(DATA_FILE)
if df.empty:
    st.warning("No data found yet. The scheduled job will populate long.csv shortly. Try Rerun in a minute.")
    st.stop()

# Expected logical split order
expected_splits = (
    ["SWIM", "T1"]
    + [f"BIKE{i}" for i in range(1, 26)]
    + ["BIKE", "T2"]
    + [f"RUN{i}" for i in range(1, 23)]
    + ["FINISH"]
)
available_splits = [s for s in expected_splits if s in df["split"].unique()]
if not available_splits:
    available_splits = sorted(df["split"].unique())

# Build current positions to order athlete list by latest split position
pos_df = compute_positions(df)
latest_split = latest_common_split(df)
if latest_split is not None:
    latest_pos = (
        pos_df[pos_df["split"] == latest_split][["name", "pos"]]
        .dropna(subset=["pos"])
        .sort_values("pos", ascending=True)
    )
    ordered_names = latest_pos["name"].tolist() + [n for n in df["name"].unique() if n not in latest_pos["name"].values]
else:
    # Fallback to alphabetical if no positions
    ordered_names = sorted(df["name"].dropna().unique().tolist())

left, right = st.columns([1, 3], gap="large")

with left:
    # Fixed metric for this XY chart (gap to leader)
    st.markdown("Metric: Time behind leader (minutes)")

    # Athlete selector ordered by current position
    default_selection = ordered_names[:10] if len(ordered_names) > 0 else []
    selected = st.multiselect("Athletes (ordered by current position)", options=ordered_names, default=default_selection)

    split_start = st.selectbox("From split", options=available_splits, index=0)
    split_end = st.selectbox("To split", options=available_splits, index=len(available_splits) - 1)

if not selected:
    st.info("Select at least one athlete to display the chart.")
    st.stop()

def idx(s):
    try:
        return available_splits.index(s)
    except ValueError:
        return 0

i0, i1 = idx(split_start), idx(split_end)
if i0 > i1:
    i0, i1 = i1, i0
range_splits = available_splits[i0 : i1 + 1]

# Compute leader elapsed and negative gap (leader - athlete)
leaders = compute_leader_by_split(df)
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
xy_df = sel.merge(leaders, on="split", how="left")
xy_df = xy_df.dropna(subset=["net_td", "leader_td"])

# X: leader elapsed minutes at split
xy_df["leader_min"] = xy_df["leader_td"].dt.total_seconds() / 60.0
# Y: negative gap in minutes (leader - athlete), so leader is 0, others negative
xy_df["neg_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0

if xy_df.empty:
    st.info("No rows to plot for the current selection. Try selecting more athletes or splits.")
    st.stop()

# Scatter + connecting lines per athlete
# Ensure neg_gap_min is <= 0 (leader=0, others negative)
xy_df["neg_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0
# Safety: clamp tiny floating noise (e.g., -1e-12) to 0
xy_df.loc[xy_df["neg_gap_min"].between(-1e-6, 1e-6), "neg_gap_min"] = 0.0
st.write({"y_min": float(xy_df["neg_gap_min"].min()), "y_max": float(xy_df["neg_gap_min"].max())})

fig = px.scatter(
    xy_df.sort_values(["name", "leader_min"]),
    x="leader_min",
    y="neg_gap_min",
    color="name",
    hover_data={
        "name": True,
        "split": True,
        "leader_min": ":.2f",
        "neg_gap_min": ":.2f",
    },
)

# Connect points per athlete as lines
for nm, grp in xy_df.sort_values(["name", "leader_min"]).groupby("name"):
    fig.add_scatter(
        x=grp["leader_min"],
        y=grp["neg_gap_min"],
        mode="lines",
        line=dict(width=1),
        name=nm,
        showlegend=False,
    )

# Axes: whole-minute ticks; Y reversed so 0 at top; labels without minus sign
x_ticks = minute_ticks(xy_df["leader_min"], min_step=1)
y_ticks = minute_ticks(xy_df["neg_gap_min"], min_step=1)  # these are <= 0 (and possibly 0)


# Force leader=0 and others negative (minutes); clamp tiny float noise
xy_df["neg_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0
xy_df.loc[xy_df["neg_gap_min"].between(-1e-6, 1e-6), "neg_gap_min"] = 0.0

# X ticks: whole minutes; thin to every 5 minutes for readability
x_ticks = minute_ticks(xy_df["leader_min"], min_step=1)
x_ticks_5 = [v for v in x_ticks if v % 5 == 0] or x_ticks

fig.update_xaxes(
    title="Leader elapsed (minutes)",
    tickmode="array",
    tickvals=x_ticks_5,
    ticktext=[str(int(v)) for v in x_ticks_5],
    showline=True,
    mirror=True,
    ticks="outside",
)

# Y ticks: build from data extents (negative up to 0), labels as absolute values
y_min_val = float(xy_df["neg_gap_min"].min())  # negative or 0
y_start = math.floor(y_min_val)                # e.g., -21
y_end = 0
y_ticks = list(range(y_start, y_end + 1, 1))   # [-21, ..., 0]

fig.update_yaxes(
    title="Time behind leader (minutes)",
    tickmode="array",
    tickvals=y_ticks,
    ticktext=[str(abs(int(v))) for v in y_ticks],  # drop minus sign
    range=[0, y_start],  # reversed: 0 at top, most negative at bottom
    autorange=False,
    zeroline=True,
    zerolinecolor="#bbb",
    showline=True,
    mirror=True,
    ticks="outside",
)

# Important: Do NOT call any further fig.update_yaxes/fig.update_xaxes after this point.

fig.update_layout(height=650, margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig, use_container_width=True)



with st.expander("Show data"):
    st.dataframe(
        sel.sort_values(["name", "split"])[["name", "split", "netTime"]].reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
