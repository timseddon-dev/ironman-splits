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

def compute_leader_by_split(df: pd.DataFrame) -> pd.DataFrame:
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

# UI
st.title("Ironman Splits Viewer")
st.caption("XY view: X = leader elapsed (minutes). Y = minutes behind leader (0 at top).")

df = load_data(DATA_FILE)
if df.empty:
    st.warning("No data found yet. The scheduled job will populate long.csv shortly. Try Rerun in a minute.")
    st.stop()

splits_order = available_splits_in_order(df)

# Athlete picker ordered by current position at latest split
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
    ordered_names = sorted(df["name"].dropna().unique().tolist())

left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown("Metric: Time behind leader (minutes)")

    default_selection = ordered_names[:10] if ordered_names else []
    selected = st.multiselect(
        "Athletes (ordered by current position)",
        options=ordered_names,
        default=default_selection,
        placeholder="Select athletes..."
    )

    split_start = st.selectbox("From split", options=splits_order, index=0)
    split_end = st.selectbox("To split", options=splits_order, index=len(splits_order) - 1)

if not selected:
    st.info("Select at least one athlete to display the chart.")
    st.stop()

def idx(s):
    try:
        return splits_order.index(s)
    except ValueError:
        return 0

i0, i1 = idx(split_start), idx(split_end)
if i0 > i1:
    i0, i1 = i1, i0
range_splits = splits_order[i0:i1+1]

# Merge leader times and compute Y = leader - athlete (minutes)
leaders = df.groupby("split", as_index=False)["net_td"].min().rename(columns={"net_td": "leader_td"})
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
xy_df = sel.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])

# X: leader elapsed minutes at split
xy_df["leader_min"] = xy_df["leader_td"].dt.total_seconds() / 60.0
# Y: leader - athlete (leader = 0; others negative)
xy_df["y_gap_min"] = (xy_df["leader_td"] - xy_df["net_td"]).dt.total_seconds() / 60.0

# Simple scatter with connecting lines; no Y-axis tweaks
import plotly.express as px
fig = px.scatter(
    xy_df.sort_values(["name", "leader_min"]),
    x="leader_min",
    y="y_gap_min",
    color="name",
    hover_data={"name": True, "split": True, "leader_min": ":.2f", "y_gap_min": ":.2f"},
    labels={"leader_min": "Leader elapsed (minutes)", "y_gap_min": "Leader - athlete (minutes)"},
)

# Connect points per athlete (no axis modifications)
for nm, grp in xy_df.sort_values(["name", "leader_min"]).groupby("name"):
    fig.add_scatter(
        x=grp["leader_min"],
        y=grp["y_gap_min"],
        mode="lines",
        line=dict(width=1),
        name=nm,
        showlegend=False,
    )

st.plotly_chart(fig, use_container_width=True)


with st.expander("Show data"):
    st.dataframe(
        sel.sort_values(["name", "split"])[["name", "split", "netTime"]].reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
