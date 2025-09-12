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

def compute_time_behind_leader(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(subset=["net_td"])
    leaders = df.groupby("split")["net_td"].min().rename("leader_td")
    df = df.merge(leaders, on="split", how="left")
    df["behind_td"] = df["net_td"] - df["leader_td"]
    return df

def format_td(td: pd.Timedelta) -> str:
    if pd.isna(td):
        return ""
    total = int(td.total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h}:{m:02d}:{s:02d}" if h else f"{sign}{m}:{s:02d}"

st.title("Ironman Splits Viewer")
st.caption("Auto-updating via GitHub Actions every 5 minutes.")

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

# Prefer expected order but fall back to what's present
available_splits = [s for s in expected_splits if s in df["split"].unique()]
if not available_splits:
    available_splits = sorted(df["split"].unique())

left, right = st.columns([1, 3], gap="large")

with left:
    metric = st.radio("Metric", ["Time behind leader", "Net time"], index=0)
    names = sorted(df["name"].dropna().unique().tolist())
    default_selection = names[:10] if len(names) > 0 else []
    selected = st.multiselect("Athletes", options=names, default=default_selection, placeholder="Select athletes...")

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

sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()

# Build a per-split leader time (elapsed) and gap to leader for selection
def compute_leader_x_and_gap(df_in: pd.DataFrame) -> pd.DataFrame:
    # df_in must already be normalized and contain net_td
    d = df_in.copy().dropna(subset=["net_td"])
    # Leader elapsed time per split
    leaders = d.groupby("split")["net_td"].min().rename("leader_td")
    d = d.merge(leaders, on="split", how="left")
    # Gap to leader as positive seconds (leader = 0)
    d["gap_td"] = d["net_td"] - d["leader_td"]
    d["gap_sec"] = d["gap_td"].dt.total_seconds()
    d["gap_min"] = d["gap_sec"] / 60.0
    # X value: leader elapsed seconds at that split
    d["leader_sec"] = d["leader_td"].dt.total_seconds()
    d["leader_min"] = d["leader_sec"] / 60.0
    return d

# Filter selection
sel = df[df["name"].isin(selected)].copy()
sel = sel[sel["split"].isin(range_splits)]

# Compute leader x-axis (elapsed) and y gap
xy_df = compute_leader_x_and_gap(sel).dropna(subset=["leader_sec", "gap_sec"])

# Safety: if no rows, explain and stop
if xy_df.empty:
    st.info("No rows to plot for the current selection. Try selecting more athletes or splits.")
    st.stop()

# Plot as XY scatter with lines connecting points per athlete
# X: leader_min (minutes), Y: negative gap (to put leader at top as 0)
xy_df["y_plot"] = -xy_df["gap_min"]  # negate so leader at 0 is at the top

fig = px.scatter(
    xy_df,
    x="leader_min",
    y="y_plot",
    color="name",
    hover_data={
        "name": True,
        "split": True,
        "leader_min": ":.2f",
        "gap_min": ":.2f",
    },
)

# Connect points per athlete with lines
for nm, grp in xy_df.sort_values(["name", "leader_min"]).groupby("name"):
    fig.add_scatter(
        x=grp["leader_min"],
        y=grp["y_plot"],
        mode="lines",
        line=dict(width=1),
        name=nm,
        showlegend=False,  # legend already present from scatter
    )

# Axis labels
fig.update_xaxes(title="Leader elapsed (minutes)")
fig.update_yaxes(title="Time behind leader (minutes)")

# Whole-minute ticks for both axes
def minute_ticks(series, min_step=1):
    if series.empty:
        return []
    lo, hi = float(series.min()), float(series.max())
    # Round outward to whole minutes
    import math
    start = math.floor(lo)
    end = math.ceil(hi)
    step = max(1, min_step)
    return list(range(start, end + 1, step))

x_ticks = minute_ticks(xy_df["leader_min"])
y_ticks = minute_ticks(xy_df["y_plot"])

# Format tick labels as whole minutes without sign on Y
fig.update_xaxes(tickvals=x_ticks, ticktext=[f"{int(v)}" for v in x_ticks])
fig.update_yaxes(
    tickvals=y_ticks,
    ticktext=[f"{abs(int(v))}" for v in y_ticks],  # remove negative sign in labels
    range=[max(y_ticks) if y_ticks else 1, min(y_ticks) if y_ticks else 0],  # keep "up is faster" (reversed)
)

fig.update_layout(height=650, margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show data"):
    st.dataframe(
        sel.sort_values(["name", "split"])[["name", "split", "netTime"]].reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
