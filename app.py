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

# Build leader X (elapsed) and negative gap Y (leader - athlete)
def compute_leader_x_and_gap(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy().dropna(subset=["net_td"])
    # Leader elapsed time per split
    leaders = d.groupby("split")["net_td"].min().rename("leader_td")
    d = d.merge(leaders, on="split", how="left")
    # Negative gap: leader - athlete (leader=0, others negative)
    d["neg_gap_td"] = d["leader_td"] - d["net_td"]
    d["neg_gap_min"] = d["neg_gap_td"].dt.total_seconds() / 60.0
    # X value: leader elapsed minutes at that split
    d["leader_min"] = d["leader_td"].dt.total_seconds() / 60.0
    return d

# Filter selection and compute XY
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
xy_df = compute_leader_x_and_gap(sel).dropna(subset=["leader_min", "neg_gap_min"])

if xy_df.empty:
    st.info("No rows to plot for the current selection. Try selecting more athletes or splits.")
    st.stop()

# Scatter + connecting lines per athlete
fig = px.scatter(
    xy_df,
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

# Connect points per athlete
for nm, grp in xy_df.sort_values(["name", "leader_min"]).groupby("name"):
    fig.add_scatter(
        x=grp["leader_min"],
        y=grp["neg_gap_min"],
        mode="lines",
        line=dict(width=1),
        name=nm,
        showlegend=False,
    )

fig.update_xaxes(title="Leader elapsed (minutes)")
fig.update_yaxes(title="Time behind leader (minutes)")

# Whole-minute ticks, with Y reversed (0 at top), and labels without minus sign
def minute_ticks(series, min_step=1):
    if series.empty:
        return []
    lo, hi = float(series.min()), float(series.max())
    import math
    start = math.floor(lo)
    end = math.ceil(hi)
    step = max(1, min_step)
    return list(range(start, end + 1, step))

x_ticks = minute_ticks(xy_df["leader_min"])
y_ticks = minute_ticks(xy_df["neg_gap_min"])  # these are <= 0 for non-leaders

# Format: show abs value on Y labels (no minus sign)
fig.update_xaxes(tickvals=x_ticks, ticktext=[f"{int(v)}" for v in x_ticks])

# Ensure 0 is on top. If y_ticks has only negatives and possibly 0, reverse the axis range.
if y_ticks:
    y_min, y_max = min(y_ticks), max(y_ticks)
    # Put 0 at top, most negative at bottom
    top = max(0, y_max)
    bottom = y_min
    fig.update_yaxes(
        tickvals=y_ticks,
        ticktext=[f"{abs(int(v))}" for v in y_ticks],
        range=[top, bottom],
    )

fig.update_layout(height=650, margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig, use_container_width=True)
with st.expander("Show data"):
    st.dataframe(
        sel.sort_values(["name", "split"])[["name", "split", "netTime"]].reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
