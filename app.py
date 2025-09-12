# /// script
# dependencies = ["streamlit", "pandas", "plotly.express"]
# ///
import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Ironman Splits Viewer", layout="wide")

# Try to autorefresh every 60 seconds so open sessions pull new data
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
    # Read very simply; the file is produced by rtrt_splits.py
    df = pd.read_csv(path)
    # Keep only expected columns if more exist
    keep = [c for c in ["name", "split", "netTime"] if c in df.columns]
    df = df[keep]

    # Convert netTime string to Timedelta
    def parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        # Normalize hour to HH if needed (e.g., 1:23:45 -> 01:23:45)
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
            s = "0" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            return pd.NaT

    df["net_td"] = df["netTime"].apply(parse_td)
    return df

def compute_time_behind_leader(df: pd.DataFrame) -> pd.DataFrame:
    # Compute per-split leader and subtract
    df = df.copy()
    df = df.dropna(subset=["net_td"])
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
st.caption("Auto-updating via GitHub Actions. Data file: long.csv in this repo.")

df = load_data(DATA_FILE)

# Normalize essential columns early
if not df.empty:
    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

# Always show a small debug/status panel on the page
st.write({
    "csv_exists": os.path.exists(DATA_FILE),
    "csv_size_bytes": os.path.getsize(DATA_FILE) if os.path.exists(DATA_FILE) else 0,
    "rows_detected": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
    "unique_names": int(df["name"].nunique()) if not df.empty else 0,
    "unique_splits": int(df["split"].nunique()) if not df.empty else 0,
    "sample_splits": sorted(df["split"].dropna().unique().tolist()[:10]) if not df.empty else [],
})

if df.empty:
    st.warning("No data found yet. The scheduled job will populate long.csv shortly. Click Rerun after a minute.")
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

left, right = st.columns([1,3])

with left:
    metric = st.radio("Metric", ["Time behind leader", "Net time"], index=0)
    names = sorted(df["name"].dropna().unique().tolist())
    default_selection = names[:10] if len(names) >= 1 else []
    selected = st.multiselect("Athletes", options=names, default=default_selection)

    split_start = st.selectbox("From split", options=available_splits, index=0 if available_splits else 0)
    split_end = st.selectbox("To split", options=available_splits, index=(len(available_splits)-1) if available_splits else 0)

if not selected:
    st.info("Select at least one athlete.")
    st.stop()

def idx(s):
    try:
        return available_splits.index(s)
    except ValueError:
        return 0

i0, i1 = idx(split_start), idx(split_end)
if i0 > i1:
    i0, i1 = i1, i0
range_splits = available_splits[i0:i1+1] if available_splits else []

sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()

if metric == "Time behind leader":
    plot_df = compute_time_behind_leader(sel)
    ycol = "behind_td"
    ytitle = "Time behind leader"
else:
    plot_df = sel.copy()
    ycol = "net_td"
    ytitle = "Net time"

plot_df = plot_df.dropna(subset=[ycol])

# Safety: if no rows after filtering, show a hint and stop
if plot_df.empty:
    st.info("No rows to plot for the current selection. Try selecting more athletes or a wider split range.")
    st.dataframe(df.head(50))
    st.stop()

# Order splits
plot_df["split"] = pd.Categorical(plot_df["split"], categories=available_splits, ordered=True)
plot_df = plot_df.sort_values(["name", "split"])
plot_df["y_seconds"] = plot_df[ycol].dt.total_seconds()

fig = px.line(
    plot_df,
    x="split",
    y="y_seconds",
    color="name",
    markers=True,
    labels={"split": "Split", "y_seconds": ytitle},
    hover_data={
        "name": True,
        "split": True,
        "y_seconds": False,
        "net": plot_df["net_td"].apply(format_td),
        "behind": plot_df.get("behind_td", pd.Series([pd.NaT]*len(plot_df))).apply(format_td) if "behind_td" in plot_df else None,
    }
)

def tickfmt(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

if len(plot_df):
    ymin, ymax = float(plot_df["y_seconds"].min()), float(plot_df["y_seconds"].max())
    span = max(1.0, ymax - ymin)
    step = max(30, int(span // 12))
    ticks = list(range(int(ymin), int(ymax) + 1, step))
    fig.update_yaxes(tickvals=ticks, ticktext=[tickfmt(v) for v in ticks], title=ytitle)

fig.update_layout(height=650, margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show data"):
    st.dataframe(
        sel.sort_values(["name", "split"])[["name", "split", "netTime"]].reset_index(drop=True),
        use_container_width=True,
        height=320
    )
