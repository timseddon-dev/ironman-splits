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
# 2) UI Setup
# ======================================
st.title("Ironman Splits Viewer")
st.caption("XY view: X = leader elapsed (minutes). Y = minutes behind leader (leader − athlete).")

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

# ======================================
# 2.5) Summary Table (Top 10 at ≤ 7h)
# ======================================
# Build ≤ 7h subset with leader times joined
leaders_full = compute_leaders(df)
df_7h = (
    df.merge(leaders_full[["split", "leader_td"]], on="split", how="left")
      .dropna(subset=["net_td", "leader_td"])
      .assign(leader_hr=lambda d: d["leader_td"].dt.total_seconds() / 3600.0)
)

# Restrict to ≤ 7h
df_7h = df_7h[df_7h["leader_hr"] <= 7.0].copy()

st.subheader("Race snapshot (Top 10 at ≤ 7h)")

if df_7h.empty:
    st.info("No data available within the first 7 hours yet.")
else:
    # Latest available row per athlete within ≤ 7h
    latest_7h = (
        df_7h.sort_values(["name", "leader_td"])
             .groupby("name", as_index=False)
             .tail(1)
             .reset_index(drop=True)
    )

    # Compute gap to leader (minutes, non-negative for display)
    latest_7h["gap_min"] = (latest_7h["net_td"] - latest_7h["leader_td"]).dt.total_seconds() / 60.0
    latest_7h["gap_min"] = latest_7h["gap_min"].clip(lower=0)

    # Order primarily by how far into the race they are (leader_td), then smallest gap
    snapshot = latest_7h.sort_values(["leader_td", "gap_min"], ascending=[False, True])

    top10 = (
        snapshot[["name", "split", "gap_min"]]
        .rename(columns={"name": "Athlete", "split": "Latest split", "gap_min": "Behind (min)"})
        .copy()
    )

    # Format gap and show only the top 10
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
# X ticks in hours but display as h:mm. Start at the beginning of the selected range,
# snapped back to the previous 30-minute mark, and extend +30 minutes past the max.
def hour_ticks(lo_h, hi_h, step=0.5):
    start = math.floor(lo_h / step) * step
    end = math.ceil(hi_h / step) * step
    vals, v = [], start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals

# Determine visible X min from data in the selected range
if xy_df.empty:
    st.info("No rows to plot for the current selection. Try selecting more athletes or splits.")
    st.stop()

x_min_data = float(xy_df["leader_hr"].min())
x_max_data = float(xy_df["leader_hr"].max())

# Snap start to previous 30-minute point relative to the first visible data point
x_left = math.floor(x_min_data / 0.5) * 0.5
# Always provide 30 minutes of padding at the right for labels
x_right = x_max_data + 0.5

# Build tick values at 30-minute spacing across the snapped domain
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
        ref_shapes.append(dict(type="line", x0=swim_x, x1=swim_x, y0=0, y1=axis_floor,
                               line=dict(color="#888", width=1, dash="dot")))
if "BIKE" in leaders["split"].values:
    bike_td = leaders.loc[leaders["split"] == "BIKE", "leader_td"].min()
    if pd.notna(bike_td):
        bike_x = bike_td.total_seconds() / 3600.0
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
    # keep your annotation labels and legend setting from earlier:
    annotations=end_annotations,
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
