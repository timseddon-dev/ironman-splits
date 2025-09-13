# app.py
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")
st.title("Race Gaps vs Leader")

# ======================================
# 1.0) Load data (from long.csv)
# ======================================
CSV_PATH = Path("long.csv")

@st.cache_data(show_spinner=True)
def load_long_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "name" not in df.columns or "split" not in df.columns or "net_td" not in df.columns:
        st.error("long.csv must include columns: name, split, net_td")
        st.stop()

    # Normalize net_td to Timedelta
    try:
        df["net_td"] = pd.to_timedelta(df["net_td"])
    except Exception:
        def to_td(x):
            try:
                return pd.to_timedelta(x)
            except Exception:
                s = str(x)
                parts = s.split(":")
                try:
                    if len(parts) == 3:
                        h, m, s = map(int, parts)
                        return pd.to_timedelta(h, unit="h") + pd.to_timedelta(m, unit="m") + pd.to_timedelta(s, unit="s")
                    elif len(parts) == 2:
                        m, s = map(int, parts)
                        return pd.to_timedelta(m, unit="m") + pd.to_timedelta(s, unit="s")
                except Exception:
                    return pd.NaT
                return pd.NaT
        df["net_td"] = df["net_td"].apply(to_td)

    # Basic cleanup
    df = df.dropna(subset=["name", "split", "net_td"]).copy()

    # Order splits by earliest leader time (earliest net_td per split)
    order = (
        df.sort_values(["split", "net_td"])
          .groupby("split", as_index=False)
          .agg(first_td=("net_td", "min"))
          .sort_values("first_td")["split"]
          .tolist()
    )
    df["split"] = pd.Categorical(df["split"], categories=order, ordered=True)
    df = df.sort_values(["split", "name", "net_td"]).reset_index(drop=True)
    return df

df = load_long_csv(CSV_PATH)
all_athletes = sorted(df["name"].unique().tolist())
splits_ordered = list(df["split"].cat.categories)

# ======================================
# 2.1) UI controls
# ======================================
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by each athlete's elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 7.0, 0.5)

colA, colB = st.columns(2)
with colA:
    default_selection = all_athletes[:6] if len(all_athletes) >= 6 else all_athletes
    selected = st.multiselect("Athletes", options=all_athletes, default=default_selection)
with colB:
    default_from = 0
    default_to = len(splits_ordered) - 1 if splits_ordered else 0
    from_split = st.selectbox("From split", options=splits_ordered, index=default_from if splits_ordered else 0)
    to_split   = st.selectbox("To split",   options=splits_ordered, index=default_to   if (splits_ordered and default_to >= 0) else 0)

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
# 2.2) Apply test filter (per-athlete)
# ======================================
if test_mode:
    max_td = pd.to_timedelta(max_hours, unit="h")
    before = len(df)
    df = df.dropna(subset=["net_td"]).copy()
    df = df[df["net_td"] < max_td].copy()
    st.caption(f"Test mode active: {len(df):,} rows (from {before:,}) with athlete elapsed < {max_hours:.1f}h")

if df.empty or len(selected) == 0 or len(range_splits) == 0:
    st.info("Please ensure there is data, select at least one athlete, and choose a valid split range.")
    st.stop()

# ======================================
# 2.5) Summary Table (Top 10 on current dataset)
# ======================================
leaders_now = (
    df.dropna(subset=["net_td"])
      .sort_values(["split", "net_td"])
      .groupby("split", as_index=False)
      .agg(leader_td=("net_td", "min"))
)

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
# 3) Plot prep
# ======================================
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
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

# ======================================
# 5) Axes and Layout
# ======================================
x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
x_right_raw = x_max_leader + 0.5  # pad 30 min
x_left = math.floor(x_min_data / 0.5) * 0.5
x_right = min(x_right_raw, float(max_hours)) if test_mode else x_right_raw

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
