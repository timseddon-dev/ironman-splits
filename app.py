# app.py
import os, re, math
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")

# Optional auto-refresh every 60 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="auto-refresh")
except Exception:
    pass

DATA_FILE = "long.csv"


# ======================================
# 1) Data load and normalization
# ======================================
import os, math, re
import pandas as pd
import streamlit as st

DATA_FILE = "long.csv"
st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")

@st.cache_data(ttl=60, show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing {path}. Ensure it exists alongside the app and contains columns: name, split, netTime or net_td, optional km.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalize text columns
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Convert km if present
    if "km" in df.columns:
        df["km"] = pd.to_numeric(df["km"], errors="coerce")

    # Ensure net_td exists (derive from netTime if needed)
    def _parse_td(x):
        if pd.isna(x): return pd.NaT
        s = str(x).strip()
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
            s = "0" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    m, sec = int(parts[0]), int(parts[1])
                    return pd.to_timedelta(m, unit="m") + pd.to_timedelta(sec, unit="s")
                elif len(parts) == 1 and parts[0].isdigit():
                    return pd.to_timedelta(int(parts[0]), unit="s")
            except Exception:
                return pd.NaT
            return pd.NaT

    if "net_td" in df.columns:
        try:
            df["net_td"] = pd.to_timedelta(df["net_td"])
        except Exception:
            df["net_td"] = df["net_td"].apply(_parse_td)
    elif "netTime" in df.columns:
        try:
            df["net_td"] = pd.to_timedelta(df["netTime"])
        except Exception:
            df["net_td"] = df["netTime"].apply(_parse_td)
    else:
        # Keep going; later logic will warn if elapsed is required
        df["net_td"] = pd.NaT

    # Drop rows missing essentials if both columns exist
    if "name" in df.columns and "split" in df.columns:
        df = df.dropna(subset=["name", "split"]).copy()

    return df

def expected_order():
    return (
        ["START", "SWIM", "T1"]
        + [f"BIKE{i}" for i in range(1, 26)] + ["BIKE", "T2"]
        + [f"RUN{i}" for i in range(1, 23)] + ["RUN", "FINISH"]
    )

def available_splits_in_order(_df: pd.DataFrame):
    if not isinstance(_df, pd.DataFrame) or "split" not in _df.columns:
        return []
    order = expected_order()
    present = [s for s in order if s in _df["split"].dropna().unique().tolist()]
    if present:
        return present
    d = _df.dropna(subset=["net_td"]).copy() if "net_td" in _df.columns else _df.copy()
    if d.empty or "net_td" not in d.columns:
        return sorted(_df["split"].dropna().unique().tolist())
    tmp = (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(first_td=("net_td", "min"))
         .sort_values("first_td")
    )
    return tmp["split"].tolist()

def compute_leaders(_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(_df, pd.DataFrame) or "net_td" not in _df.columns:
        return pd.DataFrame(columns=["split", "leader_td"])
    d = _df.dropna(subset=["net_td"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["split", "leader_td"])
    return (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(leader_td=("net_td", "min"))
    )

def friendly_split_label(split: str, km_lookup: dict | None = None) -> str:
    s = str(split).upper()
    if s == "FINISH":
        return "Finish"
    if s == "SWIM":
        if km_lookup and s in km_lookup and pd.notna(km_lookup[s]):
            return f"Swim {km_lookup[s]:.1f} km"
        return "Swim 3.8 km"
    if s in ("T1", "T2"):
        return s
    if km_lookup and s in km_lookup and pd.notna(km_lookup[s]):
        if s.startswith("BIKE"):
            return f"Bike {km_lookup[s]:.1f} km"
        if s.startswith("RUN"):
            return f"Run {km_lookup[s]:.1f} km"
    if s.startswith("BIKE"):
        return "Bike"
    if s.startswith("RUN"):
        return "Run"
    return s

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

# Load data first so df exists before Section 2
df = load_data(DATA_FILE)
if not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("No data loaded from long.csv.")
    st.stop()

# Per-split km lookup (max km per split)
km_lookup = None
if "km" in df.columns:
    km_lookup = (
        df.groupby("split", as_index=False)["km"]
          .max()
          .set_index("split")["km"]
          .to_dict()
    )

# Ensure split categorical order is set before Section 2
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass


# ======================================
# 2) UI and options (Test mode, positions, athlete selection, range)
# ======================================

# --- Test mode (safe) ---
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 5.5, 0.5)

if test_mode:
    # Ensure df exists and has elapsed time as timedelta
    if not isinstance(df, pd.DataFrame):
        st.error("Data not loaded.")
        st.stop()

    if "net_td" not in df.columns:
        # Try to derive from netTime if present
        if "netTime" in df.columns:
            try:
                df["net_td"] = pd.to_timedelta(df["netTime"])
            except Exception:
                # Best-effort parsing
                def _parse_td(_s):
                    if pd.isna(_s): return pd.NaT
                    s = str(_s).strip()
                    import re as _re
                    if _re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
                        s = "0" + s
                    try:
                        return pd.to_timedelta(s)
                    except Exception:
                        parts = s.split(":")
                        try:
                            if len(parts) == 2:
                                m, sec = int(parts[0]), int(parts[1])
                                return pd.to_timedelta(m, unit="m") + pd.to_timedelta(sec, unit="s")
                            elif len(parts) == 1 and parts[0].isdigit():
                                return pd.to_timedelta(int(parts[0]), unit="s")
                        except Exception:
                            return pd.NaT
                        return pd.NaT
                df["net_td"] = df["netTime"].apply(_parse_td)
        else:
            st.error("Test filter requires an elapsed time column: 'net_td' or 'netTime' to derive it from.")
            st.stop()

    max_td = pd.to_timedelta(max_hours, unit="h")
    before_rows = int(len(df))
    d = df.dropna(subset=["net_td"]).copy()
    df = d[d["net_td"] < max_td].copy()
    after_rows = len(df)
    st.caption(f"Test mode active: {after_rows:,} rows (from {before_rows:,}) with athlete elapsed < {max_hours:.1f} h")

# After any filtering, re-assert split category order
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass

# --- Positions and athlete options (ordered by current position) ---
leaders_for_pos = (
    df.dropna(subset=["net_td"])
      .sort_values(["split", "net_td"])
      .groupby("split", as_index=False)
      .agg(leader_td=("net_td", "min"))
)

latest_per_athlete = (
    df.merge(leaders_for_pos, on="split", how="left")
      .dropna(subset=["net_td"])
      .sort_values(["name", "net_td"])
      .groupby("name", as_index=False)
      .tail(1)
      .reset_index(drop=True)
)

# Position rank by latest net time
latest_per_athlete["pos"] = latest_per_athlete["net_td"].rank(method="first").astype(int)

athletes_ordered = latest_per_athlete.sort_values("pos")[["pos", "name"]].reset_index(drop=True)

def _label(row):
    return f"P{row['pos']:02d}  {row['name']}"

athlete_labels = athletes_ordered.apply(_label, axis=1).tolist()
athlete_names_in_order = athletes_ordered["name"].tolist()

default_selection_names = athlete_names_in_order[:20]  # top 20 by current position

label_to_name = dict(zip(athlete_labels, athlete_names_in_order))
name_to_label = {v: k for k, v in label_to_name.items()}

# --- UI controls: athlete multiselect + split range ---
colA, colB = st.columns([3, 2], gap="large")

with colA:
    st.caption("Athletes (ordered by current position)")
    selected_labels = st.multiselect(
        " ",
        options=athlete_labels,
        default=[name_to_label[n] for n in default_selection_names if n in name_to_label],
        label_visibility="collapsed",
    )
    selected = [label_to_name[lbl] for lbl in selected_labels]
    if not selected:
        selected = default_selection_names

with colB:
    st.caption("Split range")
    splits_ordered = available_splits_in_order(df)
    from_idx = splits_ordered.index("SWIM") if "SWIM" in splits_ordered else 0
    to_idx = splits_ordered.index("FINISH") if "FINISH" in splits_ordered else len(splits_ordered) - 1
    from_split = st.selectbox("From split", options=splits_ordered, index=from_idx)
    to_split = st.selectbox("To split", options=splits_ordered, index=to_idx)

# ======================================
# 2.5) Race snapshot (Top 10)
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
    top10["Latest split"] = top10["Latest split"].map(lambda s: friendly_split_label(s, km_lookup))
    top10["Behind (min)"] = top10["Behind (min)"].map(lambda x: f"{x:.1f}")

    top10_out = top10.head(10).reset_index(drop=True)
    top10_out.index = top10_out.index + 1
    st.dataframe(top10_out, use_container_width=True, height=320)

# ======================================
# 3) Plot prep
# ======================================
def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1+1]
    else:
        return splits[i1:i0+1]

master_order = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=master_order, ordered=True)
except Exception:
    pass

range_splits = split_range(master_order, from_split, to_split)
splits_in_df = set(df["split"].dropna().astype(str).unique().tolist())
range_splits = [s for s in range_splits if str(s) in splits_in_df]
if not range_splits:
    st.info("No splits remaining in the selected range after filtering. Try widening the range or turning Test mode off.")
    st.stop()

sel = df[(df["name"].isin(selected)) & (df["split"].astype(str).isin(range_splits))].copy()
if sel.empty:
    st.info("No rows match the current athlete selection and split range.")
    st.stop()

leaders_now = compute_leaders(df)
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

xy_df["split_label"] = xy_df["split"].astype(str).map(friendly_split_label)
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
        meta=g["split_label"],
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
def hour_ticks(lo_h: float, hi_h: float, step: float = 0.5) -> list:
    start = math.floor(lo_h / step) * step
    end = math.ceil(hi_h / step) * step
    vals, v = [], start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals

x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
x_right_raw = x_max_leader + 0.5
x_left = math.floor(x_min_data / 0.5) * 0.5
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
    margin=dict(l=40, r=160, t=30, b=40),
)

st.plotly_chart(fig, use_container_width=True)
