# app.py
# ======================================
# Ironman Gaps vs Leader — Full App
# ======================================
import os, math, re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

DATA_FILE = "long.csv"
st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")


# ======================================
# 1) Data load and normalization
# ======================================
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

    # Ensure net_td (elapsed) exists; derive from netTime if needed
    def _parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        # Normalize 1:23:45 -> 01:23:45
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
        df["net_td"] = pd.NaT  # downstream will guard if needed

    # Drop rows missing essentials if present
    if "name" in df.columns and "split" in df.columns:
        df = df.dropna(subset=["name", "split"]).copy()

    return df


def expected_order():
    # Master preferred split order
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
    # Format float hours as H:MM
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


# Load data once, set categorical order, build km lookup
df = load_data(DATA_FILE)
if not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("No data loaded from long.csv.")
    st.stop()

km_lookup = None
if "km" in df.columns:
    km_lookup = (
        df.groupby("split", as_index=False)["km"]
          .max()
          .set_index("split")["km"]
          .to_dict()
    )

splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass


# ======================================
# 2) UI and options (Test mode, split range)
# ======================================
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 5.5, 0.5)

if test_mode:
    if "net_td" not in df.columns:
        st.error("Test filter requires 'net_td'.")
        st.stop()
    max_td = pd.to_timedelta(max_hours, unit="h")
    before_rows = len(df)
    d = df.dropna(subset=["net_td"]).copy()
    df = d[d["net_td"] < max_td].copy()
    after_rows = len(df)
    st.caption(f"Test mode active: {after_rows:,} rows (from {before_rows:,}) with athlete elapsed < {max_hours:.1f} h")

# Re-assert split order after any filtering
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass

# Split range selectors
colA, colB = st.columns([1, 1])
with colA:
    # Default to START if present, else first available
    from_idx = splits_ordered_master.index("START") if "START" in splits_ordered_master else 0
    from_split = st.selectbox("From split", options=splits_ordered_master, index=from_idx)
with colB:
    to_idx = splits_ordered_master.index("FINISH") if "FINISH" in splits_ordered_master else len(splits_ordered_master) - 1
    to_split = st.selectbox("To split", options=splits_ordered_master, index=to_idx)

def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1 + 1]
    return splits[i1:i0 + 1]


# ======================================
# 3) Leaderboard (proper order, narrow columns, scrollable) with "Plot on chart"
# ======================================
leaders_now = compute_leaders(df)
df_now = df.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])

st.subheader("Leaderboard")

if df_now.empty:
    st.info("No data available for the current dataset.")
    selected_for_plot = []
else:
    # Latest row per athlete and gap
    latest_now = (
        df_now.sort_values(["name", "net_td"])
              .groupby("name", as_index=False)
              .tail(1)
              .reset_index(drop=True)
    )
    latest_now["gap_min"] = (latest_now["net_td"] - latest_now["leader_td"]).dt.total_seconds() / 60.0
    latest_now["gap_min"] = latest_now["gap_min"].clip(lower=0)

    # True leaderboard: smallest gap first (leader = 0.0 at top)
    latest_now = latest_now.sort_values(["gap_min", "net_td"], ascending=[True, True]).reset_index(drop=True)
    latest_now["rank"] = latest_now.index + 1

    # Friendly split labels WITH distance where available
    latest_now["Latest split"] = latest_now["split"].map(lambda s: friendly_split_label(s, km_lookup))
    latest_now["Behind (min)"] = latest_now["gap_min"].map(lambda x: f"{x:.1f}")

    # UI table fields (checkbox column rendered last)
    table_df = latest_now[["rank", "name", "Latest split", "Behind (min)"]].rename(
        columns={"rank": "#", "name": "Athlete"}
    )

    # Pagination state
    page_size = 15
    total_rows = len(table_df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    if "lb_page" not in st.session_state:
        st.session_state.lb_page = 0

    # Pagination controls
    col_prev, col_info, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️", key="lb_prev", disabled=(st.session_state.lb_page == 0)):
            st.session_state.lb_page = max(0, st.session_state.lb_page - 1)
    with col_info:
        st.caption(f"Showing rows {(st.session_state.lb_page*page_size)+1}–{min((st.session_state.lb_page+1)*page_size, total_rows)} of {total_rows}")
    with col_next:
        if st.button("➡️", key="lb_next", disabled=(st.session_state.lb_page >= total_pages - 1)):
            st.session_state.lb_page = min(total_pages - 1, st.session_state.lb_page + 1)

    start = st.session_state.lb_page * page_size
    end = min(start + page_size, total_rows)
    page_df = table_df.iloc[start:end].reset_index(drop=True)

    # Initialize checkbox state (default top 10)
    if "plot_checks" not in st.session_state:
        st.session_state.plot_checks = {}
        top10_names = latest_now.head(10)["Athlete"].tolist() if "Athlete" in table_df.columns else latest_now.head(10)["name"].tolist()
        for nm in latest_now["Athlete"] if "Athlete" in table_df.columns else latest_now["name"]:
            st.session_state.plot_checks[nm] = (nm in top10_names)

    # Narrow columns and scroll area (CSS)
    st.markdown(
        """
        <style>
        .lb-scroll-area { max-height: 520px; overflow-y: auto; padding-right: 6px; }
        .lb-row { padding: 2px 0; border-bottom: 1px solid rgba(0,0,0,0.05); }
        .lb-header { position: sticky; top: 0; background: white; z-index: 2;
                     border-bottom: 1px solid rgba(0,0,0,0.2); padding: 4px 0; }
        .lb-col-athlete { width: 26ch; }     /* slightly wider than typical longest athlete name */
        .lb-col-split   { width: 18ch; }     /* enough to include distance */
        .lb-col-gap     { width: 10ch; text-align: right; }
        .lb-col-plot    { width: 10ch; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header: Athlete | Latest split | Behind (min) | Plot on chart
    st.markdown('<div class="lb-header">', unsafe_allow_html=True)
    header_cols = st.columns([3.0, 2.2, 1.2, 1.0])
    with header_cols[0]:
        st.markdown('<div class="lb-col-athlete"><strong>Athlete</strong></div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<div class="lb-col-split"><strong>Latest split</strong></div>', unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown('<div class="lb-col-gap"><strong>Behind (min)</strong></div>', unsafe_allow_html=True)
    with header_cols[3]:
        st.markdown('<div class="lb-col-plot"><strong>Plot on chart</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Rows in a scrollable container
    st.markdown('<div class="lb-scroll-area">', unsafe_allow_html=True)
    for _, row in page_df.iterrows():
        athlete = row["Athlete"]
        latest_split = row["Latest split"]
        behind = row["Behind (min)"]

        st.markdown('<div class="lb-row">', unsafe_allow_html=True)
        row_cols = st.columns([3.0, 2.2, 1.2, 1.0])
        with row_cols[0]:
            st.markdown(f'<div class="lb-col-athlete">{athlete}</div>', unsafe_allow_html=True)
        with row_cols[1]:
            st.markdown(f'<div class="lb-col-split">{latest_split}</div>', unsafe_allow_html=True)
        with row_cols[2]:
            st.markdown(f'<div class="lb-col-gap">{behind}</div>', unsafe_allow_html=True)
        with row_cols[3]:
            st.session_state.plot_checks[athlete] = st.checkbox(
                "", value=st.session_state.plot_checks.get(athlete, False), key=f"plot_{athlete}"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bulk selection helpers
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Select top 10"):
            top10_names = latest_now.head(10)["Athlete"].tolist() if "Athlete" in table_df.columns else latest_now.head(10)["name"].tolist()
            for nm in st.session_state.plot_checks:
                st.session_state.plot_checks[nm] = (nm in top10_names)
    with c2:
        if st.button("Select none"):
            for nm in st.session_state.plot_checks:
                st.session_state.plot_checks[nm] = False
    with c3:
        if st.button("Select all"):
            for nm in st.session_state.plot_checks:
                st.session_state.plot_checks[nm] = True

    # Final list for plotting
    selected_for_plot = [nm for nm, on in st.session_state.plot_checks.items() if on]

# ======================================
# 4) Plot: Behind leader vs leader elapsed (0 at top, time behind downward)
# ======================================
# Range of splits to plot
master_order = available_splits_in_order(df)
range_splits = split_range(master_order, from_split, to_split)
if not range_splits:
    st.info("No splits available for plotting.")
    st.stop()

# Build selection dataframe
sel = df[(df["name"].isin(selected_for_plot)) & (df["split"].astype(str).isin(range_splits))].copy()
if sel.empty:
    st.info("No rows match the current selection and split range.")
    st.stop()

leaders_now = compute_leaders(df)
xy_df = sel.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])
xy_df["leader_hr"] = xy_df["leader_td"].dt.total_seconds() / 3600.0

# Make Y negative minutes so 0 is at top and time behind goes below axis
xy_df["y_gap_min"] = -((xy_df["net_td"] - xy_df["leader_td"]).dt.total_seconds() / 60.0)
xy_df["y_gap_min"] = xy_df["y_gap_min"].clip(upper=0)  # ensure never above 0

# Add START anchor for continuity
if not xy_df.empty:
    start_rows = pd.DataFrame({
        "name": list(dict.fromkeys(selected_for_plot)),
        "split": "START",
        "leader_td": pd.to_timedelta(0, unit="s"),
        "net_td": pd.to_timedelta(0, unit="s"),
        "leader_hr": 0.0,
        "y_gap_min": 0.0,
    })
    xy_df = pd.concat([start_rows, xy_df], ignore_index=True, sort=False)

# Labels with distances
xy_df["split_label"] = xy_df["split"].astype(str).map(lambda s: friendly_split_label(s, km_lookup))
xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")

import plotly.graph_objects as go
fig = go.Figure()

# One line per athlete
for nm, g in xy_df.groupby("name", sort=False):
    fig.add_trace(go.Scatter(
        x=g["leader_hr"],
        y=g["y_gap_min"],
        mode="lines",
        line=dict(width=1.8),
        name=nm,
        showlegend=False,
        hovertemplate="Athlete: %{text}<br>Split: %{meta}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{customdata:.1f} min",
        text=[nm]*len(g),
        meta=g["split_label"],
        customdata=[-v for v in g["y_gap_min"]],  # positive minutes in hover
    ))

# End labels with improved de-overlap: spread vertically and nudge right
endpoints = (
    xy_df.sort_values(["name", "leader_hr"])
         .groupby("name", as_index=False)
         .tail(1)
)
stagger_cycle = [0, -2, +2.5, -3.5, +3.2, -4.2, +4.0, -5.0, +5.0, -6.0]  # minutes offsets (y is negative)
ann = []
for i, row in endpoints.reset_index(drop=True).iterrows():
    dy_min = stagger_cycle[i % len(stagger_cycle)]
    ann.append(dict(
        x=float(row["leader_hr"]) + 0.06,
        y=float(row["y_gap_min"]) + (-dy_min),  # y is negative; invert sense
        xref="x", yref="y",
        text=str(row["name"]),
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(size=11, color="rgba(0,0,0,0.9)"),
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        opacity=0.95,
    ))

# Axes and layout
x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
x_right_raw = x_max_leader + 0.5
x_left = math.floor(x_min_data / 0.5) * 0.5
x_right = min(x_right_raw, float(max_hours)) if test_mode else x_right_raw
x_ticks_all = hour_ticks(x_left, x_right, step=0.5)

# Y ticks: from a small positive (near zero) down to min negative, labeled as positive minutes
y_min_val = float(xy_df["y_gap_min"].min())  # most negative
y_ticks = [0] + [-(i) for i in range(2, int(abs(y_min_val)) + 1, 2)]  # 0, -2, -4, ...
y_ticktext = ["0"] + [str(i) for i in range(2, int(abs(y_min_val)) + 1, 2)]

fig.update_xaxes(
    title="Leader elapsed (h)",
    tickmode="array",
    tickvals=x_ticks_all,
    ticktext=[fmt_hmm(v) for v in x_ticks_all],
    showgrid=True,
    zeroline=False,
    showline=True,
    mirror=True,
    ticks="outside",
    anchor="y",
)

fig.update_yaxes(
    title="Time behind leader (min)",
    tickmode="array",
    tickvals=y_ticks,
    ticktext=y_ticktext,
    range=[0.5, y_min_val - 1],  # 0 at top, extend below min
    showgrid=True,
    zeroline=True,
    zerolinecolor="rgba(0,0,0,0.25)",
    showline=True,
    mirror=True,
    ticks="outside",
    anchor="x",
)

fig.update_layout(
    height=520,
    margin=dict(l=50, r=30, t=30, b=40),
    annotations=ann,
)

st.plotly_chart(fig, use_container_width=True)
