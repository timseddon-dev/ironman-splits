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
        st.error(f"Missing {path}. Ensure it exists alongside the app and contains: name, split, netTime or net_td, km.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalize text columns
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Convert km
    if "km" in df.columns:
        df["km"] = pd.to_numeric(df["km"], errors="coerce")

    # Ensure net_td present (derive if needed)
    def _parse_td(x):
        if pd.isna(x):
            return pd.NaT
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
        df["net_td"] = pd.NaT

    # Drop rows missing essentials
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
    order = expected_order()
    present = [s for s in order if "split" in _df.columns and s in _df["split"].dropna().unique().tolist()]
    if present:
        return present
    d = _df.dropna(subset=["net_td"]).copy() if "net_td" in _df.columns else _df.copy()
    if d.empty or "net_td" not in d.columns:
        return sorted(_df["split"].dropna().unique().tolist()) if "split" in _df.columns else []
    tmp = (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(first_td=("net_td", "min"))
         .sort_values("first_td")
    )
    return tmp["split"].tolist()


def compute_leaders(_df: pd.DataFrame) -> pd.DataFrame:
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


# Load data
df = load_data(DATA_FILE)
if not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("No data loaded from long.csv.")
    st.stop()

# km lookup
km_lookup = None
if "km" in df.columns:
    km_lookup = (
        df.groupby("split", as_index=False)["km"]
          .max()
          .set_index("split")["km"]
          .to_dict()
    )

# Initial order for splits
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass


# ======================================
# 2) UI: Test mode and Split selectors
# ======================================
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 5.5, 0.5)

# Apply test filter BEFORE any “latest split” computation
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

# Re-assert order after filtering
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass

# Split range selectors (From defaults to START)
colA, colB = st.columns([1, 1])
with colA:
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
# 3) Leaderboard — narrow columns, sticky header, forced vertical scroll
# ======================================
leaders_now = compute_leaders(df)
df_now = df.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])

st.subheader("Leaderboard")

if df_now.empty:
    st.info("No data available for the current dataset.")
    selected_for_plot = []
else:
    # Latest per athlete (reflects applied test filter)
    latest_now = (
        df_now.sort_values(["name", "net_td"])
              .groupby("name", as_index=False)
              .tail(1)
              .reset_index(drop=True)
    )
    latest_now["gap_min"] = (latest_now["net_td"] - latest_now["leader_td"]).dt.total_seconds() / 60.0
    latest_now["gap_min"] = latest_now["gap_min"].clip(lower=0)
    latest_now = latest_now.sort_values(["gap_min", "net_td"], ascending=[True, True]).reset_index(drop=True)
    latest_now["rank"] = latest_now.index + 1
    latest_now["Latest split"] = latest_now["split"].map(lambda s: friendly_split_label(s, km_lookup))
    latest_now["Behind (min)"] = latest_now["gap_min"].map(lambda x: f"{x:.1f}")

    table_df = latest_now[["rank", "name", "Latest split", "Behind (min)"]].rename(
        columns={"rank": "#", "name": "Athlete"}
    )

    # Initialize checkbox state: top 10 selected; leader always on
    if "plot_checks" not in st.session_state:
        st.session_state.plot_checks = {}
        top10 = set(table_df.head(10)["Athlete"].tolist())
        for nm in table_df["Athlete"]:
            st.session_state.plot_checks[nm] = (nm in top10)

    leader_name = table_df.iloc[0]["Athlete"]
    st.session_state.plot_checks[leader_name] = True

    # CSS to narrow columns and force vertical scrollbar
    st.markdown(
        """
        <style>
        .lb-wrapper { max-height: 320px; overflow-y: scroll; padding-right: 10px; }
        .lb-wrapper::-webkit-scrollbar { width: 10px; }
        .lb-wrapper::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.18); border-radius: 6px; }
        .lb-row { padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); }
        .lb-header { position: sticky; top: 0; background: white; z-index: 2;
                     border-bottom: 1px solid rgba(0,0,0,0.2); padding: 6px 0; }
        .lb-col-athlete { width: 16ch; }
        .lb-col-split   { width: 14ch; }
        .lb-col-gap     { width: 6ch; text-align: right; }
        .lb-col-plot    { width: 6ch; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="lb-header">', unsafe_allow_html=True)
    header_cols = st.columns([2.0, 1.7, 1.0, 0.8])
    with header_cols[0]:
        st.markdown('<div class="lb-col-athlete"><strong>Athlete</strong></div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<div class="lb-col-split"><strong>Latest split</strong></div>', unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown('<div class="lb-col-gap"><strong>Behind (min)</strong></div>', unsafe_allow_html=True)
    with header_cols[3]:
        st.markdown('<div class="lb-col-plot"><strong>Plot</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Body
    st.markdown('<div class="lb-wrapper">', unsafe_allow_html=True)
    for _, row in table_df.iterrows():
        athlete = row["Athlete"]
        latest_split = row["Latest split"]
        behind = row["Behind (min)"]

        st.markdown('<div class="lb-row">', unsafe_allow_html=True)
        row_cols = st.columns([2.0, 1.7, 1.0, 0.8])
        with row_cols[0]:
            st.markdown(f'<div class="lb-col-athlete">{athlete}</div>', unsafe_allow_html=True)
        with row_cols[1]:
            st.markdown(f'<div class="lb-col-split">{latest_split}</div>', unsafe_allow_html=True)
        with row_cols[2]:
            st.markdown(f'<div class="lb-col-gap">{behind}</div>', unsafe_allow_html=True)
        with row_cols[3]:
            if athlete == leader_name:
                st.checkbox("", value=True, key=f"plot_{athlete}", disabled=True)
                st.session_state.plot_checks[athlete] = True
            else:
                st.session_state.plot_checks[athlete] = st.checkbox(
                    "", value=st.session_state.plot_checks.get(athlete, False), key=f"plot_{athlete}"
                )
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bulk actions
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Select top 10"):
            top10 = set(table_df.head(10)["Athlete"].tolist())
            for nm in st.session_state.plot_checks:
                st.session_state.plot_checks[nm] = (nm in top10) or (nm == leader_name)
    with c2:
        if st.button("Select none"):
            for nm in st.session_state.plot_checks:
                st.session_state.plot_checks[nm] = (nm == leader_name)
    with c3:
        if st.button("Select all"):
            for nm in st.session_state.plot_checks:
                st.session_state.plot_checks[nm] = True

    selected_for_plot = [nm for nm, on in st.session_state.plot_checks.items() if on]


# ======================================
# 4) Plot: 0 at top, downward y, labels at last point, split labels include distances
# ======================================
master_order = available_splits_in_order(df)
range_splits = split_range(master_order, from_split, to_split)
if not range_splits:
    st.info("No splits available for plotting.")
    st.stop()

sel = df[(df["name"].isin(selected_for_plot)) & (df["split"].astype(str).isin(range_splits))].copy()
if sel.empty:
    st.info("No rows match the current selection and split range.")
    st.stop()

leaders_now = compute_leaders(df)
xy_df = sel.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])
xy_df["leader_hr"] = xy_df["leader_td"].dt.total_seconds() / 3600.0

# Negative minutes so 0 is at the top; hover shows positive minutes
xy_df["y_gap_min"] = -((xy_df["net_td"] - xy_df["leader_td"]).dt.total_seconds() / 60.0)
xy_df["y_gap_min"] = xy_df["y_gap_min"].clip(upper=0)

# START anchors for continuity
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

# Split labels with distances
xy_df["split_label"] = xy_df["split"].astype(str).map(lambda s: friendly_split_label(s, km_lookup))
xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")

fig = go.Figure()

# Draw lines
for nm, g in xy_df.groupby("name", sort=False):
    g = g.sort_values("leader_hr")
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
        customdata=[-v for v in g["y_gap_min"]],
    ))

# Labels at the last point of each series
endpoints = (
    xy_df.sort_values(["name", "leader_hr"])
         .groupby("name", as_index=False)
         .tail(1)
)
labels = []
for _, row in endpoints.iterrows():
    labels.append(dict(
        x=float(row["leader_hr"]),
        y=float(row["y_gap_min"]),
        xref="x", yref="y",
        text=str(row["name"]),
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        align="left",
        font=dict(size=11, color="rgba(0,0,0,0.9)"),
        bgcolor="rgba(255,255,255,0.65)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        opacity=0.95,
    ))

# Axes setup
x_min_data = float(xy_df["leader_hr"].min())
leaders_in_xy = xy_df.groupby("split", as_index=False)["leader_hr"].max()
x_max_leader = float(leaders_in_xy["leader_hr"].max()) if not leaders_in_xy.empty else float(xy_df["leader_hr"].max())
x_right_raw = x_max_leader + 0.25
x_left = math.floor(x_min_data / 0.5) * 0.5
x_right = min(x_right_raw, float(max_hours)) if test_mode else x_right_raw
x_ticks_all = hour_ticks(x_left, x_right, step=0.5)

y_min_val = float(xy_df["y_gap_min"].min())  # negative or 0
y_span = max(1, int(abs(y_min_val)))
y_ticks = [0] + [-(i) for i in range(1, y_span + 1)]
y_ticktext = ["0"] + [str(i) for i in range(1, y_span + 1)]

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
)

fig.update_yaxes(
    title="Time behind leader (min)",
    tickmode="array",
    tickvals=y_ticks,
    ticktext=y_ticktext,        # show positive values
    range=[0.5, y_min_val - 1], # 0 at top; negative extends downward
    showgrid=True,
    zeroline=True,
    zerolinecolor="rgba(0,0,0,0.25)",
    showline=True,
    mirror=True,
    ticks="outside",
)

fig.update_layout(
    height=520,
    margin=dict(l=50, r=30, t=30, b=40),
    annotations=labels,
)

st.plotly_chart(fig, use_container_width=True)
