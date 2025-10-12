# app.py

# 0) Imports and page setup
import os
import re
import math
import statistics
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Ironman tracker (WIP)", layout="wide", initial_sidebar_state="collapsed")

DATA_FILE = "long.csv"

# 1) Split order definition (current static baseline; can be made dynamic)
ORDER = (
    ["START", "SWIM", "T1"] +
    [f"BIKE{i}" for i in range(1, 21)] + ["BIKE", "T2"] +
    [f"RUN{i}" for i in range(1, 28)] + ["RUN", "FINISH"]
)

# 2) Helpers: distance parsing and labeling
DIST_RE = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>km|k|m|mi|mile|miles)", re.IGNORECASE)

def parse_distance_km_from_label(label: str):
    if not isinstance(label, str):
        return None
    m = DIST_RE.search(label)
    if not m:
        return None
    val = float(m.group("val"))
    unit = m.group("unit").lower()
    if unit in ("km", "k"):
        return val
    if unit == "m":
        return val / 1000.0
    if unit in ("mi", "mile", "miles"):
        return val * 1.609344
    return None

def build_split_distance_map(df: pd.DataFrame) -> dict:
    dists = {}
    if "label" not in df.columns:
        df = df.assign(label=None)
    for split, g in df.groupby("split"):
        vals = []
        for lbl in g["label"].dropna():
            km = parse_distance_km_from_label(lbl)
            if km is not None:
                vals.append(round(km, 1))
        if vals:
            try:
                chosen = statistics.mode(vals)
            except statistics.StatisticsError:
                chosen = statistics.median(vals)
            dists[str(split).upper()] = float(chosen)
    return dists

def friendly_label(split: str, split_km: dict) -> str:
    s = str(split).upper()
    if s == "START":
        return "Start"
    if s == "FINISH":
        return "Finish"
    if s in ("T1", "T2"):
        return s
    km = split_km.get(s)
    if km is None:
        if s == "SWIM": return "Swim"
        if s.startswith("BIKE"): return "Bike"
        if s.startswith("RUN"): return "Run"
        return s.title()
    if s == "SWIM": return f"Swim {km:.1f} km"
    if s.startswith("BIKE"): return f"Bike {km:.1f} km"
    if s.startswith("RUN"): return f"Run {km:.1f} km"
    return f"{s} {km:.1f} km"

# 3) Data loading and normalization
@st.cache_data(ttl=30, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name", "split", "netTime", "label"])
    df = pd.read_csv(path)

    # Ensure required cols
    for c in ["name", "split", "label", "netTime"]:
        if c not in df.columns:
            df[c] = None

    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(str).fillna("").str.strip()

    # Parse net time -> timedelta
    def parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        if re.fullmatch(r"\d{1,2}:\d{2}", s):
            s = "0:" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            return pd.NaT

    df["net_td"] = df["netTime"].apply(parse_td)

    # Add START row for each athlete with 0:00 if not present
    if not df.empty:
        cur_names = df["name"].dropna().unique().tolist()
        have_start = df[df["split"] == "START"]["name"].unique().tolist()
        need_starts = sorted(set(cur_names) - set(have_start))
        if need_starts:
            starts = pd.DataFrame({
                "name": need_starts,
                "split": ["START"] * len(need_starts),
                "net_td": [pd.to_timedelta(0, unit="s")] * len(need_starts),
                "label": ["" for _ in range(len(need_starts))],
                "netTime": ["0:00:00"] * len(need_starts),
            })
            df = pd.concat([df, starts], ignore_index=True)

    # Categorical split order based on ORDER filtered to present values
    present = [s for s in ORDER if s in set(df["split"].unique())]
    if not present:
        present = ["START"]
    df["split"] = pd.Categorical(df["split"], categories=present, ordered=True)
    return df.dropna(subset=["split"])

def compute_leader(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["net_td"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["split", "leader_td"])
    return (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(leader_td=("net_td", "min"))
    )

def split_range(splits, a, b):
    if a not in splits or b not in splits:
        return splits
    i0, i1 = splits.index(a), splits.index(b)
    return splits[i0:i1+1] if i0 <= i1 else splits[i1:i0+1]

# 4) Load data and prepare utilities
df = load_data(DATA_FILE)
split_km_map = build_split_distance_map(df)

st.title("Live Gaps vs Leader")

# 5) Leaderboard data preparation

# 5.1) Compute per-split leader times (minimum net time at each split)
leaders = compute_leader(df)

# 5.2) Merge leader times into base data (all rows)
lf = df.merge(leaders, on="split", how="left")

# 5.2.1) Compute per-athlete latest split correctly
lf_valid = lf.dropna(subset=["net_td", "leader_td"]).copy()

# Build a numeric progression index from the categorical order
split_order = {s: i for i, s in enumerate(df["split"].cat.categories)}
lf_valid["split_idx"] = lf_valid["split"].map(split_order)
lf_valid["split_idx"] = pd.to_numeric(lf_valid["split_idx"], errors="coerce").astype("Int64")

# For each athlete, take the chronologically latest split with a valid time
latest = (
    lf_valid.sort_values(["name", "split_idx", "net_td"])
            .groupby("name", as_index=False)
            .tail(1)
            .reset_index(drop=True)
)
latest["split_idx"] = pd.to_numeric(latest["split_idx"], errors="coerce").astype("Int64")

# Gap to leader in minutes (non-negative)
latest["gap_min"] = (latest["net_td"] - latest["leader_td"]).dt.total_seconds() / 60.0
latest["gap_min"] = latest["gap_min"].clip(lower=0)

# Derive label for latest split
def _latest_label(row):
    lbl = str(row.get("label") or "").strip()
    if lbl:
        return lbl
    return friendly_label(row["split"], split_km_map)

latest["Latest split"] = latest.apply(_latest_label, axis=1)
latest["Time behind leader"] = latest["gap_min"].map(lambda x: f"{x:.1f}")

# 5.2.2) Compute “Places” and “Gap to in front” deltas vs previous split

# Rank within each split by net time (ascending)
ranked = (
    lf_valid.sort_values(["split_idx", "net_td"])
            .assign(place=lambda d: d.groupby("split_idx")["net_td"].rank(method="first"))
)
ranked["split_idx"] = pd.to_numeric(ranked["split_idx"], errors="coerce").astype("Int64")
ranked["place"] = ranked["place"].astype(int)

# Previous split index per row (START -> <NA>)
ranked["prev_split_idx"] = ranked["split_idx"] - 1

# Current place per athlete at latest split
current_place = ranked[["name", "split_idx", "place"]].copy().rename(columns={"place": "place_now"})

# Previous place per athlete at previous split (align by name and prev_split_idx)
prev_place = ranked.dropna(subset=["prev_split_idx"]).merge(
    current_place[["name", "split_idx"]],
    left_on=["name", "prev_split_idx"],
    right_on=["name", "split_idx"],
    how="inner",
    suffixes=("", "_curref")
)[["name", "split_idx_curref", "place"]].rename(columns={
    "split_idx_curref": "split_idx",
    "place": "place_prev"
})

# Merge places into latest
latest = latest.merge(current_place, on=["name", "split_idx"], how="left")
latest = latest.merge(prev_place, on=["name", "split_idx"], how="left")

def compute_places_delta(prev_val, cur_val):
    try:
        if pd.isna(prev_val) or pd.isna(cur_val):
            return pd.NA
        # If moved from 4th to 3rd => +1 (gained)
        return int(prev_val) - int(cur_val)
    except Exception:
        return pd.NA

latest["Places_delta"] = [
    compute_places_delta(p, c) for p, c in zip(latest["place_prev"], latest["place_now"])
]

# Gap to in front per split: for sorted rows within each split, compute time gap to the athlete ahead
def per_split_gap_to_front(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("net_td").reset_index(drop=True)
    d["gap_to_front_min"] = pd.NA
    for i in range(1, len(d)):
        delta = (d.loc[i, "net_td"] - d.loc[i - 1, "net_td"]).total_seconds() / 60.0
        d.loc[i, "gap_to_front_min"] = max(delta, 0.0)
    return d

per_split = ranked.groupby("split_idx", group_keys=False).apply(per_split_gap_to_front)
per_split["split_idx"] = pd.to_numeric(per_split["split_idx"], errors="coerce").astype("Int64")

cur_gap = per_split[["name", "split_idx", "gap_to_front_min"]].copy().rename(columns={"gap_to_front_min": "gap_front_now"})

# Previous gap_to_front aligned to current split (shift index forward)
prev_gap = per_split.copy()
prev_gap["split_idx"] = prev_gap["split_idx"] + 1
prev_gap = prev_gap.rename(columns={"gap_to_front_min": "gap_front_prev"})

latest = latest.merge(cur_gap, on=["name", "split_idx"], how="left")
latest = latest.merge(prev_gap[["name", "split_idx", "gap_front_prev"]], on=["name", "split_idx"], how="left")

def safe_gap_delta(prev_val, cur_val):
    try:
        if pd.isna(prev_val) or pd.isna(cur_val):
            return pd.NA
        return round(float(prev_val) - float(cur_val), 1)
    except Exception:
        return pd.NA

latest["Gap_to_in_front_delta"] = [
    safe_gap_delta(p, c) for p, c in zip(latest["gap_front_prev"], latest["gap_front_now"])
]

# 5.2.3) Final sort for display
latest = latest.sort_values(
    ["split_idx", "gap_min", "net_td"],
    ascending=[False, True, True]
).reset_index(drop=True)



# 5.3) Leaderboard display (native Streamlit data_editor with live selection, fixed widths, centered headings/cells)
import pandas as pd
import streamlit as st

st.subheader("Leaderboard")

if latest.empty:
    st.info("Waiting for live data...")
    selected = []
else:
    # Build view dataframe
    view = latest[["name", "Latest split", "Time behind leader", "Places_delta", "Gap_to_in_front_delta"]].copy()
    view = view.rename(columns={"name": "Athlete"})

    # Format “Places” and “Gap to in front” per spec
    def fmt_places(val):
        if pd.isna(val):
            return ""
        v = int(val)
        if v > 0:  return f"+{v}"
        if v < 0:  return f"-{abs(v)}"
        return "—"  # dash for no change

    def fmt_gap_delta(val):
        if pd.isna(val):
            return ""
        x = float(val)
        if abs(x) < 1e-9:
            return "—"  # dash for no change
        sign = "+" if x > 0 else "−"
        total_seconds = int(round(abs(x) * 60))
        m, s = divmod(total_seconds, 60)
        return f"{sign}{m}:{s:02d}"

    view["Places"] = view["Places_delta"].map(fmt_places)
    view["Gap to in front"] = view["Gap_to_in_front_delta"].map(fmt_gap_delta)
    view["Time behind leader"] = view["Time behind leader"].astype(str)

    # Selection column (Plot)
    # Initialize session selections; ensure leader selected
    if "plot_checks" not in st.session_state:
        init = {row["Athlete"]: False for _, row in view.iterrows()}
        if not view.empty:
            init[view.iloc[0]["Athlete"]] = True
        st.session_state.plot_checks = init

    # Add Plot column from session_state
    view["Plot"] = view["Athlete"].map(lambda nm: bool(st.session_state.plot_checks.get(nm, False)))

    # Display a header band to simulate the merged header row (centered except first)
    st.markdown("""
    <style>
      .hdr-band { display:grid; grid-template-columns: 26% 24% 16% 12% 14% 8%; gap:0; align-items:center;
                  border:1px solid rgba(0,0,0,0.08); border-bottom:none; border-radius:6px 6px 0 0; }
      .hdr-band div { padding: 6px 8px; font-weight:700; background:#fff; border-bottom:1px solid rgba(0,0,0,0.15); }
      .hdr-c1 { text-align:left; }
      .hdr-c2, .hdr-c3, .hdr-c6 { text-align:center; }
      .hdr-sub { display:grid; grid-template-columns: 26% 24% 16% 12% 14% 8%; gap:0; align-items:center;
                 border:1px solid rgba(0,0,0,0.08); border-top:none; border-bottom:none; }
      .hdr-sub div { padding: 6px 8px; font-weight:600; background:#fff; border-bottom:1px solid rgba(0,0,0,0.08); }
      .hdr-sub .spacer { grid-column: 1 / span 3; }
      .hdr-sub .group { grid-column: 4 / span 2; text-align:center; font-weight:700; border-bottom:1px solid rgba(0,0,0,0.15); }
      .hdr-sub .plot { grid-column: 6; text-align:center; }
    </style>
    <div class="hdr-band">
      <div class="hdr-c1">Athlete</div>
      <div class="hdr-c2">Latest split</div>
      <div class="hdr-c3">Time behind leader</div>
      <div class="hdr-c4"></div>
      <div class="hdr-c5"></div>
      <div class="hdr-c6">Plot</div>
    </div>
    <div class="hdr-sub">
      <div class="spacer"></div>
      <div class="group">Change since last split</div>
      <div class="plot"> </div>
    </div>
    """, unsafe_allow_html=True)

    # Reorder/rename for editor
    editor_df = view[["Athlete", "Latest split", "Time behind leader", "Places", "Gap to in front", "Plot"]].copy()

    # Column configuration for alignment and fixed widths
    col_config = {
        "Athlete": st.column_config.TextColumn("Athlete", width="medium"),
        "Latest split": st.column_config.TextColumn("Latest split", width="medium"),
        "Time behind leader": st.column_config.TextColumn("Time behind leader", width="small"),
        "Places": st.column_config.TextColumn("Places", width="small"),
        "Gap to in front": st.column_config.TextColumn("Gap to in front", width="small"),
        "Plot": st.column_config.CheckboxColumn("Plot", help="Select athletes to plot"),
    }

    # Data editor with fixed height, no index
    edited = st.data_editor(
        editor_df,
        hide_index=True,
        use_container_width=True,
        disabled=["Athlete", "Latest split", "Time behind leader", "Places", "Gap to in front"],
        column_config=col_config,
        height=min(520, 60 + 32 * (len(editor_df) + 2)),
        key="leaderboard_editor",
    )

    # Update session_state selections based on edits
    for _, row in edited.iterrows():
        st.session_state.plot_checks[row["Athlete"]] = bool(row["Plot"])

    # Selected list for plotting
    selected = [nm for nm, on in st.session_state.plot_checks.items() if on]
    
    
# 6) Plot controls
splits_present = list(df["split"].cat.categories)
c1, c2 = st.columns(2)
with c1:
    idx_from = splits_present.index("START") if "START" in splits_present else 0
    from_split = st.selectbox("From split", splits_present, index=idx_from, key="from_sel")
with c2:
    idx_to = splits_present.index("FINISH") if "FINISH" in splits_present else len(splits_present) - 1
    to_split = st.selectbox("To split", splits_present, index=idx_to, key="to_sel")

# 7) Data slice for plotting
range_splits = split_range(splits_present, from_split, to_split)
plot_df = df[(df["name"].isin(selected)) & (df["split"].isin(range_splits))].copy()

if plot_df.empty:
    st.info("Select athletes to plot.")
else:
    leaders2 = compute_leader(df[df["split"].isin(range_splits)])
    xy = (plot_df.merge(leaders2, on="split", how="left")
                 .dropna(subset=["net_td", "leader_td"]))
    xy["leader_hr"] = xy["leader_td"].dt.total_seconds() / 3600.0
    xy["gap_min_pos"] = ((xy["net_td"] - xy["leader_td"]).dt.total_seconds() / 60.0).clip(lower=0)
    xy["split_label"] = xy["split"].map(lambda s: friendly_label(s, split_km_map))

    fig = go.Figure()

# 8) Plot
if not plot_df.empty:
    leaders2 = compute_leader(df)
    xy = plot_df.merge(leaders2, on="split", how="left").dropna(subset=["net_td", "leader_td"])
    xy["leader_hr"] = xy["leader_td"].dt.total_seconds() / 3600.0
    xy["gap_min_pos"] = ((xy["net_td"] - xy["leader_td"]).dt.total_seconds() / 60.0).clip(lower=0)
    xy["split_label"] = xy["split"].map(lambda s: friendly_label(s, split_km_map))

    fig = go.Figure()

    # 8.1) Main athlete lines (legend off)
    for nm, g in xy.groupby("name", sort=False):
        g = g.sort_values("leader_hr")
        fig.add_trace(go.Scatter(
            x=g["leader_hr"], y=g["gap_min_pos"],
            mode="lines",
            name=str(nm),
            line=dict(width=1.8),
            showlegend=False,
            hovertemplate="Athlete: %{text}<br>Split: %{meta}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{y:.1f} min",
            text=[nm] * len(g),
            meta=g["split_label"],
        ))

    # 8.2) End labels
    ends = (xy.sort_values(["name", "leader_hr"]).groupby("name", as_index=False).tail(1))
    annotations = [
        dict(
            x=float(r["leader_hr"]), y=float(r["gap_min_pos"]),
            xref="x", yref="y", text=str(r["name"]), showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=11, color="rgba(0,0,0,0.75)")
        )
        for _, r in ends.iterrows()
    ]

    # 8.3) Axes and ticks
    if len(xy):
        x_left = math.floor(xy["leader_hr"].min() / 0.5) * 0.5
        x_right = math.ceil(xy["leader_hr"].max() / 0.5) * 0.5 + 0.25
        x_ticks = [round(x_left + 0.5 * i, 2) for i in range(int((x_right - x_left) / 0.5) + 1)]
    else:
        x_ticks = []

    fig.update_xaxes(
        title="Leader elapsed (h)",
        tickvals=x_ticks,
        ticktext=[f"{int(v)}:{int((v%1)*60):02d}" for v in x_ticks],
        showgrid=True, zeroline=False, showline=True, mirror=True, ticks="outside"
    )
    fig.update_yaxes(
        title="Behind (min)",
        autorange="reversed",
        showgrid=True, zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        showline=True, mirror=True, ticks="outside"
    )

    # 8.4) Vertical reference lines at SWIM, T1, BIKE, T2
    ref_points = {}
    for anchor in ["SWIM", "T1", "BIKE", "T2"]:
        sub = xy[xy["split"] == anchor]
        if not sub.empty:
            ref_points[anchor] = float(sub["leader_hr"].min())

    y_min = 0.0
    y_max = float(xy["gap_min_pos"].max()) if len(xy) else 1.0
    y_max = max(y_max, 1.0)

    for anchor, x_val in ref_points.items():
        color = {
            "SWIM": "rgba(0,0,0,0.45)",
            "T1":   "rgba(0,0,0,0.35)",
            "BIKE": "rgba(0,0,0,0.45)",
            "T2":   "rgba(0,0,0,0.35)",
        }[anchor]
        fig.add_trace(go.Scatter(
            x=[x_val, x_val],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color=color, width=1.6, dash="solid"),
            name=anchor,
            showlegend=False,
            hovertemplate=anchor + " at leader elapsed: %{x:.2f} h",
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=30),
        showlegend=False,
        annotations=annotations
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select athletes to plot.")
