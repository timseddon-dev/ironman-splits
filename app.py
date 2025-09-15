import os, re, math, statistics
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Live Gaps vs Leader", layout="wide", initial_sidebar_state="collapsed")

DATA_FILE = "long.csv"

# Order we’ll show, filtered to what exists in the CSV
ORDER = (
    ["START", "SWIM", "T1"] +
    [f"BIKE{i}" for i in range(1, 21)] + ["BIKE", "T2"] +
    [f"RUN{i}" for i in range(1, 28)] + ["RUN", "FINISH"]
)

# ---------- Helpers to parse distances from label ----------
DIST_RE = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>km|k|m|mi|mile|miles)", re.IGNORECASE)

def parse_distance_km_from_label(label: str):
    """
    Returns float kilometers or None.
    Accepts '3.8 km', '1500 m', '25 mi', '3.8km', etc.
    """
    if not isinstance(label, str) or not label.strip():
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
    """
    For each split, parse distances from label and pick the most common numeric value.
    """
    dists = {}
    if "split" not in df.columns:
        return dists
    # Ensure label column exists
    if "label" not in df.columns:
        df = df.assign(label=None)

    for split, g in df.groupby("split"):
        vals = []
        for lbl in g["label"].dropna().astype(str):
            km = parse_distance_km_from_label(lbl)
            if km is not None:
                # round to 1 decimal for stability (e.g., 40.899 -> 40.9)
                vals.append(round(km, 1))
        if vals:
            # Choose the mode; if tie, use median
            try:
                chosen = statistics.mode(vals)
            except statistics.StatisticsError:
                chosen = statistics.median(vals)
            dists[str(split)] = float(chosen)
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
        # Fall back to generic names if no distance available
        if s == "SWIM": return "Swim"
        if s.startswith("BIKE"): return "Bike"
        if s.startswith("RUN"): return "Run"
        return s
    if s == "SWIM":
        return f"Swim {km:.1f} km"
    if s.startswith("BIKE"):
        return f"Bike {km:.1f} km"
    if s.startswith("RUN"):
        return f"Run {km:.1f} km"
    return f"{s} {km:.1f} km"

@st.cache_data(ttl=30, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name", "split", "netTime", "label"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["name", "split", "netTime", "label"])

    # Normalize
    if "name" not in df.columns or "split" not in df.columns:
        if "athlete" in df.columns and "split" in df.columns:
            df["name"] = df["athlete"]
        else:
            return pd.DataFrame(columns=["name", "split", "netTime", "label"])

    for c in ["name", "split", "label"]:
        if c not in df.columns:
            df[c] = None

    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(str).str.strip()

    # Parse elapsed net time to timedelta
    def parse_td(s):
        if pd.isna(s):
            return pd.NaT
        x = str(s).strip()
        # Pad times like H:MM:SS and MM:SS into HH:MM:SS
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d+)?", x):
            x = "0" + x
        if re.fullmatch(r"\d{1,2}:\d{2}", x):
            x = "0:" + x
        try:
            return pd.to_timedelta(x)
        except Exception:
            return pd.NaT

    if "netTime" in df.columns:
        df["net_td"] = df["netTime"].apply(parse_td)
    else:
        df["net_td"] = pd.NaT

    # Add a zero START row per athlete to anchor lines
    starts = df[["name"]].drop_duplicates().assign(split="START", net_td=pd.to_timedelta(0, unit="s"), label="Start")
    df = pd.concat([starts, df[["name", "split", "net_td", "label"]]], ignore_index=True)

    # Order splits by our ORDER but only keep those present
    have = df["split"].dropna().unique().tolist()
    ordered = [s for s in ORDER if s in have]
    df["split"] = pd.Categorical(df["split"], categories=ordered, ordered=True)
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
    if i0 <= i1:
        return splits[i0:i1+1]
    return splits[i1:i0+1]

# Load current data
df = load_data(DATA_FILE)
splits_present = df["split"].cat.categories.tolist() if not df.empty else ORDER
split_km_map = build_split_distance_map(df)

st.title("Live Gaps vs Leader")

# Controls
c1, c2 = st.columns(2)
with c1:
    idx_from = splits_present.index("START") if "START" in splits_present else 0
    from_split = st.selectbox("From split", splits_present, index=idx_from)
with c2:
    idx_to = splits_present.index("FINISH") if "FINISH" in splits_present else len(splits_present) - 1
    to_split = st.selectbox("To split", splits_present, index=idx_to)

# Leaderboard
leaders = compute_leader(df)
lf = df.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])

st.subheader("Leaderboard")

if lf.empty:
    st.info("Waiting for live data...")
    selected = []
else:
    # Map each split to a progression index based on the categorical order
    split_order = {s: i for i, s in enumerate(df["split"].cat.categories)}
    lf["split_idx"] = lf["split"].map(split_order)

    # For each athlete, get their latest row (max split_idx, then latest time within that split)
    latest = (
        lf.sort_values(["split_idx", "net_td"])
          .groupby("name", as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )

    # Compute behind in minutes
    latest["gap_min"] = (latest["net_td"] - latest["leader_td"]).dt.total_seconds() / 60.0
    latest["gap_min"] = latest["gap_min"].clip(lower=0)

    # Sort primarily by how far they’ve progressed (split_idx), then by gap
    # Higher split_idx means farther along, so sort descending on split_idx, then ascending on gap_min
    latest = latest.sort_values(["split_idx", "gap_min", "net_td"], ascending=[False, True, True]).reset_index(drop=True)

    latest["Latest split"] = latest["split"].map(lambda s: friendly_label(s, split_km_map))
    latest["Behind (min)"] = latest["gap_min"].map(lambda x: f"{x:.1f}")

    st.markdown("""
        <style>
        .lb-wrap { max-height: 360px; overflow-y: scroll; padding-right: 8px; }
        .lb-wrap::-webkit-scrollbar { width: 10px; }
        .lb-wrap::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.3); border-radius: 6px; }
        .lb-row { display: grid; grid-template-columns: 1.6fr 1.2fr 0.6fr 0.5fr; gap: 12px;
                  padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); align-items: center; }
        .lb-head { position: sticky; top: 0; background: white; z-index: 5;
                   border-bottom: 1px solid rgba(0,0,0,0.2); padding: 6px 0; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lb-row lb-head"><strong>Athlete</strong><strong>Latest split</strong><strong>Behind</strong><strong>Plot</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="lb-wrap">', unsafe_allow_html=True)

    # Checkbox state
    if "plot_checks" not in st.session_state:
        st.session_state.plot_checks = {}
        top = set(latest.head(10)["name"])
        for nm in latest["name"]:
            st.session_state.plot_checks[nm] = nm in top

    # Ensure current leader (first row after our sort) is always selected and cannot be unselected
    leader_name = latest.iloc[0]["name"]
    st.session_state.plot_checks[leader_name] = True

    for _, r in latest.iterrows():
        ck_key = f"plot_{r['name']}"
        checked = st.session_state.plot_checks.get(r["name"], False)
        cols = st.columns([1.6, 1.2, 0.6, 0.5], gap="small")
        cols[0].markdown(f"{r['name']}")
        cols[1].markdown(f"{r['Latest split']}")
        cols[2].markdown(f"{r['Behind (min)']}")
        with cols[3]:
            st.session_state.plot_checks[r["name"]] = st.checkbox(
                "", value=True if r["name"] == leader_name else checked,
                key=ck_key, disabled=(r["name"] == leader_name)
            )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Select top 10"):
            top = set(latest.head(10)["name"])
            for k in st.session_state.plot_checks:
                st.session_state.plot_checks[k] = (k in top) or (k == leader_name)
    with c2:
        if st.button("Select none"):
            for k in st.session_state.plot_checks:
                st.session_state.plot_checks[k] = (k == leader_name)
    with c3:
        if st.button("Select all"):
            for k in st.session_state.plot_checks:
                st.session_state.plot_checks[k] = True

    selected = [nm for nm, on in st.session_state.plot_checks.items() if on]

# Plot
range_splits = split_range(splits_present, from_split, to_split)
plot_df = df[(df["name"].isin(selected)) & (df["split"].astype(str).isin(range_splits))].copy()

if not plot_df.empty:
    leaders = compute_leader(df)
    xy = plot_df.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])
    xy["leader_hr"] = xy["leader_td"].dt.total_seconds() / 3600.0
    xy["gap_min_pos"] = ((xy["net_td"] - xy["leader_td"]).dt.total_seconds() / 60.0).clip(lower=0)
    xy["split_label"] = xy["split"].map(lambda s: friendly_label(s, split_km_map))

    fig = go.Figure()
    for nm, g in xy.groupby("name", sort=False):
        g = g.sort_values("leader_hr")
        fig.add_trace(go.Scatter(
            x=g["leader_hr"], y=g["gap_min_pos"],
            mode="lines", line=dict(width=1.8),
            name=nm, showlegend=False,
            hovertemplate="Athlete: %{text}<br>Split: %{meta}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{y:.1f} min",
            text=[nm]*len(g), meta=g["split_label"],
        ))

    ends = (xy.sort_values(["name", "leader_hr"]).groupby("name", as_index=False).tail(1))
    labels = []
    for _, r in ends.iterrows():
        labels.append(dict(
            x=float(r["leader_hr"]), y=float(r["gap_min_pos"]),
            xref="x", yref="y", text=str(r["name"]), showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=11, color="rgba(0,0,0,0.9)"),
            bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
        ))

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
        title="Time behind leader (min)",
        autorange="reversed",
        showgrid=True, zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        showline=True, mirror=True, ticks="outside"
    )
    fig.update_layout(height=520, margin=dict(l=50, r=30, t=30, b=40), annotations=labels)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select athletes to plot.")
