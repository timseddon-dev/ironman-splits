import os, re, math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Live Gaps vs Leader", layout="wide", initial_sidebar_state="collapsed")

DATA_FILE = "long.csv"

# Distances from your screenshots
KM_MAP = {
    "SWIM": 3.8,

    # Bike checkpoints
    "BIKE1": 9.8, "BIKE2": 26.7, "BIKE3": 40.9, "BIKE4": 46.6, "BIKE5": 53.3,
    "BIKE6": 60.9, "BIKE7": 67.6, "BIKE8": 81.2, "BIKE9": 88.9, "BIKE10": 94.6,
    "BIKE11": 100.0, "BIKE12": 110.0, "BIKE13": 120.0, "BIKE14": 129.0,
    "BIKE15": 136.0, "BIKE16": 143.0, "BIKE17": 151.0, "BIKE18": 160.0,
    "BIKE19": 168.0, "BIKE20": 170.0,

    # Run checkpoints (confirmed)
    "RUN1": 0.15, "RUN2": 1.1,  "RUN3": 2.1,  "RUN4": 3.1,  "RUN5": 4.1,  "RUN6": 5.4,
    "RUN7": 6.7,  "RUN8": 7.7,  "RUN9": 8.7,  "RUN10": 9.7, "RUN11": 10.6, "RUN12": 11.6,
    "RUN13": 12.6, "RUN14": 13.6, "RUN15": 14.6, "RUN16": 15.9, "RUN17": 17.2,
    "RUN18": 18.2, "RUN19": 19.2, "RUN20": 20.2, "RUN21": 21.2, "RUN22": 22.2,
    "RUN23": 23.2, "RUN24": 24.2, "RUN25": 25.2,

    # Additional estimates you provided
    "RUN26": 40.2,  # Est. Run 25 mi | 40.2 km
    "RUN27": 41.2,  # Est. Run 25.6 mi | 41.2 km
}

# Full order we’ll show, filtered to what exists in the CSV
ORDER = (
    ["START", "SWIM", "T1"] +
    [f"BIKE{i}" for i in range(1, 21)] + ["BIKE", "T2"] +
    [f"RUN{i}" for i in range(1, 28)] + ["RUN", "FINISH"]
)

@st.cache_data(ttl=30, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name", "split", "netTime"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["name", "split", "netTime"])

    # Normalize
    if "name" not in df.columns or "split" not in df.columns:
        # Keep file tolerant if upstream changes keys
        if "athlete" in df.columns and "split" in df.columns:
            df["name"] = df["athlete"]
        else:
            return pd.DataFrame(columns=["name", "split", "netTime"])

    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Parse elapsed net time
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
    starts = (
        df[["name"]].drop_duplicates()
          .assign(split="START", net_td=pd.to_timedelta(0, unit="s"))
    )
    df = pd.concat([starts, df[["name", "split", "net_td"]]], ignore_index=True)

    # Order splits by our ORDER but only keep those present
    have = df["split"].dropna().unique().tolist()
    ordered = [s for s in ORDER if s in have]
    df["split"] = pd.Categorical(df["split"], categories=ordered, ordered=True)
    return df.dropna(subset=["split"])

def friendly_label(s: str) -> str:
    s = str(s).upper()
    if s == "START":
        return "Start"
    if s == "FINISH":
        return "Finish"
    if s in ("T1", "T2"):
        return s
    if s == "SWIM":
        km = KM_MAP.get("SWIM")
        return f"Swim {km:.1f} km" if km else "Swim"
    if s.startswith("BIKE"):
        km = KM_MAP.get(s)
        return f"Bike {km:.1f} km" if km else "Bike"
    if s.startswith("RUN"):
        km = KM_MAP.get(s)
        return f"Run {km:.1f} km" if km else "Run"
    return s

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
splits_present = (
    df["split"].cat.categories.tolist() if not df.empty else ORDER
)

st.title("Live Gaps vs Leader")

# Controls
c1, c2 = st.columns(2)
with c1:
    idx_from = splits_present.index("START") if "START" in splits_present else 0
    from_split = st.selectbox("From split", splits_present, index=idx_from)
with c2:
    idx_to = splits_present.index("FINISH") if "FINISH" in splits_present else len(splits_present) - 1
    to_split = st.selectbox("To split", splits_present, index=idx_to)

# Leaderboard with forced vertical scrollbar
leaders = compute_leader(df)
lf = df.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])

st.subheader("Leaderboard")

if lf.empty:
    st.info("Waiting for live data...")
    selected = []
else:
    latest = (
        lf.sort_values(["name", "net_td"])
          .groupby("name", as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )
    latest["gap_min"] = (latest["net_td"] - latest["leader_td"]).dt.total_seconds() / 60.0
    latest["gap_min"] = latest["gap_min"].clip(lower=0)
    latest = latest.sort_values(["gap_min", "net_td"]).reset_index(drop=True)

    latest["Latest split"] = latest["split"].map(friendly_label)
    latest["Behind (min)"] = latest["gap_min"].map(lambda x: f"{x:.1f}")

    # CSS to guarantee scrollbar and keep columns tight
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
    xy["split_label"] = xy["split"].map(friendly_label)

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

    # Label at last point
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

    # Axes: 0 at top using reversed autorange
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
