# ======================================
# Ironman Gaps vs Leader — Full App (for long.csv with columns: name, split, netTime)
# ======================================
import os, math, re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

DATA_FILE = "long.csv"
st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")

# --------------------------------------
# 1) Data load and normalization
# --------------------------------------
@st.cache_data(ttl=60, show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing {path}. Expect columns: name, split, netTime.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Normalize
    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Parse netTime to timedelta
    def _parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        # Pad to HH:MM:SS if M:SS or H:MM:SS
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s):
            s = "0" + s
        if re.fullmatch(r"\d{1,2}:\d{2}", s):
            s = "0:" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            try:
                parts = s.split(":")
                if len(parts) == 2:
                    m, sec = int(parts[0]), int(parts[1])
                    return pd.to_timedelta(m, unit="m") + pd.to_timedelta(sec, unit="s")
                elif len(parts) == 1 and parts[0].isdigit():
                    return pd.to_timedelta(int(parts[0]), unit="s")
            except Exception:
                return pd.NaT
            return pd.NaT

    df["net_td"] = df["netTime"].apply(_parse_td)

    # Keep only needed columns
    df = df[["name", "split", "net_td"]].dropna(subset=["name", "split"]).copy()

    return df


def expected_order():
    # We add synthetic START (0), and final FINISH already exists from feed
    return (
        ["START", "SWIM", "T1"]
        + [f"BIKE{i}" for i in range(1, 26)] + ["BIKE", "T2"]
        + [f"RUN{i}" for i in range(1, 23)] + ["RUN", "FINISH"]
    )


def available_splits_in_order(_df: pd.DataFrame):
    order = expected_order()
    present = [s for s in order if s in _df["split"].dropna().unique().tolist()]
    if present:
        return present
    # fallback by first occurrence time
    d = _df.dropna(subset=["net_td"]).copy()
    if d.empty:
        return sorted(_df["split"].dropna().unique().tolist())
    tmp = (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(first_td=("net_td", "min"))
         .sort_values("first_td")
    )
    return ["START"] + tmp["split"].tolist()


def compute_leaders(_df: pd.DataFrame) -> pd.DataFrame:
    d = _df.dropna(subset=["net_td"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["split", "leader_td"])
    return (
        d.sort_values(["split", "net_td"])
         .groupby("split", as_index=False)
         .agg(leader_td=("net_td", "min"))
    )


# Distance mapping for labels (approx course markers; adjust if needed)
DIST_KM = {
    "SWIM": 3.8,
    "BIKE": 180.2,
    "RUN": 42.2,
}
# Derive partial split distances (rough even spacing: BIKE1..25, RUN1..22)
for i in range(1, 26):
    DIST_KM[f"BIKE{i}"] = 180.2 * i / 25.0
for i in range(1, 23):
    DIST_KM[f"RUN{i}"] = 42.2 * i / 22.0

def friendly_split_label(split: str) -> str:
    s = str(split).upper()
    if s == "START":
        return "Start"
    if s == "FINISH":
        return "Finish"
    if s in ("T1", "T2"):
        return s
    if s.startswith("SWIM"):
        return f"Swim {DIST_KM.get('SWIM', 3.8):.1f} km"
    if s.startswith("BIKE"):
        return f"Bike {DIST_KM.get(s, DIST_KM.get('BIKE', 180.2)):.1f} km"
    if s.startswith("RUN"):
        return f"Run {DIST_KM.get(s, DIST_KM.get('RUN', 42.2)):.1f} km"
    return s


def fmt_hmm(hours_float: float) -> str:
    total_minutes = int(round(hours_float * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh}:{mm:02d}"


def hour_ticks(lo_h: float, hi_h: float, step: float = 0.5) -> list:
    import math
    start = math.floor(lo_h / step) * step
    end = math.ceil(hi_h / step) * step
    vals, v = [], start
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


# Load CSV
df = load_data(DATA_FILE)
if df.empty:
    st.warning("No data loaded from long.csv.")
    st.stop()

# Inject synthetic START at 0:00 for every athlete so From split can be START
start_rows = (
    df.groupby("name", as_index=False)
      .agg(dummy=("split", "first"))
      .assign(split="START", net_td=pd.to_timedelta(0, unit="s"))
      .drop(columns=["dummy"])
)
df = pd.concat([start_rows, df], ignore_index=True, sort=False)

# Ensure split ordering
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass


# --------------------------------------
# 2) UI: Test mode (max hours) + Split selectors (From defaults to START)
# --------------------------------------
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset by athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 6.0, 0.5)

if test_mode:
    max_td = pd.to_timedelta(max_hours, unit="h")
    before_rows = len(df)
    d = df.dropna(subset=["net_td"]).copy()
    df = d[d["net_td"] <= max_td].copy()
    after_rows = len(df)
    st.caption(f"Test mode active: {after_rows:,} rows (from {before_rows:,}) with athlete elapsed ≤ {max_hours:.1f} h")

# Re-assert order after filtering
splits_ordered_master = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered_master, ordered=True)
except Exception:
    pass

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


# --------------------------------------
# 3) Leaderboard — narrow columns, sticky header, vertical scrollbar
# --------------------------------------
leaders_now = compute_leaders(df)
df_now = df.merge(leaders_now, on="split", how="left").dropna(subset=["net_td", "leader_td"])

st.subheader("Leaderboard")

if df_now.empty:
    st.info("No data available for the current dataset.")
    selected_for_plot = []
else:
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

    latest_now["Latest split"] = latest_now["split"].map(friendly_split_label)
    latest_now["Behind (min)"] = latest_now["gap_min"].map(lambda x: f"{x:.1f}")

    table_df = latest_now[["rank", "name", "Latest split", "Behind (min)"]].rename(
        columns={"rank": "#", "name": "Athlete"}
    )

    # Initialize checkboxes: top 10; leader always on
    if "plot_checks" not in st.session_state:
        st.session_state.plot_checks = {}
        top10 = set(table_df.head(10)["Athlete"].tolist())
        for nm in table_df["Athlete"]:
            st.session_state.plot_checks[nm] = (nm in top10)
    leader_name = table_df.iloc[0]["Athlete"]
    st.session_state.plot_checks[leader_name] = True

    # Tight columns + forced scroll
    st.markdown(
        """
        <style>
        .lb-wrapper { max-height: 300px; overflow-y: scroll; padding-right: 8px; }
        .lb-row { padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); }
        .lb-header { position: sticky; top: 0; background: white; z-index: 2;
                     border-bottom: 1px solid rgba(0,0,0,0.2); padding: 6px 0; }
        .lb-col-athlete { width: 14ch; }
        .lb-col-split   { width: 13ch; }
        .lb-col-gap     { width: 6ch; text-align: right; }
        .lb-col-plot    { width: 6ch; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="lb-header">', unsafe_allow_html=True)
    header_cols = st.columns([1.8, 1.6, 1.0, 0.8])
    with header_cols[0]:
        st.markdown('<div class="lb-col-athlete"><strong>Athlete</strong></div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<div class="lb-col-split"><strong>Latest split</strong></div>', unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown('<div class="lb-col-gap"><strong>Behind (min)</strong></div>', unsafe_allow_html=True)
    with header_cols[3]:
        st.markdown('<div class="lb-col-plot"><strong>Plot on chart</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Body
    st.markdown('<div class="lb-wrapper">', unsafe_allow_html=True)
    for _, row in table_df.iterrows():
        athlete = row["Athlete"]
        latest_split = row["Latest split"]
        behind = row["Behind (
