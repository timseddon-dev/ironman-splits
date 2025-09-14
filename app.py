import os, re, math, time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Live Gaps vs Leader", layout="wide", initial_sidebar_state="collapsed")

DATA_FILE = "long.csv"

# =========================
# 1) Global settings and cache clear
# =========================
import os, re, math, time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Live Gaps vs Leader", layout="wide", initial_sidebar_state="collapsed")

# IMPORTANT: clear any stale cache on startup (safe to keep during live event)
try:
    st.cache_data.clear()
except Exception:
    pass

DATA_FILE = "long.csv"

# =========================
# 2) In-app updater (every 3 min) with diagnostics
# =========================
BASE = "https://track.rtrt.me"
EVENT = "IRM-WORLDCHAMPIONSHIP-MEN-2025"  # confirmed by you
CATEGORY = "MPRO"

# Ask for a broad set; API returns only existing points.
POINTS = (
    ["START", "SWIM", "T1"] +
    [f"BIKE{i}" for i in range(1, 21)] + ["BIKE", "T2"] +
    [f"RUN{i}" for i in range(1, 28)] + ["RUN", "FINISH"]
)

HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-GB,en;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://track.rtrt.me",
    "Referer": f"https://track.rtrt.me/e/{EVENT}#",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}

def _fetch_point_df(point: str) -> pd.DataFrame:
    """
    Try GET first (most trackers allow GET now). Fall back to POST without categories.
    Returns DataFrame[name, split, netTime]. Adds cache-buster to avoid CDN.
    """
    cache_bust = int(time.time())
    url = f"{BASE}/e/{EVENT}/categories/{CATEGORY}/splits/{point}?t={cache_bust}"

    def normalize(js) -> pd.DataFrame:
        rows = (js or {}).get("rows") or []
        if not rows:
            return pd.DataFrame(columns=["name", "split", "netTime"])
        df = pd.DataFrame(rows)
        if "name" not in df.columns and "athlete" in df.columns:
            df["name"] = df["athlete"]
        if "name" not in df.columns:
            df["name"] = None
        if "netTime" not in df.columns:
            if {"hh", "mm", "ss"} <= set(df.columns):
                df["netTime"] = (
                    df["hh"].astype(int).astype(str) + ":" +
                    df["mm"].astype(int).astype(str).str.zfill(2) + ":" +
                    df["ss"].astype(int).astype(str).str.zfill(2)
                )
            else:
                df["netTime"] = None
        df["split"] = point
        return df[["name", "split", "netTime"]]

    # Try GET
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return normalize(r.json())
    except Exception:
        pass

    # Fallback: POST without categories (some endpoints reject categories form data)
    try:
        r = requests.post(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return normalize(r.json())
    except Exception as e:
        # Bubble up so the caller can count/log the error
        raise e

@st.cache_data(ttl=180, show_spinner=True)
def refresh_long_csv() -> dict:
    """
    Pulls all POINTS and writes long.csv. Returns diagnostic dict.
    ttl=180s ensures the app re-runs roughly every 3 minutes while open.
    """
    frames, fetched, errors = [], 0, 0
    first_error = None
    for p in POINTS:
        try:
            dfp = _fetch_point_df(p)
            if not dfp.empty:
                frames.append(dfp)
                fetched += len(dfp)
        except Exception as e:
            errors += 1
            if first_error is None:
                first_error = str(e)
        time.sleep(0.12)  # be gentle to API

    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    wrote = False
    if frames:
        long_df = pd.concat(frames, ignore_index=True)
        long_df.to_csv(DATA_FILE, index=False, encoding="utf-8")
        wrote = True

    return {
        "timestamp_utc": ts,
        "event": EVENT,
        "fetched_rows": fetched,
        "errors": errors,
        "first_error": first_error,
        "wrote_csv": wrote,
        "sample_split_url": f"{BASE}/e/{EVENT}/categories/{CATEGORY}/splits/SWIM",
    }

# Trigger refresh on run; returns diagnostics for display
fetch_info = refresh_long_csv()

# =========================
# 3) Debug panel
# =========================
st.caption("Live source diagnostics")
with st.expander("Show fetch details"):
    st.json(fetch_info)
    st.markdown(f"[Open sample split in a new tab]({fetch_info['sample_split_url']})")
# =========================
# 5) Data loading and helpers
# =========================
@st.cache_data(ttl=30, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name", "split", "netTime"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["name", "split", "netTime"])

    # Normalize
    if "name" not in df.columns or "split" not in df.columns:
        if "athlete" in df.columns and "split" in df.columns:
            df["name"] = df["athlete"]
        else:
            return pd.DataFrame(columns=["name", "split", "netTime"])

    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Parse elapsed net time to timedelta
    def parse_td(s):
        if pd.isna(s):
            return pd.NaT
        x = str(s).strip()
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d+)?", x):
            x = "0" + x
        if re.fullmatch(r"\d{1,2}:\d{2}", x):
            x = "0:" + x
        try:
            return pd.to_timedelta(x)
        except Exception:
            return pd.NaT

    df["net_td"] = df.get("netTime", pd.Series([None]*len(df))).apply(parse_td)

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
    if s == "START": return "Start"
    if s == "FINISH": return "Finish"
    if s in ("T1", "T2"): return s
    if s == "SWIM": return f"Swim {KM_MAP.get('SWIM', 3.8):.1f} km"
    if s.startswith("BIKE"):
        km = KM_MAP.get(s); return f"Bike {km:.1f} km" if km else "Bike"
    if s.startswith("RUN"):
        km = KM_MAP.get(s); return f"Run {km:.1f} km" if km else "Run"
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
    if i0 <= i1: return splits[i0:i1+1]
    return splits[i1:i0+1]
# =========================
# 6) App UI, leaderboard and plot
# =========================
df = load_data(DATA_FILE)
splits_present = df["split"].cat.categories.tolist() if not df.empty else ORDER

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

    # Guaranteed scrollbar via CSS
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
