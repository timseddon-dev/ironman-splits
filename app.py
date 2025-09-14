import os, math, re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")
DATA_FILE = "long.csv"

@st.cache_data(ttl=30, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["name","split","netTime"])
    df = pd.read_csv(path)
    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    def _parse_td(x):
        if pd.isna(x): return pd.NaT
        s = str(x).strip()
        if re.fullmatch(r"\d:\d{2}:\d{2}(\.\d{1,3})?", s): s = "0" + s
        if re.fullmatch(r"\d{1,2}:\d{2}", s): s = "0:" + s
        try:
            return pd.to_timedelta(s)
        except Exception:
            try:
                parts = s.split(":")
                if len(parts)==2: return pd.to_timedelta(int(parts[0]), unit="m")+pd.to_timedelta(int(parts[1]), unit="s")
                if len(parts)==1 and parts[0].isdigit(): return pd.to_timedelta(int(parts[0]), unit="s")
            except Exception:
                return pd.NaT
            return pd.NaT

    df["net_td"] = df["netTime"].apply(_parse_td)

    # Inject START (0:00) per athlete
    start_rows = (
        df.groupby("name", as_index=False)
          .agg(dummy=("split","first"))
          .assign(split="START", net_td=pd.to_timedelta(0, unit="s"))
          .drop(columns=["dummy"])
    )
    df = pd.concat([start_rows, df[["name","split","net_td"]]], ignore_index=True, sort=False)
    return df.dropna(subset=["name","split"])

def expected_order():
    return (["START","SWIM","T1"]
            + [f"BIKE{i}" for i in range(1,26)] + ["BIKE","T2"]
            + [f"RUN{i}" for i in range(1,23)] + ["RUN","FINISH"])

def available_splits_in_order(df):
    order = expected_order()
    have = df["split"].dropna().unique().tolist()
    present = [s for s in order if s in have]
    return present or sorted(have)

def compute_leaders(df):
    d = df.dropna(subset=["net_td"]).copy()
    if d.empty: return pd.DataFrame(columns=["split","leader_td"])
    return (d.sort_values(["split","net_td"])
             .groupby("split", as_index=False)
             .agg(leader_td=("net_td","min")))

def friendly_split_label(s: str, km_map: dict|None=None) -> str:
    s = str(s).upper()
    if s=="START": return "Start"
    if s=="FINISH": return "Finish"
    if s in ("T1","T2"): return s
    if km_map and s in km_map:  # will wire exact distances when you send them
        if s.startswith("BIKE"): return f"Bike {km_map[s]:.1f} km"
        if s.startswith("RUN"): return f"Run {km_map[s]:.1f} km"
        if s=="SWIM": return f"Swim {km_map[s]:.1f} km"
    if s=="SWIM": return "Swim"
    if s.startswith("BIKE"): return "Bike"
    if s.startswith("RUN"): return "Run"
    return s

def fmt_hmm(h):
    total_minutes = int(round(h*60))
    return f"{total_minutes//60}:{total_minutes%60:02d}"

df = load_data(DATA_FILE)
if df.empty:
    st.info("Waiting for long.csv (updater writes every 3 minutes)...")
    st.stop()

splits_ordered = available_splits_in_order(df)
try:
    df["split"] = pd.Categorical(df["split"], categories=splits_ordered, ordered=True)
except Exception:
    pass

# Split selectors (From defaults to START)
colA, colB = st.columns([1,1])
with colA:
    from_idx = splits_ordered.index("START") if "START" in splits_ordered else 0
    from_split = st.selectbox("From split", splits_ordered, index=from_idx)
with colB:
    to_idx = splits_ordered.index("FINISH") if "FINISH" in splits_ordered else len(splits_ordered)-1
    to_split = st.selectbox("To split", splits_ordered, index=to_idx)

def split_range(splits, s0, s1):
    if s0 not in splits or s1 not in splits: return splits
    i0, i1 = splits.index(s0), splits.index(s1)
    return splits[i0:i1+1] if i0<=i1 else splits[i1:i0+1]

# Leaderboard
leaders = compute_leaders(df)
df_now = df.merge(leaders, on="split", how="left").dropna(subset=["net_td","leader_td"])
st.subheader("Leaderboard")

if df_now.empty:
    st.info("No data available yet.")
    selected_for_plot = []
else:
    latest = (df_now.sort_values(["name","net_td"])
                    .groupby("name", as_index=False)
                    .tail(1)
                    .reset_index(drop=True))
    latest["gap_min"] = (latest["net_td"] - latest["leader_td"]).dt.total_seconds()/60.0
    latest["gap_min"] = latest["gap_min"].clip(lower=0)
    latest = latest.sort_values(["gap_min","net_td"], ascending=[True,True]).reset_index(drop=True)
    latest["Latest split"] = latest["split"].map(friendly_split_label)
    latest["Behind (min)"] = latest["gap_min"].map(lambda x: f"{x:.1f}")
    table = latest.rename(columns={"name":"Athlete"})[["Athlete","Latest split","Behind (min)"]]

    # Force visible vertical scrollbar and tight columns
    st.markdown("""
        <style>
        .lb-wrap { max-height: 300px; overflow-y: scroll; padding-right: 8px; }
        .lb-row { padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); }
        .lb-col-athlete { width: 16ch; display: inline-block; }
        .lb-col-split { width: 14ch; display: inline-block; }
        .lb-col-gap { width: 8ch; display: inline-block; text-align: right; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lb-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="lb-row"><strong class="lb-col-athlete">Athlete</strong><strong class="lb-col-split">Latest split</strong><strong class="lb-col-gap">Behind</strong></div>', unsafe_allow_html=True)
    for _, r in table.iterrows():
        st.markdown(f'<div class="lb-row"><span class="lb-col-athlete">{r["Athlete"]}</span><span class="lb-col-split">{r["Latest split"]}</span><span class="lb-col-gap">{r["Behind (min)"]}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Defaults: top 10 checked, leader always on
    if "plot_checks" not in st.session_state:
        st.session_state.plot_checks = {}
        top10 = set(table.head(10)["Athlete"].tolist())
        for nm in table["Athlete"]:
            st.session_state.plot_checks[nm] = (nm in top10)
    leader_name = table.iloc[0]["Athlete"]
    st.session_state.plot_checks[leader_name] = True

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Select top 10"):
            top10 = set(table.head(10)["Athlete"].tolist())
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

# Plot
range_splits = split_range(splits_ordered, from_split, to_split)
sel = df[(df["name"].isin(selected_for_plot)) & (df["split"].astype(str).isin(range_splits))].copy()
if sel.empty:
    st.stop()

xy = sel.merge(leaders, on="split", how="left").dropna(subset=["net_td","leader_td"])
xy["leader_hr"] = xy["leader_td"].dt.total_seconds()/3600.0

# Plot positive minutes and reverse the y-axis so 0 is at the top
xy["gap_min_pos"] = ((xy["net_td"] - xy["leader_td"]).dt.total_seconds()/60.0).clip(lower=0)

fig = go.Figure()
for nm, g in xy.groupby("name"):
    g = g.sort_values("leader_hr")
    fig.add_trace(go.Scatter(
        x=g["leader_hr"], y=g["gap_min_pos"], mode="lines",
        line=dict(width=1.8), name=nm, showlegend=False,
        hovertemplate="Athlete: %{text}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{y:.1f} min",
        text=[nm]*len(g),
    ))

# Labels at last point
endpoints = (xy.sort_values(["name","leader_hr"]).groupby("name", as_index=False).tail(1))
labels = []
for _, row in endpoints.iterrows():
    labels.append(dict(
        x=float(row["leader_hr"]), y=float(row["gap_min_pos"]),
        xref="x", yref="y",
        text=str(row["name"]), showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=11, color="rgba(0,0,0,0.9)"),
        bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1, opacity=0.95
    ))

x_left = math.floor(float(xy["leader_hr"].min())/0.5)*0.5
x_right = float(xy.groupby("split", as_index=False)["leader_hr"].max()["leader_hr"].max()) + 0.25
x_ticks = [round(x_left + 0.5*i, 6) for i in range(int((x_right - x_left)/0.5)+1)]

fig.update_xaxes(title="Leader elapsed (h)", tickmode="array", tickvals=x_ticks, ticktext=[fmt_hmm(v) for v in x_ticks], showgrid=True, zeroline=False, showline=True, mirror=True, ticks="outside")
fig.update_yaxes(title="Time behind leader (min)", autorange="reversed", showgrid=True, zeroline=True, zerolinecolor="rgba(0,0,0,0.25)", showline=True, mirror=True, ticks="outside")

fig.update_layout(height=520, margin=dict(l=50, r=30, t=30, b=40), annotations=labels)
st.plotly_chart(fig, use_container_width=True)
