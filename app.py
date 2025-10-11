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

# 1) Split order definition (static for now; will be made dynamic next update)
ORDER = (
    ["START", "SWIM", "T1"] +
    [f"BIKE{i}" for i in range(1, 21)] + ["BIKE", "T2"] +
    [f"RUN{i}" for i in range(1, 28)] + ["RUN", "FINISH"]
)

# 2) Helpers: distance parsing and labeling
DIST_RE = re.compile(r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>km|k|m|mi|mile|miles)", re.IGNORECASE)

def parse_distance_km_from_label(label: str):
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
            dists[str(split)] = float(chosen)
    return dists

def friendly_label(split: str, split_km: dict) -> str:
    s = str(split).upper()
    if s == "START": return "Start"
    if s == "FINISH": return "Finish"
    if s in ("T1", "T2"): return s
    km = split_km.get(s)
    if km is None:
        if s == "SWIM": return "Swim"
        if s.startswith("BIKE"): return "Bike"
        if s.startswith("RUN"): return "Run"
        return s
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
    starts = df[["name"]].drop_duplicates().assign(
        split="START", net_td=pd.to_timedelta(0, unit="s"), label="Start"
    )
    df = pd.concat([starts, df[["name", "split", "net_td", "label"]]], ignore_index=True)

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
# If duplicates at the same split exist, pick the smallest net_td
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

# 5.2.2) Compute place and "gap to in front" deltas vs previous split

# Rank within each split by net time (ascending)
ranked = (
    lf_valid.sort_values(["split_idx", "net_td"])
            .assign(place=lambda d: d.groupby("split_idx")["net_td"].rank(method="first"))
)
ranked["split_idx"] = pd.to_numeric(ranked["split_idx"], errors="coerce").astype("Int64")

# Previous split index per row (START -> <NA>)
ranked["prev_split_idx"] = ranked["split_idx"] - 1

# Current place per athlete at their latest split
current_place = latest[["name", "split_idx"]].merge(
    ranked[["name", "split_idx", "place"]], on=["name", "split_idx"], how="left"
).rename(columns={"place": "place_now"})

# Previous place for the same athlete (only where previous exists)
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
latest = latest.merge(current_place[["name", "place_now"]], on="name", how="left")
latest = latest.merge(prev_place, on=["name", "split_idx"], how="left")

# Places delta (positive = gained)
latest["Places_delta"] = latest["place_prev"] - latest["place_now"]

# Per-split "gap to in front"
def per_split_gap_to_front(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("net_td").reset_index(drop=True)
    d["gap_to_front_min"] = pd.NA
    for i in range(1, len(d)):
        delta = (d.loc[i, "net_td"] - d.loc[i - 1, "net_td"]).total_seconds() / 60.0
        d.loc[i, "gap_to_front_min"] = max(delta, 0.0)
    return d

per_split = ranked.groupby("split_idx", group_keys=False).apply(per_split_gap_to_front)
per_split["split_idx"] = pd.to_numeric(per_split["split_idx"], errors="coerce").astype("Int64")

# Current gap_to_front
cur_gap = latest[["name", "split_idx"]].merge(
    per_split[["name", "split_idx", "gap_to_front_min"]],
    on=["name", "split_idx"], how="left"
).rename(columns={"gap_to_front_min": "gap_front_now"})

# Previous gap_to_front aligned to current split (shift index forward)
prev_gap = per_split.copy()
prev_gap["split_idx"] = prev_gap["split_idx"] + 1
prev_gap = prev_gap.rename(columns={"gap_to_front_min": "gap_front_prev"})

latest = latest.merge(cur_gap, on=["name", "split_idx"], how="left")
latest = latest.merge(prev_gap[["name", "split_idx", "gap_front_prev"]], on=["name", "split_idx"], how="left")

# Delta: positive = getting closer (previous gap - current gap)
def _safe_delta(prev_val, cur_val):
    try:
        if pd.isna(prev_val) or pd.isna(cur_val):
            return pd.NA
        return round(float(prev_val) - float(cur_val), 1)
    except Exception:
        return pd.NA

latest["Gap_to_in_front_delta"] = [
    _safe_delta(p, c) for p, c in zip(latest["gap_front_prev"], latest["gap_front_now"])
]

# 5.2.3) Final sort for display
latest = latest.sort_values(
    ["split_idx", "gap_min", "net_td"],
    ascending=[False, True, True]
).reset_index(drop=True)



# 5.3) Leaderboard display (two-level header with merged group, fixed widths, inline selection)
st.subheader("Leaderboard")

if latest.empty:
    st.info("Waiting for live data...")
    selected = []
else:
    view = latest[["name", "Latest split", "Time behind leader", "Places_delta", "Gap_to_in_front_delta"]].copy()
    view = view.rename(columns={"name": "Athlete"})

    # Persist selections in query params for stability across reruns
    qp = st.experimental_get_query_params()
    selected_qp = set(qp.get("sel", []))  # list of names in query params
    if "plot_checks" not in st.session_state:
        # Initialize from query params; ensure current leader is always selected
        init = {nm: (nm in selected_qp) for nm in view["Athlete"]}
        leader_name = latest.iloc[0]["Athlete"] if "Athlete" in latest.columns else latest.iloc[0]["name"]
        init[leader_name] = True
        st.session_state.plot_checks = init
    else:
        # Ensure leader stays selected
        leader_name = latest.iloc[0]["Athlete"] if "Athlete" in latest.columns else latest.iloc[0]["name"]
        st.session_state.plot_checks[leader_name] = True

    # CSS: fixed column widths, no wrapping, unified table layout
    st.markdown("""
        <style>
        .lb-wrap { border: 1px solid rgba(0,0,0,0.08); border-radius: 6px; }
        .lb-scroll { max-height: 460px; overflow-y: auto; }
        table.lb { border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 14px; }
        table.lb th, table.lb td { padding: 8px; border-bottom: 1px solid rgba(0,0,0,0.06); vertical-align: middle; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        table.lb thead th { position: sticky; top: 0; background: #fff; z-index: 2; border-bottom: 1px solid rgba(0,0,0,0.15); }
        table.lb thead tr.top th { height: 28px; font-weight: 700; text-align: left; }
        table.lb thead tr.bottom th { height: 28px; font-weight: 600; text-align: left; border-bottom: 1px solid rgba(0,0,0,0.08); }
        /* Fixed widths per column to keep all rows aligned */
        .col-athlete  { width: 26%; }
        .col-latest   { width: 24%; }
        .col-gap      { width: 16%; }
        .col-places   { width: 12%; text-align: left; }
        .col-gapfront { width: 14%; text-align: left; }
        .col-plot     { width: 8%;  text-align: center; }
        /* Pills and colors */
        .pill-pos { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#1aa260; color:#fff; font-weight:700; }
        .pill-neg { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#d93025; color:#fff; font-weight:700; }
        .txt-pos  { color:#1aa260; font-weight:700; }
        .txt-neg  { color:#d93025; font-weight:700; }
        /* Inline checkbox styling */
        .sel-box { transform: scale(1.2); cursor: pointer; }
        </style>
    """, unsafe_allow_html=True)

    # Build header and rows as pure HTML so the checkbox is inside the same row
    # We’ll emit a form with checkboxes named sel and submit it via JS on change to keep Streamlit state in sync.
    rows_html = []
    for nm, latest_split, gap_txt, places_delta, gap_delta in view.itertuples(index=False, name=None):
        # Places pill
        if pd.isna(places_delta):
            places_html = ""
        else:
            if places_delta > 0:
                places_html = f'<span class="pill-pos">+{int(places_delta)}</span>'
            elif places_delta < 0:
                places_html = f'<span class="pill-neg">-{int(abs(places_delta))}</span>'
            else:
                places_html = "0"

        # Gap-to-in-front delta
        if pd.isna(gap_delta):
            gapfront_html = ""
        else:
            if gap_delta > 0:
                gapfront_html = f'<span class="txt-pos">+{gap_delta:.1f}</span>'
            elif gap_delta < 0:
                gapfront_html = f'<span class="txt-neg">{gap_delta:.1f}</span>'
            else:
                gapfront_html = "0.0"

        checked = "checked" if st.session_state.plot_checks.get(nm, False) else ""
        row = f"""
            <tr>
              <td class="col-athlete">{nm}</td>
              <td class="col-latest">{latest_split}</td>
              <td class="col-gap">{gap_txt}</td>
              <td class="col-places">{places_html}</td>
              <td class="col-gapfront">{gapfront_html}</td>
              <td class="col-plot">
                <input type="checkbox" class="sel-box" name="sel" value="{nm}" {checked} />
              </td>
            </tr>
        """
        rows_html.append(row)

    # Full table HTML with merged header cells:
    # - Top row: merge col 1,2,3,6 with rowspan=2; group header (colspan=2) over cols 4-5
    table_html = f"""
    <div class="lb-wrap">
      <form id="lb-form" method="get">
        <div class="lb-scroll">
          <table class="lb">
            <thead>
              <tr class="top">
                <th class="col-athlete" rowspan="2">Athlete</th>
                <th class="col-latest"  rowspan="2">Latest split</th>
                <th class="col-gap"     rowspan="2">Time behind leader</th>
                <th colspan="2" style="text-align:center;">Change since last split</th>
                <th class="col-plot"    rowspan="2">Plot</th>
              </tr>
              <tr class="bottom">
                <th class="col-places">Places</th>
                <th class="col-gapfront">Gap to in front</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
      </form>
    </div>
    <script>
      // Auto-submit selection via query params on change
      const form = document.getElementById('lb-form');
      if (form) {{
        form.addEventListener('change', (e) => {{
          const boxes = form.querySelectorAll('input[name="sel"]');
          const vals = [];
          boxes.forEach(b => {{ if (b.checked) vals.push(b.value); }});
          const url = new URL(window.location);
          url.searchParams.delete('sel');
          vals.forEach(v => url.searchParams.append('sel', v));
          window.history.replaceState(null, '', url.toString());
          // Trigger a Streamlit rerun by programmatically clicking the hidden anchor
          const anchor = document.createElement('a');
          anchor.href = url.toString();
          document.body.appendChild(anchor);
          anchor.click();
          document.body.removeChild(anchor);
        }});
      }}
    </script>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    # After render, read query params to produce selected list for plotting and sync session_state
    qp = st.experimental_get_query_params()
    selected_names = qp.get("sel", [])
    # Ensure leader always included
    leader_name = latest.iloc[0]["Athlete"] if "Athlete" in latest.columns else latest.iloc[0]["name"]
    if leader_name not in selected_names:
        selected_names = list(selected_names) + [leader_name]
        st.experimental_set_query_params(sel=selected_names)
    # Update session state
    st.session_state.plot_checks = {nm: (nm in set(selected_names)) for nm in view["Athlete"]}
    selected = [nm for nm, on in st.session_state.plot_checks.items() if on]

# 6) From/To split controls
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
plot_df = df[(df["name"].isin(selected)) & (df["split"].astype(str).isin(range_splits))].copy()

# 8) Plot
if not plot_df.empty:
    leaders2 = compute_leader(df)
    xy = plot_df.merge(leaders2, on="split", how="left").dropna(subset=["net_td", "leader_td"])
    xy["leader_hr"] = xy["leader_td"].dt.total_seconds() / 3600.0
    xy["gap_min_pos"] = ((xy["net_td"] - xy["leader_td"]).dt.total_seconds() / 60.0).clip(lower=0)
    xy["split_label"] = xy["split"].map(lambda s: friendly_label(s, split_km_map))

    fig = go.Figure()

    # 8.1) Main athlete lines
    for nm, g in xy.groupby("name", sort=False):
        g = g.sort_values("leader_hr")
        fig.add_trace(go.Scatter(
            x=g["leader_hr"], y=g["gap_min_pos"],
            mode="lines", line=dict(width=1.8),
            name=nm, showlegend=False,
            hovertemplate="Athlete: %{text}<br>Split: %{meta}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{y:.1f} min",
            text=[nm]*len(g), meta=g["split_label"],
        ))

    # 8.2) End labels
    ends = (xy.sort_values(["name", "leader_hr"]).groupby("name", as_index=False).tail(1))
    annotations = [
        dict(
            x=float(r["leader_hr"]), y=float(r["gap_min_pos"]),
            xref="x", yref="y", text=str(r["name"]), showarrow=False,
            xanchor="left", yanchor="middle", font=dict(size=11, color="rgba(0,0,0,0.9)")
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
        title="Time behind leader (min)",
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
        margin=dict(l=50, r=30, t=30, b=40),
        annotations=annotations
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select athletes to plot.")
