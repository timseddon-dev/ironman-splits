# app.py
import json
import math
import re
from html.parser import HTMLParser

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Race Gaps vs Leader", layout="wide")
st.title("Race Gaps vs Leader")

RTRT_URL = "https://app.rtrt.me/IRM-WORLDCHAMPIONSHIP-MEN-2024/leaderboard/pro-men-ironman/"

# ======================================
# Utilities
# ======================================
def parse_hms_to_timedelta(s: str) -> pd.Timedelta | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        else:
            sec = int(float(s))
            h, m = 0, 0
        return pd.to_timedelta(hours=h, minutes=m, seconds=sec)
    except Exception:
        return None

def available_splits_in_order(df: pd.DataFrame) -> list:
    tmp = (
        df.dropna(subset=["net_td"])
          .sort_values(["split", "net_td"])
          .groupby("split", as_index=False)
          .agg(first_td=("net_td", "min"))
          .sort_values("first_td")
    )
    return tmp["split"].tolist()

def compute_leaders(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["net_td"])
          .sort_values(["split", "net_td"])
          .groupby("split", as_index=False)
          .agg(leader_td=("net_td", "min"))
    )

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

def split_range(splits, start_key, end_key):
    if start_key not in splits or end_key not in splits:
        return splits
    i0, i1 = splits.index(start_key), splits.index(end_key)
    if i0 <= i1:
        return splits[i0:i1+1]
    else:
        return splits[i1:i0+1]

# ======================================
# Lightweight HTML extraction (no bs4)
# ======================================
class SimpleTableParser(HTMLParser):
    # Collect text content cell-by-cell in order; we’ll rebuild logical rows later
    def __init__(self):
        super().__init__()
        self.in_tr = False
        self.in_td = False
        self.curr_row = []
        self.rows = []
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_tr = True
            self.curr_row = []
        elif tag in ("td", "th"):
            self.in_td = True
            self._buf = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            if self.in_td:
                text = "".join(self._buf).strip()
                self.curr_row.append(re.sub(r"\s+", " ", text))
                self._buf = []
                self.in_td = False
        elif tag == "tr":
            if self.in_tr:
                # Only keep non-empty rows with some useful content
                if any(cell for cell in self.curr_row):
                    self.rows.append(self.curr_row)
                self.curr_row = []
                self.in_tr = False

    def handle_data(self, data):
        if self.in_td:
            self._buf.append(data)

def extract_embedded_json_blobs(html: str) -> list[str]:
    # Capture candidate JSON-like blobs inside <script> tags
    # Heuristic: largest {...} blocks that contain time-ish keys
    blobs = []
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE):
        script = m.group(1) or ""
        if any(k in script for k in ["leaderboard", "athlete", "split", "elapsed", "time"]):
            # try to find JSON object(s)
            for obj in re.finditer(r"\{[\s\S]*?\}", script):
                text = obj.group(0)
                if any(k in text for k in ["name", "split", "elapsed", "time"]):
                    blobs.append(text)
    # Sort by length descending to try richer blobs first
    blobs.sort(key=len, reverse=True)
    return blobs

def parse_embedded_json_to_df(html: str) -> pd.DataFrame | None:
    blobs = extract_embedded_json_blobs(html)
    if not blobs:
        return None

    rows = []
    def walk(node, name_ctx=None):
        if isinstance(node, dict):
            if "name" in node and isinstance(node["name"], str):
                name_ctx = node["name"]
            # elapsed in various fields
            for key in ("elapsed", "time", "net", "total"):
                if key in node and isinstance(node[key], (str, int, float)):
                    td = parse_hms_to_timedelta(str(node[key]))
                    if td is not None:
                        sp = node.get("split") or node.get("label") or None
                        rows.append({"name": name_ctx or "Unknown", "split": sp, "net_td": td})
            for v in node.values():
                walk(v, name_ctx=name_ctx)
        elif isinstance(node, list):
            for v in node:
                walk(v, name_ctx=name_ctx)

    for blob in blobs[:6]:  # try a few biggest blobs
        try_text = re.sub(r",\s*}", "}", blob)
        try_text = re.sub(r",\s*]", "]", try_text)
        try:
            data = json.loads(try_text)
        except Exception:
            continue
        walk(data)

    if not rows:
        return None

    df = pd.DataFrame(rows).dropna(subset=["net_td"])
    # If splits are missing, we’ll assign generic sequence per athlete
    if "split" not in df.columns or df["split"].isna().all():
        out = []
        for nm, g in df.groupby("name"):
            g = g.sort_values("net_td")
            out.append(pd.DataFrame({
                "name": nm,
                "split": ["START"] + [f"S{i}" for i in range(1, len(g) + 1)],
                "net_td": [pd.to_timedelta(0, unit="s")] + g["net_td"].tolist()
            }))
        df2 = pd.concat(out, ignore_index=True)
        order = df2.groupby("split")["net_td"].min().sort_values().index.tolist()
        df2["split"] = pd.Categorical(df2["split"], categories=order, ordered=True)
        return df2.sort_values(["split", "name"]).reset_index(drop=True)

    # Ensure START exists per athlete
    out = []
    for nm, g in df.groupby("name"):
        g = g.sort_values("net_td")
        out.append(pd.DataFrame([{"name": nm, "split": "START", "net_td": pd.to_timedelta(0, unit="s")}]))
        out.append(g[["name", "split", "net_td"]])
    df2 = pd.concat(out, ignore_index=True)
    order = df2.groupby("split")["net_td"].min().sort_values().index.tolist()
    df2["split"] = pd.Categorical(df2["split"], categories=order, ordered=True)
    return df2.sort_values(["split", "name"]).reset_index(drop=True)

def fallback_parse_table_to_df(html: str) -> pd.DataFrame:
    parser = SimpleTableParser()
    parser.feed(html)
    table_rows = [r for r in parser.rows if len(r) >= 3]

    people = []
    time_re = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
    for row in table_rows:
        # guess a name cell (2+ words, not just times)
        name = None
        for cell in row:
            if time_re.search(cell):
                continue
            if len(cell.split()) >= 2:
                name = cell
                break
        if not name:
            continue
        times = [c for c in row if time_re.search(c)]
        if times:
            people.append((name, times))

    if not people:
        raise RuntimeError("Could not parse any athlete rows from table.")

    out = []
    for nm, times in people:
        out.append({"name": nm, "split": "START", "net_td": pd.to_timedelta(0, unit="s")})
        for i, ts in enumerate(times, start=1):
            td = parse_hms_to_timedelta(ts)
            if td is not None:
                out.append({"name": nm, "split": f"S{i}", "net_td": td})

    df = pd.DataFrame(out)
    order = df.groupby("split")["net_td"].min().sort_values().index.tolist()
    df["split"] = pd.Categorical(df["split"], categories=order, ordered=True)
    return df.sort_values(["split", "name"]).reset_index(drop=True)

@st.cache_data(show_spinner=True, ttl=300)
def load_rtrt_df(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BearlyFocus/1.0)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    html = r.text

    # Try embedded JSON first (more structured)
    df = parse_embedded_json_to_df(html)
    if df is not None and not df.empty:
        return df

    # Fallback to table-only parse
    return fallback_parse_table_to_df(html)

# ======================================
# Load data
# ======================================
try:
    df = load_rtrt_df(RTRT_URL)
    st.caption("Data source: RTRT leaderboard (live parsed, no bs4).")
except Exception as e:
    st.error(f"Failed to fetch/parse RTRT data. Details: {e}")
    st.stop()

# ======================================
# UI controls
# ======================================
with st.expander("Test mode", expanded=True):
    test_mode = st.checkbox("Enable test mode (limit dataset to athlete elapsed < Max hours)", value=False)
    max_hours = st.slider("Max hours", 1.0, 12.0, 7.0, 0.5)

all_athletes = sorted(df["name"].dropna().unique().tolist())
splits_ordered = available_splits_in_order(df)
default_athletes = all_athletes[:6] if len(all_athletes) >= 6 else all_athletes

colA, colB = st.columns(2)
with colA:
    selected = st.multiselect("Athletes", options=all_athletes, default=default_athletes)
with colB:
    default_from = 0
    default_to = len(splits_ordered) - 1 if splits_ordered else 0
    from_split = st.selectbox("From split", options=splits_ordered, index=default_from if splits_ordered else 0)
    to_split = st.selectbox("To split", options=splits_ordered, index=default_to if (splits_ordered and default_to is not None and default_to >= 0) else 0)

# Apply per-athlete elapsed filter if test mode is on
if test_mode:
    max_td = pd.to_timedelta(max_hours, unit="h")
    before = len(df)
    df = df.dropna(subset=["net_td"]).copy()
    df = df[df["net_td"] < max_td].copy()
    st.caption(f"Test mode active: {len(df):,} rows (from {before:,}) with athlete elapsed < {max_hours:.1f}h")

if df.empty or len(selected) == 0 or len(splits_ordered) == 0:
    st.info("Please ensure there is data and at least one athlete selected.")
    st.stop()

range_splits = split_range(splits_ordered, from_split, to_split)

# ======================================
# Snapshot table (Top 10)
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
    top10["Behind (min)"] = top10["Behind (min)"].map(lambda x: f"{x:.1f}")
    st.dataframe(top10.head(10).reset_index(drop=True), use_container_width=True, height=320)

# ======================================
# Plot prep
# ======================================
sel = df[df["name"].isin(selected) & df["split"].isin(range_splits)].copy()
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

xy_df = xy_df.sort_values(["name", "leader_hr"], kind="mergesort")
if xy_df.empty:
    st.info("No rows to plot for the current selection.")
    st.stop()

# ======================================
# Plot
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
        meta=g["split"],
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

# X axis bounds
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
