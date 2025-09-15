# rebuild

import streamlit as st
st.write("Boot: app.py loaded")

try:
    import plotly
    st.write("Plotly version:", getattr(plotly, "__version__", "unknown"))
    import plotly.graph_objects as go
    import plotly.express as px  # only if you also use px; otherwise omit
    st.write("Plotly imports OK")
except Exception as e:
    st.error("Plotly failed to import. Check requirements.txt and redeploy.")
    st.exception(e)

# app.py — bs4-free version (regex HTML parsing)
import re
import math
import time
from html import unescape
import pandas as pd
import requests
# keep `st` already imported above; you can remove this second import if you like

# =========================
# 0) Streamlit setup and cache reset
# =========================
st.set_page_config(page_title="Live Gaps vs Leader", layout="wide", initial_sidebar_state="collapsed")
try:
    st.cache_data.clear()
except Exception:
    pass

DATA_FILE = "long.csv"

# =========================
# 1) Constants and headers
# =========================
BASE = "https://track.rtrt.me"
EVENT = "IRM-WORLDCHAMPIONSHIP-MEN-2025"     # event path
CATEGORY_UI = "pro-men-ironman"              # UI route segment used in URLs (hash/SEO)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

HEADERS_HTML = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": f"{BASE}/e/{EVENT}#",
    "Connection": "keep-alive",
}

# =========================
# 2) HTML helpers (regex-based, no bs4)
# =========================
def _extract_table(html: str) -> str | None:
    m = re.search(r"<table\b[^>]*>.*?</table>", html, flags=re.I | re.S)
    return m.group(0) if m else None

def _extract_headers(table_html: str) -> list[str]:
    headers = re.findall(r"<th\b[^>]*>(.*?)</th>", table_html, flags=re.I | re.S)
    headers = [unescape(re.sub(r"<[^>]+>", " ", h)).strip() for h in headers]
    return headers

def _extract_rows(table_html: str) -> list[list[str]]:
    rows = []
    for tr_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.I | re.S):
        tds = re.findall(r"<td\b[^>]*>(.*?)</td>", tr_html, flags=re.I | re.S)
        cells = [unescape(re.sub(r"<[^>]+>", " ", td)).strip() for td in tds]
        if cells:
            rows.append(cells)
    return rows

def _find_name_time_indices(headers_lower: list[str]) -> tuple[int, int]:
    # Heuristics to find Athlete/Name and Time columns
    name_idx = 0
    time_idx = len(headers_lower) - 1 if headers_lower else 0
    for i, h in enumerate(headers_lower):
        if ("athlete" in h) or (h == "name") or ("bib" in h and "name" in h):
            name_idx = i
            break
    for i, h in enumerate(headers_lower):
        if (("net" in h and "time" in h) or h == "time" or "split time" in h):
            time_idx = i
            break
    return name_idx, time_idx

def _normalize_rows_from_table(headers: list[str], rows: list[list[str]], split: str) -> list[dict]:
    headers_lower = [h.lower() for h in headers]
    name_idx, time_idx = _find_name_time_indices(headers_lower)
    out = []
    for tds in rows:
        if not tds:
            continue
        name = tds[name_idx] if name_idx < len(tds) else ""
        ntime = tds[time_idx] if time_idx < len(tds) else ""
        if not name or name.lower() in ("athlete", "name"):
            continue
        out.append({"name": name.strip(), "split": split, "netTime": ntime.strip()})
    return out

def _extract_jsonish_rows(html: str) -> list[dict]:
    # Look for inline arrays like rows: [{name:'...'}]
    m = re.search(r'rows\s*:\s*($$[\s\S]*?$$)', html)
    if not m:
        return []
    arr = m.group(1)
    # Tolerant conversion to JSON: quote bare keys; single->double quotes
    arr2 = re.sub(r"(\w+)\s*:", r'"\1":', arr)
    arr2 = arr2.replace("'", '"')
    # Remove dangling commas before ] or }
    arr2 = re.sub(r",\s*([}\]])", r"\1", arr2)
    try:
        import json
        data = json.loads(arr2)
    except Exception:
        return []
    out = []
    for row in data if isinstance(data, list) else []:
        name = (row.get("name") or row.get("athlete") or "").strip()
        net = (row.get("netTime") or row.get("time") or "").strip()
        if name:
            out.append({"name": name, "netTime": net})
    return out

# =========================
# 3) Split discovery and scraping (HTML-only, no JSON APIs)
# =========================
def _discover_splits_html(session: requests.Session) -> list[str]:
    url = f"{BASE}/e/{EVENT}"
    r = session.get(url, headers=HEADERS_HTML, timeout=25)
    r.raise_for_status()
    html = r.text

    tokens = set(re.findall(r'\b(RUN\d+|BIKE\d+|SWIM|START|FINISH|T1|T2|RUN|BIKE)\b', html, flags=re.I))
    tokens = {t.upper() for t in tokens}

    # Ensure wide run range to capture RUN37 etc.
    for i in range(1, 61):
        tokens.add(f"RUN{i}")

    ordered = []
    for k in ["START", "SWIM", "T1"]:
        if k in tokens:
            ordered.append(k)
    bikes = sorted([t for t in tokens if t.startswith("BIKE") and t[4:].isdigit()], key=lambda x: int(x[4:]))
    ordered += bikes
    if "BIKE" in tokens:
        ordered.append("BIKE")
    if "T2" in tokens:
        ordered.append("T2")
    runs = sorted([t for t in tokens if t.startswith("RUN") and len(t) > 3 and t[3:].isdigit()], key=lambda x: int(x[3:]))
    ordered += runs
    if "RUN" in tokens:
        ordered.append("RUN")
    if "FINISH" in tokens:
        ordered.append("FINISH")

    # Deduplicate preserving order
    seen, final = set(), []
    for x in ordered:
        if x not in seen:
            seen.add(x)
            final.append(x)

    if not final:
        final = ["START", "SWIM", "T1"] + [f"BIKE{i}" for i in range(1, 21)] + ["BIKE", "T2"] + [f"RUN{i}" for i in range(1, 60)] + ["RUN", "FINISH"]
    return final[:80]

def _split_url_ui(split: str) -> str:
    return f"{BASE}/e/{EVENT}#/leaderboard/{CATEGORY_UI}/{split}"

def _split_url_seo(split: str) -> str:
    return f"{BASE}/e/{EVENT}/leaderboard/{CATEGORY_UI}/{split}"

def _fetch_split_rows_html(session: requests.Session, split: str) -> list[dict]:
    urls = [
        _split_url_ui(split),
        _split_url_seo(split),
    ]
    for url in urls:
        try:
            r = session.get(url, headers=HEADERS_HTML, timeout=25)
            if r.status_code != 200 or not r.text:
                continue
            html = r.text

            # Parse table if present
            table = _extract_table(html)
            if table:
                headers = _extract_headers(table)
                rows = _extract_rows(table)
                out = _normalize_rows_from_table(headers, rows, split)
                if out:
                    return out

            # Fallback: parse JSON-ish blob
            blob = _extract_jsonish_rows(html)
            if blob:
                return [{"name": rr["name"], "split": split, "netTime": rr.get("netTime", "")} for rr in blob]
        except Exception:
            continue
    return []

@st.cache_data(ttl=180, show_spinner=True)
def refresh_long_csv() -> dict:
    """
    HTML-only pipeline: discover split IDs from the event page, scrape each split,
    and build long.csv. No BeautifulSoup; regex-based parsing.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    discovered = _discover_splits_html(session)

    frames, fetched, empties = [], 0, 0
    first_empty = None

    for sp in discovered:
        try:
            rows = _fetch_split_rows_html(session, sp)
            if rows:
                frames.append(pd.DataFrame(rows))
                fetched += len(rows)
            else:
                empties += 1
                if first_empty is None:
                    first_empty = sp
        except Exception:
            empties += 1
            if first_empty is None:
                first_empty = sp
        time.sleep(0.12)

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
        "empty_pages": empties,
        "first_empty_split": first_empty,
        "wrote_csv": wrote,
        "discovered_splits": discovered[:80],
        "sample_ui_split": _split_url_ui("RUN37"),
        "note": "bs4-free HTML scraper path. If fetched_rows stays 0, a headless browser is likely required.",
    }

# Kick off refresh
fetch_info = refresh_long_csv()

# =========================
# 4) Distances, ordering, and friendly labels
# =========================
KM_MAP = {
    "SWIM": 3.8,
    "T1": 3.8,
    "BIKE1": 5.0, "BIKE2": 10.0, "BIKE3": 15.0, "BIKE4": 20.0, "BIKE5": 25.0,
    "BIKE6": 30.0, "BIKE7": 35.0, "BIKE8": 40.0, "BIKE9": 45.0, "BIKE10": 50.0,
    "BIKE11": 60.0, "BIKE12": 70.0, "BIKE13": 80.0, "BIKE14": 90.0, "BIKE15": 100.0,
    "BIKE16": 110.0, "BIKE17": 120.0, "BIKE18": 130.0, "BIKE19": 140.0, "BIKE20": 150.0,
    "BIKE": 180.2, "T2": 180.2,
    # Run km approximations (tweak as needed)
    **{f"RUN{i}": i for i in range(1, 60)},
    "RUN": 42.2,
}

def friendly_label(s: str) -> str:
    s = str(s).upper()
    if s == "START": return "Start"
    if s == "FINISH": return "Finish"
    if s in ("T1", "T2"): return s
    if s == "SWIM": return f"Swim {KM_MAP.get('SWIM', 3.8):.1f} km"
    if s.startswith("BIKE"):
        km = KM_MAP.get(s)
        return f"Bike {km:.1f} km" if km else "Bike"
    if s.startswith("RUN"):
        km = KM_MAP.get(s)
        return f"Run {km:.1f} km" if km else "Run"
    return s.title()

# =========================
# 5) Data loading and helpers
# =========================
@st.cache_data(ttl=30, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["name", "split", "netTime"])

    if "name" not in df.columns or "split" not in df.columns:
        return pd.DataFrame(columns=["name", "split", "netTime"])

    df["name"] = df["name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.upper()

    # Parse netTime into timedelta (accept H:MM:SS, MM:SS, etc.)
    def parse_td(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        if not s:
            return pd.NaT
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, sec = parts
                return pd.to_timedelta(int(h)) + pd.to_timedelta(int(m), unit="m") + pd.to_timedelta(float(sec), unit="s")
            if len(parts) == 2:
                m, sec = parts
                return pd.to_timedelta(int(m), unit="m") + pd.to_timedelta(float(sec), unit="s")
            # Fallback: seconds
            return pd.to_timedelta(float(s), unit="s")
        except Exception:
            return pd.NaT

    df["net_td"] = df.get("netTime", pd.Series([None]*len(df))).apply(parse_td)

    # Add a zero START row per athlete to anchor lines
    if not df.empty:
        starts = df[["name"]].drop_duplicates().assign(split="START", net_td=pd.to_timedelta(0, unit="s"))
        df = pd.concat([starts, df[["name", "split", "net_td"]]], ignore_index=True)

    # Split order based on discovery
    splits_present = (
        df["split"].dropna().unique().tolist()
        if df is not None and len(df)
        else []
    )
    return df

def compute_leader(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["split", "leader_td"])
    return (
        df.dropna(subset=["net_td"])
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
# 6) UI — diagnostics, leaderboard, and plot
# =========================
st.caption("Live source diagnostics")
with st.expander("Show fetch details"):
    st.json(fetch_info)
    st.markdown(f"[Open sample UI split (RUN37)]({fetch_info['sample_ui_split']})")

df = load_data(DATA_FILE)

st.subheader("Leaderboard")
if df.empty or df["net_td"].dropna().empty:
    st.info("Waiting for live data...")
else:
    leaders = compute_leader(df)
    lf = df.merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])
    latest = (lf.sort_values(["name", "net_td"])
                .groupby("name", as_index=False)
                .last())
    latest["gap_min"] = (latest["net_td"] - latest["leader_td"]).dt.total_seconds() / 60.0
    latest["gap_min"] = latest["gap_min"].clip(lower=0)
    latest = latest.sort_values(["gap_min", "net_td"]).reset_index(drop=True)
    latest["Latest split"] = latest["split"].map(friendly_label)
    latest["Behind (min)"] = latest["gap_min"].map(lambda x: f"{x:.1f}")

    # Scrollable leaderboard
    st.markdown("""
        <style>
        .lb-wrap { max-height: 360px; overflow-y: auto; padding-right: 8px; }
        .lb-row { display: grid; grid-template-columns: 1.8fr 1.2fr 0.7fr 0.6fr; gap: 12px;
                  padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); align-items: center; }
        .lb-head { position: sticky; top: 0; background: white; z-index: 5;
                   border-bottom: 1px solid rgba(0,0,0,0.2); padding: 6px 0; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lb-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="lb-row lb-head"><strong>Athlete</strong><strong>Latest split</strong><strong>Behind</strong><strong>Plot</strong></div>', unsafe_allow_html=True)

    if "plot_checks" not in st.session_state:
        st.session_state.plot_checks = {}
    top = set(latest.head(10)["name"])
    leader_name = latest.iloc[0]["name"] if len(latest) else None
    for nm in latest["name"]:
        st.session_state.plot_checks.setdefault(nm, nm in top or nm == leader_name)

    for _, r in latest.iterrows():
        cols = st.columns([1.6, 1.2, 0.6, 0.5], gap="small")
        cols[0].markdown(f"**{r['name']}**")
        cols[1].markdown(r["Latest split"])
        cols[2].markdown(r["Behind (min)"])
        disabled = (r["name"] == leader_name)
        with cols[3]:
            st.session_state.plot_checks[r["name"]] = st.checkbox("", value=st.session_state.plot_checks[r["name"]], key=f"plot_{r['name']}", disabled=disabled)

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

    selected = [k for k, v in st.session_state.plot_checks.items() if v]

    # Plot
    if selected:
        leaders = compute_leader(df)
        xy = df[df["name"].isin(selected)].merge(leaders, on="split", how="left").dropna(subset=["net_td", "leader_td"])
        xy["leader_hr"] = xy["leader_td"].dt.total_seconds() / 3600.0
        xy["gap_min"] = (xy["net_td"] - xy["leader_td"]).dt.total_seconds() / 60.0
        xy["gap_min_pos"] = xy["gap_min"].clip(lower=0)
        xy["split_label"] = xy["split"].map(friendly_label)

        fig = go.Figure()
        for nm, g in xy.sort_values(["leader_hr"]).groupby("name"):
            fig.add_trace(go.Scatter(
                x=g["leader_hr"], y=g["gap_min_pos"],
                mode="lines", line=dict(width=1.8),
                name=nm, showlegend=False,
                hovertemplate="Athlete: %{text}<br>Split: %{meta}<br>Leader elapsed: %{x:.2f} h<br>Behind: %{y:.1f} min",
                text=[nm]*len(g), meta=g["split_label"],
            ))
        # Label at last point per athlete
        ends = (xy.sort_values(["leader_hr"]).groupby("name").tail(1))
        for _, r in ends.iterrows():
            fig.add_annotation(
                x=r["leader_hr"], y=r["gap_min_pos"],
                text=str(r["name"]),
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=11, color="rgba(0,0,0,0.9)"),
                bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
            )
        # Axes formatting with y reversed (0 at top)
        if len(xy):
            x_left = math.floor(xy["leader_hr"].min() / 0.5) * 0.5
            x_right = math.ceil(xy["leader_hr"].max() / 0.5) * 0.5
            x_ticks = [t/2 for t in range(int(x_left*2), int(x_right*2)+1)]
        else:
            x_ticks = []

        fig.update_layout(
            height=520,
            margin=dict(l=60, r=20, t=30, b=50),
            showlegend=False,
        )
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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select athletes above to plot gap lines.")

# Footer info
st.caption("Tip: If data remains empty, the site likely renders via client-side JavaScript only. Ask for the Playwright version to execute the page JS and extract the DOM reliably.")
