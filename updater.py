# updater.py
# One-shot data pull from RTRT split API and write long.csv
# Zero-config: adjust EVENT/CATEGORY/POINTS if needed.

import time
import requests
import pandas as pd

# 1) Configuration
BASE = "https://api.rtrt.me"
EVENT = "IRM-WORLDCHAMPIONSHIP-MEN-2025"
CATEGORY = "pro-men-ironman"
OUT = "long.csv"

# Known split points in approximate race order.
POINTS = (
    ["SWIM", "T1"]
    + [f"BIKE{i}" for i in range(1, 26)]
    + ["BIKE", "T2"]
    + [f"RUN{i}" for i in range(1, 23)]
    + ["FINISH"]
)

# Headers and form body commonly used by the web tracker
HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://app.rtrt.me",
    "Referer": "https://app.rtrt.me/",
    "User-Agent": "Mozilla/5.0",
}

FORM_BASE = {
    "timesort": "1",
    "nohide": "1",
    "appid": "5824c5c948fd08c23a8b4567",
    "token": "8DDD292572147587A6E0",
    "max": "500",
    "catloc": "1",
    "cattotal": "1",
    "units": "standard",
    "source": "webtracker",
}

# 2) Helpers
def _get_first(d: dict, keys: list, default=""):
    for k in keys:
        if k in d and d[k] not in (None, "", "null"):
            return d[k]
    return default

# 3) Fetch a single split point and return a DataFrame of rows
def fetch_point(point: str) -> pd.DataFrame:
    url = f"{BASE}/events/{EVENT}/categories/{CATEGORY}/splits/{point}"
    r = requests.post(url, headers=HEADERS, data=FORM_BASE, timeout=30)
    r.raise_for_status()
    js = r.json()

    items = js.get("list") or js.get("items") or []
    rows = []
    for it in items:
        # Robust field extraction with fallbacks
        name = str(_get_first(it, ["name", "athlete", "athletename"], "")).strip()
        bib = str(_get_first(it, ["bib", "bibnum", "athleteBib"], "")).strip()
        split_point = _get_first(it, ["point", "split", "location"], point)
        net_time = _get_first(it, ["netTime", "time", "elapsed"], "")
        label = _get_first(it, ["label"], "")
        place_change = str(_get_first(it, ["placeChange", "place_change"], "")).strip()
        pace = _get_first(it, ["pace", "runPace", "swimPace", "bikePace"], "")
        speed = _get_first(it, ["speed", "avgSpeed"], "")
        rank_overall = _get_first(it, ["overallRank", "rankOverall", "overall_rank"], "")
        rank_gender = _get_first(it, ["genderRank", "rankGender", "gender_rank"], "")
        rank_div = _get_first(it, ["divisionRank", "rankDivision", "division_rank"], "")
        timestamp = _get_first(it, ["timestamp", "timeStamp", "ts"], "")

        rows.append(
            {
                "name": name,
                "bib": bib,
                "split": split_point,
                "netTime": net_time,
                "label": label,
                "placeChange": place_change,
                "pace": pace,
                "speed": speed,
                "rankOverall": rank_overall,
                "rankGender": rank_gender,
                "rankDivision": rank_div,
                "timestamp": timestamp,
            }
        )
    return pd.DataFrame(rows)

# 4) Fetch all points once, write CSV, return row count
def run_once() -> int:
    frames = []
    for p in POINTS:
        try:
            dfp = fetch_point(p)
            if not dfp.empty:
                frames.append(dfp)
        except requests.RequestException:
            # ignore transient network/API errors; continue
            continue
        # be polite to API
        time.sleep(0.15)

    if not frames:
        # Write an empty file with headers for consistency
        empty_cols = [
            "name",
            "bib",
            "split",
            "netTime",
            "label",
            "placeChange",
            "pace",
            "speed",
            "rankOverall",
            "rankGender",
            "rankDivision",
            "timestamp",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(OUT, index=False, encoding="utf-8")
        return 0

    long_df = pd.concat(frames, ignore_index=True)

    # Ensure consistent columns/order even if some splits lacked fields
    cols = [
        "name",
        "bib",
        "split",
        "netTime",
        "label",
        "placeChange",
        "pace",
        "speed",
        "rankOverall",
        "rankGender",
        "rankDivision",
        "timestamp",
    ]
    for c in cols:
        if c not in long_df.columns:
            long_df[c] = ""
    long_df = long_df[cols].copy()

    # Light cleanup
    for c in ["name", "bib", "split", "label", "placeChange", "pace", "speed", "rankOverall", "rankGender", "rankDivision", "timestamp"]:
        long_df[c] = long_df[c].astype(str).str.strip()

    long_df.to_csv(OUT, index=False, encoding="utf-8")
    return len(long_df)

# 5) One-shot main
if __name__ == "__main__":
    n = run_once()
    print(f"Wrote {n} rows to {OUT}")
