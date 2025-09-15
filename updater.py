# updater.py — no rankOverall/rankGender/rankDivision

import time
import requests
import pandas as pd

BASE = "https://api.rtrt.me"
EVENT = "IRM-WORLDCHAMPIONSHIP-MEN-2025"
CATEGORY = "pro-men-ironman"
OUT = "long.csv"

POINTS = (
    ["SWIM", "T1"]
    + [f"BIKE{i}" for i in range(1, 26)]
    + ["BIKE", "T2"]
    + [f"RUN{i}" for i in range(1, 23)]
    + ["FINISH"]
)

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

def _get_first(d: dict, keys: list, default=""):
    for k in keys:
        if k in d and d[k] not in (None, "", "null"):
            return d[k]
    return default

def fetch_point(point: str) -> pd.DataFrame:
    url = f"{BASE}/events/{EVENT}/categories/{CATEGORY}/splits/{point}"
    r = requests.post(url, headers=HEADERS, data=FORM_BASE, timeout=30)
    r.raise_for_status()
    js = r.json()

    items = js.get("list") or js.get("items") or []
    rows = []
    for it in items:
        rows.append(
            {
                "name": str(_get_first(it, ["name", "athlete", "athletename"], "")).strip(),
                "bib": str(_get_first(it, ["bib", "bibnum", "athleteBib"], "")).strip(),
                "split": _get_first(it, ["point", "split", "location"], point),
                "netTime": _get_first(it, ["netTime", "time", "elapsed"], ""),
                "label": _get_first(it, ["label"], ""),
                "placeChange": str(_get_first(it, ["placeChange", "place_change"], "")).strip(),
                "pace": _get_first(it, ["pace", "runPace", "swimPace", "bikePace"], ""),
                "speed": _get_first(it, ["speed", "avgSpeed"], ""),
                "timestamp": _get_first(it, ["timestamp", "timeStamp", "ts"], ""),
            }
        )
    return pd.DataFrame(rows)

def run_once() -> int:
    frames = []
    for p in POINTS:
        try:
            dfp = fetch_point(p)
            if not dfp.empty:
                frames.append(dfp)
        except requests.RequestException:
            continue
        time.sleep(0.15)

    cols = ["name", "bib", "split", "netTime", "label", "placeChange", "pace", "speed", "timestamp"]

    if not frames:
        pd.DataFrame(columns=cols).to_csv(OUT, index=False, encoding="utf-8")
        return 0

    long_df = pd.concat(frames, ignore_index=True)
    for c in cols:
        if c not in long_df.columns:
            long_df[c] = ""
    long_df = long_df[cols].copy()

    for c in ["name", "bib", "split", "label", "placeChange", "pace", "speed", "timestamp"]:
        long_df[c] = long_df[c].astype(str).str.strip()

    long_df.to_csv(OUT, index=False, encoding="utf-8")
    return len(long_df)

if __name__ == "__main__":
    n = run_once()
    print(f"Wrote {n} rows to {OUT}")
