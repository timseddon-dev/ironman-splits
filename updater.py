# 1) Imports, constants, and configuration
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

# 2) Fetch one split point and return a DataFrame
def fetch_point(point: str) -> pd.DataFrame:
    url = f"{BASE}/events/{EVENT}/categories/{CATEGORY}/splits/{point}"
    r = requests.post(url, headers=HEADERS, data=FORM_BASE, timeout=30)
    r.raise_for_status()
    js = r.json()
    items = js.get("list") or js.get("items") or []
    rows = []
    for it in items:
        rows.append({
            "name": str(it.get("name") or "").strip(),
            "split": (it.get("point") or point),
            "netTime": it.get("netTime") or it.get("time"),
        })
    return pd.DataFrame(rows)

# 3) Fetch all points once, write long.csv, and return row count
def run_once() -> int:
    frames = []
    for p in POINTS:
        try:
            dfp = fetch_point(p)
            if not dfp.empty:
                frames.append(dfp)
        except requests.RequestException:
            # Ignore network/API blips for robustness
            continue
        time.sleep(0.15)
    if not frames:
        return 0
    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df[["name", "split", "netTime"]].copy()
    long_df["name"] = long_df["name"].astype(str).str.strip()
    long_df.to_csv(OUT, index=False, encoding="utf-8")
    return len(long_df)

# 4) One-shot run
if __name__ == "__main__":
    n = run_once()
    print(f"Wrote {n} rows to {OUT}")
