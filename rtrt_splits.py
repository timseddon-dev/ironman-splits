# /// script
# dependencies = ["requests", "pandas", "openpyxl"]
# ///
import requests
import pandas as pd
from pathlib import Path
import time

# Event configuration
BASE = "https://api.rtrt.me"
EVENT = "IRM-WORLDCHAMPIONSHIP-MEN-2024"
CATEGORY = "pro-men-ironman"

# Desired splits
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

# Replace with fresh values if they ever change
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
            "split": it.get("point") or point,
            "netTime": it.get("netTime") or it.get("time"),
        })
    return pd.DataFrame(rows)

def main():
    frames = []
    for p in POINTS:
        try:
            dfp = fetch_point(p)
            if not dfp.empty:
                frames.append(dfp)
                print(f"Fetched {p}: {len(dfp)} rows")
            else:
                print(f"{p}: no rows")
            time.sleep(0.2)
        except requests.HTTPError as e:
            print(f"Failed {p}: {e}")

    if not frames:
        print("No data returned. Check appid/token or split names.")
        return

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df[["name", "split", "netTime"]].copy()
    long_df["name"] = long_df["name"].astype(str).str.strip()

    out = "long.csv"
    long_df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(long_df)} rows to {out}")

if __name__ == "__main__":
    main()
