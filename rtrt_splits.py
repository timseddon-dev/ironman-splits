# /// script
# dependencies = ["requests", "pandas", "time", "openpyxl"]
# ///
import time
import requests
import pandas as pd

BASE = "https://track.rtrt.me"
EVENT = "IRM-WORLDCHAMPIONSHIP-MEN-2025"  # live event id
CATEGORY = "MPRO"

# Full ordered split set we poll. Extend if the tracker adds more RUN markers.
POINTS = (
    ["START", "SWIM", "T1"] +
    [f"BIKE{i}" for i in range(1, 21)] +  # we will prune to available
    ["BIKE", "T2"] +
    [f"RUN{i}" for i in range(1, 30)] +   # include predicted markers; server returns those that exist
    ["RUN", "FINISH"]
)

HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://track.rtrt.me",
    "Referer": "https://track.rtrt.me/",
}

def fetch_point(point: str) -> pd.DataFrame:
    url = f"{BASE}/e/{EVENT}/categories/{CATEGORY}/splits/{point}"
    r = requests.post(url, headers=HEADERS, data={"categories": "4,8,16,32"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = js.get("rows") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize expected columns
    if "name" not in df.columns:
        # some payloads use 'athlete'
        if "athlete" in df.columns:
            df["name"] = df["athlete"]
    df["split"] = point
    # prefer 'netTime' if available; otherwise compose from parts if provided
    if "netTime" not in df.columns:
        if {"hh", "mm", "ss"} <= set(df.columns):
            df["netTime"] = df["hh"].astype(str) + ":" + df["mm"].astype(str).str.zfill(2) + ":" + df["ss"].astype(str).str.zfill(2)
        else:
            df["netTime"] = None
    return df[["name", "split", "netTime"]]

def write_long_csv(out_path="long.csv"):
    frames = []
    for p in POINTS:
        try:
            dfp = fetch_point(p)
            if not dfp.empty:
                frames.append(dfp)
        except Exception:
            continue
        time.sleep(0.15)  # be polite
    if not frames:
        return 0
    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(out_path, index=False, encoding="utf-8")
    return len(long_df)

def main():
    print("Live updater started. Writing long.csv every 3 minutes.")
    while True:
        try:
            n = write_long_csv("long.csv")
            print(f"[update] wrote {n} rows")
        except Exception as e:
            print(f"[update] error: {e}")
        time.sleep(180)  # 3 minutes

if __name__ == "__main__":
    main()
