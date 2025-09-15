# 1) Imports and constants
import asyncio
from datetime import datetime, timezone
import pandas as pd

from playwright.async_api import async_playwright

EVENT_URL = "https://track.rtrt.me/e/IRM-WORLDCHAMPIONSHIP-MEN-2025#/leaderboard/pro-men-ironman"
OUT_CSV = "data.csv"

# 2) Utilities
def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def normalize_rows(rows):
    # Make rectangular and add metadata
    max_len = max((len(r) for r in rows), default=0)
    cols = [f"c{i}" for i in range(max_len)]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "fetched_at_utc", utc_now_str())
    return df

# 3) Core scraping with Playwright
async def fetch_table_rows():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Navigate and allow client-side app to hydrate
        await page.goto(EVENT_URL, wait_until="domcontentloaded")
        # RTRT is a SPA; wait for any table-like element to appear
        candidate_selectors = [
            "table",
            "table.table",
            "[data-testid='leaderboard-table']",
            "table.table-striped",
            "tbody tr"
        ]
        found = False
        for sel in candidate_selectors:
            try:
                await page.wait_for_selector(sel, timeout=20000)
                found = True
                break
            except Exception:
                continue

        # Give a bit more time for rows to render/virtualize
        await page.wait_for_timeout(2000)

        # Collect rows
        rows = []
        tr_nodes = await page.query_selector_all("tbody tr")
        for tr in tr_nodes:
            tds = await tr.query_selector_all("td")
            cells = [(await td.inner_text()).strip() for td in tds]
            if cells and any(c for c in cells):
                rows.append(cells)

        await context.close()
        await browser.close()
        return rows

# 4) Orchestrator
async def main():
    print(f"[{utc_now_str()}] Fetching: {EVENT_URL}")
    rows = await fetch_table_rows()
    print(f"[{utc_now_str()}] Rows scraped: {len(rows)}")

    df = normalize_rows(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[{utc_now_str()}] Wrote {OUT_CSV} with {len(df)} rows, {len(df.columns)} cols")

# 5) Entrypoint
if __name__ == "__main__":
    # Prereqs (run locally once): 
    #   pip install playwright pandas
    #   playwright install chromium
    asyncio.run(main())
