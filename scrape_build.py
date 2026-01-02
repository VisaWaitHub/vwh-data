import json
import re
from datetime import datetime, timezone
from dateutil import parser as dateparser

import pandas as pd
import requests

GLOBAL_URL = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/global-visa-wait-times.html"

# Convert State Dept "Month" style strings into an estimated day-count
def months_text_to_days(val: str):
    if val is None:
        return None
    s = str(val).strip()

    if s.upper() in ("NA", "N/A", ""):
        return None

    # "< 0.5 Month"
    m = re.search(r"<\s*0\.5\s*Month", s, re.I)
    if m:
        return 14  # conservative estimate ~2 weeks

    # "0.5 Month"
    m = re.search(r"\b0\.5\s*Month\b", s, re.I)
    if m:
        return 15

    # "1 Month", "2 Months", "14.5 Months"
    m = re.search(r"(\d+(\.\d+)?)\s*Month", s, re.I)
    if m:
        months = float(m.group(1))
        return int(round(months * 30))

    return None

def find_page_date(html: str):
    # The page shows a date near the top like "Dec 12, 2025 — ..."
    m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", html)
    if not m:
        return None
    try:
        return dateparser.parse(m.group(1)).date().isoformat()
    except Exception:
        return None

def main():
    r = requests.get(GLOBAL_URL, timeout=30)
    r.raise_for_status()
    html = r.text

    as_of_date = find_page_date(html)  # ISO date string or None
    checked_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Pull the first HTML table found on the page
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found on the Global Visa Wait Times page.")

    df = tables[0].copy()

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Expect a "Post" column
    if "Post" not in df.columns:
        # Sometimes it's labeled slightly differently; fall back to first column
        df.rename(columns={df.columns[0]: "Post"}, inplace=True)

    # Build records in a schema that’s easy for your WP frontend to adapt.
    # NOTE: This is a “Phase 1” dataset. It contains:
    # - post (city/post name)
    # - visa_category
    # - wait_days_est (int or null)
    # - wait_display (original text)
    records = []

    # Determine which visa columns exist (varies)
    visa_cols = [c for c in df.columns if c != "Post"]

    for _, row in df.iterrows():
        post = str(row.get("Post", "")).strip()
        if not post or post.lower() == "nan":
            continue

        for col in visa_cols:
            raw = row.get(col)
            raw_txt = "" if pd.isna(raw) else str(raw).strip()

            rec = {
                "post": post,
                "visa_category": col.strip(),
                "wait_display": raw_txt,
                "wait_days_est": months_text_to_days(raw_txt),
                "source_url": GLOBAL_URL,
                "source_as_of_date": as_of_date,
                "last_checked_utc": checked_utc,
            }
            records.append(rec)

    out = {
        "dataset": "vwh_us_global_wait_times_phase1",
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,
        "source_as_of_date": as_of_date,
        "last_checked_utc": checked_utc,
        "records": records,
    }

    # Write to docs/ so GitHub Pages can serve it
    import os
    os.makedirs("docs", exist_ok=True)
    with open("docs/us_posts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote docs/us_posts.json with {len(records)} records")

if __name__ == "__main__":
    main()
