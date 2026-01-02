import json
import os
import re
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests
from dateutil import parser as dateparser

GLOBAL_URL = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/global-visa-wait-times.html"
POSTS_URL  = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/visa-issuing-posts.html"

DOCS_DIR = "docs"
RAW_OUT  = os.path.join(DOCS_DIR, "us_posts_raw.json")
MAP_OUT  = os.path.join(DOCS_DIR, "post_map.json")
FINAL_OUT= os.path.join(DOCS_DIR, "us_posts.json")

CONTROLLED = os.getenv("CONTROLLED", "0").strip() == "1"

def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def norm_key(s: str) -> str:
    # normalize for matching: lower, trim, collapse spaces, strip punctuation
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-']", "", s)
    return s

def months_text_to_days(val: str):
    if val is None:
        return None
    s = str(val).strip()
    if s.upper() in ("NA", "N/A", ""):
        return None

    if re.search(r"<\s*0\.5\s*Month", s, re.I):
        return 14
    if re.search(r"\b0\.5\s*Month\b", s, re.I):
        return 15

    m = re.search(r"(\d+(\.\d+)?)\s*Month", s, re.I)
    if m:
        months = float(m.group(1))
        return int(round(months * 30))

    return None

def find_page_date(html: str):
    m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", html)
    if not m:
        return None
    try:
        return dateparser.parse(m.group(1)).date().isoformat()
    except Exception:
        return None

def fetch(url: str) -> str:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.text

def read_first_table(html: str):
    # Wrap HTML to avoid pandas "literal html" warning
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("No tables found.")
    return tables[0].copy()

def build_post_to_country_map():
    """
    Scrape visa-issuing-posts.html to get mapping of 'Post' -> 'Country' (and maybe more).
    The exact columns can vary, so we try to find plausible ones.
    Returns dict keyed by normalized post name: {post_key: {"post":..., "country":..., "country_code": ""}}
    (We will fill country_code via a small internal mapping pass where possible; otherwise blank.)
    """
    html = fetch(POSTS_URL)
    df = read_first_table(html)
    df.columns = [str(c).strip() for c in df.columns]

    # Heuristic: try to find "Post" and "Country" columns
    post_col = None
    country_col = None

    for c in df.columns:
        cl = c.lower()
        if post_col is None and "post" in cl:
            post_col = c
        if country_col is None and "country" in cl:
            country_col = c

    if post_col is None:
        # fall back to first column
        post_col = df.columns[0]
    if country_col is None:
        # if there's no country column, we can't map automatically
        raise RuntimeError("Visa issuing posts table did not contain a Country column.")

    out = {}
    for _, row in df.iterrows():
        post = str(row.get(post_col, "")).strip()
        country = str(row.get(country_col, "")).strip()

        if not post or post.lower() == "nan":
            continue
        if not country or country.lower() == "nan":
            country = ""

        k = norm_key(post)
        out[k] = {
            "post": post,
            "country": country,
            "country_code": ""  # filled later (optional)
        }

    return out

def load_existing_post_map():
    if not os.path.exists(MAP_OUT):
        return None
    with open(MAP_OUT, "r", encoding="utf-8") as f:
        return json.load(f)

def save_post_map(data):
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_json(path, data):
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def is_complete_mapping(m):
    return bool((m.get("country_code") or "").strip() and (m.get("country") or "").strip())

def main():
    checked_utc = now_utc_iso()

    # 1) SCRAPE GLOBAL WAIT TIMES
    global_html = fetch(GLOBAL_URL)
    as_of_date = find_page_date(global_html)

    df = read_first_table(global_html)
    df.columns = [str(c).strip() for c in df.columns]

    if "Post" not in df.columns:
        df.rename(columns={df.columns[0]: "Post"}, inplace=True)

    visa_cols = [c for c in df.columns if c != "Post"]

    records = []
    all_posts = []

    for _, row in df.iterrows():
        post = str(row.get("Post", "")).strip()
        if not post or post.lower() == "nan":
            continue
        all_posts.append(post)

        for col in visa_cols:
            raw = row.get(col)
            raw_txt = "" if pd.isna(raw) else str(raw).strip()
            records.append({
                "post": post,
                "visa_category": col.strip(),
                "wait_display": raw_txt,
                "wait_days_est": months_text_to_days(raw_txt),
                "source_url": GLOBAL_URL,
                "source_as_of_date": as_of_date,
                "last_checked_utc": checked_utc,
            })

    raw_out = {
        "dataset": "vwh_us_global_wait_times_raw",
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,
        "source_as_of_date": as_of_date,
        "last_checked_utc": checked_utc,
        "records": records,
    }
    save_json(RAW_OUT, raw_out)
    print(f"[OK] Wrote {RAW_OUT} with {len(records)} records and {len(set(all_posts))} unique posts")

    # 2) BUILD/UPDATE POST MAP (CONTROLLED STORAGE)
    # Build an auto map from Visa Issuing Posts list
    try:
        auto_map = build_post_to_country_map()
        print(f"[OK] Loaded visa-issuing-posts map: {len(auto_map)} rows")
    except Exception as e:
        auto_map = {}
        print(f"[WARNING] Could not build auto post->country map from visa-issuing-posts: {e}")

    existing = load_existing_post_map()
    if not existing:
        post_map = {
            "version": 1,
            "updated_utc": checked_utc,
            "posts": {}
        }
    else:
        post_map = existing
        post_map.setdefault("posts", {})
        post_map["updated_utc"] = checked_utc

    # Ensure every global post exists in post_map (COMPLETE LIST)
    for p in sorted(set(all_posts)):
        if p not in post_map["posts"]:
            post_map["posts"][p] = {
                "country_code": "",
                "country": "",
                "region": "",
                "state_province": "",
                "city": p,
                "city_slug": slugify(p),
                "embassy_name": "",
                "source_hint": ""
            }

        # Auto-fill country if we can match by normalized key
        k = norm_key(p)
        if k in auto_map:
            # only fill blanks (controlled)
            if not post_map["posts"][p].get("country"):
                post_map["posts"][p]["country"] = auto_map[k].get("country", "").strip()

    save_post_map(post_map)
    print(f"[OK] Wrote/updated {MAP_OUT} with {len(post_map['posts'])} posts (complete list)")

    # 3) BUILD FINAL MAPPED DATASET
    missing = []
    mapped = 0

    # Fold records by post
    by_post = {}
    for rec in records:
        by_post.setdefault(rec["post"], []).append(rec)

    final_posts = []
    for p, recs in by_post.items():
        m = post_map["posts"].get(p, {})
        if is_complete_mapping(m):
            mapped += 1
        else:
            missing.append(p)

        final_posts.append({
            "id": f"{(m.get('country_code') or 'xx').lower()}-{slugify(m.get('city') or p)}",
            "country_code": (m.get("country_code") or "").upper(),
            "country": m.get("country") or "",
            "region": m.get("region") or "",
            "state_province": m.get("state_province") or "",
            "city": m.get("city") or p,
            "city_slug": m.get("city_slug") or slugify(m.get("city") or p),
            "embassy_name": m.get("embassy_name") or "",
            "source_url": GLOBAL_URL,
            "source_as_of_date": as_of_date,
            "last_checked_utc": checked_utc,
            "waits": recs
        })

    if missing:
        print(f"[WARNING] Missing mapping fields for {len(missing)} posts.")
        print("First 50 missing:")
        for p in missing[:50]:
            mm = post_map["posts"].get(p, {})
            print(f"  - {p} | country='{mm.get('country','')}' country_code='{mm.get('country_code','')}'")
        if len(missing) > 50:
            print(f"  ...and {len(missing)-50} more")

    print(f"[INFO] Mapping status: mapped_complete={mapped} missing={len(missing)} controlled={CONTROLLED}")

    final_out = {
        "dataset": "vwh_us_global_wait_times_phase1_mapped",
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,
        "source_as_of_date": as_of_date,
        "last_checked_utc": checked_utc,
        "posts": final_posts
    }

    # In controlled mode, refuse to publish FINAL if any mappings are missing
    if CONTROLLED and missing:
        print("[FAIL] CONTROLLED mode enabled and mappings are incomplete. Not publishing us_posts.json.")
        raise SystemExit(f"CONTROLLED mode: {len(missing)} posts missing mapping fields (country/country_code).")

    save_json(FINAL_OUT, final_out)
    print(f"[OK] Wrote {FINAL_OUT} with {len(final_posts)} posts")

if __name__ == "__main__":
    main()
