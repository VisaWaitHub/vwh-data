import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dateutil import parser as dateparser

# -----------------------------
# SOURCES
# -----------------------------
GLOBAL_URL = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/global-visa-wait-times.html"

# This page contains the canonical "Visa Issuing Posts" country -> posts mapping.
# (State Dept page that lists countries and their visa-issuing posts.)
POSTMAP_URL = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/wait-times.html"


# -----------------------------
# HELPERS
# -----------------------------

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def normalize_key(s: str) -> str:
    """
    Normalize post names aggressively so joins are deterministic.
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = strip_accents(s)
    s = s.lower()

    # Normalize common punctuation and whitespace
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2013\u2014–—]", "-", s)  # dashes
    s = re.sub(r"[^\w\s\-]", " ", s)         # remove punctuation (keep words/dashes)
    s = re.sub(r"\s+", " ", s).strip()

    # Some posts include "u.s. embassy" in other contexts; keep post bare
    s = s.replace("u s", "us")
    return s

def slugify(s: str) -> str:
    if s is None:
        return ""
    s = strip_accents(str(s)).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s

def safe_get_first_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found on page.")
    return tables[0].copy()

def find_page_date(html: str) -> Optional[str]:
    # The page often shows a date near the top like "Dec 12, 2025"
    m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", html)
    if not m:
        return None
    try:
        return dateparser.parse(m.group(1)).date().isoformat()
    except Exception:
        return None

# Convert State Dept "Month" style strings into an estimated day-count
def months_text_to_days(val: str) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()

    if s.upper() in ("NA", "N/A", ""):
        return None

    # "< 0.5 Month"
    if re.search(r"<\s*0\.5\s*Month", s, re.I):
        return 14  # ~2 weeks

    # "0.5 Month"
    if re.search(r"\b0\.5\s*Month\b", s, re.I):
        return 15

    # "1 Month", "2 Months", "14.5 Months"
    m = re.search(r"(\d+(\.\d+)?)\s*Month", s, re.I)
    if m:
        months = float(m.group(1))
        return int(round(months * 30))

    # Sometimes global page may show "Days"
    m = re.search(r"(\d+)\s*Day", s, re.I)
    if m:
        return int(m.group(1))

    return None


# -----------------------------
# POST MAP (Option B)
# -----------------------------

def parse_posts_cell(cell: str) -> List[str]:
    """
    The post map table often lists multiple posts per country.
    We split on separators conservatively.
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []

    # Common separators in these kinds of tables
    # Examples: "Paris, Marseille" or "Paris; Marseille"
    # Keep hyphenated names intact, split on commas/semicolons/slashes/newlines
    parts = re.split(r"[;\n/]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # If comma-separated list is used for multiple posts:
        # Only split if it looks like a list (contains multiple commas and not "City, State" patterns).
        # In practice for posts, commas usually indicate lists, so split on comma too.
        subparts = [x.strip() for x in p.split(",") if x.strip()]
        out.extend(subparts)
    # de-dupe while preserving order
    seen = set()
    deduped = []
    for p in out:
        k = normalize_key(p)
        if k and k not in seen:
            seen.add(k)
            deduped.append(p)
    return deduped

def build_post_map() -> Tuple[Dict[str, Dict], List[str]]:
    """
    Scrape POSTMAP_URL for the Country -> Visa Issuing Posts mapping.
    Returns:
      - map_by_post_norm: dict[normalized_post] -> {"country":..., "country_code":..., "raw_post":...}
      - warnings: list[str] of notes (duplicates/ambiguous/etc.)
    """
    warnings = []
    r = requests.get(POSTMAP_URL, timeout=30)
    r.raise_for_status()
    html = r.text

    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found on the Visa Wait Times post map page.")

    # Find the best candidate table: must have a Country-ish column and a Posts-ish column
    best = None
    best_score = -1

    for t in tables:
        df = t.copy()
        df.columns = [str(c).strip() for c in df.columns]

        cols = [c.lower() for c in df.columns]
        has_country = any("country" in c for c in cols)
        has_post = any("post" in c or "embassy" in c or "consulate" in c for c in cols)

        score = int(has_country) + int(has_post)
        if score > best_score:
            best = df
            best_score = score

    if best is None or best_score < 2:
        raise RuntimeError("Could not locate a suitable Country/Post mapping table on POSTMAP_URL.")

    df = best.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify columns
    country_col = None
    posts_col = None

    for c in df.columns:
        cl = c.lower()
        if country_col is None and "country" in cl:
            country_col = c
        if posts_col is None and ("post" in cl or "embassy" in cl or "consulate" in cl):
            posts_col = c

    if not country_col or not posts_col:
        raise RuntimeError("Post map table was found but columns could not be identified.")

    map_by_post_norm: Dict[str, Dict] = {}
    collisions: Dict[str, List[Dict]] = {}

    for _, row in df.iterrows():
        country = str(row.get(country_col, "")).strip()
        posts_cell = row.get(posts_col)

        if not country or country.lower() == "nan":
            continue

        posts = parse_posts_cell(posts_cell)
        for post in posts:
            post_norm = normalize_key(post)
            if not post_norm:
                continue

            entry = {
                "country": country,
                # Country codes are not consistently provided on this page.
                # We'll compute later if you decide to add ISO mapping; for now keep null.
                "country_code": None,
                "post": post,
                "post_norm": post_norm,
            }

            if post_norm in map_by_post_norm:
                # Collision: same normalized post appears under multiple countries (rare but possible).
                collisions.setdefault(post_norm, []).append(entry)
            else:
                map_by_post_norm[post_norm] = entry

    # Record collisions for visibility
    for post_norm, entries in collisions.items():
        existing = map_by_post_norm.get(post_norm)
        if existing:
            # Keep the first; warn about ambiguity
            all_countries = [existing["country"]] + [e["country"] for e in entries]
            warnings.append(
                f"AMBIGUOUS post '{existing['post']}' matches multiple countries: {sorted(set(all_countries))}. Keeping '{existing['country']}'."
            )

    return map_by_post_norm, warnings


# -----------------------------
# GLOBAL WAIT TIMES PARSE
# -----------------------------

def parse_global_wait_times() -> Tuple[List[Dict], Optional[str], str]:
    r = requests.get(GLOBAL_URL, timeout=30)
    r.raise_for_status()
    html = r.text

    as_of_date = find_page_date(html)  # ISO date string or None
    checked_utc = utc_now_iso()

    df = safe_get_first_table(html)
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure "Post" column exists
    if "Post" not in df.columns:
        df.rename(columns={df.columns[0]: "Post"}, inplace=True)

    visa_cols = [c for c in df.columns if c != "Post"]

    records = []
    for _, row in df.iterrows():
        post = str(row.get("Post", "")).strip()
        if not post or post.lower() == "nan":
            continue

        for col in visa_cols:
            raw = row.get(col)
            raw_txt = "" if pd.isna(raw) else str(raw).strip()

            rec = {
                "post": post,
                "post_norm": normalize_key(post),
                "visa_category": col.strip(),
                "wait_display": raw_txt,
                "wait_days_est": months_text_to_days(raw_txt),
                "source_url": GLOBAL_URL,
                "source_as_of_date": as_of_date,
                "last_checked_utc": checked_utc,
            }
            records.append(rec)

    return records, as_of_date, checked_utc


# -----------------------------
# OUTPUT SHAPING (WP-friendly)
# -----------------------------

def infer_visa_code(visa_category: str) -> Tuple[str, str, str]:
    """
    Map the column title into:
      - visa_code (for URL schema)
      - visa_label (display)
      - metric (avg vs next)
    We keep this conservative and transparent.
    """
    s = (visa_category or "").lower()

    metric = "avg"
    if "next available" in s:
        metric = "next"

    # Default fallback
    visa_code = "unknown"
    visa_label = visa_category.strip()

    # Detect major groups
    if "b1/b2" in s or "visitor" in s:
        visa_code = "b1b2"
        visa_label = "B1/B2 Visitor Visa"
    elif re.search(r"\bf\b", s) or "student" in s:
        visa_code = "f"
        visa_label = "F Student Visa"
    elif re.search(r"\bh\b", s) or "temporary worker" in s:
        visa_code = "h"
        visa_label = "H Temporary Worker Visa"
    elif re.search(r"\bl\b", s) or "intracompany" in s:
        visa_code = "l"
        visa_label = "L Intracompany Transferee Visa"
    elif re.search(r"\bo\b", s) or "extraordinary" in s:
        visa_code = "o"
        visa_label = "O Extraordinary Ability Visa"

    return visa_code, visa_label, metric

def choose_primary_wait(wait_records: List[Dict]) -> Optional[int]:
    """
    Prefer 'next available' when present; else fall back to avg.
    """
    next_vals = [r["wait_days_est"] for r in wait_records if r.get("metric") == "next" and r.get("wait_days_est") is not None]
    if next_vals:
        return next_vals[0]
    avg_vals = [r["wait_days_est"] for r in wait_records if r.get("metric") == "avg" and r.get("wait_days_est") is not None]
    if avg_vals:
        return avg_vals[0]
    return None

def build_posts(global_records: List[Dict], post_map: Dict[str, Dict], checked_utc: str, as_of_date: Optional[str]) -> Tuple[List[Dict], List[str]]:
    """
    Transform raw global_records into WP-friendly posts list with deterministic country joins.
    """
    warnings = []

    # Group by (post_norm, visa_code)
    grouped: Dict[Tuple[str, str], List[Dict]] = {}

    for r in global_records:
        visa_code, visa_label, metric = infer_visa_code(r.get("visa_category", ""))
        r2 = dict(r)
        r2["visa_code"] = visa_code
        r2["visa_label"] = visa_label
        r2["metric"] = metric

        key = (r2.get("post_norm", ""), visa_code)
        grouped.setdefault(key, []).append(r2)

    posts_out: List[Dict] = []

    for (post_norm, visa_code), items in grouped.items():
        if not post_norm:
            continue

        # Canonical country join
        pm = post_map.get(post_norm)
        if not pm:
            # Unmapped: keep record but flag clearly
            warnings.append(f"UNMAPPED post '{items[0].get('post')}' (norm='{post_norm}') not found in post map.")
            country = None
            country_code = "xx"
        else:
            country = pm.get("country")
            country_code = pm.get("country_code") or "xx"  # you can upgrade to ISO mapping later

        post_name = items[0].get("post")
        post_slug = slugify(post_name)

        # Your site uses /wait-times/{cc}/{city}/{visa}/ pattern. We keep cc even if unknown.
        # If you later add ISO mapping, country_code becomes real and URLs auto-fix.
        cc = (country_code or "xx").lower()

        # Use "us-{visa_code}" scheme to match earlier examples like us-b1b2
        visa_slug = f"us-{visa_code}"

        slug = f"{cc}/{post_slug}/{visa_slug}"

        # choose primary wait
        current_wait_days = choose_primary_wait(items)

        # Expose underlying metrics too
        metrics = []
        for it in items:
            metrics.append({
                "visa_category": it.get("visa_category"),
                "metric": it.get("metric"),
                "wait_display": it.get("wait_display"),
                "wait_days_est": it.get("wait_days_est"),
            })

        rec = {
            "id": f"{cc}-{post_slug}-{visa_slug}",
            "country_code": (country_code or "xx").upper(),
            "country": country,
            "city": post_name,
            "city_slug": post_slug,
            "post": slug,          # legacy key your earlier sample used
            "slug": slug,          # keep both for convenience
            "visa_code": visa_code,
            "visa_label": items[0].get("visa_label"),
            "current_wait_days": current_wait_days,
            "last_updated": as_of_date or (checked_utc[:10] if checked_utc else None),
            "scraped_at": checked_utc,
            "source_url": GLOBAL_URL,
            "status": "ok" if country else "unmapped_post",
            "metrics": metrics,

            # Placeholders for future phases (history + change notes)
            "history_values": [],
            "history_dates": [],
            "change_notes": [],
        }

        posts_out.append(rec)

    # Stable ordering: country then city then visa
    def sort_key(x):
        return (
            (x.get("country") or "ZZZ"),
            (x.get("city") or "ZZZ"),
            (x.get("visa_code") or "zzz")
        )
    posts_out.sort(key=sort_key)

    return posts_out, warnings


# -----------------------------
# MAIN
# -----------------------------

def main():
    checked_utc = utc_now_iso()

    # 1) Build canonical Post Map
    post_map, pm_warnings = build_post_map()

    # 2) Parse Global Visa Wait Times
    global_records, as_of_date, checked_utc2 = parse_global_wait_times()
    checked_utc = checked_utc2 or checked_utc

    # 3) Build WP-friendly posts output with deterministic joins
    posts, join_warnings = build_posts(global_records, post_map, checked_utc, as_of_date)

    out = {
        "version": "phase1-postmap-join-1.0.0",
        "dataset": "vwh_us_global_wait_times_phase1",
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,
        "postmap_url": POSTMAP_URL,
        "source_as_of_date": as_of_date,
        "last_checked_utc": checked_utc,
        "counts": {
            "posts_out": len(posts),
            "post_map_entries": len(post_map),
            "global_records_raw": len(global_records),
            "warnings": len(pm_warnings) + len(join_warnings),
        },
        "warnings": (pm_warnings + join_warnings)[:200],  # cap to avoid huge JSON
        "posts": posts,
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/us_posts.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote docs/us_posts.json")
    print(f"posts_out: {len(posts)}")
    print(f"post_map_entries: {len(post_map)}")
    print(f"raw_global_records: {len(global_records)}")
    print(f"warnings: {len(pm_warnings) + len(join_warnings)}")
    if pm_warnings or join_warnings:
        print("First warnings:")
        for w in (pm_warnings + join_warnings)[:10]:
            print(" -", w)


if __name__ == "__main__":
    main()
