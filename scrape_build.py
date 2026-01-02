import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import pycountry
except Exception:
    pycountry = None

GLOBAL_URL = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/global-visa-wait-times.html"

# Official directory of country pages (contains Visa Issuing Posts sections)
RECIPROCITY_INDEX = "https://travel.state.gov/content/travel/en/us-visas/Visa-Reciprocity-and-Civil-Documents-by-Country.html"

# Output files (served by GitHub Pages from /docs)
DOCS_DIR = "docs"
OUT_POSTS = os.path.join(DOCS_DIR, "us_posts.json")
OUT_POST_MAP = os.path.join(DOCS_DIR, "post_map.json")

UA = "VisaWaitHubBot/1.0 (+https://visawaithub.com; data informational; respects robots; rate-limited)"

def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

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

def normalize_post_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    # remove common noise
    s = s.replace("u.s. ", "").replace("us ", "")
    s = s.replace("embassy", "").replace("consulate", "").replace("consulate general", "")
    s = s.replace("consulate-general", "")
    s = re.sub(r"[^a-z0-9\s\-']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

def http_get(url: str, timeout=30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text

def country_name_to_iso2(name: str) -> Optional[str]:
    if not name:
        return None
    # If pycountry is available, use it (best).
    if pycountry:
        try:
            c = pycountry.countries.lookup(name)
            return c.alpha_2
        except Exception:
            pass

    # Fallback small manual normalization for common cases
    n = name.strip().lower()
    manual = {
        "côte d’ivoire": "CI",
        "cote d'ivoire": "CI",
        "cote d’ivoire": "CI",
        "bolivia": "BO",
        "tanzania": "TZ",
        "vietnam": "VN",
        "laos": "LA",
        "russia": "RU",
        "iran": "IR",
        "syria": "SY",
        "venezuela": "VE",
        "moldova": "MD",
        "macedonia": "MK",
        "korea, south": "KR",
        "korea, north": "KP",
    }
    return manual.get(n)

def extract_country_links(index_html: str) -> List[Tuple[str, str]]:
    """
    Returns list of (country_name, country_url)
    """
    soup = BeautifulSoup(index_html, "html.parser")

    links = []
    # The directory page contains many country links; keep only ones that look like country pages.
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        text = (a.get_text() or "").strip()
        if not text or len(text) < 3:
            continue
        if "/Visa-Reciprocity-and-Civil-Documents-by-Country/" not in href:
            continue
        if not href.endswith(".html"):
            continue

        url = href
        if url.startswith("/"):
            url = "https://travel.state.gov" + url

        # Filter out obvious non-country pages if any
        if any(x in url.lower() for x in ["view-all", "search", "index"]):
            continue

        links.append((text, url))

    # Deduplicate by URL
    seen = set()
    out = []
    for name, url in links:
        if url in seen:
            continue
        seen.add(url)
        out.append((name, url))

    return out

def extract_visa_issuing_posts(country_html: str) -> List[str]:
    """
    Tries to find 'Visa Issuing Posts' section and returns a list of post names.
    The structure varies, so we use a few heuristics.
    """
    soup = BeautifulSoup(country_html, "html.parser")
    text = soup.get_text("\n")

    # Heuristic 1: find heading containing "Visa Issuing Posts"
    # Then capture following list items or lines until next heading.
    headings = soup.find_all(["h1", "h2", "h3", "h4", "strong"])
    target = None
    for h in headings:
        t = (h.get_text() or "").strip().lower()
        if "visa issuing" in t and "post" in t:
            target = h
            break

    posts = []

    if target:
        # walk siblings and gather list items / lines
        for sib in target.parent.find_all_next():
            # stop at next big heading
            if sib.name in ("h1", "h2", "h3") and sib is not target:
                break
            if sib.name in ("li",):
                val = (sib.get_text() or "").strip()
                if val:
                    posts.append(val)

        # If we found list items, return them
        if posts:
            return clean_post_list(posts)

    # Heuristic 2: Some pages embed content as plain text blocks.
    # Extract lines after the phrase.
    m = re.search(r"Visa Issuing Posts\s*\n(.+)", text, re.I)
    if m:
        # Take a slice after the match and split lines; stop at empty run.
        start = m.start()
        chunk = text[start:start + 2000]
        lines = [ln.strip() for ln in chunk.splitlines()]
        # Keep lines after the header line
        out_lines = []
        seen_header = False
        for ln in lines:
            if not ln:
                continue
            if not seen_header:
                if re.search(r"visa issuing posts", ln, re.I):
                    seen_header = True
                continue
            # stop if we hit another obvious section header
            if re.search(r"^((police|court|marriage|birth|divorce|death|fees|comments|document|issuing authority))", ln, re.I):
                break
            out_lines.append(ln)
        if out_lines:
            return clean_post_list(out_lines)

    return []

def clean_post_list(items: List[str]) -> List[str]:
    cleaned = []
    for it in items:
        s = (it or "").strip()
        s = re.sub(r"\s+", " ", s)
        # Remove bullets and numbering
        s = re.sub(r"^[\-\*\u2022]+\s*", "", s)
        s = re.sub(r"^\d+[\.\)]\s*", "", s)
        # Ignore very short / generic lines
        if len(s) < 3:
            continue
        cleaned.append(s)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in cleaned:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def build_post_map(controlled: bool) -> Tuple[Dict[str, dict], List[str]]:
    """
    Returns:
      - map keyed by normalized post name -> mapping dict
      - warnings list
    """
    warnings = []
    index_html = http_get(RECIPROCITY_INDEX)

    country_links = extract_country_links(index_html)
    if not country_links:
        msg = "WARNING: Could not find any country links on reciprocity index."
        warnings.append(msg)
        if controlled:
            raise RuntimeError(msg)
        return {}, warnings

    post_map: Dict[str, dict] = {}

    # Rate limit: this is a "build map" step; keep it polite
    # (If you decide this is too heavy daily, we can cache and rebuild weekly.)
    for i, (country_name, country_url) in enumerate(country_links, start=1):
        try:
            html = http_get(country_url)
        except Exception as e:
            warnings.append(f"WARNING: Failed to fetch country page {country_url}: {e}")
            continue

        posts = extract_visa_issuing_posts(html)
        if not posts:
            # Not all pages expose it consistently; don't fail for this alone.
            continue

        cc = country_name_to_iso2(country_name) or ""
        for p in posts:
            norm = normalize_post_name(p)
            if not norm:
                continue
            # Keep first mapping if collision
            if norm in post_map:
                continue
            post_map[norm] = {
                "post": p,
                "country": country_name,
                "country_code": cc,
                "source_country_page": country_url,
            }

        # small sleep every few pages
        if i % 20 == 0:
            time.sleep(0.6)

    if not post_map:
        msg = "WARNING: Built post_map is empty (no Visa Issuing Posts sections found)."
        warnings.append(msg)
        if controlled:
            raise RuntimeError(msg)

    return post_map, warnings

def scrape_global_wait_times() -> Tuple[dict, pd.DataFrame]:
    html = http_get(GLOBAL_URL)
    checked_utc = now_utc_iso()

    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found on the Global Visa Wait Times page.")

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Post" not in df.columns:
        df.rename(columns={df.columns[0]: "Post"}, inplace=True)

    meta = {
        "dataset": "vwh_us_global_wait_times_phase1",
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,
        "last_checked_utc": checked_utc,
    }
    return meta, df

def main():
    controlled = os.getenv("CONTROLLED", "").strip() in ("1", "true", "TRUE", "yes", "YES")
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Always rebuild mapping in this version (fast enough for now).
    # Later we can cache and only rebuild weekly.
    post_map, pm_warnings = build_post_map(controlled)

    # Write post_map.json for inspection
    pm_out = {
        "generated_utc": now_utc_iso(),
        "source_index": RECIPROCITY_INDEX,
        "count": len(post_map),
        "items": post_map,
        "warnings": pm_warnings,
    }
    with open(OUT_POST_MAP, "w", encoding="utf-8") as f:
        json.dump(pm_out, f, ensure_ascii=False, indent=2)

    meta, df = scrape_global_wait_times()

    visa_cols = [c for c in df.columns if c != "Post"]
    records = []

    for _, row in df.iterrows():
        post = str(row.get("Post", "")).strip()
        if not post or post.lower() == "nan":
            continue

        post_norm = normalize_post_name(post)
        mapping = post_map.get(post_norm)

        for col in visa_cols:
            raw = row.get(col)
            raw_txt = "" if pd.isna(raw) else str(raw).strip()

            rec = {
                "post": post,
                "post_norm": post_norm,
                "visa_category": col.strip(),
                "wait_display": raw_txt,
                "wait_days_est": months_text_to_days(raw_txt),
                "source_url": GLOBAL_URL,
                "last_checked_utc": meta["last_checked_utc"],
            }

            if mapping:
                rec["country"] = mapping.get("country", "")
                rec["country_code"] = mapping.get("country_code", "")
            else:
                rec["country"] = ""
                rec["country_code"] = ""

            records.append(rec)

    # Print mapping coverage info (this is what you saw in logs)
    unique_posts = sorted(set(r["post"] for r in records))
    missing = []
    for p in unique_posts:
        if not post_map.get(normalize_post_name(p)):
            missing.append(p)

    print(f"[OK] Built post_map.json with {len(post_map)} unique normalized posts")
    if pm_warnings:
        print(f"[WARN] post_map warnings: {len(pm_warnings)} (see docs/post_map.json warnings)")

    print(f"[OK] Parsed global wait table into {len(records)} records across {len(unique_posts)} unique posts")

    if missing:
        print(f"[WARNING] Missing mapping for {len(missing)} posts")
        print("Missing list (ALL):")
        for p in missing[:50]:
            print(f" - {p}")

        # --- NEW: write missing list to docs/ so you can download it ---
        
        os.makedirs("docs", exist_ok=True)

        with open("docs/missing_posts.txt", "w", encoding="utf-8") as f:
            for p in missing:
                f.write(p + "\n")

        with open("docs/missing_posts.json", "w", encoding="utf-8") as f:
            json.dump({"missing_count": len(missing), "missing_posts": missing}, f, ensure_ascii=False, indent=2)

        print("[OK] Wrote docs/missing_posts.txt and docs/missing_posts.json")
        # --- END NEW ---

        if controlled:
            raise RuntimeError(f"CONTROLLED=1: Missing mapping for {len(missing)} posts")

    out = {**meta, "record_count": len(records), "records": records}

    with open(OUT_POSTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {OUT_POSTS}")

if __name__ == "__main__":
    main()
