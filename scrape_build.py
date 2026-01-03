import csv
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
RECIPROCITY_INDEX = "https://travel.state.gov/content/travel/en/us-visas/Visa-Reciprocity-and-Civil-Documents-by-Country.html"

DOCS_DIR = "docs"
OUT_POSTS = os.path.join(DOCS_DIR, "us_posts.json")
OUT_POST_MAP = os.path.join(DOCS_DIR, "post_map.json")
OUT_MISSING_TXT = os.path.join(DOCS_DIR, "missing_posts.txt")
OUT_MISSING_JSON = os.path.join(DOCS_DIR, "missing_posts.json")
OVERRIDES_CSV = os.path.join(DOCS_DIR, "post_overrides.csv")

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
    s = s.replace("consulate general", "")
    s = s.replace("consulate-general", "")
    s = s.replace("consulate", "")
    s = s.replace("embassy", "")

    # normalize punctuation
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9\s\-'\(\)]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def http_get(url: str, timeout=30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text

def country_name_to_iso2(name: str) -> Optional[str]:
    if not name:
        return None

    if pycountry:
        try:
            c = pycountry.countries.lookup(name)
            return c.alpha_2
        except Exception:
            pass

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
    soup = BeautifulSoup(index_html, "html.parser")
    links = []

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

        if any(x in url.lower() for x in ["view-all", "search", "index"]):
            continue

        links.append((text, url))

    seen = set()
    out = []
    for name, url in links:
        if url in seen:
            continue
        seen.add(url)
        out.append((name, url))

    return out

def clean_post_list(items: List[str]) -> List[str]:
    cleaned = []
    for it in items:
        s = (it or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"^[\-\*\u2022]+\s*", "", s)
        s = re.sub(r"^\d+[\.\)]\s*", "", s)
        if len(s) < 3:
            continue
        cleaned.append(s)

    seen = set()
    out = []
    for s in cleaned:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def extract_visa_issuing_posts(country_html: str) -> List[str]:
    soup = BeautifulSoup(country_html, "html.parser")
    text = soup.get_text("\n")

    headings = soup.find_all(["h1", "h2", "h3", "h4", "strong"])
    target = None
    for h in headings:
        t = (h.get_text() or "").strip().lower()
        if "visa issuing" in t and "post" in t:
            target = h
            break

    posts = []

    if target:
        for sib in target.parent.find_all_next():
            if sib.name in ("h1", "h2", "h3") and sib is not target:
                break
            if sib.name in ("li",):
                val = (sib.get_text() or "").strip()
                if val:
                    posts.append(val)
        if posts:
            return clean_post_list(posts)

    m = re.search(r"Visa Issuing Posts\s*\n(.+)", text, re.I)
    if m:
        start = m.start()
        chunk = text[start:start + 2500]
        lines = [ln.strip() for ln in chunk.splitlines()]
        out_lines = []
        seen_header = False
        for ln in lines:
            if not ln:
                continue
            if not seen_header:
                if re.search(r"visa issuing posts", ln, re.I):
                    seen_header = True
                continue
            if re.search(r"^(police|court|marriage|birth|divorce|death|fees|comments|document|issuing authority)", ln, re.I):
                break
            out_lines.append(ln)
        if out_lines:
            return clean_post_list(out_lines)

    return []

def build_post_map(controlled: bool) -> Tuple[Dict[str, dict], List[str]]:
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

    for i, (country_name, country_url) in enumerate(country_links, start=1):
        try:
            html = http_get(country_url)
        except Exception as e:
            warnings.append(f"WARNING: Failed to fetch country page {country_url}: {e}")
            continue

        posts = extract_visa_issuing_posts(html)
        if not posts:
            continue

        cc = country_name_to_iso2(country_name) or ""
        for p in posts:
            norm = normalize_post_name(p)
            if not norm:
                continue
            if norm in post_map:
                continue
            post_map[norm] = {
                "post": p,
                "country": country_name,
                "country_code": cc,
                "source_country_page": country_url,
            }

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

def ensure_overrides_file_exists():
    if os.path.exists(OVERRIDES_CSV):
        return
    with open(OVERRIDES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["post", "country", "country_code"])

def load_overrides() -> Dict[str, dict]:
    ensure_overrides_file_exists()
    overrides: Dict[str, dict] = {}
    with open(OVERRIDES_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post = (row.get("post") or "").strip()
            if not post:
                continue
            country = (row.get("country") or "").strip()
            cc = (row.get("country_code") or "").strip().upper()

            if country and not cc:
                cc_guess = country_name_to_iso2(country)
                if cc_guess:
                    cc = cc_guess

            overrides[normalize_post_name(post)] = {
                "post": post,
                "country": country,
                "country_code": cc,
                "source": "override_csv",
            }
    return overrides

def main():
    controlled = os.getenv("CONTROLLED", "").strip() in ("1", "true", "TRUE", "yes", "YES")
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 1) Auto map from reciprocity pages (best-effort)
    post_map, pm_warnings = build_post_map(controlled)

    # 2) Manual overrides (makes system complete)
    overrides = load_overrides()

    # Merge overrides into post_map (overrides win)
    for k, v in overrides.items():
        if v.get("country") and v.get("country_code"):
            post_map[k] = {
                "post": v.get("post", ""),
                "country": v.get("country", ""),
                "country_code": v.get("country_code", ""),
                "source_country_page": "override_csv",
            }

    print(f"[DEBUG] Overrides loaded={len(overrides)} | post_map_after_overrides={len(post_map)}")
# DEBUG: check whether specific missing posts are now mapped
for t in ["Abidjan","Accra","Adana","Almaty","Amsterdam"]:
    print("[DEBUG] override_check", t, "->", post_map.get(normalize_post_name(t)))

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
                "country": (mapping or {}).get("country", ""),
                "country_code": (mapping or {}).get("country_code", ""),
            }
            records.append(rec)

    unique_posts = sorted(set(r["post"] for r in records))
    missing = []
    for p in unique_posts:
        if not post_map.get(normalize_post_name(p)):
            missing.append(p)

    print(f"[OK] Built post_map.json with {len(post_map)} unique normalized posts (after overrides)")
    if pm_warnings:
        print(f"[WARN] post_map warnings: {len(pm_warnings)} (see docs/post_map.json warnings)")

    print(f"[OK] Parsed global wait table into {len(records)} records across {len(unique_posts)} unique posts")

    # Always write missing files (even if empty) so they can be committed and served
    with open(OUT_MISSING_TXT, "w", encoding="utf-8") as f:
        for p in missing:
            f.write(p + "\n")

    with open(OUT_MISSING_JSON, "w", encoding="utf-8") as f:
        json.dump({"missing_count": len(missing), "missing_posts": missing}, f, ensure_ascii=False, indent=2)

    if missing:
        print(f"[WARNING] Missing mapping for {len(missing)} posts")
        print("First 50 missing:")
        for p in missing[:50]:
            print(f" - {p}")
        print(f"[OK] Wrote {OUT_MISSING_TXT} and {OUT_MISSING_JSON}")
        if controlled:
            raise RuntimeError(f"CONTROLLED=1: Missing mapping for {len(missing)} posts")
    else:
        print("[OK] No missing mappings. You are complete.")

    out = {**meta, "record_count": len(records), "records": records}
    with open(OUT_POSTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {OUT_POSTS}")

if __name__ == "__main__":
    main()
