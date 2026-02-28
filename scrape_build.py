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

# --- SAFE HTTP FETCH HELPER (prevents GitHub Action from hanging) ---



# ---------------------------
# Build detail-page sitemap (Phase 1 pilot)
# ---------------------------
def build_detail_sitemap(posts, out_path):
    """
    Phase 1 sitemap:
    Only include selected pilot countries.
    Later we will remove this filter to include ALL posts.
    """

    # ---------------------------
# Build detail-page sitemap (FULL — all detail pages)
# ---------------------------
def build_detail_sitemap(posts, out_path):
    """
    Full sitemap:
    Includes ALL detail pages (no country filtering).
    """

    urls = []

    for p in posts:
        cc = (p.get("country_code") or "").lower()
        city_slug = p.get("city_slug")
        visa = (p.get("visa_code") or "").lower()

        if not cc or not city_slug or not visa:
            continue

        url = f"https://visawaithub.com/wait-times/{cc}/{city_slug}/us-{visa}/"
        lastmod = p.get("last_updated") or ""

        urls.append((url, lastmod))

    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]

    for url, lastmod in urls:
        xml_parts.append("<url>")
        xml_parts.append(f"<loc>{url}</loc>")
        if lastmod:
            xml_parts.append(f"<lastmod>{lastmod}</lastmod>")
        xml_parts.append("</url>")

    xml_parts.append("</urlset>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_parts))

    print(f"[OK] Wrote FULL detail sitemap → {out_path} | urls={len(urls)}")

# ---------------------------
# VWH: History + Change Notes helpers (daily)
# ---------------------------

from datetime import datetime, timezone

def _vwh_today_iso():
    # Example: "2026-02-02"
    return datetime.now(timezone.utc).date().isoformat()

def _vwh_load_prev_posts_map(path):
    """
    Reads the previous OUT_POSTS JSON (if it exists) and returns:
      { "post_id": previous_post_dict, ... }
    If missing or invalid, returns {}.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prev_posts = data.get("posts") or []
        m = {}
        for p in prev_posts:
            pid = p.get("id")
            if pid is not None:
                m[str(pid)] = p
        return m
    except Exception:
        return {}

def _vwh_coerce_history(prev_post):
    """
    Returns safe (history_values, history_dates, change_notes).
    Ensures lists + aligned lengths.
    """
    if not prev_post:
        return [], [], []

    hv = prev_post.get("history_values")
    hd = prev_post.get("history_dates")
    cn = prev_post.get("change_notes")

    if not isinstance(hv, list): hv = []
    if not isinstance(hd, list): hd = []
    if not isinstance(cn, list): cn = []

    # Align history arrays so they are same length
    n = min(len(hv), len(hd))
    hv = hv[-n:]
    hd = hd[-n:]

    return hv, hd, cn

def _vwh_append_daily(hv, hd, cn, today_iso, wait_days_int, visa_label):
    """
    Append exactly one history point per day (no duplicates).
    If today's value changed vs yesterday, append a human-readable change note.
    """
    hv = list(hv)
    hd = list(hd)
    cn = list(cn)

    # Only append if we have a real integer wait
    if wait_days_int is None:
        return hv, hd, cn

    last_date = hd[-1] if hd else None
    if last_date != today_iso:
        prev_val = hv[-1] if hv else None

        hd.append(today_iso)
        hv.append(int(wait_days_int))

        # Change note only if changed vs previous
        if prev_val is not None and int(wait_days_int) != int(prev_val):
            direction = "increased" if int(wait_days_int) > int(prev_val) else "decreased"
            text = f"{visa_label} wait time {direction} from {int(prev_val)} to {int(wait_days_int)} days."

            # Avoid duplicate note for same day
            last_note_date = None
            if cn and isinstance(cn[-1], dict):
                last_note_date = cn[-1].get("date")

            if last_note_date != today_iso:
                cn.append({"date": today_iso, "text": text})

    # Keep history bounded (optional safety)
    if len(hv) > 365:
        hv = hv[-365:]
        hd = hd[-365:]
    if len(cn) > 50:
        cn = cn[-50:]

    return hv, hd, cn


GLOBAL_URL = "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/global-visa-wait-times.html"
RECIPROCITY_INDEX = "https://travel.state.gov/content/travel/en/us-visas/Visa-Reciprocity-and-Civil-Documents-by-Country.html"

DOCS_DIR = "docs"
CACHE_DIR = os.path.join(DOCS_DIR, "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_POSTS = os.path.join(DOCS_DIR, "us_posts.json")
OUT_RECORDS = os.path.join(DOCS_DIR, "us_records.json")
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

import time


def http_get(url: str, headers=None, timeout=(10, 30), retries=2, backoff=2.0) -> str:
    """
    timeout=(connect_seconds, read_seconds)
    retries=2 => up to 3 attempts total
    """
    hdrs = headers or {"User-Agent": UA}
    last_err = None

    for attempt in range(retries + 1):
        try:
            print(f"[INFO] Fetching ({attempt+1}/{retries+1}): {url}")
            r = requests.get(url, headers=hdrs, timeout=timeout)
            r.raise_for_status()
            print(f"[INFO] Done: {url} ({len(r.text)} bytes)")
            return r.text
        except Exception as e:
            last_err = e
            print(f"[WARN] Fetch failed: {url} — {e}")
            if attempt < retries:
                sleep_s = backoff * (attempt + 1)
                print(f"[INFO] Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch after retries: {url}") from last_err
    # --- CACHE HELPER FOR RECIPROCITY PAGES ---
def cached_get(url: str, cache_key: str, max_age_hours: int = 168) -> str:
    """
    Disk cache for expensive reciprocity pages.
    max_age_hours=168 means refresh once every 7 days.
    """
    path = os.path.join(CACHE_DIR, cache_key)

    if os.path.exists(path):
        age_s = time.time() - os.path.getmtime(path)
        if age_s < max_age_hours * 3600:
            print(f"[CACHE] Hit: {cache_key} ({age_s/3600:.1f}h old)")
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    print(f"[CACHE] Miss: {cache_key} — fetching")
    html = http_get(url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return html




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
    index_html = cached_get(RECIPROCITY_INDEX, "recip_index.html", max_age_hours=168)

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
            cache_key = "recip_" + re.sub(r"[^a-zA-Z0-9]+", "_", country_url.split("/")[-1])[:80] + ".html"
            html = cached_get(country_url, cache_key, max_age_hours=168)

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

    print("[INFO] Parsing tables with pandas.read_html...")
    t0 = time.time()
    from io import StringIO
    tables = pd.read_html(StringIO(html), flavor="lxml")
    print(f"[INFO] pandas.read_html done in {time.time() - t0:.2f}s. tables={len(tables)}")

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
import re
from typing import Optional

def city_to_slug(city: str) -> str:
    """
    Convert a post/city display name into a stable URL slug.
    Examples:
      "New Delhi" -> "new-delhi"
      "Chennai (Madras)" -> "chennai-madras"
      "Osaka/Kobe" -> "osaka-kobe"
      "N`Djamena" -> "n-djamena"
    """
    s = (city or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[’'`]", "", s)         # strip apostrophes/backticks
    s = re.sub(r"[^a-z0-9]+", "-", s)   # non-alnum -> hyphen
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def visa_category_to_code(label: str) -> Optional[str]:
    """
    Map the global wait-times table column header to our Big Five visa_code.

    IMPORTANT:
    - We intentionally prefer "Next available appointment" (not "Average wait times")
      to match what users usually mean by "current wait".
    - We only emit Big Five: b1b2, f, h, l, o
    """
    t = (label or "").strip().lower()

    # We only want "Next available appointment" columns
    if "next available appointment" not in t:
        return None

    # B1/B2 (Tourism/Business)
    if "(b1/b2)" in t or "b1/b2" in t:
        return "b1b2"

    # F (Student) — State groups as (F,M,J); we only keep F for Big Five
    if "(f,m,j)" in t or "f,m,j" in t:
        return "f"

    # H / L / O — State groups petition-based as (H,L,O,P,Q); we keep H/L/O for Big Five
    if "(h,l,o,p,q)" in t or "h,l,o,p,q" in t:
        # We'll map this grouped column to each of H/L/O later if we want separate,
        # but for now choose ONE behavior:
        # Option 1 (recommended v1): map to "h" as the representative petition bucket
        return "petition"

    return None



def main():
    controlled = os.getenv("CONTROLLED", "").strip() in ("1", "true", "TRUE", "yes", "YES")
    FORCE_POST_MAP = os.getenv("FORCE_POST_MAP", "0") == "1"
    os.makedirs(DOCS_DIR, exist_ok=True)


    # 1) Auto map from reciprocity pages (best-effort)
    POST_MAP_CACHE_JSON = os.path.join(DOCS_DIR, "_cache", "post_map_cache.json")
    POST_MAP_JSON = os.path.join(DOCS_DIR, "post_map.json")


    post_map = None
    pm_warnings = []

    # Reuse existing post_map cache to save GitHub minutes (rebuild only if missing)
    if os.path.exists(POST_MAP_CACHE_JSON) and not FORCE_POST_MAP:
        try:
            with open(POST_MAP_CACHE_JSON, "r", encoding="utf-8") as f:
                post_map = json.load(f)
            print(f"[OK] Loaded cached post_map from {POST_MAP_CACHE_JSON} | posts={len(post_map)}")
        except Exception as e:
            print(f"[WARN] Failed to load {POST_MAP_CACHE_JSON}, rebuilding: {e}")
            post_map = None

    # Build fresh only when needed
    if post_map is None:
        post_map, pm_warnings = build_post_map(controlled)
        with open(POST_MAP_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(post_map, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote post_map cache → {POST_MAP_CACHE_JSON} | posts={len(post_map)}")

    # 1b) Always apply overrides (even when reusing cached post_map)
    try:
        overrides = load_overrides()
    except Exception as e:
        overrides = {}
        print(f"[WARN] Failed to load overrides CSV: {e}")

    applied = 0
    if overrides:
        for norm, rec in overrides.items():
            if not norm:
                continue

            base = post_map.get(norm)
            if not base:
                post_map[norm] = dict(rec)
                applied += 1
                continue

            changed = False
            for k in ("post", "country", "country_code"):
                if rec.get(k) and not base.get(k):
                    base[k] = rec[k]
                    changed = True
            if changed:
                applied += 1

    print(f"[OK] Applied overrides to post_map: {applied} entries")

    # If overrides changed anything, persist back so future runs are stable
    if applied > 0:
        with open(POST_MAP_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(post_map, f, ensure_ascii=False, indent=2)
        print(f"[OK] Updated cached post_map → {POST_MAP_JSON} | posts={len(post_map)}")



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

        # Filter obvious non-post junk tokens
        if normalize_post_name(post) in ("department",):
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
    # --- Option D posts[] build (Big Five only) ---
    prev_posts_map = _vwh_load_prev_posts_map(OUT_POSTS)
    today_iso = _vwh_today_iso()
    posts = []
    for r in records:
        visa_code = visa_category_to_code(r.get("visa_category", ""))
        if not visa_code:
            continue  # skip non-Big-Five columns

        post_name = (r.get("post") or "").strip()
        if not post_name:
            continue

        country_code = (r.get("country_code") or "").strip().upper()
        country = (r.get("country") or "").strip()

        city_slug = city_to_slug(post_name)

        # Stable ID: cc-city-visa (good enough for v1)
        stable_id = f"{country_code.lower()}-{city_slug}-{visa_code}"

        wait_days = r.get("wait_days_est")
        try:
            wait_days_int = int(wait_days) if wait_days is not None else None
        except Exception:
            wait_days_int = None

        wait_display = (r.get("wait_display") or "").strip()

        # Basic availability heuristic
        raw_status = wait_display
        is_available = True
        if (not wait_display) or ("not available" in wait_display.lower()) or (wait_display.strip() in ("—", "-", "N/A")):
            is_available = False

        # --- VWH history persistence (per post id) ---
        prev_post = prev_posts_map.get(str(stable_id))
        hv_prev, hd_prev, cn_prev = _vwh_coerce_history(prev_post)

        # label used inside change notes
        visa_label_for_note = (visa_code.upper() if visa_code else "VISA")

        hv_new, hd_new, cn_new = _vwh_append_daily(
            hv_prev,
            hd_prev,
            cn_prev,
            today_iso,
            wait_days_int,
            visa_label_for_note,
        )

        posts.append({
            # identity / location
            "id": stable_id,
            "country_code": country_code,
            "country": country,
            "region": "",
            "state_province": "",
            "city": post_name,
            "city_slug": city_slug,
            "embassy_name": post_name,   # placeholder for v1
            "post_title": post_name,
            "post_type": "post",
            "post": post_name,

            # visa
            "visa_code": visa_code,
            "visa_label": visa_code.upper(),
            "visa_group": "big5",

            # live wait-time fields
            "current_wait_days": wait_days_int,
            "raw_status": raw_status,
            "is_available": is_available,
            "last_updated": r.get("last_checked_utc") or "",

            # history (persisted + daily appended)
            "history_values": hv_new,
            "history_dates": hd_new,
            "change_notes": cn_new,


            # scraper metadata
            "source_url": r.get("source_url") or GLOBAL_URL,
            "scraped_at": r.get("last_checked_utc") or "",
            "status": "ok",
            "note": "",
        })
    # ---- DERIVED FIELDS (Phase 1: per-post deltas + last change) ----
    from datetime import datetime

    def _iso_to_date(s):
        try:
            if not s:
                return None
            # Accept "YYYY-MM-DD" or full ISO "YYYY-MM-DDTHH:MM:SS+00:00"
            return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
        except Exception:
            return None

    def _find_value_on_or_before(history_dates, history_values, target_date):
        """
        Returns value for the latest snapshot on or before target_date.
        Assumes history_dates is oldest->newest and aligned with history_values.
        """
        if (not history_dates) or (not history_values) or (len(history_dates) != len(history_values)):
            return None

        best = None
        for ds, vs in zip(history_dates, history_values):
            d = _iso_to_date(ds)
            if not d:
                continue
            if d <= target_date:
                best = vs
            else:
                # dates are ascending; once we pass target_date we can stop
                break
        return best

    def _compute_post_derivatives(p):
        hv = p.get("history_values") or []
        hd = p.get("history_dates") or []

        # defaults
        p["delta_7d"] = None
        p["delta_30d"] = None
        p["last_change_at"] = ""
        p["has_recent_change"] = False
        p["trend_direction"] = "unknown"

        if (not hv) or (not hd) or (len(hv) != len(hd)):
            return

        last_date = _iso_to_date(hd[-1])
        try:
            last_val = int(hv[-1])
        except Exception:
            return

        if not last_date:
            return

        # target dates
        d7 = last_date.fromordinal(last_date.toordinal() - 7)
        d30 = last_date.fromordinal(last_date.toordinal() - 30)

        v7 = _find_value_on_or_before(hd, hv, d7)
        v30 = _find_value_on_or_before(hd, hv, d30)

        # positive = worse (wait increased); negative = improved
        try:
            p["delta_7d"] = (last_val - int(v7)) if v7 is not None else None
        except Exception:
            p["delta_7d"] = None

        try:
            p["delta_30d"] = (last_val - int(v30)) if v30 is not None else None
        except Exception:
            p["delta_30d"] = None

        # last_change_at: most recent snapshot where value changed vs prior
        last_change = ""
        for i in range(len(hv) - 1, 0, -1):
            try:
                a = int(hv[i])
                b = int(hv[i - 1])
            except Exception:
                continue
            if a != b:
                last_change = str(hd[i])
                break

        p["last_change_at"] = last_change
        p["has_recent_change"] = bool(last_change)

        # trend_direction uses delta_7d if available else delta_30d
        d_primary = p["delta_7d"] if p["delta_7d"] is not None else p["delta_30d"]
        if d_primary is None:
            p["trend_direction"] = "unknown"
        elif d_primary > 0:
            p["trend_direction"] = "up"
        elif d_primary < 0:
            p["trend_direction"] = "down"
        else:
            p["trend_direction"] = "flat"

    for p in posts:
        _compute_post_derivatives(p)

    print("[OK] Derived fields added: delta_7d, delta_30d, last_change_at, has_recent_change, trend_direction")
    # ---- END DERIVED FIELDS ----
    # ---- HOMEPAGE HIGHLIGHTS (Phase 2) ----

    # Fastest available (lowest wait days)
    fastest_available = [
        p for p in posts
        if p.get("is_available") and isinstance(p.get("current_wait_days"), int)
    ]
    fastest_available.sort(key=lambda x: x["current_wait_days"])
    highlights_fastest = fastest_available[:10]

    # Recently changed (most recent change first)
    recently_changed = [
        p for p in posts
        if p.get("has_recent_change") and p.get("last_change_at")
    ]
    recently_changed.sort(key=lambda x: x["last_change_at"], reverse=True)
    highlights_recent = recently_changed[:10]

    print("[OK] Homepage highlight lists built")

    # ---- END HOMEPAGE HIGHLIGHTS ----    
    # ---------------------------
    # INSIGHTS (Global Intelligence Layer) — v1
    # ---------------------------
    import math
    from datetime import datetime, timezone

    def _iso_to_dt(s: str):
        if not s:
            return None
        try:
            # Handles: 2026-02-26T19:07:00+00:00
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    def _pct(values, p):
        """Percentile (0-100) with linear interpolation; returns None if empty."""
        if not values:
            return None
        xs = sorted(values)
        if len(xs) == 1:
            return xs[0]
        k = (len(xs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs[int(k)]
        d0 = xs[f] * (c - k)
        d1 = xs[c] * (k - f)
        return d0 + d1

    def _summary_stats(rows):
        """
        rows: list of post dicts
        returns: robust stats for waits + delta_7d + delta_30d + availability rate
        """
        waits = [r["current_wait_days"] for r in rows if isinstance(r.get("current_wait_days"), int)]
        d7 = [r["delta_7d"] for r in rows if isinstance(r.get("delta_7d"), int)]
        d30 = [r["delta_30d"] for r in rows if isinstance(r.get("delta_30d"), int)]
        avail = [r.get("is_available") for r in rows if isinstance(r.get("is_available"), bool)]

        def _rates(arr):
            if not arr:
                return {"up_rate": None, "down_rate": None, "flat_rate": None}
            up = sum(1 for x in arr if x > 0)
            down = sum(1 for x in arr if x < 0)
            flat = sum(1 for x in arr if x == 0)
            n = len(arr)
            return {
                "up_rate": round(up / n, 4),
                "down_rate": round(down / n, 4),
                "flat_rate": round(flat / n, 4),
            }

        def _delta_block(arr):
            if not arr:
                return {
                    "avg": None, "median": None, "p90_abs": None,
                    "up_rate": None, "down_rate": None, "flat_rate": None
                }
            abs_arr = [abs(x) for x in arr]
            return {
                "avg": round(sum(arr) / len(arr), 4),
                "median": _pct(arr, 50),
                "p90_abs": _pct(abs_arr, 90),
                **_rates(arr)
            }

        wait_block = None
        if waits:
            wait_block = {
                "avg": round(sum(waits) / len(waits), 4),
                "median": _pct(waits, 50),
                "p10": _pct(waits, 10),
                "p25": _pct(waits, 25),
                "p75": _pct(waits, 75),
                "p90": _pct(waits, 90),
                "min": min(waits),
                "max": max(waits),
            }

        available_rate = None
        if avail:
            available_rate = round(sum(1 for x in avail if x) / len(avail), 4)

        return {
            "count": len(rows),
            "available_rate": available_rate,
            "wait_days": wait_block,
            "delta_7d": _delta_block(d7),
            "delta_30d": _delta_block(d30),
        }

    def _compact(p):
        """Compact ranking entry; keep file size sane."""
        cc = (p.get("country_code") or "").lower()
        city = p.get("city") or ""
        city_slug = p.get("city_slug") or ""
        visa = (p.get("visa_code") or "").lower()
        href = f"/wait-times/{cc}/{city_slug}/us-{visa}/"
        return {
            "cc": cc,
            "city": city,
            "city_slug": city_slug,
            "visa": visa,
            "wait": p.get("current_wait_days"),
            "available": p.get("is_available"),
            "last_updated": p.get("last_updated") or "",
            "href": href,
            "delta_7d": p.get("delta_7d"),
            "delta_30d": p.get("delta_30d"),
            "last_change_at": p.get("last_change_at") or "",
        }

    def _top_n(rows, key_fn, n=20, reverse=False, where_fn=None):
        xs = rows
        if where_fn:
            xs = [r for r in xs if where_fn(r)]
        xs = sorted(xs, key=key_fn, reverse=reverse)
        return [_compact(r) for r in xs[:n]]

    # Totals
    posts_total = len(posts)
    posts_with_wait = sum(1 for p in posts if isinstance(p.get("current_wait_days"), int))
    posts_available = sum(1 for p in posts if p.get("is_available") is True)
    posts_unavailable = sum(1 for p in posts if p.get("is_available") is False)
    countries_total = len(set((p.get("country_code") or "").lower() for p in posts if p.get("country_code")))
    cities_total = len(set((p.get("city_slug") or "").lower() for p in posts if p.get("city_slug")))

    # By visa (all + each visa_code)
    posts_all = posts[:]
    visa_codes = sorted(set((p.get("visa_code") or "").lower() for p in posts if p.get("visa_code")))
    by_visa = {"all": _summary_stats(posts_all)}
    for v in visa_codes:
        by_visa[v] = _summary_stats([p for p in posts if (p.get("visa_code") or "").lower() == v])

    # Time diagnostics (latest last_updated; history span)
    latest_last_updated = ""
    last_updated_vals = [p.get("last_updated") for p in posts if p.get("last_updated")]
    if last_updated_vals:
        latest_last_updated = sorted(last_updated_vals)[-1]

    span_min = None
    span_max = None
    for p in posts:
        hd = p.get("history_dates") or []
        if not isinstance(hd, list) or len(hd) < 2:
            continue
        a = _iso_to_dt(hd[0])
        b = _iso_to_dt(hd[-1])
        if not a or not b:
            continue
        days = int((b - a).total_seconds() // 86400)
        if span_min is None or days < span_min:
            span_min = days
        if span_max is None or days > span_max:
            span_max = days

    # Rankings (Top 20 each)
    # NOTE: these intentionally use simple filters and rely on your computed delta fields.
    rankings = {
        "top_longest_wait": _top_n(posts, key_fn=lambda r: (r.get("current_wait_days") is None, r.get("current_wait_days", -1)), n=20, reverse=True,
                                   where_fn=lambda r: isinstance(r.get("current_wait_days"), int)),
        "top_shortest_wait": _top_n(posts, key_fn=lambda r: r.get("current_wait_days", 10**9), n=20, reverse=False,
                                    where_fn=lambda r: isinstance(r.get("current_wait_days"), int)),
        "top_fastest_available": _top_n(posts, key_fn=lambda r: r.get("current_wait_days", 10**9), n=20, reverse=False,
                                        where_fn=lambda r: r.get("is_available") is True and isinstance(r.get("current_wait_days"), int)),

        "top_increase_30d": _top_n(posts, key_fn=lambda r: r.get("delta_30d", -10**9), n=20, reverse=True,
                                   where_fn=lambda r: isinstance(r.get("delta_30d"), int) and r.get("delta_30d") > 0),
        "top_decrease_30d": _top_n(posts, key_fn=lambda r: r.get("delta_30d", 10**9), n=20, reverse=False,
                                   where_fn=lambda r: isinstance(r.get("delta_30d"), int) and r.get("delta_30d") < 0),
        "top_movers_30d_abs": _top_n(posts, key_fn=lambda r: abs(r.get("delta_30d", 0)), n=20, reverse=True,
                                     where_fn=lambda r: isinstance(r.get("delta_30d"), int) and r.get("delta_30d") != 0),

        "top_increase_7d": _top_n(posts, key_fn=lambda r: r.get("delta_7d", -10**9), n=20, reverse=True,
                                  where_fn=lambda r: isinstance(r.get("delta_7d"), int) and r.get("delta_7d") > 0),
        "top_decrease_7d": _top_n(posts, key_fn=lambda r: r.get("delta_7d", 10**9), n=20, reverse=False,
                                  where_fn=lambda r: isinstance(r.get("delta_7d"), int) and r.get("delta_7d") < 0),
        "top_movers_7d_abs": _top_n(posts, key_fn=lambda r: abs(r.get("delta_7d", 0)), n=20, reverse=True,
                                    where_fn=lambda r: isinstance(r.get("delta_7d"), int) and r.get("delta_7d") != 0),

        "most_recently_changed": _top_n(posts, key_fn=lambda r: (r.get("last_change_at") or ""), n=20, reverse=True,
                                        where_fn=lambda r: bool(r.get("last_change_at"))),
        "most_unavailable_now": _top_n(posts, key_fn=lambda r: r.get("current_wait_days", 10**9), n=20, reverse=True,
                                       where_fn=lambda r: r.get("is_available") is False and isinstance(r.get("current_wait_days"), int)),
    }

    # Region mapping (DICT-based; v1 defaults to unknown; we’ll fill this out next step)
    REGION_BY_CC = {
        # We'll expand this mapping in the next baby step without changing schema.
        # Example entries (not exhaustive):
        "us": "north_america",
        "ca": "north_america",
        "mx": "north_america",
        "gb": "europe",
        "fr": "europe",
        "de": "europe",
        "in": "asia",
        "ph": "asia",
        "ng": "africa",
        "za": "africa",
        "au": "oceania",
        "nz": "oceania",
        "br": "south_america",
        "ar": "south_america",
    }

    def _region_for_cc(cc):
        cc = (cc or "").lower()
        return REGION_BY_CC.get(cc, "unknown")

    # By region (v1, will be more complete as REGION_BY_CC fills out)
    region_buckets = {}
    for p in posts:
        cc = (p.get("country_code") or "").lower()
        r = _region_for_cc(cc)
        region_buckets.setdefault(r, []).append(p)

    by_region = {}
    for r, rows in region_buckets.items():
        by_region[r] = _summary_stats(rows)

    insights = {
        "meta": {
            "schema_version": "insights-1.0",
            "window_days": {"d7": 7, "d30": 30, "d90": 90},
            "notes": [
                "All waits are in days.",
                "Deltas: positive means longer waits; negative means shorter waits."
            ]
        },
        "totals": {
            "posts_total": posts_total,
            "posts_with_wait": posts_with_wait,
            "posts_available": posts_available,
            "posts_unavailable": posts_unavailable,
            "countries_total": countries_total,
            "cities_total": cities_total
        },
        "time": {
            "generated_at": now_utc_iso(),
            "latest_post_last_updated": latest_last_updated,
            "history_span_days": {"min": span_min, "max": span_max}
        },
        "by_visa": by_visa,
        "by_region": by_region,
        "rankings": rankings
    }

    print(f"[OK] insights built: visa={len(by_visa)} regions={len(by_region)} rankings={len(rankings)}")
    out_posts = {
        "version": "1.0",
        "generated_at": now_utc_iso(),
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,

        "insights": insights,

        "highlights_fastest_available": highlights_fastest,
        "highlights_recently_changed": highlights_recent,

        "posts": posts,
    }

    with open(OUT_POSTS, "w", encoding="utf-8") as f:
        json.dump(out_posts, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote Option D posts[] to {OUT_POSTS} | posts={len(posts)}")
    # Build Phase-1 pilot sitemap (detail pages)
    OUT_SITEMAP = os.path.join(DOCS_DIR, "vwh-sitemap-details.xml")
    build_detail_sitemap(posts, OUT_SITEMAP)
    print(f"[INFO] Expected sitemap path: {OUT_SITEMAP}")
    if not os.path.exists(OUT_SITEMAP):
        raise RuntimeError(f"Sitemap was not created at {OUT_SITEMAP}")
    print(f"[INFO] Sitemap created OK: {OUT_SITEMAP} ({os.path.getsize(OUT_SITEMAP)} bytes)")



    
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
    with open(OUT_RECORDS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {OUT_POSTS}")

if __name__ == "__main__":
    main()
