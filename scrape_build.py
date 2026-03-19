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
# ==============================
# B3 SNAPSHOT ARCHIVE HELPERS
# ==============================

def vwh_month_key_from_iso(iso_str: str) -> str:
    s = (iso_str or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m")

def write_json_file(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def archive_daily_raw_records(records: list, meta: dict, docs_dir: str = "docs"):
    """
    Write an immutable daily raw-records snapshot before posts/history logic.
    This is the audit truth source for what the parser produced on a given day.
    """
    day = str((meta or {}).get("last_checked_utc") or now_utc_iso())[:10]  # YYYY-MM-DD
    out_dir = os.path.join(docs_dir, "daily-raw")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{day}-records.json")

    force = os.environ.get("VWH_DAILY_RAW_FORCE", "").strip() == "1"
    if os.path.exists(out_path) and not force:
        print(f"[daily-raw] exists (immutable) → {out_path} (set VWH_DAILY_RAW_FORCE=1 to overwrite)")
        return out_path

    payload = {
        "generated_at": now_utc_iso(),
        "source": "U.S. Department of State (travel.state.gov)",
        "source_url": GLOBAL_URL,
        "last_checked_utc": (meta or {}).get("last_checked_utc"),
        "record_count": len(records or []),
        "records": records or [],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[daily-raw] wrote {out_path} | records={len(records or [])}")
    return out_path
def build_daily_raw_audit_report(records: list, meta: dict, docs_dir: str = "docs", days_back: int = 7):
    """
    Compare today's raw records to the raw snapshot from N days ago.
    Writes a compact audit report showing whether raw source rows changed.
    """
    from datetime import timedelta

    last_checked = str((meta or {}).get("last_checked_utc") or now_utc_iso())
    today_date = datetime.fromisoformat(last_checked.replace("Z", "+00:00")).date()
    prior_date = today_date - timedelta(days=days_back)

    raw_dir = os.path.join(docs_dir, "daily-raw")
    prior_path = os.path.join(raw_dir, f"{prior_date.isoformat()}-records.json")

    out_dir = os.path.join(docs_dir, "audit")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{today_date.isoformat()}-vs-{days_back}d.json")

    def rec_key(r):
        return (
            str(r.get("post") or "").strip(),
            str(r.get("visa_category") or "").strip(),
        )

    def rec_sig(r):
        return {
            "wait_display": str(r.get("wait_display") or "").strip(),
            "wait_days_est": r.get("wait_days_est"),
            "country": str(r.get("country") or "").strip(),
            "country_code": str(r.get("country_code") or "").strip(),
        }

    today_map = {rec_key(r): rec_sig(r) for r in (records or [])}

    if not os.path.exists(prior_path):
        payload = {
            "generated_at": now_utc_iso(),
            "days_back": days_back,
            "today_date": today_date.isoformat(),
            "prior_date": prior_date.isoformat(),
            "prior_snapshot_found": False,
            "today_record_count": len(today_map),
            "summary": {
                "changed": 0,
                "unchanged": 0,
                "new": len(today_map),
                "missing": 0,
            },
            "examples": {
                "changed": [],
                "new": [],
                "missing": [],
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[audit] wrote {out_path} (no prior snapshot found)")
        return out_path

    with open(prior_path, "r", encoding="utf-8") as f:
        prior_payload = json.load(f)

    prior_records = prior_payload.get("records") or []
    prior_map = {rec_key(r): rec_sig(r) for r in prior_records}

    changed = []
    unchanged = 0
    new = []
    missing = []

    all_keys = set(today_map.keys()) | set(prior_map.keys())

    for k in sorted(all_keys):
        t = today_map.get(k)
        p = prior_map.get(k)

        if t is None and p is not None:
            missing.append({
                "post": k[0],
                "visa_category": k[1],
                "prior": p,
            })
            continue

        if t is not None and p is None:
            new.append({
                "post": k[0],
                "visa_category": k[1],
                "today": t,
            })
            continue

        if t == p:
            unchanged += 1
        else:
            changed.append({
                "post": k[0],
                "visa_category": k[1],
                "today": t,
                "prior": p,
            })

    payload = {
        "generated_at": now_utc_iso(),
        "days_back": days_back,
        "today_date": today_date.isoformat(),
        "prior_date": prior_date.isoformat(),
        "prior_snapshot_found": True,
        "today_record_count": len(today_map),
        "prior_record_count": len(prior_map),
        "summary": {
            "changed": len(changed),
            "unchanged": unchanged,
            "new": len(new),
            "missing": len(missing),
        },
        "examples": {
            "changed": changed[:25],
            "new": new[:25],
            "missing": missing[:25],
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[audit] wrote {out_path} | changed={len(changed)} unchanged={unchanged} new={len(new)} missing={len(missing)}")
    return out_path    
def archive_monthly_snapshot(out_posts: dict, docs_dir: str = "docs"):
    """
    Write an immutable monthly snapshot JSON (no posts[] inside).
    Snapshot should preserve ALL insights fields so report pages stay consistent
    as insights evolve over time.
    """
    import copy

    # Determine month from out_posts["generated_at"] if present, else now_utc_iso()
    gen = (out_posts or {}).get("generated_at") or now_utc_iso()
    month = str(gen)[:7]  # "YYYY-MM"

    snapshots_dir = os.path.join(docs_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    snap_path = os.path.join(snapshots_dir, f"{month}.json")

    # Immutability by default (do not overwrite)
    force = os.environ.get("VWH_SNAPSHOT_FORCE", "").strip() == "1"
    if os.path.exists(snap_path) and not force:
        print(f"[snapshots] exists (immutable) → {snap_path} (set VWH_SNAPSHOT_FORCE=1 to overwrite)")
        return None

    # Snapshot = full out_posts minus posts[] (keeps all insights, rankings_meta, movement, etc.)
    snap = copy.deepcopy(out_posts) if out_posts else {}
    if "posts" in snap:
        snap.pop("posts", None)

    # Optional: keep highlights (small), but you can remove them if you want an even slimmer file.
    # snap.pop("highlights_fastest_available", None)
    # snap.pop("highlights_recently_changed", None)

    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)

    return snap_path

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

def write_mapping_suspects_report(records: list, overrides: Dict[str, dict], out_path: str):
    """
    Write a simple mapping audit report for manual review.
    This is a first-pass sanity checker, not a full geocoder.
    """
    suspects = []

    override_keys = set((overrides or {}).keys())

    # Known suspicious patterns we want to surface for manual review
    known_post_country_mismatches = {
        ("abu dhabi", "IR"),
    }

    seen = set()

    for r in (records or []):
        post = str(r.get("post") or "").strip()
        post_norm = str(r.get("post_norm") or "").strip()
        country = str(r.get("country") or "").strip()
        cc = str(r.get("country_code") or "").strip().upper()

        key = (post, cc)
        if key in seen:
            continue
        seen.add(key)

        reason = None

        if not post:
            continue

        if post_norm in override_keys:
            continue  # trusted by explicit override

        if not country or not cc:
            reason = "missing_country_mapping"
        elif (post.lower(), cc) in known_post_country_mismatches:
            reason = "known_post_country_mismatch"

        if reason:
            suspects.append({
                "post": post,
                "post_norm": post_norm,
                "country": country,
                "country_code": cc,
                "reason": reason,
            })

    payload = {
        "generated_at": now_utc_iso(),
        "suspect_count": len(suspects),
        "suspects": suspects,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[audit] wrote {out_path} | suspects={len(suspects)}")

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
    archive_daily_raw_records(records, meta, docs_dir=DOCS_DIR)
    build_daily_raw_audit_report(records, meta, docs_dir=DOCS_DIR, days_back=7)
    build_daily_raw_audit_report(records, meta, docs_dir=DOCS_DIR, days_back=1)
    write_mapping_suspects_report(records, overrides, os.path.join(DOCS_DIR, "audit", "mapping-suspects.json"))
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
    seen_ids = {}
    for r in records:
        visa_code = visa_category_to_code(r.get("visa_category", ""))
        if not visa_code:
            continue  # skip non-Big-Five columns

        post_name = (r.get("post") or "").strip()
        if not post_name:
            continue
        # --- TEMP DEBUG: inspect selected raw records before posts[] build ---
        debug_targets = {
            ("Abu Dhabi", "f"),
            ("Abuja", "b1b2"),
            ("Accra", "b1b2"),
            ("Asuncion", "b1b2"),
        }

        if (post_name, visa_code) in debug_targets:
            print("[DEBUG raw record]", {
                "post": r.get("post"),
                "visa_category": r.get("visa_category"),
                "visa_code": visa_code,
                "country": r.get("country"),
                "country_code": r.get("country_code"),
                "wait_days_est": r.get("wait_days_est"),
                "wait_display": r.get("wait_display"),
                "last_checked_utc": r.get("last_checked_utc"),
                "source_url": r.get("source_url"),
            })
        country_code = (r.get("country_code") or "").strip().upper()
        country = (r.get("country") or "").strip()

        city_slug = city_to_slug(post_name)

        # Stable ID: cc-city-visa (good enough for v1)
        stable_id = f"{country_code.lower()}-{city_slug}-{visa_code}"
        seen_ids[stable_id] = seen_ids.get(stable_id, 0) + 1
        if seen_ids[stable_id] > 1:
            print("[DEBUG duplicate stable_id]", {
                "stable_id": stable_id,
                "count": seen_ids[stable_id],
                "post": post_name,
                "visa_code": visa_code,
                "country": country,
                "country_code": country_code,
                "wait_days_est": r.get("wait_days_est"),
                "wait_display": r.get("wait_display"),
            })
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
    # ---- DEDUPLICATE POSTS BY ID (CRITICAL FIX) ----
    deduped = {}
    for p in posts:
        pid = str(p.get("id") or "")
        if not pid:
            continue
        deduped[pid] = p  # last occurrence wins

    posts = list(deduped.values())

    print(f"[OK] Deduplicated posts: {len(posts)} remaining")
    # ---- END DEDUP ----
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
        Canonical aggregate stats for a list of post records.
        """

        waits = []
        posts_available = 0
        posts_unavailable = 0

        for r in (rows or []):
            w = r.get("current_wait_days")
            if isinstance(w, int):
                waits.append(w)

            if r.get("is_available") is True:
                posts_available += 1
            elif r.get("is_available") is False:
                posts_unavailable += 1

        total_posts = len(rows or [])
        posts_with_wait = len(waits)

        def _pct(x, denom):
            if denom <= 0:
                return None
            return float(x) / float(denom)

        def _median(nums):
            if not nums:
                return None
            s = sorted(nums)
            n = len(s)
            mid = n // 2
            if n % 2 == 1:
                return float(s[mid])
            return (float(s[mid - 1]) + float(s[mid])) / 2.0

        def _pctl(nums, p):
            if not nums:
                return None
            s = sorted(nums)
            if len(s) == 1:
                return float(s[0])
            k = int(round((p / 100.0) * (len(s) - 1)))
            k = max(0, min(k, len(s) - 1))
            return float(s[k])

        avg_wait = (sum(waits) / float(len(waits))) if waits else None
        median_wait = _median(waits)
        p90_wait = _pctl(waits, 90)

        availability_rate = _pct(posts_available, total_posts)

        longest_wait = float(max(waits)) if waits else None
        shortest_wait = float(min(waits)) if waits else None

        return {
            "avg_wait": avg_wait,
            "median_wait": median_wait,
            "p90_wait": p90_wait,
            "availability_rate": availability_rate,

            "total_posts": total_posts,
            "posts_with_wait": posts_with_wait,
            "posts_available": posts_available,
            "posts_unavailable": posts_unavailable,

            "longest_wait": longest_wait,
            "shortest_wait": shortest_wait,
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

    # -------------------------------------------------
    # AGGREGATES (Global / Visa / Region / Movement)
    # -------------------------------------------------

    def _pctl(vals, p: float):
        """Nearest-rank percentile (stable, no numpy). p in [0,1]."""
        if not vals:
            return None
        s = sorted(vals)
        if len(s) == 1:
            return s[0]
        # nearest-rank: ceil(p*n)
        import math
        k = int(math.ceil(p * len(s)))
        k = max(1, min(len(s), k))
        return s[k - 1]

    def _wait_ints(rows):
        return [r.get("current_wait_days") for r in rows if isinstance(r.get("current_wait_days"), int)]

    def _count_available(rows):
        return sum(1 for r in rows if r.get("is_available") is True)

    def _count_unavailable(rows):
        return sum(1 for r in rows if r.get("is_available") is False)

    def _agg_block(rows):
        # Backwards-compatible alias (older code may still call _agg_block)
        return _summary_stats(rows)

    # -------------------------
    # Global totals (basic)
    # -------------------------
    posts_total = len(posts)
    posts_with_wait_count = sum(1 for p in posts if isinstance(p.get("current_wait_days"), int))
    posts_available = sum(1 for p in posts if p.get("is_available") is True)
    posts_unavailable = sum(1 for p in posts if p.get("is_available") is False)
    countries_total = len(set((p.get("country_code") or "").lower() for p in posts if p.get("country_code")))
    cities_total = len(set((p.get("city_slug") or "").lower() for p in posts if p.get("city_slug")))

    # -------------------------
    # By visa (all + each visa_code) with completed aggregates
    # -------------------------
    visa_codes = sorted(set((p.get("visa_code") or "").lower() for p in posts if p.get("visa_code")))
    by_visa = {"all": _agg_block(posts)}
    for v in visa_codes:
        by_visa[v] = _agg_block([p for p in posts if (p.get("visa_code") or "").lower() == v])

    # -------------------------
    # Diagnostics (latest last_updated; history span)
    # -------------------------
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

    # -------------------------
    # Movement aggregates (global)
    # -------------------------
    d30_vals = [p.get("delta_30d") for p in posts if isinstance(p.get("delta_30d"), int)]
    total_with_delta_30d = len(d30_vals)
    avg_delta_30d = (sum(d30_vals) / total_with_delta_30d) if total_with_delta_30d else None

    inc = sum(1 for x in d30_vals if x > 0)
    dec = sum(1 for x in d30_vals if x < 0)
    same = sum(1 for x in d30_vals if x == 0)

    pct_increasing_30d = (inc / total_with_delta_30d) if total_with_delta_30d else None
    pct_decreasing_30d = (dec / total_with_delta_30d) if total_with_delta_30d else None
    pct_unchanged_30d = (same / total_with_delta_30d) if total_with_delta_30d else None

    movement = {
        "total_with_delta_30d": total_with_delta_30d,
        "avg_delta_30d": avg_delta_30d,
        "pct_increasing_30d": pct_increasing_30d,
        "pct_decreasing_30d": pct_decreasing_30d,
        "pct_unchanged_30d": pct_unchanged_30d,
    }

    # -------------------------------------------------
    # GLOBAL TIE INTELLIGENCE (Authority-grade context)
    # (IMPORTANT: do NOT overwrite posts_with_wait_count)
    # -------------------------------------------------
    posts_with_wait_rows = [r for r in posts if isinstance(r.get("current_wait_days"), int)]
    posts_available_with_wait_rows = [
        r for r in posts
        if r.get("is_available") is True and isinstance(r.get("current_wait_days"), int)
    ]

    rankings_meta = {}

    if posts_with_wait_rows:
        min_wait = min(r["current_wait_days"] for r in posts_with_wait_rows)
        ties = [r for r in posts_with_wait_rows if r["current_wait_days"] == min_wait]
        tie_by_visa = {}
        for r in ties:
            vc = (r.get("visa_code") or "unknown").lower()
            tie_by_visa[vc] = tie_by_visa.get(vc, 0) + 1
        rankings_meta["shortest_wait"] = {
            "min_wait_days": min_wait,
            "tie_posts_total": len(ties),
            "tie_by_visa": tie_by_visa,
        }

    if posts_available_with_wait_rows:
        min_wait_avail = min(r["current_wait_days"] for r in posts_available_with_wait_rows)
        ties_avail = [r for r in posts_available_with_wait_rows if r["current_wait_days"] == min_wait_avail]
        tie_by_visa_avail = {}
        for r in ties_avail:
            vc = (r.get("visa_code") or "unknown").lower()
            tie_by_visa_avail[vc] = tie_by_visa_avail.get(vc, 0) + 1
        rankings_meta["fastest_available"] = {
            "min_wait_days": min_wait_avail,
            "tie_posts_total": len(ties_avail),
            "tie_by_visa": tie_by_visa_avail,
        }
    # ---------------------------
    # Region mapping (SET-based, no deps)
    # ---------------------------
    AFRICA = {
        "dz","ao","bj","bw","bf","bi","cv","cm","cf","td","km","cg","cd","ci","dj","eg","gq","er","sz","et",
        "ga","gm","gh","gn","gw","ke","ls","lr","ly","mg","mw","ml","mr","mu","ma","mz","na","ne","ng","rw",
        "st","sn","sc","sl","so","za","ss","sd","tz","tg","tn","ug","zm","zw"
    }

    ASIA = {
        "af","am","az","bh","bd","bt","bn","kh","cn","cy","ge","in","id","ir","iq","il","jp","jo","kz","kw",
        "kg","la","lb","my","mv","mn","mm","np","kp","om","pk","ps","ph","qa","sa","sg","kr","lk","sy","tw",
        "tj","th","tl","tr","tm","ae","uz","vn","ye","hk","mo"
    }

    EUROPE = {
        "al","ad","at","by","be","ba","bg","hr","cz","dk","ee","fi","fr","de","gr","hu","is","ie","it","lv",
        "li","lt","lu","mt","md","mc","me","nl","mk","no","pl","pt","ro","ru","sm","rs","sk","si","es","se",
        "ch","ua","gb","va","xk"
    }

    NORTH_AMERICA = {
        "ag","bs","bb","bz","ca","cr","cu","dm","do","sv","gd","gt","ht","hn","jm","mx","ni","pa","kn","lc",
        "vc","tt","us"
    }

    SOUTH_AMERICA = {
        "ar","bo","br","cl","co","ec","gy","py","pe","sr","uy","ve","fk"
    }

    OCEANIA = {
        "au","fj","ki","mh","fm","nr","nz","pw","pg","ws","sb","to","tv","vu"
    }

    def _region_for_cc(cc):
        cc = (cc or "").lower()
        if cc in AFRICA: return "africa"
        if cc in ASIA: return "asia"
        if cc in EUROPE: return "europe"
        if cc in NORTH_AMERICA: return "north_america"
        if cc in SOUTH_AMERICA: return "south_america"
        if cc in OCEANIA: return "oceania"
        return "unknown"
        
    # -------------------------
    # Rankings (Top 20 each)
    # NOTE: these intentionally use simple filters and rely on your computed delta fields.
    # -------------------------
    rankings = {
        "top_longest_wait": _top_n(
            posts,
            key_fn=lambda r: (r.get("current_wait_days") is None, r.get("current_wait_days", -1)),
            n=20,
            reverse=True,
            where_fn=lambda r: isinstance(r.get("current_wait_days"), int),
        ),
        "top_shortest_wait": _top_n(
            posts,
            key_fn=lambda r: r.get("current_wait_days", 10**9),
            n=20,
            reverse=False,
            where_fn=lambda r: isinstance(r.get("current_wait_days"), int),
        ),
        "top_fastest_available": _top_n(
            posts,
            key_fn=lambda r: r.get("current_wait_days", 10**9),
            n=20,
            reverse=False,
            where_fn=lambda r: r.get("is_available") is True and isinstance(r.get("current_wait_days"), int),
        ),

        "top_increase_30d": _top_n(
            posts,
            key_fn=lambda r: r.get("delta_30d", -10**9),
            n=20,
            reverse=True,
            where_fn=lambda r: isinstance(r.get("delta_30d"), int) and r.get("delta_30d") > 0,
        ),
        "top_decrease_30d": _top_n(
            posts,
            key_fn=lambda r: r.get("delta_30d", 10**9),
            n=20,
            reverse=False,
            where_fn=lambda r: isinstance(r.get("delta_30d"), int) and r.get("delta_30d") < 0,
        ),
        "top_movers_30d_abs": _top_n(
            posts,
            key_fn=lambda r: abs(r.get("delta_30d", 0)),
            n=20,
            reverse=True,
            where_fn=lambda r: isinstance(r.get("delta_30d"), int) and r.get("delta_30d") != 0,
        ),

        "top_increase_7d": _top_n(
            posts,
            key_fn=lambda r: r.get("delta_7d", -10**9),
            n=20,
            reverse=True,
            where_fn=lambda r: isinstance(r.get("delta_7d"), int) and r.get("delta_7d") > 0,
        ),
        "top_decrease_7d": _top_n(
            posts,
            key_fn=lambda r: r.get("delta_7d", 10**9),
            n=20,
            reverse=False,
            where_fn=lambda r: isinstance(r.get("delta_7d"), int) and r.get("delta_7d") < 0,
        ),
        "top_movers_7d_abs": _top_n(
            posts,
            key_fn=lambda r: abs(r.get("delta_7d", 0)),
            n=20,
            reverse=True,
            where_fn=lambda r: isinstance(r.get("delta_7d"), int) and r.get("delta_7d") != 0,
        ),

        "most_recently_changed": _top_n(
            posts,
            key_fn=lambda r: (r.get("last_change_at") or ""),
            n=20,
            reverse=True,
            where_fn=lambda r: bool(r.get("last_change_at")),
        ),
        "most_unavailable_now": _top_n(
            posts,
            key_fn=lambda r: r.get("current_wait_days", 10**9),
            n=20,
            reverse=True,
            where_fn=lambda r: r.get("is_available") is False and isinstance(r.get("current_wait_days"), int),
        ),
    }    
    # -------------------------
    # By region with completed aggregates
    # -------------------------
    region_buckets = {}
    for p in posts:
        cc = (p.get("country_code") or "").lower()
        rgn = _region_for_cc(cc)
        region_buckets.setdefault(rgn, []).append(p)

    by_region = {}
    for rgn, rows in region_buckets.items():
        by_region[rgn] = _agg_block(rows)

    # -------------------------
    # Insights object (now complete)
    # IMPORTANT: insights MUST be built before out_posts
    # -------------------------
    insights = {
        "meta": {
            "schema_version": "insights-1.1",
            "window_days": {"d7": 7, "d30": 30, "d90": 90},
            "notes": [
                "All waits are in days.",
                "Deltas: positive means longer waits; negative means shorter waits."
            ]
        },
        "totals": {
            "posts_total": posts_total,
            "posts_with_wait": posts_with_wait_count,
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
        "movement": movement,
        "by_visa": by_visa,
        "by_region": by_region,
        "rankings": rankings,
        "rankings_meta": rankings_meta,
    }

    print(f"[OK] insights built: visa={len(by_visa)} regions={len(by_region)} rankings={len(rankings)}")

    # -------------------------
    # Build final dataset JSON
    # -------------------------
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

    # B3: archive immutable monthly snapshot (no posts[] inside)
    snap_path = archive_monthly_snapshot(out_posts, docs_dir=DOCS_DIR)
    if snap_path:
        print(f"[snapshots] wrote {snap_path}")
    else:
        print("[snapshots] no write (already exists or missing generated_at)")
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
