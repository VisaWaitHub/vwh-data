#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import requests


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
TEST_OUTPUT_PATH = DOCS_DIR / "_country_context_test.json"
LIVE_OUTPUT_PATH = DOCS_DIR / "country_context.json"

# Put your real source URLs here later.
# For the first safe manual test, we allow fallback sample mode.
ISSUANCE_SOURCE_URL = None
REFUSAL_SOURCE_URL = None

MIN_COUNTRY_COUNT = 20
REQUEST_TIMEOUT = 60

# Locked thresholds from your current system
HIGH_DEMAND_THRESHOLD = 500_000
MODERATE_DEMAND_THRESHOLD = 150_000


# =========================
# HELPERS
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def demand_tier(value: int) -> str:
    if value >= HIGH_DEMAND_THRESHOLD:
        return "high"
    if value >= MODERATE_DEMAND_THRESHOLD:
        return "moderate"
    return "lower"


def format_int(value: int) -> str:
    return f"{value:,}"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def ensure_docs_dir() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def safe_get_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"Could not read existing JSON file at {path}: {e}")


# =========================
# COUNTRY MAPPING
# Expand this as needed
# =========================

COUNTRY_NAME_TO_CODE = {
    "India": "IN",
    "Brazil": "BR",
    "Nigeria": "NG",
    "Mexico": "MX",
    "France": "FR",
    "Germany": "DE",
    "Colombia": "CO",
    "Turkey": "TR",
    "Chile": "CL",
    "Egypt": "EG",
    "United Kingdom": "GB",
    "United Arab Emirates": "AE",
    "Canada": "CA",
    "Australia": "AU",
    "Japan": "JP",
    "South Korea": "KR",
    "China": "CN",
    "Pakistan": "PK",
    "South Africa": "ZA",
    "Argentina": "AR",
}


# =========================
# FETCH
# =========================

def fetch_text(url: str) -> str:
    log(f"Fetching text: {url}")
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def fetch_bytes(url: str) -> bytes:
    log(f"Fetching binary: {url}")
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content


# =========================
# PARSE PLACEHOLDERS
# Replace these later with real source parsers.
# For the first manual test, they can use sample data.
# =========================

def parse_issuance_data() -> Dict[str, Dict[str, Any]]:
    """
    Return shape:
    {
      "IN": {"country": "India", "issuance_volume_value": 123456},
      ...
    }
    """
    if not ISSUANCE_SOURCE_URL:
        log("ISSUANCE_SOURCE_URL not set. Using sample issuance data for safe manual test.")
        return {
            "IN": {"country": "India", "issuance_volume_value": 1200000},
            "BR": {"country": "Brazil", "issuance_volume_value": 700000},
            "NG": {"country": "Nigeria", "issuance_volume_value": 80000},
            "MX": {"country": "Mexico", "issuance_volume_value": 950000},
            "FR": {"country": "France", "issuance_volume_value": 180000},
            "DE": {"country": "Germany", "issuance_volume_value": 160000},
            "CO": {"country": "Colombia", "issuance_volume_value": 300000},
            "TR": {"country": "Turkey", "issuance_volume_value": 90000},
            "CL": {"country": "Chile", "issuance_volume_value": 60000},
            "EG": {"country": "Egypt", "issuance_volume_value": 110000},
            "GB": {"country": "United Kingdom", "issuance_volume_value": 400000},
            "AE": {"country": "United Arab Emirates", "issuance_volume_value": 170000},
            "CA": {"country": "Canada", "issuance_volume_value": 350000},
            "AU": {"country": "Australia", "issuance_volume_value": 140000},
            "JP": {"country": "Japan", "issuance_volume_value": 200000},
            "KR": {"country": "South Korea", "issuance_volume_value": 190000},
            "CN": {"country": "China", "issuance_volume_value": 500000},
            "PK": {"country": "Pakistan", "issuance_volume_value": 130000},
            "ZA": {"country": "South Africa", "issuance_volume_value": 100000},
            "AR": {"country": "Argentina", "issuance_volume_value": 125000},
        }

    # Real parser will go here later.
    # For now we deliberately fail if a URL is set but parser is not implemented.
    fail("Real issuance parser not implemented yet. Leave ISSUANCE_SOURCE_URL unset for the first manual test.")


def parse_refusal_data() -> Dict[str, Dict[str, Any]]:
    """
    Return shape:
    {
      "IN": {"country": "India", "refusal_rate_value": 22.04},
      ...
    }
    """
    if not REFUSAL_SOURCE_URL:
        log("REFUSAL_SOURCE_URL not set. Using sample refusal data for safe manual test.")
        return {
            "IN": {"country": "India", "refusal_rate_value": 22.04},
            "BR": {"country": "Brazil", "refusal_rate_value": 14.87},
            "NG": {"country": "Nigeria", "refusal_rate_value": 57.00},
            "MX": {"country": "Mexico", "refusal_rate_value": 19.20},
            "FR": {"country": "France", "refusal_rate_value": 12.30},
            "DE": {"country": "Germany", "refusal_rate_value": 10.80},
            "CO": {"country": "Colombia", "refusal_rate_value": 28.40},
            "TR": {"country": "Turkey", "refusal_rate_value": 24.70},
            "CL": {"country": "Chile", "refusal_rate_value": 11.20},
            "EG": {"country": "Egypt", "refusal_rate_value": 39.60},
            "GB": {"country": "United Kingdom", "refusal_rate_value": 9.90},
            "AE": {"country": "United Arab Emirates", "refusal_rate_value": 13.40},
            "CA": {"country": "Canada", "refusal_rate_value": 8.70},
            "AU": {"country": "Australia", "refusal_rate_value": 7.80},
            "JP": {"country": "Japan", "refusal_rate_value": 6.40},
            "KR": {"country": "South Korea", "refusal_rate_value": 8.10},
            "CN": {"country": "China", "refusal_rate_value": 25.50},
            "PK": {"country": "Pakistan", "refusal_rate_value": 36.20},
            "ZA": {"country": "South Africa", "refusal_rate_value": 16.90},
            "AR": {"country": "Argentina", "refusal_rate_value": 15.10},
        }

    # Real parser will go here later.
    fail("Real refusal parser not implemented yet. Leave REFUSAL_SOURCE_URL unset for the first manual test.")


# =========================
# MERGE + BUILD
# =========================

def build_country_context(
    issuance_map: Dict[str, Dict[str, Any]],
    refusal_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    all_codes = sorted(set(issuance_map.keys()) | set(refusal_map.keys()))
    output: Dict[str, Dict[str, Any]] = {}

    for code in all_codes:
        issuance = issuance_map.get(code, {})
        refusal = refusal_map.get(code, {})

        country = issuance.get("country") or refusal.get("country")
        if not country:
            fail(f"Missing country name for code {code}")

        issuance_value = issuance.get("issuance_volume_value")
        refusal_value = refusal.get("refusal_rate_value")

        if issuance_value is None:
            fail(f"Missing issuance_volume_value for {code} / {country}")

        if refusal_value is None:
            fail(f"Missing refusal_rate_value for {code} / {country}")

        output[code] = {
            "country": country,
            "issuance_volume": format_int(int(issuance_value)),
            "issuance_volume_value": int(issuance_value),
            "refusal_rate": format_pct(float(refusal_value)),
            "refusal_rate_value": float(refusal_value),
            "demand_tier": demand_tier(int(issuance_value)),
        }

    return output


# =========================
# VALIDATION
# =========================

def validate_country_context(data: Dict[str, Dict[str, Any]]) -> None:
    if len(data) < MIN_COUNTRY_COUNT:
        fail(f"Country count too low: {len(data)} found, need at least {MIN_COUNTRY_COUNT}")

    required_fields = {
        "country",
        "issuance_volume",
        "issuance_volume_value",
        "refusal_rate",
        "refusal_rate_value",
        "demand_tier",
    }

    for code, item in data.items():
        missing = required_fields - set(item.keys())
        if missing:
            fail(f"{code} missing fields: {sorted(missing)}")

        if not item["country"]:
            fail(f"{code} has blank country")

        if item["issuance_volume_value"] is None:
            fail(f"{code} has null issuance_volume_value")

        if not isinstance(item["issuance_volume_value"], int):
            fail(f"{code} issuance_volume_value must be int")

        refusal = item["refusal_rate_value"]
        if refusal is None:
            fail(f"{code} has null refusal_rate_value")

        if not isinstance(refusal, (int, float)):
            fail(f"{code} refusal_rate_value must be numeric")

        if refusal < 0 or refusal > 100:
            fail(f"{code} refusal_rate_value out of range: {refusal}")

        tier = item["demand_tier"]
        if tier not in {"high", "moderate", "lower"}:
            fail(f"{code} invalid demand_tier: {tier}")

    log(f"Validation passed for {len(data)} countries.")


# =========================
# WRITE
# =========================

def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    log(f"Wrote {path}")


# =========================
# MAIN
# =========================

def main() -> None:
    ensure_docs_dir()

    log("Starting build_country_context.py")
    issuance_map = parse_issuance_data()
    refusal_map = parse_refusal_data()

    country_context = build_country_context(issuance_map, refusal_map)
    validate_country_context(country_context)

    write_json(TEST_OUTPUT_PATH, country_context)
    log("Manual test build complete. Live file was NOT touched.")


if __name__ == "__main__":
    main()
