#!/usr/bin/env python3

import csv
import json
from datetime import date
from pathlib import Path
from typing import Dict, Any


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = BASE_DIR / "source_data" / "uscis_source.csv"
OUTPUT_PATH = BASE_DIR / "docs" / "uscis_context.json"

MIN_ROWS = 4


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def format_range(min_value: float, max_value: float) -> str:
    min_clean = int(min_value) if float(min_value).is_integer() else min_value
    max_clean = int(max_value) if float(max_value).is_integer() else max_value
    return f"{min_clean}–{max_clean} months"


def read_source_csv() -> Dict[str, Dict[str, Any]]:
    if not SOURCE_PATH.exists():
        fail(f"Missing USCIS source CSV: {SOURCE_PATH}")

    output = {}

    with SOURCE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "visa_key",
            "label",
            "processing_time_min",
            "processing_time_max",
            "source",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            fail(f"USCIS source CSV missing columns: {sorted(missing)}")

        for row in reader:
            key = (row.get("visa_key") or "").strip().lower()
            label = (row.get("label") or "").strip()
            source = (row.get("source") or "").strip() or "U.S. Citizenship and Immigration Services"

            if not key or not label:
                continue

            try:
                min_val = float(row["processing_time_min"])
                max_val = float(row["processing_time_max"])
            except Exception:
                fail(f"Invalid processing range for {key}")

            if min_val < 0 or max_val < 0:
                fail(f"Negative processing range for {key}")

            if max_val < min_val:
                fail(f"Max is less than min for {key}")

            output[key] = {
                "label": label,
                "processing_time_min": min_val,
                "processing_time_max": max_val,
                "source": source,
            }

    return output


def validate(rows: Dict[str, Dict[str, Any]]) -> None:
    if len(rows) < MIN_ROWS:
        fail(f"Too few USCIS rows: {len(rows)}")

    for required_key in ["combined", "h", "l", "o"]:
        if required_key not in rows:
            fail(f"Missing required USCIS key: {required_key}")

    log(f"Validation passed for {len(rows)} USCIS rows.")


def build_context(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    today = date.today().isoformat()

    petition_based = {}

    for key in ["combined", "h", "l", "o"]:
        row = rows[key]

        if key == "combined":
            scope = "Broad USCIS processing context for H/L/O petition-based cases"
        else:
            scope = f"USCIS processing for {key.upper()} petition-based cases"

        petition_based[key] = {
            "label": row["label"],
            "uscis_form": "I-129",
            "category": "Temporary worker petition",
            "processing_time": format_range(
                row["processing_time_min"],
                row["processing_time_max"],
            ),
            "processing_time_min": row["processing_time_min"],
            "processing_time_max": row["processing_time_max"],
            "processing_time_unit": "months",
            "period": "latest available",
            "scope": scope,
            "status": "dynamic_range",
            "last_updated": today,
            "source": row["source"],
        }

    return {
        "version": "1.0",
        "generated_at": today,
        "source": "USCIS Case Processing Times",
        "source_url": "https://egov.uscis.gov/processing-times",
        "scope_note": (
            "USCIS processing times vary by form, petition category, and processing office. "
            "USCIS processing is separate from embassy interview wait times and may affect "
            "the total timeline before a petition-based visa interview can be scheduled."
        ),
        "petition_based": petition_based,
    }


def write_json(data: Dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"Wrote {OUTPUT_PATH}")


def main() -> None:
    log("Starting build_uscis_context.py")
    rows = read_source_csv()
    validate(rows)
    context = build_context(rows)
    write_json(context)
    log("USCIS context build complete.")


if __name__ == "__main__":
    main()
