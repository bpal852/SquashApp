#!/usr/bin/env python3
"""
show_latest_awaiting_results.py

- Loads divisions from config/divisions/{season}.json
- Scans {REPO_ROOT}/{season}/awaiting_results/week_*
- For each division, loads the most recent available {division}_awaiting_results.csv
- Concatenates and sorts by Date, prints a compact view, and writes a combined CSV
"""

import os
import re
import json
import glob
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------
# Config / Paths / Logging
# ---------------------------

DEFAULT_SEASON = "2025-2026"  # single source of truth (overridable via --season)

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("awaiting_results")

logger = setup_logging()

REPO_ROOT = Path(os.getenv("SQUASHAPP_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_DIR = REPO_ROOT / "config" / "divisions"

def season_dir(season: str) -> Path:
    return REPO_ROOT / season

# ---------------------------
# Divisions loader
# ---------------------------

def load_divisions_for_season(season: str) -> Dict[str, int]:
    """
    Load divisions JSON and coerce into {division_name: division_id}.
    Accepts either a mapping or a list of {name,id} objects.
    """
    path = CONFIG_DIR / f"{season}.json"
    if not path.exists():
        raise FileNotFoundError(f"Divisions JSON not found: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # Handle new format: {"season": "2025-2026", "divisions": [...]}
    if isinstance(data, dict) and "divisions" in data:
        data = data["divisions"]

    if isinstance(data, dict):
        return {str(k): int(v) for k, v in data.items()}

    if isinstance(data, list):
        out: Dict[str, int] = {}
        for row in data:
            name = row.get("name") or row.get("division") or row.get("title")
            div_id = row.get("id") or row.get("division_id")
            if name is None or div_id is None:
                raise ValueError(
                    "Divisions JSON entries must include 'name' and 'id' (or 'division'/'division_id')."
                )
            out[str(name)] = int(div_id)
        return out

    raise TypeError(f"Unsupported divisions JSON shape in {path}")

# ---------------------------
# Helpers
# ---------------------------

WEEK_DIR_RE = re.compile(r"^week_(\d+)$")
FILE_RE = re.compile(r"^(.*)_awaiting_results\.csv$")

def find_week_dirs(awaiting_dir: Path) -> List[Path]:
    """Return week_* dirs sorted DESC by week number."""
    candidates = [p for p in awaiting_dir.glob("week_*") if p.is_dir()]
    parsed = []
    for p in candidates:
        m = WEEK_DIR_RE.match(p.name)
        if m:
            parsed.append((int(m.group(1)), p))
    parsed.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in parsed]

def parse_week_number(p: Path) -> Optional[int]:
    m = WEEK_DIR_RE.match(p.name)
    return int(m.group(1)) if m else None

# ---------------------------
# Core
# ---------------------------

def load_latest_awaiting_per_division(season: str, write_combined: bool = True) -> pd.DataFrame:
    """
    For each division, walk week_* in DESC order and load the first {division}_awaiting_results.csv seen.
    """
    divisions = load_divisions_for_season(season)
    logger.info("Loaded %d divisions for season %s", len(divisions), season)

    sdir = season_dir(season)
    awaiting_dir = sdir / "awaiting_results"
    if not awaiting_dir.exists():
        logger.info("No awaiting_results directory found at %s; nothing to show.", awaiting_dir)
        return pd.DataFrame()

    week_dirs = find_week_dirs(awaiting_dir)
    if not week_dirs:
        logger.info("No week_* folders found under %s; nothing to show.", awaiting_dir)
        return pd.DataFrame()

    divisions_to_load = set(divisions.keys())
    frames: List[pd.DataFrame] = []

    for wdir in week_dirs:
        if not divisions_to_load:
            break

        csvs = list(wdir.glob("*.csv"))
        if not csvs:
            continue

        for csv_path in csvs:
            m = FILE_RE.match(csv_path.name)
            if not m:
                logger.warning("Unexpected file in %s: %s", wdir, csv_path.name)
                continue

            division_name = m.group(1)
            # only accept division files that exist in our season's list
            if division_name not in divisions_to_load:
                continue

            try:
                # Parse dates robustly if 'Date' column exists; otherwise load raw.
                parse_dates = ["Date"] if "Date" in pd.read_csv(csv_path, nrows=0).columns else None
                df = pd.read_csv(csv_path, parse_dates=parse_dates)
                df["Division"] = division_name
                df["Week"] = parse_week_number(wdir)
                frames.append(df)
                divisions_to_load.remove(division_name)
                logger.info("Loaded %s (week %s)", division_name, df["Week"].iloc[0] if "Week" in df else "?")
            except Exception as e:
                logger.error("Failed reading %s: %s", csv_path, e)

    if not frames:
        logger.info("No awaiting results found for any division.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Sort by Date if present; otherwise by Week then Division then Home Team
    if "Date" in combined.columns:
        combined.sort_values(by=["Date", "Division"], inplace=True, kind="mergesort")
    else:
        sort_cols = [c for c in ["Week", "Division", "Home Team"] if c in combined.columns]
        if sort_cols:
            combined.sort_values(by=sort_cols, inplace=True, kind="mergesort")

    # Optionally write a combined artifact
    if write_combined:
        out_dir = awaiting_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latest_awaiting_results_all_divisions.csv"
        try:
            combined.to_csv(out_path, index=False)
            logger.info("Wrote combined awaiting results -> %s", out_path)
        except Exception as e:
            logger.error("Could not write combined CSV: %s", e)

    return combined

# ---------------------------
# CLI / Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Show latest awaiting results across all divisions.")
    ap.add_argument("--season", default=DEFAULT_SEASON, help="Season string, e.g. 2025-2026")
    ap.add_argument("--no-write", action="store_true", help="Do not write the combined CSV artifact")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    df = load_latest_awaiting_per_division(args.season, write_combined=not args.no_write)

    if df.empty:
        print("No awaiting results to display.")
        return

    cols = [c for c in ["Home Team", "Away Team", "Division", "Date"] if c in df.columns]
    if cols:
        print("\nAwaiting Results:")
        print(df[cols].to_string(index=False))
    else:
        # Fallback: show a compact head if expected columns are absent
        print("\nAwaiting Results (first 20 rows):")
        print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
