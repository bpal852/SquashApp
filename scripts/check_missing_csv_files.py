"""
check_missing_csv_files.py

Searches for missing or empty CSV files across all subfolders of the 2025-2026 season
directory. Division names are loaded from config/divisions/2025-2026.json.

Structure handled:
  - Standard:  2025-2026/{category}/week_x/{division}_{suffix}.csv
  - Nested:    2025-2026/{category}/{subcategory}/week_x/{division}_{suffix}.csv
               e.g. team_win_percentage_breakdown/Away/week_1/

A CSV is considered "empty" if it contains no data rows (only a header, or is 0 bytes).
"""

import csv
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_divisions(config_path: Path) -> list:
    """Return a list of division name strings from the JSON config."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [div["name"] for div in data["divisions"]]


def is_csv_empty(file_path: Path) -> bool:
    """
    Return True if the CSV file has no data rows.
    A file is considered empty when it is 0 bytes, contains no lines at all,
    or contains only a single header row with no data beneath it.
    """
    if file_path.stat().st_size == 0:
        return True
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if any(cell.strip() for cell in row)]
        # <= 1 row means at most just a header
        return len(rows) <= 1
    except Exception:
        # If the file can't be read treat it as empty
        return True


def find_division_csv(week_dir: Path, division: str) -> Path | None:
    """
    Find the CSV file for *division* inside *week_dir*.
    Files follow the pattern: {division}_{anything}.csv
    A plain {division}.csv match is also accepted as a fallback.
    """
    for f in week_dir.iterdir():
        if f.suffix.lower() != ".csv":
            continue
        # Primary pattern: "Division 7_results_df.csv"
        if f.name.startswith(division + "_"):
            return f
        # Fallback: exact stem match
        if f.stem == division:
            return f
    return None


def load_division_max_weeks(season_dir: Path, divisions: list) -> dict:
    """
    Read schedules_df/week_0/{division}_schedules_df.csv for each division and
    return a dict mapping division name -> max value of the 'Match Week' column.
    Divisions whose schedule file is missing or unreadable are excluded from the
    returned dict (they will not be filtered out during checks).
    """
    schedules_week0 = season_dir / "schedules_df" / "week_0"
    max_weeks: dict = {}

    if not schedules_week0.is_dir():
        print(f"WARNING: schedules_df/week_0 not found – per-division week limits disabled.")
        return max_weeks

    for division in divisions:
        csv_file = find_division_csv(schedules_week0, division)
        if csv_file is None:
            continue
        try:
            with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                weeks = [
                    int(row["Match Week"])
                    for row in reader
                    if row.get("Match Week", "").strip().isdigit()
                ]
            if weeks:
                max_weeks[division] = max(weeks)
        except Exception as exc:
            print(f"WARNING: Could not read {csv_file.name}: {exc}")

    return max_weeks


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

def check_week_dir(
    week_dir: Path,
    divisions: list,
    week_num: int,
    division_max_weeks: dict,
) -> dict:
    """
    Check a single week_x directory for missing / empty division CSV files.

    Divisions are skipped when week_num exceeds their known maximum match week
    (sourced from schedules_df/week_0).

    Returns
    -------
    dict with keys:
        "missing"  – divisions that have no CSV file at all
        "empty"    – divisions whose CSV file exists but has no data rows
    """
    missing, empty = [], []
    for division in divisions:
        max_week = division_max_weeks.get(division)
        if max_week is not None and week_num > max_week:
            continue  # this week is beyond the division's schedule
        csv_file = find_division_csv(week_dir, division)
        if csv_file is None:
            missing.append(division)
        elif is_csv_empty(csv_file):
            empty.append(division)
    return {"missing": missing, "empty": empty}


def scan_for_week_dirs(folder: Path) -> list:
    """
    Recursively collect all week_x directories under *folder*, returning each
    as a (relative_display_path, absolute_path) tuple.

    The display path uses the folder's name as the root so that the output
    reads naturally, e.g. ``teams_df/week_3`` or
    ``team_win_percentage_breakdown/Away/week_1``.
    """
    results = []

    def _recurse(current: Path, rel: str):
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except PermissionError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.startswith("week_"):
                if entry.name == "week_0":
                    continue
                results.append((f"{rel}/{entry.name}", entry))
            else:
                _recurse(entry, f"{rel}/{entry.name}")

    _recurse(folder, folder.name)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    config_path = root_dir / "config" / "divisions" / "2025-2026.json"
    season_dir = root_dir / "2025-2026"

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not config_path.exists():
        print(f"ERROR: Config file not found:\n  {config_path}")
        return
    if not season_dir.exists():
        print(f"ERROR: Season directory not found:\n  {season_dir}")
        return

    divisions = load_divisions(config_path)
    print(f"Loaded {len(divisions)} divisions from config:")
    print(f"  {', '.join(divisions)}\n")

    # ------------------------------------------------------------------
    # Load per-division maximum match weeks from schedules_df/week_0
    # ------------------------------------------------------------------
    division_max_weeks = load_division_max_weeks(season_dir, divisions)
    if division_max_weeks:
        print("Division match-week limits (excluded beyond these weeks):")
        for div, mw in sorted(division_max_weeks.items(), key=lambda kv: kv[1]):
            print(f"  {div}: {mw}")
        print()

    # ------------------------------------------------------------------
    # Walk top-level category folders inside 2025-2026/
    # ------------------------------------------------------------------
    IGNORED_FOLDERS = {
        "played_every_game",
        "neutral_fixtures",
        "logs",
        "validation_reports",
        "awaiting_results",
        "unbeaten_players",
        "debug_html",
        "simulated_fixtures",
        "simulated_tables",
    }

    all_issues = []  # list of (display_path, missing_list, empty_list)

    try:
        top_level = sorted(season_dir.iterdir(), key=lambda p: p.name)
    except PermissionError:
        print(f"ERROR: Cannot read season directory: {season_dir}")
        return

    for category_dir in top_level:
        if not category_dir.is_dir():
            continue
        if category_dir.name in IGNORED_FOLDERS:
            continue

        week_entries = scan_for_week_dirs(category_dir)
        if not week_entries:
            # No week_x subdirectories found – skip this folder
            continue

        for display_path, week_dir in week_entries:
            week_num = int(week_dir.name.split("_")[1])
            issues = check_week_dir(week_dir, divisions, week_num, division_max_weeks)
            if issues["missing"] or issues["empty"]:
                all_issues.append((display_path, issues["missing"], issues["empty"]))

    # Sort results: first by category path, then by numeric week
    def sort_key(item):
        parts = item[0].split("/")
        week_num = int(parts[-1].split("_")[1]) if parts[-1].startswith("week_") else 0
        return (parts[:-1], week_num)

    all_issues.sort(key=sort_key)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_missing = sum(len(m) for _, m, _ in all_issues)
    total_empty = sum(len(e) for _, _, e in all_issues)

    separator = "=" * 70

    if not all_issues:
        print(f"{separator}")
        print("All CSV files are present and non-empty. No issues found.")
        print(separator)
        return

    print(separator)
    print(
        f"SUMMARY  |  {total_missing} missing CSV file(s)  |  "
        f"{total_empty} empty CSV file(s)  |  "
        f"{len(all_issues)} affected location(s)"
    )
    print(separator)
    print()

    current_category = None
    for display_path, missing, empty in all_issues:
        # Print a category header when the category changes
        category = display_path.rsplit("/week_", 1)[0]
        if category != current_category:
            current_category = category
            print(f"[ {category} ]")

        week_label = display_path.split("/")[-1]
        print(f"  {week_label}")
        if missing:
            print(f"    MISSING ({len(missing)}): {', '.join(missing)}")
        if empty:
            print(f"    EMPTY   ({len(empty)}): {', '.join(empty)}")

    print()
    print(separator)
    print(f"Total missing: {total_missing}  |  Total empty: {total_empty}")
    print(separator)


if __name__ == "__main__":
    main()
