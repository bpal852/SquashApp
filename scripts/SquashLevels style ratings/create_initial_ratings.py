import pandas as pd
import numpy as np
import re
import os
import glob

# ---------------------------------------------------------------------
# 1. Define division -> base rating mapping
# ---------------------------------------------------------------------
division_base_ratings = {
    # Numeric divisions
    '2': 8000,
    '3': 6000,
    '4': 4000,
    '5': 3000,
    '6': 2000,
    '7': 1800,
    '8': 1600,
    '9': 1400,
    '10': 1200,
    '11': 1000,
    '12': 800,
    '13': 600,
    '14': 400,
    '15': 200,
    # Special names
    'premier main': 18000,
    'premier masters': 6000,
    'm2': 2000,
    'm3': 1600,
    'm4': 1000,
    'premier ladies': 6000,
    'l2': 2000,
    'l3': 1200,
    'l4': 800,
}

# ---------------------------------------------------------------------
# 1a. Exceptions: manual overrides for certain players
# ---------------------------------------------------------------------
# Map HKS_No (int) to forced initial rating
exception_ratings = {
    29779: 8000,   # Tulloch Gillem
    32455: 2500,   # Gates Stuart
}

# ---------------------------------------------------------------------
# League-type helper functions
# ---------------------------------------------------------------------
def get_league_type(division_str_lower):
    main_divs = {
        'premier main', '2', '3', '4', '5', '6', '7', '8',
        '9', '10', '11', '12', '13', '14', '15'
    }
    masters_divs = {'premier masters', 'm2', 'm3', 'm4'}
    ladies_divs = {'premier ladies', 'l2', 'l3', 'l4'}

    if division_str_lower in main_divs:
        return 'main'
    elif division_str_lower in masters_divs:
        return 'masters'
    elif division_str_lower in ladies_divs:
        return 'ladies'
    else:
        return 'unknown'

def parse_division_from_filename(filepath):
    base = os.path.basename(filepath)
    name, _ = os.path.splitext(base)
    division_str = name.replace("_players_df", "").strip().lower()
    match = re.match(r'^(\d+)', division_str)
    if match:
        division_str = match.group(1)
    return division_str

# ---------------------------------------------------------------------
# Base rating lookup
# ---------------------------------------------------------------------
def get_base_rating(division_str):
    if division_str in division_base_ratings:
        return division_base_ratings[division_str]
    for d in division_base_ratings:
        if d in division_str:
            return division_base_ratings[d]
    return 200

# ---------------------------------------------------------------------
# Initial rating calculation
# ---------------------------------------------------------------------
def compute_linear_median_rating(base_rating, rank_i, total_in_team, alpha=0.15):
    if total_in_team <= 1:
        return base_rating
    m = (total_in_team + 1) / 2.0
    factor = 1.0 + alpha * ((m - rank_i) / (m - 1))
    return base_rating * factor

# ---------------------------------------------------------------------
# Process one division file, applying overrides
# ---------------------------------------------------------------------
def process_division_file(filepath, player_ratings, name_to_hks_set, hks_to_name_set):
    division_str = parse_division_from_filename(filepath)
    df_div = pd.read_csv(filepath)
    if df_div.empty:
        return
    if "HKS No." in df_div.columns:
        df_div.rename(columns={"HKS No.": "HKS_No"}, inplace=True)

    base = get_base_rating(division_str)
    for team_name, group in df_div.groupby("Team", as_index=False):
        N = len(group)
        group_sorted = group.sort_values("Order", ascending=True)
        for row in group_sorted.itertuples(index=False):
            player_name = (row.Player or "").strip()
            hks_no = int(row.HKS_No)
            if not player_name:
                continue

            # Track conflicts
            name_to_hks_set.setdefault(player_name, set()).add(hks_no)
            hks_to_name_set.setdefault(hks_no, set()).add(player_name)
            if len(name_to_hks_set[player_name]) > 1:
                print(f"Warning: Multiple HKS numbers for player '{player_name}': {name_to_hks_set[player_name]}")
            if len(hks_to_name_set[hks_no]) > 1:
                print(f"Warning: Multiple names for HKS number '{hks_no}': {hks_to_name_set[hks_no]}")

            player_key = (player_name, hks_no)
            if player_key in player_ratings:
                continue

            # Check for manual override first
            if hks_no in exception_ratings:
                init_rating = exception_ratings[hks_no]
            else:
                init_rating = compute_linear_median_rating(base, int(row.Order), N, alpha=0.2)

            player_ratings[player_key] = {
                "Player": player_name,
                "HKS_No": hks_no,
                "Division": division_str,
                "Team": team_name,
                "Initial Rating": init_rating
            }

# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
folder = r"C:\Users\bpali\PycharmProjects\SquashApp\2024-2025\players_df\week_1"
pattern = os.path.join(folder, "*_players_df.csv")
filepaths = glob.glob(pattern)

main_files, masters_files, ladies_files = [], [], []
for fp in filepaths:
    d_str = parse_division_from_filename(fp)
    lt = get_league_type(d_str)
    if lt == 'main': main_files.append(fp)
    elif lt == 'masters': masters_files.append(fp)
    elif lt == 'ladies': ladies_files.append(fp)

player_ratings = {}
name_to_hks_set, hks_to_name_set = {}, {}

for fp in main_files:
    process_division_file(fp, player_ratings, name_to_hks_set, hks_to_name_set)
for fp in masters_files:
    process_division_file(fp, player_ratings, name_to_hks_set, hks_to_name_set)
for fp in ladies_files:
    process_division_file(fp, player_ratings, name_to_hks_set, hks_to_name_set)

rows = []
for (player_name, hks_no), info in player_ratings.items():
    rows.append([player_name, hks_no, info["Division"], info["Team"], info["Initial Rating"]])
final_df = pd.DataFrame(rows, columns=["Player", "HKS_No", "Division", "Team", "Initial Rating"])
final_df.sort_values(by=["Initial Rating", "Division"], ascending=[False, True], inplace=True, ignore_index=True)

outpath = os.path.join(r"C:\Users\bpali\PycharmProjects\SquashApp\2024-2025", "all_initial_ratings.csv")
final_df.to_csv(outpath, index=False)
print(f"Done. {len(final_df)} players in all_initial_ratings.csv.")
