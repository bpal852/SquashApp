import os
import pandas as pd
import logging
import re

# Clear any existing handlers
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

# Set level to INFO
logger.setLevel(logging.INFO)

# Create a stream handler for the terminal and set a formatter
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ------------------------------------------------------------------
# Base ratings by division (same as initial‐ratings script)
# ------------------------------------------------------------------
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

def get_base_rating(division_str):
    d = str(division_str).lower().strip()
    if d in division_base_ratings:
        return division_base_ratings[d]
    m = re.match(r'^(\d+)', d)
    if m and m.group(1) in division_base_ratings:
        return division_base_ratings[m.group(1)]
    return 1000.0

# ------------------------------------------------------------------
# 1) Ratio-based helpers
# ------------------------------------------------------------------
MARGIN_SCORES = {
    '3-0': (0.67, 0.33),
    '3-1': (0.6, 0.4),
    '3-2': (0.55, 0.45),
}

def expected_fraction(LA, LB):
    return LA / (LA + LB)

def ratio_update(LA, LB, A_actual, B_actual, k=0.3):
    EA = expected_fraction(LA, LB)
    EB = 1 - EA

    LA_new = LA * (1 + k*(A_actual - EA))
    LB_new = LB * (1 + k*(B_actual - EB))

    # floor to avoid negative/zero
    if LA_new < 1: LA_new = 1
    if LB_new < 1: LB_new = 1

    return LA_new, LB_new

def main():
    folder = r"C:\Users\bpali\PycharmProjects\SquashApp\2024-2025"

    # ------------------------------------------------------------------
    # 2) Read initial ratings => (player, team) -> rating
    #    Also store matches played = 0
    # ------------------------------------------------------------------
    init_csv = os.path.join(folder, "all_initial_ratings.csv")
    df_init = pd.read_csv(init_csv)

    # Make sure column is integer
    df_init["HKS_No"] = pd.to_numeric(df_init["HKS_No"], errors="coerce")

    # Rename column name to avoid spaces
    df_init.rename(columns={"Initial Rating": "Initial_Rating"}, inplace=True)

    player_data = {}
    for row in df_init.itertuples(index=False):
        p = getattr(row, "Player", "").strip()  # proper name
        t = getattr(row, "Team", "").strip()
        r = getattr(row, "Initial_Rating", 1000.0)  # default rating is 1000.0
        hks = getattr(row, "HKS_No", None)
        if pd.isna(hks):
            continue  # If HKS_No is missing, you may want to skip this player entirely.
        key = int(hks)  # use the integer value as the key
        player_data[key] = {
            "player": p,
            "rating": float(r),
            "matches_played": 0,
            "teams": {t}
        }

    # ------------------------------------------------------------------
    # 3) Read combined_player_results_df => skip unwanted rows
    # ------------------------------------------------------------------
    results_csv = os.path.join(folder, "combined_player_results_df.csv")
    df = pd.read_csv(results_csv)

    # Rename "HKS No." and "Opponent HKS No." to avoid spaces
    df.rename(columns={"HKS No.": "HKS_No", "Opponent HKS No.": "Opponent_HKS_No"}, inplace=True) 

    # Make sure columns are integer
    df["HKS_No"] = pd.to_numeric(df["HKS_No"], errors="coerce")
    df["Opponent_HKS_No"] = pd.to_numeric(df["Opponent_HKS_No"], errors="coerce")

    # Sort by Match Date, then Division, then Rubber Number
    df = df.sort_values(['Match Date', 'Division', 'Team', 'Rubber Number']).reset_index(drop=True)

    # If Score is CR/WO => skip
    # If Player Name or Opponent Name = 'Unknown', skip
    skip_mask = (
        df["Player Name"].eq("Unknown") | 
        df["Opponent Name"].eq("Unknown") | 
        df["Score"].isin(["CR","WO"])
    )
    df = df[~skip_mask].copy()

    # Avoid converting date or now to see if it's a problem
    # df["Match Date"] = pd.to_datetime(df["Match Date"], dayfirst=True, errors="coerce")
    df.sort_values("Match Date", inplace=True)

    # We'll add new columns to track rating changes in the updated CSV
    df["Pre Winner Rating"] = pd.NA
    df["Post Winner Rating"] = pd.NA
    df["Pre Loser Rating"] = pd.NA
    df["Post Loser Rating"] = pd.NA
    df["Winner Matches Played"] = pd.NA
    df["Loser Matches Played"] = pd.NA

    def get_or_create_player(hks, player_name, team_name, division):
        # If HKS_No is missing, skip the row.
        if pd.isna(hks):
            logging.warning(f"HKS_No is missing for player '{player_name}' on team '{team_name}'. Skipping this row.")
            return None

        key = int(hks)
        logging.info(f"Processing player: '{player_name}' on team '{team_name}', key = '{key}'")

        if key not in player_data:
            # NEW: use division base rating instead of flat 1000
            initial = get_base_rating(division)
            player_data[key] = {
                "player": player_name,
                "rating": initial,
                "matches_played": 0,
                "teams": {team_name}
            }
        else:
            player_data[key]["teams"].add(team_name)

        return key


    # ------------------------------------------------------------------
    # 4) Process matches => only update on row where Result == "Win"
    # ------------------------------------------------------------------
    for idx, row in df.iterrows():
        # Only update ratings if this row is the "Win" row
        if row["Result"] != "Win":
            # We'll skip rating updates => leave columns blank/NaN
            continue

        # For the winner:
        winner_hks = row["HKS_No"]
        winner_name = row["Player Name"]
        winner_team = row["Team"]
        w_key = get_or_create_player(winner_hks, winner_name, winner_team, row["Division"])
        if w_key is None:
            continue  # Skip this row if winner's HKS_No is missing

        # For the loser:
        loser_hks = row["Opponent_HKS_No"]
        loser_name = row["Opponent Name"]
        loser_team = row["Opponent Team"]
        l_key = get_or_create_player(loser_hks, loser_name, loser_team, row["Division"])
        if l_key is None:
            continue  # Skip this row if loser's HKS_No is missing

        w_rating_before = player_data[w_key]["rating"]
        l_rating_before = player_data[l_key]["rating"]

        score_str = row["Score"]
        if score_str not in MARGIN_SCORES:
            continue
        w_actual, l_actual = MARGIN_SCORES[score_str]

        # ratio update
        w_rating_after, l_rating_after = ratio_update(w_rating_before, l_rating_before, w_actual, l_actual)

        # increment matches
        player_data[w_key]["matches_played"] += 1
        player_data[l_key]["matches_played"] += 1

        # store updated ratings
        player_data[w_key]["rating"] = w_rating_after
        player_data[l_key]["rating"] = l_rating_after

        # fill in new columns
        df.at[idx, "Pre Winner Rating"]   = w_rating_before
        df.at[idx, "Pre Loser Rating"]    = l_rating_before
        df.at[idx, "Post Winner Rating"]  = w_rating_after
        df.at[idx, "Post Loser Rating"]   = l_rating_after
        df.at[idx, "Winner Matches Played"] = player_data[w_key]["matches_played"]
        df.at[idx, "Loser Matches Played"]  = player_data[l_key]["matches_played"]

    # ------------------------------------------------------------------
    # 5) Output #1: Updated version of combined_player_results_df
    # ------------------------------------------------------------------
    # Drop rows where Result != "Win"
    df = df[df["Result"] == "Win"].copy()

    # Create relative difference column
    df["rel_diff"] = (df["Post Winner Rating"] - df["Pre Winner Rating"]) / df["Pre Winner Rating"] * 100.0

    updated_csv = os.path.join(folder, "combined_player_results_df_updated.csv")
    df.to_csv(updated_csv, index=False)

    # ------------------------------------------------------------------
    # 6) Output #2: Final ratio results => each player's final rating + matches
    # ------------------------------------------------------------------
    final_rows = []
    for key, info in player_data.items():
        final_rows.append({
            "Player": info.get("player", ""),
            "HKS_No": key,
            "Teams": ", ".join(sorted(info.get("teams", []))),
            "Final Rating": info["rating"],
            "Matches Played": info["matches_played"]
        })


    final_df = pd.DataFrame(final_rows)
    final_df.sort_values("Final Rating", ascending=False, inplace=True, ignore_index=True)

    ratio_csv = os.path.join(folder, "ratio_results.csv")
    final_df.to_csv(ratio_csv, index=False)
    print("Done!")
    print(f"Updated matches CSV => {updated_csv}")
    print(f"Final ratio results => {ratio_csv}")


if __name__ == "__main__":
    main()