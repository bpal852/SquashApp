import os
import pandas as pd
import logging
import re
from pathlib import Path

# ------------------------------------------------------------------
# Base ratings by division (same as initialâ€ratings script)
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
    'premier masters': 4000, # div 4
    'm2': 2000, # div 6
    'm3': 1800, # div 7
    'm4': 1200, # div 10
    'premier ladies': 8000, # div 2
    'l2': 2000, # div 6
    'l3': 1600, # div 8
    'l4': 600, # div 13
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

def process_ratings_algorithm(base_folder=None, current_season="2025-2026", previous_season="2024-2025"):
    """
    Process player ratings using ratio-based algorithm.
    Carries over ratings from previous season and processes current season matches.
    
    Args:
        base_folder: Base folder path (defaults to script location's parent)
        current_season: Current season folder name
        previous_season: Previous season folder name
    """
    # ------------------------------------------------------------------
    # Season configuration
    # ------------------------------------------------------------------
    if base_folder is None:
        base_folder = Path(__file__).parent.parent.parent  # Go up to repo root
    else:
        base_folder = Path(base_folder)
    
    previous_season_folder = base_folder / "previous_seasons" / previous_season
    current_season_folder = base_folder / current_season

    # ------------------------------------------------------------------
    # 2) Read initial ratings from previous season's final results
    #    If a player doesn't exist, use division-based initial rating
    # ------------------------------------------------------------------
    logging.info("="*70)
    logging.info("PROCESSING PLAYER RATINGS ALGORITHM")
    logging.info("="*70)
    logging.info(f"Loading final ratings from {previous_season} season...")
    
    # Try to load previous season's final ratings
    previous_ratings_csv = previous_season_folder / "ratio_results.csv"
    
    player_data = {}
    
    if os.path.exists(previous_ratings_csv):
        df_prev = pd.read_csv(previous_ratings_csv)
        logging.info(f"Loaded {len(df_prev)} players from previous season")
        logging.info(f"Columns in previous season data: {df_prev.columns.tolist()}")
        
        # Load previous season's ratings as starting point
        # Use iterrows instead of itertuples to avoid column name conversion issues
        for idx, row in df_prev.iterrows():
            hks = row.get("HKS_No", None)
            if pd.isna(hks):
                continue
            
            key = int(hks)
            # Access column with exact name including space
            final_rating = row.get("Final Rating", 1000.0)
            
            player_data[key] = {
                "player": row.get("Player", "").strip(),
                "rating": float(final_rating),
                "matches_played": 0,  # Reset match count for new season
                "teams": set()  # Will be populated from current season data
            }
        
        logging.info(f"Initialized {len(player_data)} players with ratings from 2024-2025")
        # Log a sample of loaded players
        sample_keys = list(player_data.keys())[:5]
        for key in sample_keys:
            logging.info(f"  Sample: HKS {key} = {player_data[key]['player']} with rating {player_data[key]['rating']:.2f}")
    else:
        logging.warning(f"Previous season ratings not found at {previous_ratings_csv}")
        logging.warning("Will use division-based initial ratings for all players")
    
    # Also check if current season has initial ratings for NEW players
    current_init_csv = current_season_folder / "all_initial_ratings.csv"
    if os.path.exists(current_init_csv):
        df_current_init = pd.read_csv(current_init_csv)
        logging.info(f"Found {len(df_current_init)} players in current season initial ratings")
        
        # For any NEW players not in player_data, add them with initial ratings
        df_current_init["HKS_No"] = pd.to_numeric(df_current_init["HKS No."], errors="coerce")
        df_current_init.rename(columns={"Initial Rating": "Initial_Rating"}, inplace=True)
        
        new_players_count = 0
        for row in df_current_init.itertuples(index=False):
            hks = getattr(row, "HKS_No", None)
            if pd.isna(hks):
                continue
            
            key = int(hks)
            if key not in player_data:
                player_data[key] = {
                    "player": getattr(row, "Player", "").strip(),
                    "rating": float(getattr(row, "Initial_Rating", 1000.0)),
                    "matches_played": 0,
                    "teams": {getattr(row, "Team", "").strip()}
                }
                new_players_count += 1
        
        logging.info(f"Added {new_players_count} new players from current season initial ratings")
    else:
        logging.info(f"No initial ratings file found at {current_init_csv}")

    # ------------------------------------------------------------------
    # 3) Read combined_player_results_df from CURRENT SEASON => skip unwanted rows
    # ------------------------------------------------------------------
    results_csv = current_season_folder / "combined_player_results_df.csv"
    logging.info(f"Loading match results from {results_csv}")
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

        if key not in player_data:
            # NEW player not in previous season data: use division base rating
            initial = get_base_rating(division)
            player_data[key] = {
                "player": player_name,
                "rating": initial,
                "matches_played": 0,
                "teams": {team_name}
            }
            logging.info(f"New player: '{player_name}' (HKS {key}) initialized with rating {initial} based on division {division}")
        else:
            # Existing player: just add the team
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

    logging.info(f"Processed {len(df[df['Result'] == 'Win'])} matches")
    logging.info(f"Total players in database: {len(player_data)}")
    
    # Count how many players have played matches this season
    players_with_matches = sum(1 for p in player_data.values() if p['matches_played'] > 0)
    logging.info(f"Players with matches in 2025-2026: {players_with_matches}")

    # ------------------------------------------------------------------
    # 5) Output #1: Updated version of combined_player_results_df
    # ------------------------------------------------------------------
    # Drop rows where Result != "Win"
    df = df[df["Result"] == "Win"].copy()

    # Create relative difference column
    df["rel_diff"] = (df["Post Winner Rating"] - df["Pre Winner Rating"]) / df["Pre Winner Rating"] * 100.0

    updated_csv = current_season_folder / "combined_player_results_df_updated.csv"
    df.to_csv(updated_csv, index=False)
    logging.info(f"Saved updated matches to {updated_csv}")

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

    ratio_csv = current_season_folder / "ratio_results.csv"
    final_df.to_csv(ratio_csv, index=False)
    logging.info("="*70)
    logging.info(f"Updated matches CSV => {updated_csv}")
    logging.info(f"Final ratio results => {ratio_csv}")
    logging.info("="*70)
    logging.info("PLAYER RATINGS ALGORITHM COMPLETED")
    logging.info("="*70)


if __name__ == "__main__":
    process_ratings_algorithm()
