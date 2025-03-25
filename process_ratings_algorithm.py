import os
import pandas as pd

# ------------------------------------------------------------------
# 1) Ratio-based helpers
# ------------------------------------------------------------------
MARGIN_SCORES = {
    '3-0': (1.0, 0.0),
    '3-1': (0.85, 0.15),
    '3-2': (0.7, 0.3),
}

def expected_fraction(LA, LB):
    return LA / (LA + LB)

def ratio_update(LA, LB, A_actual, B_actual, k=0.05):
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

    # Rename column to match combined_player_results_df
    df_init.rename(columns={"Player": "Player Name"}, inplace=True)

    # Example: columns = ["Player", "Team", "Initial Rating", ...]
    player_data = {}  
    # structure: player_data[(player_name, team_name)] = {
    #   "rating": float,
    #   "matches_played": int
    # }

    for row in df_init.itertuples(index=False):
        p = getattr(row, "Player_Name", "").strip().lower()
        t = getattr(row, "Team", "").strip().lower()
        r = getattr(row, "Initial_Rating", 1000.0)
        if p and t:
            player_data[(p, t)] = {"rating": float(r), "matches_played": 0}

    # ------------------------------------------------------------------
    # 3) Read combined_player_results_df => skip unwanted rows
    # ------------------------------------------------------------------
    results_csv = os.path.join(folder, "combined_player_results_df.csv")
    df = pd.read_csv(results_csv)

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

    def get_or_create_player(player_name, team_name):
        p = player_name.strip().lower()
        t = team_name.strip().lower()
        key = (p, t)
        if key not in player_data:
            player_data[key] = {"rating": 1000.0, "matches_played": 0}
        return key

    # ------------------------------------------------------------------
    # 4) Process matches => only update on row where Result == "Win"
    # ------------------------------------------------------------------
    for idx, row in df.iterrows():
        # Only update ratings if this row is the "Win" row
        if row["Result"] != "Win":
            # We'll skip rating updates => leave columns blank/NaN
            continue

        winner_name = row["Player Name"]
        winner_team = row["Team"]
        loser_name  = row["Opponent Name"]
        loser_team  = row["Opponent Team"]
        score_str   = row["Score"]

        if score_str not in MARGIN_SCORES:
            continue

        # Margin fractions
        w_actual, l_actual = MARGIN_SCORES[score_str]

        # get or create each player's data
        w_key = get_or_create_player(winner_name, winner_team)
        l_key = get_or_create_player(loser_name, loser_team)

        w_rating_before = player_data[w_key]["rating"]
        l_rating_before = player_data[l_key]["rating"]

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
    updated_csv = os.path.join(folder, "combined_player_results_df_updated.csv")
    df.to_csv(updated_csv, index=False)

    # ------------------------------------------------------------------
    # 6) Output #2: Final ratio results => each player's final rating + matches
    # ------------------------------------------------------------------
    final_rows = []
    for (p, t), info in player_data.items():
        final_rows.append({
            "Player": p,
            "Team": t,
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