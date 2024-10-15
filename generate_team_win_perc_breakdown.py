import pandas as pd
import os
import re
import glob
import logging
import numpy as np
import ast

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("squash_app_debug.log"),
        logging.StreamHandler()
    ]
)

# Base path for previous seasons' data
base_path = r"C:\Users\bpali\PycharmProjects\SquashApp\previous_seasons"

# Exclude the 2023-2024 season
excluded_season = '2023-2024'

# Get all season folders except for 2023-2024
season_folders = [folder for folder in glob.glob(os.path.join(base_path, '*')) if '2023-2024' not in folder]

# Iterate through each season directory
for season in season_folders:
    season_path = os.path.join(base_path, season)
    if not os.path.isdir(season_path) or season == excluded_season:
        continue

    # Create team_win_percentage_breakdown folder for the current season if it doesn't exist
    team_win_percentage_breakdown_path = os.path.join(season_path, "team_win_percentage_breakdown")
    if not os.path.exists(team_win_percentage_breakdown_path):
        os.makedirs(team_win_percentage_breakdown_path)

    # Iterate through each division for the given season
    results_path = os.path.join(season_path, "results_df")

    # Find all results_df files in the season
    for results_file in glob.glob(os.path.join(results_path, "*.csv")):
        # Extract the division number from the filename
        match = re.search(r"([A-Za-z0-9]*)_results_df.csv", os.path.basename(results_file))
        if match:
            division_number = match.group(1)
        else:
            logging.warning(f"File name does not match expected pattern: {results_file}")
            continue  # Skip this file if it doesn't match the pattern

        # Load DataFrames
        results_df = pd.read_csv(results_file)

        # Define functions

        def determine_winner(rubber_score, home_team, away_team):
            """
            Function to determine the winner of a rubber
            """
            if pd.isna(rubber_score) or rubber_score in ['CR', 'WO']:
                return pd.NA
            home_score, away_score = map(int, rubber_score.split('-'))
            return home_team if home_score > away_score else away_team
        
        def count_valid_matches(df, rubber_index):
            """
            Function to count matches excluding 'NA', 'CR', and 'WO'
            """
            valid_matches_count = {}
            for _, row in df.iterrows():
                if (rubber_index < len(row['Rubbers']) and
                    pd.notna(row['Rubbers'][rubber_index]) and
                        row['Rubbers'][rubber_index] not in ['CR', 'WO']):
                    valid_matches_count[row['Home Team']] = valid_matches_count.get(row['Home Team'], 0) + 1
                    valid_matches_count[row['Away Team']] = valid_matches_count.get(row['Away Team'], 0) + 1
            return valid_matches_count
        

        # Convert string representation of list to actual list if needed
        results_df['Rubbers'] = results_df['Rubbers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Find the maximum number of rubbers in any match
        max_rubbers = results_df['Rubbers'].apply(len).max()

        # Apply the determine_winner function to each rubber in the list
        for i in range(1, max_rubbers + 1):
            rubber_column = f'Rubber {i}'
            results_df[f'Winner {rubber_column}'] = results_df.apply(
                lambda row: determine_winner(row['Rubbers'][i - 1] if i - 1 < len(row['Rubbers']) else pd.NA,
                                            row['Home Team'], row['Away Team']), axis=1)
            
        # Aggregate the number of wins for each team in each rubber
        aggregate_wins = pd.DataFrame()
        for i in range(1, max_rubbers + 1):
            rubber_column = f'Rubber {i}'
            winner_column = f'Winner {rubber_column}'
            wins = results_df[winner_column].value_counts().rename(f'Wins in {rubber_column}')
            aggregate_wins = pd.concat([aggregate_wins, wins], axis=1)

        # Fill NaN values in aggregate wins with zeros
        aggregate_wins.fillna(0, inplace=True)

        # Convert wins to integer type
        aggregate_wins = aggregate_wins.astype(int)

        # Calculate total matches for each rubber excluding 'NA', 'CR', and 'WO'
        total_matches_per_rubber = {
            f'Rubber {i}': count_valid_matches(results_df, i - 1) for i in range(1, max_rubbers + 1)
        }

        # Convert the dictionary to a DataFrame with teams as index
        total_matches_df = pd.DataFrame(total_matches_per_rubber)

        # Properly merge total matches and aggregate wins based on team names
        combined = aggregate_wins.merge(total_matches_df, left_index=True, right_index=True, how='outer')

        # Replace NaN in wins and total matches with 0
        combined.fillna(0, inplace=True)

        # Calculate win percentage
        for i in range(1, max_rubbers + 1):
            rubber_column = f'Rubber {i}'
            combined[f'{rubber_column} Win %'] = (combined[f'Wins in {rubber_column}'] / combined[rubber_column]) * 100

        # Replace NaN in win % columns 0
        combined.fillna(0, inplace=True)

        # Calculate average win percentage
        combined["avg_win_perc"] = combined[[f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)]].mean(axis=1)

        # Calculate total games played by summing all the rubber matches for each team
        combined["Total Rubbers"] = total_matches_df.sum(axis=1)

        # Sort by total wins
        combined_sorted = combined.sort_values("avg_win_perc", ascending=False)

        # Reset the index
        combined_sorted = combined_sorted.reset_index().rename(columns={'index': 'Team'})

        # Filter out unnecessary columns
        keep_columns = (
                ["Team"] +
                [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] +
                ['avg_win_perc', "Total Rubbers"]
        )

        # Select only the win percentage columns and the avg_win_perc column
        win_percentage_df = combined_sorted[keep_columns]

        # Save the win percentage DataFrame to a CSV file
        output_file = os.path.join(team_win_percentage_breakdown_path, f"{division_number}_team_win_percentage_breakdown.csv")
        win_percentage_df.to_csv(output_file, index=False)

        # Add logging
        logging.info(f"Saved team win percentage breakdown for division {division_number} in season {season} to {output_file}")
