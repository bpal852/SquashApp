# Imports
import os
import glob
import re
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define paths
base_directory = os.path.dirname(os.path.abspath(__file__))
current_season = "2025-2026"
out_dir = os.path.join(base_directory, current_season)
os.makedirs(out_dir, exist_ok=True)


def load_all_results_and_player_results(season_base_path):
    """
    Load all results_df and player_results CSVs from all week_* folders and combine them, removing duplicates.
    """
    all_results = []
    all_player_results = []

    # Paths to the results_df and player_results directories
    results_df_dir = os.path.join(season_base_path, "results_df")
    player_results_dir = os.path.join(season_base_path, "player_results")

    # Find all week_* folders
    results_week_folders = glob.glob(os.path.join(results_df_dir, "week_*"))
    player_results_week_folders = glob.glob(os.path.join(player_results_dir, "week_*"))

    # Load all results_df files
    for week_folder in results_week_folders:
        csv_files = glob.glob(os.path.join(week_folder, "*.csv"))
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Extract division from filename
                filename = os.path.basename(file)
                match = re.match(r"(.*)_results_df\.csv", filename)
                if match:
                    division = match.group(1)
                else:
                    division = "Unknown"
                df['Division'] = division
                # Optionally, add week information if needed
                all_results.append(df)
                # Check file to see if a row is empty except for Division column
                if df.drop(columns=['Division']).isnull().all(axis=1).any():
                    logging.warning(f"File {file} contains empty rows except for Division column.")
            except Exception as e:
                logging.warning(f"Error reading file {file}: {e}")

    # Load all player_results files
    for week_folder in player_results_week_folders:
        csv_files = glob.glob(os.path.join(week_folder, "*.csv"))
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Extract division from filename
                filename = os.path.basename(file)
                match = re.match(r"(.*)_player_results\.csv", filename)
                if match:
                    division = match.group(1)
                else:
                    division = "Unknown"
                df['Division'] = division
                # Optionally, add week information if needed
                all_player_results.append(df)
            except Exception as e:
                logging.warning(f"Error reading file {file}: {e}")

    # Combine all results into a single DataFrame, remove duplicates
    if all_results:
        combined_results_df = pd.concat(all_results, ignore_index=True).drop_duplicates()
    else:
        combined_results_df = pd.DataFrame()

    if all_player_results:
        combined_player_results_df = pd.concat(all_player_results, ignore_index=True).drop_duplicates()
    else:
        combined_player_results_df = pd.DataFrame()

    # Convert Division to string and strip spaces
    for df in (combined_results_df, combined_player_results_df):
        if not df.empty and 'Division' in df.columns:
            df['Division'] = df['Division'].astype(str).str.strip()

    # Make sure Match Date column is in datetime format (and day first)
    if not combined_player_results_df.empty:
        if 'Match Date' in combined_player_results_df.columns:
            combined_player_results_df['Match Date'] = pd.to_datetime(
                combined_player_results_df['Match Date'], dayfirst=True, errors='coerce'
            )
        sort_cols_pr = [c for c in ['Match Date', 'Division', 'Team', 'Rubber Number']
                        if c in combined_player_results_df.columns]
        if sort_cols_pr:
            combined_player_results_df = combined_player_results_df.sort_values(sort_cols_pr).reset_index(drop=True)

    # Sort combined_results_df by Date, then Match Week, then Division, then Home Team
    if not combined_results_df.empty:
        if 'Date' in combined_results_df.columns:
            combined_results_df['Date'] = pd.to_datetime(
                combined_results_df['Date'], dayfirst=True, errors='coerce'
            )
        sort_cols_r = [c for c in ['Date', 'Match Week', 'Division', 'Home Team']
                    if c in combined_results_df.columns]
        if sort_cols_r:
            combined_results_df = combined_results_df.sort_values(sort_cols_r).reset_index(drop=True)

    return combined_results_df, combined_player_results_df


if __name__ == "__main__":
    season_base_path = out_dir
    combined_results_df, combined_player_results_df = load_all_results_and_player_results(season_base_path)
    combined_results_df.to_csv(os.path.join(season_base_path, "combined_results_df.csv"), index=False)
    combined_player_results_df.to_csv(os.path.join(season_base_path, "combined_player_results_df.csv"), index=False)
    logging.info("Combined data saved.")