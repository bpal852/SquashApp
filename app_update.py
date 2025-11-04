import glob
import json
import logging
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define the log format
    handlers=[logging.StreamHandler()],  # Also output logs to the console (optional)
)

# Suppress Matplotlib's DEBUG logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Suppress DEBUG messages from the 'watchdog' logger
logging.getLogger("watchdog").setLevel(logging.WARNING)

# Set page configurations
st.set_page_config(page_title="HK Squash App", page_icon="🇭🇰", layout="wide")

today = pd.Timestamp(datetime.now().date())

# Define the season
current_season = "2025-2026"

# Define the base directory
base_directory = os.path.dirname(os.path.abspath(__file__))

# Define the season directory
season_dir = os.path.join(base_directory, current_season)
os.makedirs(season_dir, exist_ok=True)  # ensure folder exists for writes


def get_available_seasons(base_directory, current_season):
    """
    Function to get list of available seasons
    """
    previous_seasons_dir = os.path.join(base_directory, "previous_seasons")
    if os.path.exists(previous_seasons_dir):
        previous_seasons = [
            d for d in os.listdir(previous_seasons_dir) if os.path.isdir(os.path.join(previous_seasons_dir, d))
        ]
    else:
        previous_seasons = []
    seasons = previous_seasons + [current_season]
    seasons.sort(reverse=True)  # Sort seasons in descending order
    return seasons


# List of available seasons
available_seasons = get_available_seasons(base_directory, current_season)

# Define the season base path
season_base_path = os.path.join(base_directory, current_season)

# Define the team win percentage base path
team_win_percentage_breakdown_path = os.path.join(season_base_path, "team_win_percentage_breakdown")

# Paths to subdirectories
team_win_percentage_breakdown_overall_path = os.path.join(team_win_percentage_breakdown_path, "Overall")
team_win_percentage_breakdown_home_path = os.path.join(team_win_percentage_breakdown_path, "Home")
team_win_percentage_breakdown_away_path = os.path.join(team_win_percentage_breakdown_path, "Away")
team_win_percentage_breakdown_delta_path = os.path.join(team_win_percentage_breakdown_path, "Delta")


def load_divisions_simple(base_dir: str, season: str) -> dict[str, int | None]:
    """
    Reads config/divisions/<season>.json and returns {"5": 476, ...}.
    Keeps ALL divisions; ignores 'enabled'.
    """
    cfg = Path(base_dir) / "config" / "divisions" / f"{season}.json"
    if not cfg.exists():
        return {}

    with cfg.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    out: dict[str, int | None] = {}
    for d in obj.get("divisions", []):
        name = d.get("name")
        if not name:
            continue
        _id = d.get("id")
        try:
            _id = int(_id) if _id is not None else None
        except Exception:
            _id = None
        out[name] = _id
    return out


all_divisions = (base_directory, current_season)

# List of clubs
clubs = [
    "Hong Kong Cricket Club",
    "Hong Kong Football Club",
    "Kowloon Cricket Club",
    "Ladies Recreation Club",
    "Royal Hong Kong Yacht Club",
    "United Services Recreation Club",
    "Fusion Squash Club",
    "Sha Tin",
    "X-Alpha",
    "TNG",
    "RELAY",
    "JESSICA",
    "i-Mask Advance Squash Club",
    "Vitality Squash",
    "Friend Club",
    "North District Sports Association",
    "Physical Chess",
    "Electrify Squash",
    "Global Squash",
    "Squashathon",
    "Hong Kong Racketlon Association",
    "The Squash Club",
    "Happy Squash",
    "Star River",
    "Kinetic",
    "The Hong Kong Jockey Club",
    "Young Player",
    "Hong Kong Club",
    "The Best Group",
    "Bravo",
    "Energy Squash Club",
    "HKIS",
    "NEXUS",
]


def find_latest_file_for_division(data_folder, division, filename_pattern):
    """
    Search for the latest file for a given division in the week folders under data_folder.
    If week_* folders do not exist, search directly in the data_folder.

    Args:
        data_folder (str): The base path to the data folder.
        division (str): The division name.
        filename_pattern (str): The filename pattern, e.g., "{}_detailed_league_table.csv"

    Returns:
        str or None: The full path to the latest file found for the division, or None if not found.
    """
    # Ensure data_folder exists
    if not os.path.exists(data_folder):
        logging.warning(f"Data folder does not exist: {data_folder}")
        return None

    # Check if week_* folders exist
    week_folders = glob.glob(os.path.join(data_folder, "week_*"))
    if week_folders:
        # Proceed as before, searching within week_* folders
        week_numbers = []
        for folder in week_folders:
            week_name = os.path.basename(folder)
            match = re.match(r"week_(\d+)", week_name)
            if match:
                week_number = int(match.group(1))
                week_numbers.append((week_number, folder))
        # Sort week folders by week number in descending order
        week_numbers.sort(reverse=True)
        sorted_week_folders = [folder for _, folder in week_numbers]
        # Loop over week folders
        for week_folder in sorted_week_folders:
            # Construct the file path
            filename = filename_pattern.format(division)
            file_path = os.path.join(week_folder, filename)
            if os.path.exists(file_path):
                logging.debug(f"Found file for division {division} at {file_path}")
                return file_path
        logging.warning(f"No file found for division {division} in week folders of {data_folder}")
    else:
        # No week_* folders; search directly in data_folder
        filename = filename_pattern.format(division)
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            logging.debug(f"Found file for division {division} at {file_path}")
            return file_path
        else:
            logging.warning(f"No file found for division {division} directly in {data_folder}")

    return None


def load_overall_home_away_data(division, season_base_path):
    """
    Load the overall home/away data for a given division.
    """
    # Define the base path for home_away_data
    home_away_data_base_path = os.path.join(season_base_path, "home_away_data")

    # Use find_latest_file_for_division to get the file path
    overall_home_away_path = find_latest_file_for_division(home_away_data_base_path, division, "{}_overall_scores.csv")

    if overall_home_away_path is None:
        logging.warning(f"No home/away data available for division {division}.")
        return None  # Or return an empty DataFrame/list as appropriate

    try:
        # Read the CSV file using Pandas
        overall_home_away = pd.read_csv(overall_home_away_path, header=None)
        logging.debug(f"Loaded overall_home_away data from {overall_home_away_path}")

        # If you need to extract the scores as a list
        # Assuming the scores are in the first row
        scores = overall_home_away.iloc[0].tolist()
        return scores

    except FileNotFoundError:
        logging.warning(f"File not found: {overall_home_away_path}")
        return None  # Or return an empty DataFrame/list as appropriate

    except Exception as e:
        logging.error(f"Error loading overall_home_away data from {overall_home_away_path}: {e}")
        return None  # Or return an empty DataFrame/list as appropriate


def load_csvs(division, season_base_path, is_current_season=True):
    try:
        logging.info(f"Loading CSVs for division {division} for season {season_base_path}")
        # Define the paths to the data folders
        detailed_league_tables_path = os.path.join(season_base_path, "detailed_league_tables")
        home_away_data_path = os.path.join(season_base_path, "home_away_data")
        team_win_percentage_breakdown_path = os.path.join(season_base_path, "team_win_percentage_breakdown")
        summarized_player_tables_path = os.path.join(season_base_path, "summarized_player_tables")
        ranking_df_path = os.path.join(season_base_path, "ranking_df")
        # For current season only
        simulated_tables_path = os.path.join(season_base_path, "simulated_tables") if is_current_season else None
        simulated_fixtures_path = os.path.join(season_base_path, "simulated_fixtures") if is_current_season else None
        awaiting_results_path = os.path.join(season_base_path, "awaiting_results") if is_current_season else None

        # Check if subfolders exist under team_win_percentage_breakdown_path
        overall_subfolder = os.path.join(team_win_percentage_breakdown_path, "Overall")
        if os.path.exists(overall_subfolder):
            # Subfolders exist
            team_win_percentage_breakdown_overall_path = overall_subfolder
            team_win_percentage_breakdown_home_path = os.path.join(team_win_percentage_breakdown_path, "Home")
            team_win_percentage_breakdown_away_path = os.path.join(team_win_percentage_breakdown_path, "Away")
            team_win_percentage_breakdown_delta_path = os.path.join(team_win_percentage_breakdown_path, "Delta")
            # Indicate that multiple breakdowns are available
            multiple_breakdowns_available = True
        else:
            # Subfolders do not exist; use the main folder and set others to None
            team_win_percentage_breakdown_overall_path = team_win_percentage_breakdown_path
            team_win_percentage_breakdown_home_path = None
            team_win_percentage_breakdown_away_path = None
            team_win_percentage_breakdown_delta_path = None
            multiple_breakdowns_available = False

        # Define the data files to load
        data_files = [
            {
                "key": "overall_home_away",
                "data_folder": home_away_data_path,
                "filename_pattern": "{}_overall_scores.csv",
                "read_params": {"header": None},
            },
            {
                "key": "home_away_df",
                "data_folder": home_away_data_path,
                "filename_pattern": "{}_team_average_scores.csv",
                "read_params": {},
            },
            {
                "key": "team_win_breakdown_overall",
                "data_folder": team_win_percentage_breakdown_overall_path,
                "filename_pattern": "{}_team_win_percentage_breakdown.csv",
                "read_params": {},
            },
        ]

        # Only add Home, Away, and Delta if they are available
        if multiple_breakdowns_available:
            data_files.extend(
                [
                    {
                        "key": "team_win_breakdown_home",
                        "data_folder": team_win_percentage_breakdown_home_path,
                        "filename_pattern": "{}_team_win_percentage_breakdown_home.csv",
                        "read_params": {},
                    },
                    {
                        "key": "team_win_breakdown_away",
                        "data_folder": team_win_percentage_breakdown_away_path,
                        "filename_pattern": "{}_team_win_percentage_breakdown_away.csv",
                        "read_params": {},
                    },
                    {
                        "key": "team_win_breakdown_delta",
                        "data_folder": team_win_percentage_breakdown_delta_path,
                        "filename_pattern": "{}_team_win_percentage_breakdown_delta.csv",
                        "read_params": {},
                    },
                ]
            )
        else:
            # Assign None or empty DataFrames to these keys
            data_files.extend(
                [
                    {
                        "key": "team_win_breakdown_home",
                        "data_folder": None,
                        "filename_pattern": None,
                        "read_params": {},
                    },
                    {
                        "key": "team_win_breakdown_away",
                        "data_folder": None,
                        "filename_pattern": None,
                        "read_params": {},
                    },
                    {
                        "key": "team_win_breakdown_delta",
                        "data_folder": None,
                        "filename_pattern": None,
                        "read_params": {},
                    },
                ]
            )

        # Add detailed league table
        data_files.append(
            {
                "key": "detailed_league_table",
                "data_folder": detailed_league_tables_path,
                "filename_pattern": "{}_detailed_league_table.csv",
                "read_params": {},
            }
        )

        # Handle summarized players differently based on the existence of summarized_player_tables_path
        if os.path.exists(summarized_player_tables_path):
            # Use summarized_player_tables_path with the format "{}_summarized_players.csv"
            data_files.append(
                {
                    "key": "summarized_players",
                    "data_folder": summarized_player_tables_path,
                    "filename_pattern": "{}_summarized_players.csv",
                    "read_params": {},
                }
            )
        else:
            # Use ranking_df_path with the format "{}_summarized_df.csv"
            data_files.append(
                {
                    "key": "summarized_players",
                    "data_folder": ranking_df_path,
                    "filename_pattern": "{}_summarized_df.csv",
                    "read_params": {},
                }
            )

        # Only load projections and awaiting results for current season
        if is_current_season:
            data_files.extend(
                [
                    {
                        "key": "final_table",
                        "data_folder": simulated_tables_path,
                        "filename_pattern": "{}_proj_final_table.csv",
                        "read_params": {},
                    },
                    {
                        "key": "fixtures",
                        "data_folder": simulated_fixtures_path,
                        "filename_pattern": "{}_proj_fixtures.csv",
                        "read_params": {},
                    },
                    {
                        "key": "awaiting_results",
                        "data_folder": awaiting_results_path,
                        "filename_pattern": "{}_awaiting_results.csv",
                        "read_params": {},
                    },
                    {
                        "key": "simulation_date",
                        "data_folder": simulated_tables_path,
                        "filename_pattern": "{}_simulation_date.txt",
                        "read_params": {},
                    },
                ]
            )
        else:
            # For previous seasons, assign empty DataFrames
            data_files.extend(
                [
                    {"key": "final_table", "data_folder": None, "filename_pattern": None, "read_params": {}},
                    {"key": "fixtures", "data_folder": None, "filename_pattern": None, "read_params": {}},
                    {"key": "awaiting_results", "data_folder": None, "filename_pattern": None, "read_params": {}},
                    {"key": "simulation_date", "data_folder": None, "filename_pattern": None, "read_params": {}},
                ]
            )

        data = {}
        for file_info in data_files:
            key = file_info["key"]
            data_folder = file_info["data_folder"]
            filename_pattern = file_info["filename_pattern"]
            read_params = file_info.get("read_params", {})
            if data_folder and filename_pattern:
                file_path = find_latest_file_for_division(data_folder, division, filename_pattern)
                if file_path:
                    try:
                        if key == "simulation_date":
                            # Read the date from the text file
                            with open(file_path, "r") as f:
                                data[key] = f.read().strip()
                            logging.debug(f"Loaded {key} data from {file_path}")
                        else:
                            data[key] = pd.read_csv(file_path, **read_params)
                            logging.debug(f"Loaded {key} data from {file_path}")
                    except Exception as e:
                        logging.warning(f"Could not load {key} data from {file_path}: {e}")
                        data[key] = pd.DataFrame() if key != "simulation_date" else None
                else:
                    logging.warning(f"No file found for {key} for division {division}; setting as empty DataFrame.")
                    data[key] = pd.DataFrame() if key != "simulation_date" else None
            else:
                logging.debug(f"Data folder or filename pattern is None for {key}; assigning empty DataFrame.")
                data[key] = pd.DataFrame() if key != "simulation_date" else None

        # Return the data in the expected order
        return (
            data.get("final_table", pd.DataFrame()),
            data.get("fixtures", pd.DataFrame()),
            data.get("home_away_df", pd.DataFrame()),
            data.get("team_win_breakdown_overall", pd.DataFrame()),
            data.get("team_win_breakdown_home", pd.DataFrame()),
            data.get("team_win_breakdown_away", pd.DataFrame()),
            data.get("team_win_breakdown_delta", pd.DataFrame()),
            data.get("awaiting_results", pd.DataFrame()),
            data.get("detailed_league_table", pd.DataFrame()),
            data.get("overall_home_away", pd.DataFrame()),
            data.get("summarized_players", pd.DataFrame()),
            multiple_breakdowns_available,  # Return this flag
            data.get("simulation_date", None),  # Add this line
        )

    except Exception as e:
        logging.exception(f"An error occurred while loading CSVs for division {division}: {e}")
        st.error(f"Data not found for division {division}. Error: {e}")
        # Return a tuple of empty DataFrames
        empty_df = pd.DataFrame()
        return (empty_df,) * 11 + (False, None)  # Return False for multiple_breakdowns_available and simulation_date


def load_txts(division, season_base_path):
    """
    Load the lists of unbeaten players and players who have played every game for a given division.
    Since the main script now always creates a TXT file (even a blank one), we simply read from it.
    """
    logging.info(f"Loading TXTs for division {division}")

    # Define the paths for the season’s unbeaten_players and played_every_game directories
    unbeaten_players_base_path = os.path.join(season_base_path, "unbeaten_players")
    played_every_game_base_path = os.path.join(season_base_path, "played_every_game")

    # Get the file paths using find_latest_file_for_division
    unbeaten_file_path = find_latest_file_for_division(unbeaten_players_base_path, division, "{}.txt")
    played_every_game_file_path = find_latest_file_for_division(played_every_game_base_path, division, "{}.txt")

    # Always try to open the unbeaten players file; if it exists but is empty, we just return an empty list.
    try:
        with open(unbeaten_file_path, "r") as file:
            unbeaten_players = [line.strip() for line in file if line.strip()]
        logging.debug(f"Loaded unbeaten players from {unbeaten_file_path}")
    except Exception as e:
        logging.info(f"Unbeaten players file not found or error for division {division}. Returning an empty list.")
        unbeaten_players = []

    # Always try to open the played every game file; a blank file yields an empty list.
    try:
        with open(played_every_game_file_path, "r") as file:
            played_every_game = [line.strip() for line in file if line.strip()]
        logging.debug(f"Loaded players who have played every game from {played_every_game_file_path}")
    except Exception as e:
        logging.info(f"Played every game file not found or error for division {division}. Returning an empty list.")
        played_every_game = []

    return unbeaten_players, played_every_game


def load_player_rankings(season_base_path, divisions_for_season):
    """
    Function to load player rankings CSVs from the most recent weeks for each division,
    ensuring data is loaded for all divisions even if they didn't play in the latest week.
    Handles both cases where data is stored under week_* folders (current season)
    and where data files are stored directly under the data directories (previous seasons).
    """
    logging.info(f"Loading player rankings for season {season_base_path}")

    # Define the path to the ranking data for the season
    ranking_df_path = os.path.join(season_base_path, "ranking_df")

    # Ensure the path exists
    if not os.path.exists(ranking_df_path):
        logging.warning(f"Ranking data path does not exist: {ranking_df_path}")
        st.error("Ranking data not available.")
        return pd.DataFrame()

    # Initialize variables
    ranking_dataframes = []

    # Check if week_* folders exist
    week_folders = glob.glob(os.path.join(ranking_df_path, "week_*"))
    if week_folders:
        # Week folders exist; proceed to load data from them
        week_numbers = []
        for folder in week_folders:
            week_name = os.path.basename(folder)
            match = re.match(r"week_(\d+)", week_name)
            if match:
                week_number = int(match.group(1))
                week_numbers.append((week_number, folder))
        # Sort week folders by week number in descending order
        week_numbers.sort(reverse=True)
        sorted_week_folders = [folder for _, folder in week_numbers]

        divisions_loaded = set()
        divisions_to_load = set(divisions_for_season)

        # Iterate over week folders
        for week_folder in sorted_week_folders:
            if not divisions_to_load:
                break  # All divisions have been loaded

            # Get all CSV files in the week folder
            ranking_files = glob.glob(os.path.join(week_folder, "*.csv"))
            for file in ranking_files:
                # Extract division from filename
                filename = os.path.basename(file)
                logging.debug(f"Processing file: {filename}")
                division_match = re.match(r"(.*)_ranking_df\.csv", filename)
                if division_match:
                    division = division_match.group(1)
                    logging.debug(f"Matched division: {division}")
                    if division in divisions_to_load:
                        try:
                            df = pd.read_csv(file)
                            ranking_dataframes.append(df)
                            divisions_loaded.add(division)
                            divisions_to_load.remove(division)
                            logging.debug(f"Loaded ranking data for division {division} from {file}")
                        except Exception as e:
                            logging.warning(f"Error reading file {file}: {e}")
                            continue
                else:
                    logging.warning(f"Filename {filename} does not match expected pattern.")
            if not divisions_to_load:
                break  # All divisions loaded
    else:
        # No week_* folders; load CSV files directly from ranking_df_path
        ranking_files = glob.glob(os.path.join(ranking_df_path, "*.csv"))
        if not ranking_files:
            logging.warning(f"No ranking files found in {ranking_df_path}")
            st.error("No ranking data available.")
            return pd.DataFrame()

        for file in ranking_files:
            # Extract division from filename
            filename = os.path.basename(file)
            logging.debug(f"Processing file: {filename}")
            division_match = re.match(r"(.*)_ranking_df\.csv", filename)
            if division_match:
                division = division_match.group(1)
                try:
                    df = pd.read_csv(file)
                    ranking_dataframes.append(df)
                    logging.debug(f"Loaded ranking data for division {division} from {file}")
                except Exception as e:
                    logging.warning(f"Error reading file {file}: {e}")
                    continue
            else:
                logging.warning(f"Filename {filename} does not match expected pattern.")

    if not ranking_dataframes:
        logging.error("No valid ranking data files loaded.")
        st.error("No valid ranking data files loaded.")
        return pd.DataFrame()

    # Concatenate all ranking DataFrames
    ranking_df_all = pd.concat(ranking_dataframes, ignore_index=True)
    logging.info(f"Combined all rankings data into a single DataFrame with shape {ranking_df_all.shape}")

    # Now repeat similar steps for players_df
    # Define the path to the players data for the season
    players_df_path = os.path.join(season_base_path, "players_df")

    # Ensure the path exists
    if not os.path.exists(players_df_path):
        logging.warning(f"Players data path does not exist: {players_df_path}")
        st.error("Players data not available.")
        return pd.DataFrame()

    players_dataframes = []

    # Check if week_* folders exist
    week_folders = glob.glob(os.path.join(players_df_path, "week_*"))
    if week_folders:
        # Week folders exist; proceed to load data from them
        week_numbers = []
        for folder in week_folders:
            week_name = os.path.basename(folder)
            match = re.match(r"week_(\d+)", week_name)
            if match:
                week_number = int(match.group(1))
                week_numbers.append((week_number, folder))
        # Sort week folders by week number descending
        week_numbers.sort(reverse=True)
        sorted_week_folders = [folder for _, folder in week_numbers]

        divisions_loaded = set()
        divisions_to_load = set(divisions_for_season)

        # Iterate over week folders
        for week_folder in sorted_week_folders:
            if not divisions_to_load:
                break  # All divisions loaded

            # Get all CSV files in the week folder
            players_files = glob.glob(os.path.join(week_folder, "*.csv"))
            for file in players_files:
                # Extract division from filename
                filename = os.path.basename(file)
                logging.debug(f"Processing file: {filename}")
                division_match = re.match(r"(.*)_players_df\.csv", filename)
                if division_match:
                    division = division_match.group(1)
                    logging.debug(f"Matched division: {division}")
                    if division in divisions_to_load:
                        try:
                            df = pd.read_csv(file)
                            players_dataframes.append(df)
                            divisions_loaded.add(division)
                            divisions_to_load.remove(division)
                            logging.debug(f"Loaded players data for division {division} from {file}")
                        except Exception as e:
                            logging.warning(f"Error reading file {file}: {e}")
                            continue
                else:
                    logging.warning(f"Filename {filename} does not match expected pattern.")
            if not divisions_to_load:
                break  # All divisions loaded
    else:
        # No week_* folders; load CSV files directly from players_df_path
        players_files = glob.glob(os.path.join(players_df_path, "*.csv"))
        if not players_files:
            logging.warning(f"No players files found in {players_df_path}")
            st.error("No players data available.")
            return pd.DataFrame()

        for file in players_files:
            # Extract division from filename
            filename = os.path.basename(file)
            logging.debug(f"Processing file: {filename}")
            division_match = re.match(r"(.*)_players_df\.csv", filename)
            if division_match:
                division = division_match.group(1)
                try:
                    df = pd.read_csv(file)
                    players_dataframes.append(df)
                    logging.debug(f"Loaded players data for division {division} from {file}")
                except Exception as e:
                    logging.warning(f"Error reading file {file}: {e}")
                    continue
            else:
                logging.warning(f"Filename {filename} does not match expected pattern.")

    if not players_dataframes:
        logging.error("No valid players data files loaded.")
        st.error("No valid players data files loaded.")
        return pd.DataFrame()

    # Concatenate all players DataFrames
    players_df_all = pd.concat(players_dataframes, ignore_index=True)
    logging.info(f"Combined all players data into a single DataFrame with shape {players_df_all.shape}")

    # Standardize player name column in players_df_all
    if "Player" in players_df_all.columns:
        players_df_all = players_df_all.rename(columns={"Player": "Name of Player"})
    elif "Name of Players" in players_df_all.columns:
        players_df_all = players_df_all.rename(columns={"Name of Players": "Name of Player"})
    elif "Name of Player" in players_df_all.columns:
        pass  # Column is already named 'Name of Player'
    else:
        logging.error("No player name column found in players_df_all.")
        st.error("Player name column not found in players data.")
        return pd.DataFrame()

    # Standardize player name column in ranking_df_all
    if "Player" in ranking_df_all.columns:
        ranking_df_all = ranking_df_all.rename(columns={"Player": "Name of Player"})
    elif "Name of Players" in ranking_df_all.columns:
        ranking_df_all = ranking_df_all.rename(columns={"Name of Players": "Name of Player"})
    elif "Name of Player" in ranking_df_all.columns:
        pass  # Column is already named 'Name of Player'
    else:
        logging.error("No player name column found in ranking_df_all.")
        st.error("Player name column not found in ranking data.")
        return pd.DataFrame()

    # Now proceed to merge
    try:
        # Ensure required columns are present
        required_columns = ["Name of Player", "Team", "HKS No."]
        missing_columns = [col for col in required_columns if col not in players_df_all.columns]
        if missing_columns:
            logging.error(f"Missing columns in players_df_all: {', '.join(missing_columns)}")
            st.error(f"Missing columns in players data: {', '.join(missing_columns)}")
            return pd.DataFrame()

        # Precompute lookups to support filling identifiers and identifying "playing up" fixtures
        players_df_all["_team_norm"] = players_df_all["Team"].astype(str).str.strip().str.lower()
        registered_teams_lookup = (
            players_df_all.groupby("Name of Player")["_team_norm"]
            .apply(lambda s: {team for team in s if team and team != "nan"})
            .to_dict()
        )
        players_df_all.drop(columns="_team_norm", inplace=True)

        if "Division" in players_df_all.columns:
            registered_divisions_lookup = (
                players_df_all.groupby("Name of Player")["Division"]
                .apply(lambda s: {str(div).strip() for div in s.dropna().astype(str)})
                .to_dict()
            )
        else:
            registered_divisions_lookup = {}

        player_hks_lookup = (
            players_df_all.dropna(subset=["HKS No."])
            .groupby("Name of Player")["HKS No."]
            .first()
        )

        merged_df = pd.merge(
            ranking_df_all,
            players_df_all[["Name of Player", "Team", "HKS No."]],
            on=["Name of Player", "Team"],
            how="left",
        )
    except Exception as e:
        logging.exception("Error merging ranking_df_all and players_df_all")
        st.error("An error occurred while merging player data.")
        return pd.DataFrame()

    # Check columns after merging
    if "HKS No." not in merged_df.columns:
        logging.error("'HKS No.' column is missing after merging.")

    # Fill missing HKS numbers using the player's registered information, then flag remaining gaps
    if not player_hks_lookup.empty:
        merged_df["HKS No."] = merged_df["HKS No."].fillna(merged_df["Name of Player"].map(player_hks_lookup))

    merged_df["_team_norm"] = merged_df["Team"].astype(str).str.strip().str.lower()

    def _is_playing_up(row: pd.Series) -> bool:
        player_name = row["Name of Player"]
        team_norm = row["_team_norm"]
        if team_norm == "nan":
            team_norm = ""
        registered_teams = registered_teams_lookup.get(player_name)
        if registered_teams:
            if team_norm and team_norm in registered_teams:
                return False
            if team_norm and team_norm not in registered_teams:
                return True
        registered_divisions = registered_divisions_lookup.get(player_name)
        division_value = row.get("Division")
        if pd.notna(division_value) and registered_divisions is not None:
            division_str = str(division_value).strip()
            if division_str and division_str not in registered_divisions:
                return True
        return False

    merged_df["Playing Up"] = merged_df.apply(_is_playing_up, axis=1)
    merged_df.drop(columns="_team_norm", inplace=True)

    def _format_division_label(row: pd.Series):
        division_value = row.get("Division")
        if pd.isna(division_value):
            return pd.NA
        division_text = str(division_value).strip()
        if not division_text or division_text.lower() == "nan":
            return pd.NA
        return f"{division_text} (Playing Up)" if row.get("Playing Up") else division_text

    merged_df["Division Display"] = merged_df.apply(_format_division_label, axis=1)

    # Handle missing HKS No. (these should be players who remain unmatched after all lookups)
    missing_hksno = merged_df["HKS No."].isnull().sum()
    if missing_hksno > 0:
        logging.warning(f"{missing_hksno} entries have missing HKS No. after merge.")

    # Identify rows with missing 'HKS No.'
    missing_hksno_df = merged_df[merged_df["HKS No."].isnull()]
    missing_hksno_df.to_csv(os.path.join(season_dir, "missing_hksno.csv"), index=False)

    def determine_club(team_name):
        """
        Function to determine the club based on the team name.
        """
        # Allow for special case
        if team_name == "HKCC Tuesday Night Rockers":
            return "Hong Kong Cricket Club"
        for club in clubs:
            if club.lower() in team_name.lower():
                return club
        return team_name  # Return the team name if no club is matched

    # Apply the function to the 'Team' column
    if "Team" in merged_df.columns:
        merged_df["Club"] = merged_df["Team"].apply(determine_club)
    else:
        st.error("Column 'Team' not found in ranking data.")
        merged_df["Club"] = "Unknown"

    return merged_df


def get_division_sort_key(division_name, is_current_season):
    """
    Returns a sort key for a division name to sort divisions in the desired order.
    """
    if is_current_season:
        # Map renamed divisions to their old equivalents
        division_mapping = {
            "Premier Main": ("Main", 1),
            "Premier Masters": ("Masters", 1),
            "Premier Ladies": ("Ladies", 1),
        }
        if division_name in division_mapping:
            division_type, main_number = division_mapping[division_name]
            letter_value = -1  # No letter
        else:
            # Use existing logic for other divisions
            division_type = None
            main_number = None
            letter_value = None
    else:
        # For previous seasons, no special mapping
        division_type = None
        main_number = None
        letter_value = None

    if division_type is None:
        # Existing logic for determining division type and number
        if division_name.startswith("M"):
            division_type = "Masters"
            category = 2
            rest = division_name[1:]
        elif division_name.startswith("L"):
            division_type = "Ladies"
            category = 3
            rest = division_name[1:]
        else:
            division_type = "Main"
            category = 1
            rest = division_name

        # Extract the main number and any letter suffix
        match = re.match(r"(\d+)([A-Z]*)", rest, re.I)
        if match:
            main_number = int(match.group(1))
            letter_suffix = match.group(2).upper()
        else:
            # Handle divisions without a number
            main_number = float("inf")
            letter_suffix = ""

        # Map letter suffix to a number for sorting ('A' -> 0, 'B' -> 1, etc.)
        letter_value = ord(letter_suffix) - ord("A") if letter_suffix else -1
    else:
        # Assign category based on division type
        category = {"Main": 1, "Masters": 2, "Ladies": 3}[division_type]

    # Return the sort key as a tuple
    return (category, main_number, letter_value)


def get_divisions_for_season(season_base_path, is_current_season):
    """
    Function to get a list of divisions available for a given season.
    """
    if is_current_season:
        # For current season, use the divisions from the all_divisions dictionary
        divisions_list = sorted(all_divisions.keys(), key=lambda x: get_division_sort_key(x, is_current_season))
        return divisions_list
    else:
        # Existing code for previous seasons
        detailed_league_tables_path = os.path.join(season_base_path, "detailed_league_tables")

        # Check if week_* folders exist
        week_folders = glob.glob(os.path.join(detailed_league_tables_path, "week_*"))
        divisions = set()

        if week_folders:
            # Week folders exist; collect divisions from the most recent week
            week_numbers = []
            for folder in week_folders:
                week_name = os.path.basename(folder)
                match = re.match(r"week_(\d+)", week_name)
                if match:
                    week_number = int(match.group(1))
                    week_numbers.append((week_number, folder))
            # Sort week folders by week number in descending order
            week_numbers.sort(reverse=True)
            latest_week_folder = week_numbers[0][1]  # Get the folder for the latest week

            # Get all CSV files in the latest week folder
            csv_files = glob.glob(os.path.join(latest_week_folder, "*.csv"))
        else:
            # No week_* folders; collect divisions from files directly under detailed_league_tables_path
            csv_files = glob.glob(os.path.join(detailed_league_tables_path, "*.csv"))

        # Extract division names from filenames
        for file in csv_files:
            filename = os.path.basename(file)
            match = re.match(r"(.*)_detailed_league_table\.csv", filename)
            if match:
                division_name = match.group(1)
                divisions.add(division_name)

        # Convert the set to a sorted list
        divisions_list = sorted(divisions, key=lambda x: get_division_sort_key(x, is_current_season))

        return divisions_list


# Start the main application
def main():
    logging.info("Application started")
    # Initialize 'data' in session_state with default empty DataFrames if it doesn't exist
    if "data" not in st.session_state:
        empty_df = pd.DataFrame()
        st.session_state["data"] = {
            "division_data": {},
            "all_rankings_df": pd.DataFrame(),
            "data_loaded": False,
            "current_division": None,
            "current_season": None,
        }
        logging.debug("Initialized session state data")

    # Title
    st.title("HK Squash League App")

    with st.sidebar:
        # Season selection
        selected_season = st.selectbox("**Select Season:**", available_seasons)
        is_current_season = selected_season == current_season

        # Set the season_base_path based on the selected season
        if selected_season == current_season:
            season_base_path = os.path.join(base_directory, selected_season)
        else:
            season_base_path = os.path.join(base_directory, "previous_seasons", selected_season)

        # Try to load divisions from config, but for previous seasons allow fallback to data discovery
        all_divisions = load_divisions_simple(base_directory, selected_season)

        if not all_divisions:
            if selected_season == current_season:
                # For current season, config is required
                st.error(f"Divisions config not found or empty for {selected_season}.")
                st.stop()
            else:
                # For previous seasons, try to infer divisions from available data files
                st.warning(f"Divisions config not found for {selected_season}. Inferring divisions from data files...")

                # Try multiple patterns to find division files
                patterns_to_try = [
                    (os.path.join(season_base_path, "schedules"), "*_schedules.csv", "_schedules.csv"),
                    (os.path.join(season_base_path, "schedules"), "schedules_*.csv", "schedules_"),
                    (os.path.join(season_base_path, "schedules_df"), "*_schedules_df.csv", "_schedules_df.csv"),
                    (os.path.join(season_base_path, "ranking_df"), "*_ranking_df.csv", "_ranking_df.csv"),
                ]

                all_divisions = {}
                for folder_path, pattern, suffix in patterns_to_try:
                    if os.path.exists(folder_path):
                        files = glob.glob(os.path.join(folder_path, pattern))
                        if files:
                            for file in files:
                                basename = os.path.basename(file)
                                # Handle both "10_schedules.csv" and "schedules_10.csv" patterns
                                if suffix.startswith("_"):
                                    div_name = basename.replace(suffix, "")
                                else:
                                    div_name = basename.replace(suffix, "").replace(".csv", "")
                                all_divisions[div_name] = None  # No ID available for old seasons
                            logging.info(
                                f"Inferred {len(all_divisions)} divisions from {folder_path} for {selected_season}"
                            )
                            break

                if not all_divisions:
                    st.error(f"Could not find divisions for {selected_season}.")
                    st.stop()

        # Your requested line — sort with the season-aware key
        divisions_for_season = sorted(all_divisions.keys(), key=lambda x: get_division_sort_key(x, is_current_season))

        # Stats selection
        stats_selection = st.radio("**Select Stats Type:**", ["Player Stats", "Team Stats"])

        # Sections depending on stats_selection
        if stats_selection == "Player Stats":
            sections = ["Player Info", "Division Player Stats"]
        else:
            if is_current_season:
                sections = [
                    "Detailed Division Table",
                    "Rubber Win Percentage",
                    "Home/Away Splits",
                    "Projections",
                    "Match Results",
                ]
            else:
                sections = ["Detailed Division Table", "Rubber Win Percentage", "Home/Away Splits"]

        # Select a Section
        selected_section = st.selectbox("**Select a Section:**", sections)

        # If selected_section is 'Player Info', we don't need to select a division
        if selected_section in ["Player Info", "Match Results"]:
            division = None
        else:
            division = st.selectbox("**Select a Division:**", divisions_for_season)

        # Logging statements
        logging.info(f"User selected season: {selected_season}")
        logging.info(f"User selected stats type: {stats_selection}")
        logging.info(f"User selected section: {selected_section}")
        logging.info(f"User selected division: {division}")

        about = st.expander("**About**")
        about.write(
            """The aim of this application is to take publicly available data from 
            [hksquash.org.hk](https://www.hksquash.org.hk/public/index.php/leagues/index/league/Squash/pages_id/25.html)
            and provide insights to players and convenors involved in the Hong Kong Squash League.
            \nThis application is not affiliated with the Squash Association of Hong Kong, China."""
        )

        contact = st.expander("**Contact**")
        contact.write("For any queries, email bpalitherland@gmail.com")

    # Data loading logic based on selected section
    if selected_section == "Player Info":
        logging.debug("Processing data for 'Player Info'")
        # ---------- NEW SAFEGUARD ---------------
        if "data" not in st.session_state:
            st.session_state["data"] = {
                "division_data": {},
                "all_rankings_df": pd.DataFrame(),
                "data_loaded": False,
                "current_division": None,
                "current_season": None,
            }
        # ----------------------------------------
        # Handle the "Player Info" case
        season_key = f"{selected_season}_all"
        if not st.session_state["data"].get(f"all_rankings_loaded_{season_key}", False):
            # Load all_rankings_df
            all_rankings_df = load_player_rankings(season_base_path, divisions_for_season)
            if all_rankings_df.empty:
                st.error("No player ranking data is available.")
                return  # Exit the function to prevent further errors
            st.session_state["data"][f"all_rankings_df_{season_key}"] = all_rankings_df
            st.session_state["data"][f"all_rankings_loaded_{season_key}"] = True
        else:
            all_rankings_df = st.session_state["data"][f"all_rankings_df_{season_key}"]

    elif selected_section == "Match Results":
        # Load data
        season_base_path = os.path.join(base_directory, selected_season)
        # We load the CSVs rather than generating them in the script
        combined_results_df = pd.read_csv(os.path.join(season_dir, "combined_results_df.csv"))
        combined_player_results_df = pd.read_csv(os.path.join(season_dir, "combined_player_results_df.csv"))

        # Convert 'Date' and 'Match Date' to datetime objects
        combined_results_df["Date"] = pd.to_datetime(combined_results_df["Date"], format="%Y-%m-%d")
        combined_player_results_df["Match Date"] = pd.to_datetime(
            combined_player_results_df["Match Date"], format="%Y-%m-%d"
        )

        # Log unique parsed dates for verification
        parsed_results_dates = combined_results_df["Date"].dropna().unique()
        parsed_player_dates = combined_player_results_df["Match Date"].dropna().unique()
        logging.debug(f"Unique parsed Dates in combined_results_df: {parsed_results_dates}")
        logging.debug(f"Unique parsed Match Dates in combined_player_results_df: {parsed_player_dates}")

        # Make sure Division columns are consistent
        combined_results_df["Division"] = combined_results_df["Division"].astype(str).str.strip()
        combined_player_results_df["Division"] = combined_player_results_df["Division"].astype(str).str.strip()

        # Drop duplicates in combined player results
        combined_player_results_df = combined_player_results_df[combined_player_results_df["Home/Away"] == "Home"]

        if combined_results_df.empty:
            st.error("No match results data available.")
            return

        if combined_player_results_df.empty:
            st.warning("Player results data is not available.")
            # You can proceed without player results, but you won't be able to display them

        # Define the determine_club function
        def determine_club(team_name):
            """
            Function to determine the club based on the team name.
            """
            # Allow for special case
            if team_name == "HKCC Tuesday Night Rockers":
                return "Hong Kong Cricket Club"
            for club in clubs:
                if club.lower() in team_name.lower():
                    return club
            return "Other"  # Assign 'Other' if no club is matched

        # Apply the function to determine the club for each team
        combined_results_df["Home Club"] = combined_results_df["Home Team"].apply(determine_club)
        combined_results_df["Away Club"] = combined_results_df["Away Team"].apply(determine_club)

        # Sidebar for filter option
        filter_option = st.sidebar.radio("Filter by:", ["Club", "Division"])

        if filter_option == "Club":

            # Include 'Other' in the list of clubs if necessary
            all_clubs_in_data = set(combined_results_df["Home Club"]).union(set(combined_results_df["Away Club"]))
            clubs_in_data = [club for club in clubs if club in all_clubs_in_data]
            if "Other" in all_clubs_in_data:
                clubs_in_data.append("Other")

            # Add 'Overall' option to the list of clubs
            list_of_clubs = ["Overall"] + sorted(clubs_in_data)

            # Set default club to 'Hong Kong Cricket Club'
            default_club = "Hong Kong Cricket Club"

            # Sidebar for club selection
            selected_club = st.sidebar.selectbox(
                "Select a Club:", list_of_clubs, index=list_of_clubs.index(default_club)
            )

        elif filter_option == "Division":

            # Get list of divisions from the combined_results_df
            divisions_in_data = combined_results_df["Division"].unique()

            # Convert to list and sort
            divisions_list = ["Overall"] + sorted(
                divisions_in_data, key=lambda x: get_division_sort_key(x, is_current_season)
            )

            # Set default division to '7B'
            default_division = "7"

            # Sidebar for division selection
            selected_division = st.sidebar.selectbox(
                "Select a Division:", divisions_list, index=divisions_list.index(default_division)
            )

        if filter_option == "Club":
            # Filter matches involving the selected club
            if selected_club == "Overall":
                filtered_results = combined_results_df
            else:
                filtered_results = combined_results_df[
                    (combined_results_df["Home Club"] == selected_club)
                    | (combined_results_df["Away Club"] == selected_club)
                ]
        elif filter_option == "Division":
            # Filter matches in the selected division
            if selected_division == "Overall":
                filtered_results = combined_results_df
            else:
                filtered_results = combined_results_df[combined_results_df["Division"] == selected_division]

        # Sort by Date
        filtered_results = filtered_results.sort_values(
            by=["Date", "Division", "Home Team"], ascending=[False, True, True]
        )

        if filtered_results.empty:
            st.info(f"No matches found for club {selected_club}.")
            return

        # Adjust the header based on the selected filter option
        if filter_option == "Club":
            st.header(f"Matches Involving {selected_club}")
        elif filter_option == "Division":
            st.header(f"Matches in {selected_division}")

        # Display the filtered results
        for idx, row in filtered_results.iterrows():
            # Match summary
            match_summary = f'### **Division {row["Division"]}:** {row["Home Team"]} **{row["Home Score"]}** - **{row["Away Score"]}** {row["Away Team"]}  ({row["Date"].date()})'

            with st.expander(match_summary):
                # Filter player_results where Division and Match Date match, and Team is either Home or Away Team
                player_results = combined_player_results_df[
                    (combined_player_results_df["Division"] == row["Division"])
                    & (combined_player_results_df["Match Date"] == row["Date"])
                    & (combined_player_results_df["Team"].isin([row["Home Team"], row["Away Team"]]))
                ].drop_duplicates()
                logging.debug(f"Found {len(player_results)} player results for this match.")

                if not player_results.empty:
                    st.subheader("Player Results")

                    for rubber_number, rubber_results in player_results.groupby("Rubber Number"):
                        result_row = rubber_results.iloc[0]

                        # Determine the home and away players
                        if result_row["Team"] == row["Home Team"]:  # Change from 'Home Team' to 'Team'
                            home_player = result_row["Player Name"]
                            away_player = result_row["Opponent Name"]
                            score = result_row["Score"]
                            home_won = result_row["Result"] == "Win"
                        else:
                            home_player = result_row["Opponent Name"]
                            away_player = result_row["Player Name"]
                            score = result_row["Score"]
                            home_won = result_row["Result"] == "Loss"

                        if score in ["CR", "WO"]:
                            # Use your existing logic to format CR/WO results.
                            if home_won:
                                line = f"<p><strong>{rubber_number}</strong>: {home_player} beat {away_player} ({score}).</p>"
                            else:
                                line = f"<p><strong>{rubber_number}</strong>: {home_player} lost to {away_player} ({score}).</p>"
                        else:
                            # Now handle the numeric scores, and swap if needed.
                            if "-" in score:
                                home_score, away_score = score.split("-")

                                if home_won:
                                    line = f"<p><strong>{rubber_number}</strong>: {home_player} beat {away_player} {home_score}-{away_score}.</p>"
                                else:
                                    line = f"<p><strong>{rubber_number}</strong>: {home_player} lost to {away_player} {home_score}-{away_score}.</p>"

                        st.markdown(line, unsafe_allow_html=True)

                logging.debug(
                    f"Displaying player results for match: {row['Home Team']} vs {row['Away Team']} on {row['Date']}"
                )

        # else:
        #    logging.debug(f"No player results found for match: {row['Home Team']} vs {row['Away Team']} on {row['Date']}")
        #    st.write("Player results not available for this match.")

    else:
        # Process division-specific data
        if division is not None:
            logging.debug(f"Processing division-specific data for {division}")
            # Check if data has already been loaded for the selected division
            division_key = f"{selected_season}_{division}"
            if st.session_state["data"].get("current_division") != division_key or not st.session_state["data"].get(
                "data_loaded"
            ):
                # Load data for the selected division and season
                division_data = load_csvs(division, season_base_path, is_current_season)
                st.session_state["data"]["division_data"][division_key] = division_data
                st.session_state["data"]["current_division"] = division_key
                st.session_state["data"]["data_loaded"] = True

                # Load TXTs only for a specific division
                try:
                    # Load TXT files
                    unbeaten_players, played_every_game = load_txts(division, season_base_path)
                except Exception as e:
                    logging.exception(f"Error loading TXT files for division {division}")
                    unbeaten_players, played_every_game = [], []
                st.session_state["data"]["unbeaten_players"] = unbeaten_players
                st.session_state["data"]["played_every_game"] = played_every_game
            else:
                # Retrieve data from session state
                division_data = st.session_state["data"]["division_data"].get(division_key)
                unbeaten_players = st.session_state["data"].get("unbeaten_players", [])
                played_every_game = st.session_state["data"].get("played_every_game", [])

            # Unpack the division data
            (
                simulated_table,
                simulated_fixtures,
                home_away_df,
                team_win_breakdown_overall,
                team_win_breakdown_home,
                team_win_breakdown_away,
                team_win_breakdown_delta,
                awaiting_results,
                detailed_league_table,
                overall_home_away,
                summarized_players,
                multiple_breakdowns_available,
                simulation_date,
            ) = division_data

            # Assign the simulation_date (already loaded from the file)
            if not simulation_date:
                simulation_date = "Date not available"

            # Initialize date variable
            date = None

            # Handle cases where data might be empty
            if overall_home_away is not None and not overall_home_away.empty and overall_home_away.shape[1] > 3:
                # Extract the raw date value
                date_value = overall_home_away.iloc[0, 3]

                # Convert to datetime with error handling
                date_raw = pd.to_datetime(date_value, errors="coerce")

                # Check and handle NaT values for date
                if pd.isnull(date_raw):
                    date = None  # Do not assign today's date
                    logging.warning("Date is not available.")
                else:
                    date = date_raw.strftime("%Y-%m-%d")
            else:
                date = None
                logging.warning(
                    "overall_home_away DataFrame is empty or does not have enough columns; date is not available."
                )
        else:
            st.error("Please select a division.")
            return

    # Now proceed based on the selected section
    if selected_section == "Player Info":
        # Handle 'Player Info' section
        # Load combined_player_results_df for date filtering
        try:
            combined_player_results_df = pd.read_csv(os.path.join(season_base_path, "combined_player_results_df.csv"))
            combined_player_results_df["Match Date"] = pd.to_datetime(
                combined_player_results_df["Match Date"], format="%Y-%m-%d"
            )
            logging.info(
                f"Loaded combined_player_results_df with {len(combined_player_results_df)} records for {selected_season}"
            )
        except Exception as e:
            logging.warning(f"Could not load combined_player_results_df for {selected_season}: {e}")
            combined_player_results_df = pd.DataFrame()

        try:
            ratio_results_path = os.path.join(season_base_path, "ratio_results.csv")
            ratio_results_df = pd.read_csv(ratio_results_path)
            ratio_results_df = ratio_results_df.rename(columns={"HKS_No": "HKS No.", "Final Rating": "Elo Rating"})
            ratio_results_df["HKS No."] = pd.to_numeric(ratio_results_df.get("HKS No."), errors="coerce")
            ratio_results_df["Elo Rating"] = pd.to_numeric(ratio_results_df.get("Elo Rating"), errors="coerce")
            logging.info(f"Loaded ratio_results_df with {len(ratio_results_df)} records for {selected_season}")
        except Exception as e:
            logging.warning(f"Could not load ratio_results.csv for {selected_season}: {e}")
            ratio_results_df = pd.DataFrame()

        def aggregate_club(x):
            """
            Function to aggregate clubs in Club column
            """
            unique_clubs = x.unique()
            if len(unique_clubs) == 1:  # If only one unique club
                return unique_clubs[0]
            else:
                return ", ".join(sorted(unique_clubs))

        def aggregate_club_overall(x):
            """
            Function to aggregate clubs in Club column for 'Overall' selection
            """
            unique_clubs = x.unique()
            if len(unique_clubs) == 1:  # If only one unique club
                return unique_clubs[0]
            else:
                return ", ".join(sorted(unique_clubs))  # Join multiple club names with commas

        # List of clubs
        list_of_clubs = ["Overall"] + sorted(clubs)

        # Extract unique months from combined_player_results_df for month filtering
        if not combined_player_results_df.empty:
            # Get unique months from Match Date
            unique_dates = combined_player_results_df["Match Date"].dropna().dt.to_period("M").unique()
            unique_months = sorted([pd.Period(m) for m in unique_dates], reverse=True)
            month_options = ["All Months"] + [m.strftime("%B %Y") for m in unique_months]
            logging.info(f"Found {len(unique_months)} unique months in player results")
        else:
            month_options = ["All Months"]
            logging.warning("combined_player_results_df is empty, only 'All Months' option available")

        # Title
        st.title("**Player Rankings**")

        # Line break
        st.write("<br>", unsafe_allow_html=True)

        # Adjust the fractions to control the width of each column
        col1, col2, col3 = st.columns([1, 1, 2])

        # Find the index for "Hong Kong Cricket Club"
        default_club_index = list_of_clubs.index("Hong Kong Cricket Club")

        # Plotting the chart in the first column
        with col1:
            selected_month = st.selectbox("**Select month:**", month_options, index=0)

        with col2:
            club = st.selectbox("**Select club:**", list_of_clubs, index=default_club_index)

        # Filter rankings by selected month
        filtered_rankings_df = all_rankings_df.copy()

        if selected_month != "All Months" and not combined_player_results_df.empty:
            # Parse the selected month
            selected_month_period = pd.Period(selected_month, freq="M")

            # Filter player results to selected month
            month_mask = combined_player_results_df["Match Date"].dt.to_period("M") == selected_month_period
            players_in_month = combined_player_results_df[month_mask][
                ["HKS No.", "Player Name", "Team"]
            ].drop_duplicates()

            logging.info(f"Found {len(players_in_month)} unique players in {selected_month}")

            # Filter rankings to only include players who played in the selected month
            # Match by HKS No. if available, otherwise by Name
            if "HKS No." in filtered_rankings_df.columns:
                filtered_rankings_df = filtered_rankings_df[
                    filtered_rankings_df["HKS No."].isin(players_in_month["HKS No."])
                ]
            else:
                filtered_rankings_df = filtered_rankings_df[
                    filtered_rankings_df["Name of Player"].isin(players_in_month["Player Name"])
                ]

            st.info(f"Showing rankings for players who competed in {selected_month}")

        division_display_col = "Division Display" if "Division Display" in filtered_rankings_df.columns else "Division"

        def aggregate_division_labels(series: pd.Series) -> str:
            ordered: list[str] = []
            for value in series:
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if not text or text.lower() == "nan":
                    continue
                if text not in ordered:
                    ordered.append(text)
            return ", ".join(ordered)

        aggregation_common: dict[str, object] = {
            division_display_col: aggregate_division_labels,
            "Average Points": "mean",
            "Total Game Points": "sum",
            "Games Played": "sum",
            "Won": "sum",
            "Lost": "sum",
        }

        if "Playing Up" in filtered_rankings_df.columns:
            aggregation_common["Playing Up"] = "sum"

        if club != "Overall":
            aggregated_df = (
                filtered_rankings_df.groupby(["HKS No.", "Name of Player", "Club"])
                .agg(aggregation_common)
                .reset_index()
            )
        else:
            aggregation_with_club = {"Club": aggregate_club_overall, **aggregation_common}
            aggregated_df = (
                filtered_rankings_df.groupby(["HKS No.", "Name of Player"])
                .agg(aggregation_with_club)
                .reset_index()
            )

        aggregated_df = aggregated_df.rename(columns={division_display_col: "Division"})
        if "Playing Up" in aggregated_df.columns:
            aggregated_df = aggregated_df.rename(columns={"Playing Up": "Played Up"})
            aggregated_df["Played Up"] = (
                pd.to_numeric(aggregated_df["Played Up"], errors="coerce").fillna(0).astype(int)
            )

        # Now calculate 'Win Percentage' outside the aggregation step
        aggregated_df["Win Percentage"] = (aggregated_df["Won"] / aggregated_df["Games Played"] * 100).fillna(0)

        # Create Avg Pts column
        aggregated_df["Avg Pts"] = (aggregated_df["Total Game Points"] / aggregated_df["Games Played"]).fillna(0)

        # Merge Elo ratings from ratio_results.csv when available
        if "HKS No." in aggregated_df.columns:
            aggregated_df["HKS No."] = pd.to_numeric(aggregated_df["HKS No."], errors="coerce")
        if not ratio_results_df.empty and "HKS No." in aggregated_df.columns:
            aggregated_df = aggregated_df.merge(ratio_results_df[["HKS No.", "Elo Rating"]], on="HKS No.", how="left")
        if "Elo Rating" not in aggregated_df.columns:
            aggregated_df["Elo Rating"] = pd.NA

        # Continue with reduced dataframe
        columns_to_keep = [
            "HKS No.",
            "Name of Player",
            "Club",
            "Division",
            "Games Played",
            "Won",
            "Lost",
            "Win Percentage",
            "Avg Pts",
        ]
        if "Played Up" in aggregated_df.columns:
            insert_at = columns_to_keep.index("Division") + 1 if "Division" in columns_to_keep else len(columns_to_keep)
            columns_to_keep.insert(insert_at, "Played Up")
        if "Elo Rating" in aggregated_df.columns:
            columns_to_keep.append("Elo Rating")

        aggregated_df_reduced = aggregated_df[columns_to_keep].rename(
            columns={"Name of Player": "Player", "Games Played": "Games", "Win Percentage": "Win %"}
        )

        if "Elo Rating" in aggregated_df_reduced.columns:
            column_order = list(aggregated_df_reduced.columns)
            column_order.insert(column_order.index("Avg Pts"), column_order.pop(column_order.index("Elo Rating")))
            aggregated_df_reduced = aggregated_df_reduced[column_order]

        # Sort functionality
        with col1:
            sort_column = st.selectbox(
                "**Sort by:**", aggregated_df_reduced.columns, index=aggregated_df_reduced.columns.get_loc("Games")
            )
        sort_order = st.radio("**Sort order**", ["Ascending", "Descending"], index=1)

        # Filter DataFrame based on selected club
        if club != "Overall":
            filtered_df = aggregated_df_reduced[aggregated_df_reduced["Club"] == club]
            # Drop Club column if needed
            filtered_df = filtered_df.drop(columns="Club", errors="ignore")
        else:
            filtered_df = aggregated_df_reduced

        # Sort the DataFrame based on user selection
        ascending = True if sort_order == "Ascending" else False
        sorted_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

        numeric_alignment_columns = [
            col
            for col in ["Games", "Won", "Lost", "Win %", "Avg Pts", "Elo Rating", "Played Up"]
            if col in sorted_df.columns
        ]

        # Apply styles and formatting to sorted_df
        sorted_df_styled = sorted_df.style.set_properties(subset=["Player", "Division"], **{"text-align": "left"}).hide(
            axis="index"
        )
        sorted_df_styled = sorted_df_styled.set_properties(subset=numeric_alignment_columns, **{"text-align": "right"})

        # Format the columns to display as desired
        format_map = {"HKS No.": "{:.0f}", "Win %": "{:.1f}", "Avg Pts": "{:.1f}"}
        if "Played Up" in sorted_df.columns:
            format_map["Played Up"] = "{:.0f}"
        if "Elo Rating" in sorted_df.columns:
            format_map["Elo Rating"] = "{:,.0f}"

        sorted_df_styled = sorted_df_styled.format(format_map)

        # Convert DataFrame to HTML, hide the index, and apply minimal styling for spacing
        html = sorted_df_styled.to_html()

        # Line break
        st.write("<br>", unsafe_allow_html=True)

        # Display the sorted DataFrame
        st.write(html, unsafe_allow_html=True)

    elif selected_section == "Division Player Stats":
        # Handle 'Division Player Stats' section
        # Use division-specific data

        # Header
        st.header(f"Division Player Stats - {division}")

        def extract_names(cell_value):
            """
            Function to extract names from the cell
            """
            try:
                # Convert cell_value to string and handle possible NaN values
                cell_str = str(cell_value) if pd.notnull(cell_value) else ""
                # Split the cell value by commas
                parts = [part.strip() for part in cell_str.split(",")]
                names = []
                for part in parts:
                    # Extract the name before the parenthesis
                    match = re.match(r"([^\(]+)", part)
                    if match:
                        name = match.group(1).strip()
                        names.append(name)
                return set(names)
            except Exception as e:
                logging.exception(f"Error in extract_names for cell_value: {cell_value}")
                return set()

        # Custom styling function
        def highlight_row_if_same_player(row):
            try:
                # Extract player names from each column
                games_names = extract_names(row["Most Games"])
                wins_names = extract_names(row["Most Wins"])
                win_percentage_names = extract_names(row["Highest Win Percentage"])

                # Highlight color
                highlight_color = "background-color: #FFF2CC"
                # Default color
                default_color = ""

                # Find the intersection of names across all columns
                common_names = games_names & wins_names & win_percentage_names

                if common_names:
                    # Highlight the cells if there is at least one common name
                    return [
                        (
                            highlight_color
                            if col in ["Most Games", "Most Wins", "Highest Win Percentage"]
                            else default_color
                        )
                        for col in row.index
                    ]
                else:
                    # Return default color for all columns if no common names
                    return [default_color for _ in row.index]
            except Exception as e:
                logging.exception(f"Error in highlight_row_if_same_player for row: {row}")
                return ["" for _ in row.index]

        # Load and display overall scores
        if date:
            st.write(f"**Last Updated:** {date}")
        else:
            st.write("**Last Updated:** Date not available")

        if summarized_players.empty:
            st.info(f"No player stats available for Division {division}.")
        else:
            # Ensure required columns are present
            required_columns = ["Team", "Most Games", "Most Wins", "Highest Win Percentage"]
            missing_columns = [col for col in required_columns if col not in summarized_players.columns]

            if missing_columns:
                st.error(f"Missing columns in player stats: {', '.join(missing_columns)}")
            else:
                try:
                    # Apply styles to the DataFrame
                    styled_df = summarized_players.style.set_properties(**{"text-align": "left"}).hide(axis="index")

                    # Apply the styling function to the DataFrame
                    styled_df = styled_df.apply(highlight_row_if_same_player, axis=1)

                    # Convert styled DataFrame to HTML
                    html = styled_df.to_html(escape=False)

                    st.write(html, unsafe_allow_html=True)
                except Exception as e:
                    logging.exception("Error while processing and displaying summarized player stats")
                    st.error("An error occurred while displaying player stats.")
                    st.text(f"Exception: {str(e)}")
                    st.text(traceback.format_exc())

        # Line break
        st.write("<br>", unsafe_allow_html=True)

        # Show list of unbeaten players
        if unbeaten_players:
            st.subheader("Unbeaten Players")
            if len(unbeaten_players) == 0:
                st.write("**There are no unbeaten players.**")
            elif len(unbeaten_players) == 1:
                st.write(f"**The following player is unbeaten:**  \n{', '.join(unbeaten_players)}")
            else:
                unbeaten_players_list = "<br>".join(unbeaten_players)
                st.markdown(
                    f"**The following players " f"are unbeaten:**<br>{unbeaten_players_list}", unsafe_allow_html=True
                )

        # Show list of players who have played in every game for their team
        if played_every_game:
            st.subheader("Players Who Have Played Every Game")
            if len(played_every_game) == 0:
                st.write("No player has played in all of their team's games")
            elif len(played_every_game) == 1:
                st.write(
                    f"**The following player has played in all of their "
                    f"team's games:**  \n{', '.join(played_every_game)}"
                )
            else:
                every_game_list = "<br>".join(played_every_game)
                st.markdown(
                    f"**The following players " f"have played every game:**<br>{every_game_list}",
                    unsafe_allow_html=True,
                )

        # Line break
        st.write("**Note:** Players must have played 5+ games to qualify.")

    elif selected_section == "Detailed Division Table":
        # Handle 'Detailed Division Table' section
        # Use division-specific data

        # Header
        st.header(f"Detailed Division Table - Division {division}")
        st.write(f"**Last updated:** {date}")

        if not awaiting_results.empty:
            # Line break
            st.write("<br>", unsafe_allow_html=True)
            st.subheader("Still awaiting these results:")
            styled_awaiting_results = awaiting_results.style.hide(axis="index")
            st.write(styled_awaiting_results.to_html(escape=False), unsafe_allow_html=True)
            st.write("<br>", unsafe_allow_html=True)

        # Line break
        st.write("<br>", unsafe_allow_html=True)

        if not detailed_league_table.empty:
            # Apply styles to the DataFrame
            styled_df = detailed_league_table.style.set_properties(**{"text-align": "right"}).hide(axis="index")
            styled_df = styled_df.set_properties(subset=["Team"], **{"text-align": "left"})
            styled_df = styled_df.bar(subset=["Points"], color="#87CEEB", vmin=0)

            # Convert styled DataFrame to HTML
            html = styled_df.to_html(escape=False)

            # Display in Streamlit
            st.write(html, unsafe_allow_html=True)
        else:
            st.info(f"No detailed league table available for Division {division}.")

            # Note
            st.write("<br>", unsafe_allow_html=True)
            st.write(
                "**Note:**  \nCR stands for Conceded Rubber.  \nWO stands for Walkover. Teams are penalized \
                     one point for each walkover given."
            )

    elif selected_section == "Home/Away Splits":
        # Handle 'Home/Away Splits' section
        # Use division-specific data

        # Header
        st.header(f"Home/Away Splits - Division {division}")

        # Load and display overall scores
        overall_scores = load_overall_home_away_data(division, season_base_path)
        if overall_scores:
            # Line break
            st.write("<br>", unsafe_allow_html=True)
            st.subheader("Overall split:")

            # Sizes for the pie chart
            sizes = [float(overall_scores[0]), float(overall_scores[1])]

            # Update labels and colors
            labels = ["Home", "Away"]
            colors = ["#ff9999", "#66b3ff"]

            # Set font properties to Calibri
            prop = fm.FontProperties(family="Calibri")

            # Create columns using st.columns
            # Adjust the fractions to control the width of each column
            col1, col2 = st.columns([1, 1])

            # Plotting the chart in the first column
            with col1:
                # Create a pie chart with larger font size for labels
                fig, ax = plt.subplots(figsize=(8, 6))
                pie, texts, autotexts = ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    startangle=90,
                    autopct="",
                    textprops={"fontsize": 16, "fontproperties": prop},
                )

                # Draw a white circle in the middle to create the donut shape
                centre_circle = plt.Circle((0, 0), 0.70, fc="white")
                fig = plt.gcf()
                fig.gca().add_artist(centre_circle)

                # Place the absolute value texts inside their respective segments
                for i, (slice, value) in enumerate(zip(pie, sizes)):
                    angle = (slice.theta2 + slice.theta1) / 2
                    x, y = slice.r * np.cos(np.deg2rad(angle)), slice.r * np.sin(np.deg2rad(angle))
                    ax.text(
                        x * 0.85,
                        y * 0.85,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontproperties=prop,
                        fontsize=14,
                        color="black",
                    )

                # Add text in the center of the circle
                percentage = float(overall_scores[2]) * 100 if len(overall_scores) > 2 else 0
                plt.text(
                    0, 0, f"Home Win\n{percentage:.1f}%", ha="center", va="center", fontproperties=prop, fontsize=14
                )

                # Add title
                plt.title(f"Average Home/Away Rubbers Won in Division {division}", fontproperties=prop, size=16)

                # Ensure the pie chart is a circle
                ax.axis("equal")
                plt.tight_layout()

                # Display the plot in the Streamlit app
                st.pyplot(fig)

            # Use the second column for other content or leave it empty
            with col2:
                st.write("")  # You can add other content here if needed

            # Line break
            st.write("<br>", unsafe_allow_html=True)
            st.subheader("Split by team:")

            if not home_away_df.empty:
                # Rename columns appropriately
                home_away_df = home_away_df.rename(
                    columns={
                        "home_away_diff": "Difference",
                        "Home": "Home Venue",  # Renaming existing 'Home' column to 'Home Venue'
                        "Average Home Score": "Home",
                        "Average Away Score": "Away",
                    }
                )

                # Ensure that the 'Home', 'Away', and 'Difference' columns are numeric
                home_away_df[["Home", "Away", "Difference"]] = home_away_df[["Home", "Away", "Difference"]].apply(
                    pd.to_numeric, errors="coerce"
                )

                # Determine the range for the colormap
                vmin = home_away_df[["Home", "Away"]].min().min()
                vmax = home_away_df[["Home", "Away"]].max().max()

                # Apply a color gradient using 'Blues' colormap to 'Home' and 'Away' columns
                colormap_blues = "Blues"
                styled_home_away_df = (
                    home_away_df.style.background_gradient(
                        cmap=colormap_blues, vmin=vmin, vmax=vmax, subset=["Home", "Away"]
                    )
                    .set_properties(subset=["Home", "Away"], **{"text-align": "right"})
                    .format("{:.2f}", subset=["Home", "Away", "Difference"])
                )

                # Apply a color gradient using 'OrRd' colormap to 'Difference' column
                colormap_orrd = "OrRd"
                if "Difference" in home_away_df.columns:
                    styled_home_away_df = styled_home_away_df.background_gradient(
                        cmap=colormap_orrd, subset=["Difference"]
                    ).set_properties(subset=["Difference"], **{"text-align": "right"})

                # Hide the index
                styled_home_away_df = styled_home_away_df.hide(axis="index")

                # Display the styled DataFrame in Streamlit
                st.write(styled_home_away_df.to_html(escape=False), unsafe_allow_html=True)
            else:
                st.info(f"No home/away data available for Division {division}.")

            # Line break
            st.write("<br>", unsafe_allow_html=True)

            # Note
            st.write(
                "**Note:**  \nMatches where the home team and away team share a "
                "home venue are ignored in the calculation"
            )

            st.write(
                """
                Since 2016-17 (ignoring the incomplete 2019-20 and 2021-22 seasons), 
                the overall home advantage factor is 0.5294, meaning that in a best-of-5-rubber division
                teams win an average of 2.65 rubbers at home compared to 2.35 away.
                """
            )

        else:
            st.info(f"No overall home/away data available.")

    elif selected_section == "Rubber Win Percentage":
        # Handle 'Rubber Win Percentage' section
        # Use division-specific data

        # Header
        st.header(f"Rubber Win Percentage - Division {division}")

        # Function to apply common formatting
        def format_dataframe(df):
            if df is not None and not df.empty:

                # Rename avg_win_perc column
                df = df.rename(columns={"avg_win_perc": "Average"})
                # Select only numeric columns for vmin and vmax calculation
                numeric_cols_raw = [col for col in df.columns if "Win" in col]

                # Convert these columns to numeric type, handling non-numeric values
                for col in numeric_cols_raw:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Determine the range for the colormap
                vmin = df[numeric_cols_raw].min().min()
                vmax = df[numeric_cols_raw].max().max()

                # Format numeric columns for display
                def format_float(x):
                    try:
                        return f"{float(x):.1f}"
                    except ValueError:
                        return x

                # Build the list of columns to format (include 'Average' only if present)
                cols_to_format = list(numeric_cols_raw)
                if "Average" in df.columns:
                    # ensure 'Average' is numeric before formatting
                    df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
                    cols_to_format.append("Average")

                # Element-wise format on a DataFrame
                df[cols_to_format] = df[cols_to_format].applymap(format_float)

                # Check if "Total Rubbers" is in the DataFrame and format it as integer
                if "Total Rubbers" in df.columns:
                    df["Total Rubbers"] = df["Total Rubbers"].astype(int)

                colormap_blues = "Blues"
                cols_for_blues_gradient = numeric_cols_raw
                styled_df = df.style.background_gradient(
                    cmap=colormap_blues, vmin=vmin, vmax=vmax, subset=cols_for_blues_gradient
                ).set_properties(subset=cols_for_blues_gradient, **{"text-align": "right"})

                # Set right alignment for "Total Rubbers"
                if "Total Rubbers" in df.columns:
                    styled_df = styled_df.set_properties(subset=["Total Rubbers"], **{"text-align": "right"})

                colormap_oranges = "OrRd"
                if "Average" in df.columns:
                    styled_df = styled_df.background_gradient(cmap=colormap_oranges, subset=["Average"]).set_properties(
                        subset=["Average"], **{"text-align": "right"}
                    )

                styled_df = styled_df.hide(axis="index")
                return styled_df
            else:
                return "DataFrame is empty or not loaded."

        def format_dataframe_delta(df):
            if df is not None and not df.empty:

                # Rename avg_win_perc column
                df = df.rename(columns={"avg_win_perc": "Average"})
                # Select only numeric columns for vmin and vmax calculation
                numeric_cols_raw = [col for col in df.columns if "Win" in col]

                # Convert these columns to numeric type, handling non-numeric values
                for col in numeric_cols_raw:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Determine the range for the colormap
                vmin = df[numeric_cols_raw].min().min()
                vmax = df[numeric_cols_raw].max().max()

                # Format numeric columns for display
                def format_float(x):
                    try:
                        return f"{float(x):.1f}"
                    except ValueError:
                        return x

                df[numeric_cols_raw + ["Average"]] = df[numeric_cols_raw + ["Average"]].map(format_float)

                # Check if "Total Rubbers" is in the DataFrame and format it as integer
                if "Total Rubbers" in df.columns:
                    df["Total Rubbers"] = df["Total Rubbers"].astype(int)

                colormap_blues = "RdYlBu"
                cols_for_blues_gradient = numeric_cols_raw
                styled_df = df.style.background_gradient(
                    cmap=colormap_blues, vmin=vmin, vmax=vmax, subset=cols_for_blues_gradient
                ).set_properties(subset=cols_for_blues_gradient, **{"text-align": "right"})

                # Set right alignment for "Total Rubbers"
                if "Total Rubbers" in df.columns:
                    styled_df = styled_df.set_properties(subset=["Total Rubbers"], **{"text-align": "right"})

                colormap_oranges = "RdYlBu"
                if "Average" in df.columns:
                    styled_df = styled_df.background_gradient(cmap=colormap_oranges, subset=["Average"]).set_properties(
                        subset=["Average"], **{"text-align": "right"}
                    )

                styled_df = styled_df.hide(axis="index")
                return styled_df
            else:
                return "DataFrame is empty or not loaded."

        if multiple_breakdowns_available:
            # Radio button for user to choose the DataFrame
            option = st.radio(
                "Select Team Win Breakdown View:", ["Overall", "Home", "Away", "H/A Delta"], horizontal=True
            )
        else:
            option = "Overall"

        st.write("<br>", unsafe_allow_html=True)
        st.subheader("Team win percentage by rubber:")

        # Apply formatting and display the selected DataFrame
        dataframes = {
            "Overall": team_win_breakdown_overall,
            "Home": team_win_breakdown_home,
            "Away": team_win_breakdown_away,
            "H/A Delta": team_win_breakdown_delta,
        }

        selected_df = dataframes.get(option)
        if selected_df is not None and not selected_df.empty:
            if option == "H/A Delta":
                st.write(format_dataframe_delta(selected_df).to_html(escape=False), unsafe_allow_html=True)
            else:
                st.write(format_dataframe(selected_df).to_html(escape=False), unsafe_allow_html=True)
        else:
            st.info(f"No data available for {option} breakdown in Division {division}.")

        # Note
        st.write("<br>", unsafe_allow_html=True)
        if multiple_breakdowns_available:
            st.write(
                "**Note:**  \nOnly rubbers that were played are included. Conceded Rubbers "
                "and Walkovers are ignored.  \nMatches where the home team and away team share "
                "a home venue are ignored in the Home and Away tables."
            )
        else:
            st.write(
                "**Note:**  \nOnly rubbers that were played are included. Conceded Rubbers "
                "and Walkovers are ignored."
            )

    elif selected_section == "Projections":
        # Handle 'Projections' section (only available for current season)
        if not is_current_season:
            st.error("Projections are only available for the current season.")
        else:
            # Use division-specific data

            # Load and display overall scores
            st.header(f"Projections - Division {division}")
            if simulation_date and simulation_date != "Date not available":
                st.write(f"**Date of last simulation:** {simulation_date}")
            else:
                st.write("**Date of last simulation:** Date not available")

            if not awaiting_results.empty:
                # Line break
                st.write("<br>", unsafe_allow_html=True)
                st.subheader("Still awaiting these results:")
                styled_awaiting_results = awaiting_results.style.hide(axis="index")
                st.write(styled_awaiting_results.to_html(escape=False), unsafe_allow_html=True)
                st.write("<br>", unsafe_allow_html=True)
                st.write("<br>", unsafe_allow_html=True)

            if simulated_fixtures.empty:
                st.write("No more remaining fixtures!")
            else:
                # Convert the "Match Week" column to integers
                simulated_fixtures["Match Week"] = simulated_fixtures["Match Week"].astype(int)

                # Adjust the "Date" column format
                simulated_fixtures["Date"] = pd.to_datetime(simulated_fixtures["Date"]).dt.date

                # Rename columns
                simulated_fixtures = simulated_fixtures.rename(
                    columns={
                        "Avg Simulated Home Points": "Proj. Home Pts",
                        "Avg Simulated Away Points": "Proj. Away Pts",
                    }
                )

                # Round values in simulated_fixtures DataFrame except for "Match Week"
                numeric_cols_simulated_fixtures = simulated_fixtures.select_dtypes(
                    include=["float", "int"]
                ).columns.drop("Match Week")
                simulated_fixtures[numeric_cols_simulated_fixtures] = simulated_fixtures[
                    numeric_cols_simulated_fixtures
                ].applymap(lambda x: f"{x:.2f}")

                # Ensure the columns are numeric for vmin and vmax calculation
                simulated_fixtures_numeric = simulated_fixtures.copy()
                simulated_fixtures_numeric[numeric_cols_simulated_fixtures] = simulated_fixtures_numeric[
                    numeric_cols_simulated_fixtures
                ].apply(pd.to_numeric, errors="coerce")

                # Get the range of match weeks
                min_week = simulated_fixtures_numeric["Match Week"].min()
                max_week = simulated_fixtures_numeric["Match Week"].max()

                # Create columns for the slider or display the week directly if only one is available
                col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed
                with col1:
                    if min_week < max_week:
                        selected_week = st.slider(
                            "**Select Match Week:**", int(min_week), int(max_week), value=int(min_week), step=1
                        )
                    else:
                        selected_week = int(min_week)
                        st.write(f"**Match Week:** {selected_week}")  # Display the week directly without a slider

                # Filter the fixtures based on the selected match week
                filtered_fixtures = simulated_fixtures_numeric[
                    simulated_fixtures_numeric["Match Week"] == selected_week
                ]

                # Determine the range for the colormap
                vmin = filtered_fixtures[["Proj. Home Pts", "Proj. Away Pts"]].min().min()
                vmax = filtered_fixtures[["Proj. Home Pts", "Proj. Away Pts"]].max().max()

                # Apply styling to the filtered DataFrame
                styled_filtered_fixtures = (
                    filtered_fixtures.style.background_gradient(
                        cmap="Blues", vmin=vmin, vmax=vmax, subset=numeric_cols_simulated_fixtures
                    )
                    .set_properties(
                        subset=["Match Week", "Proj. Home Pts", "Proj. Away Pts"], **{"text-align": "right"}
                    )
                    .format("{:.2f}", subset=["Proj. Home Pts", "Proj. Away Pts"])
                    .hide(axis="index")
                )

                # Display the styled DataFrame in Streamlit
                st.subheader(f"Projected Fixtures for Match Week {selected_week}:")
                st.write(styled_filtered_fixtures.to_html(escape=False), unsafe_allow_html=True)

            if not simulated_table.empty:
                # Line break and subheader
                st.write("<br>", unsafe_allow_html=True)
                st.subheader("Projected Final Table:")

                # Convert 'Played' column to integers
                if "Played" in simulated_table.columns:
                    simulated_table["Played"] = simulated_table["Played"].astype(int)

                # Round values in simulated_table DataFrame except for 'Played'
                numeric_cols_simulated_table = simulated_table.select_dtypes(include=["float", "int"]).columns
                cols_to_round = numeric_cols_simulated_table.drop("Played")

                # Columns to exclude from gradient formatting
                cols_to_exclude = {"Played", "Won", "Lost", "Points", "Playoffs"}
                cols_for_blues_gradient = [col for col in cols_to_round if col not in cols_to_exclude]

                # Determine the range for the colormap
                vmin = simulated_table[cols_for_blues_gradient].min().min()
                vmax = simulated_table[cols_for_blues_gradient].max().max()

                # Apply a color gradient using 'Blues' colormap to selected numeric columns
                styled_simulated_table = simulated_table.style.background_gradient(
                    cmap="Blues", vmin=vmin, vmax=vmax, subset=cols_for_blues_gradient
                ).set_properties(subset=cols_for_blues_gradient + ["Played", "Won", "Lost"], **{"text-align": "right"})

                # Apply a color gradient using 'OrRd' colormap to 'Playoffs' column
                if "Playoffs" in simulated_table.columns:
                    styled_simulated_table = styled_simulated_table.background_gradient(
                        cmap="OrRd", subset=["Playoffs"]
                    ).set_properties(subset=["Playoffs"], **{"text-align": "right"})

                # Apply bar chart formatting to the 'Points' column
                styled_simulated_table = styled_simulated_table.bar(subset=["Points"], color="#87CEEB")

                # Round all numeric columns
                styled_simulated_table = styled_simulated_table.format("{:.1f}", subset=cols_to_round)

                # Apply custom formatting for zero values in cols_for_blues_gradient
                styled_simulated_table = styled_simulated_table.format(
                    lambda x: f"<span style='color: #f7fbff;'>{x:.1f}</span>" if x == 0 else f"{x:.1f}",
                    subset=cols_for_blues_gradient,
                )

                # Hide the index
                styled_simulated_table = styled_simulated_table.hide(axis="index")

                # Display the styled DataFrame in Streamlit
                st.write(styled_simulated_table.to_html(escape=False), unsafe_allow_html=True)

                # Note
                st.write("<br>", unsafe_allow_html=True)
                st.write(
                    "**Note:**  \nThe projected final table is the average result of simulating the remaining "
                    "fixtures 10,000 times.  \nFixtures are simulated using teams' average rubber win percentage, "
                    "factoring in home advantage."
                )

    elif selected_section == "Match Results":
        # We have already handled "Match Results" above, so do nothing here
        pass

    else:
        st.error("Invalid section selected.")


if __name__ == "__main__":
    main()
