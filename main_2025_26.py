# Imports
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import configuration
from config import get_config, print_config_summary

# Import parser functions
from parsers import (
    _parse_summary_row_text,
    count_games_won,
    count_valid_matches,
    determine_winner,
    home_team_won,
    normalize_rubber,
    parse_result,
    split_overall_score,
)

# Import scraper functions
from scrapers import (
    scrape_players_page,
    scrape_ranking_page,
    scrape_schedules_and_results_page,
    scrape_summary_page,
    scrape_teams_page,
)
from scripts.create_combined_results import load_all_results_and_player_results

# Import functions from other scripts
from scripts.create_player_results_database_all_divisions import run_player_results_pipeline
from utils.divisions_export import save_divisions_json

# Import validator functions
from validators import (
    PlayersValidator,
    RankingValidator,
    SchedulesValidator,
    SummaryValidator,
    TeamsValidator,
    ValidationReport,
)

# Global variables
_SUMMARY_NUMS_RE = re.compile(r"(.*?)[^\d]*?(\d+)\s*(\d+)\s*(\d+)\s*(\d+)$")

# Load configuration
config = get_config()

# Print configuration summary
print_config_summary(config)


def build_session() -> requests.Session:
    """Build a requests session with retry logic based on config."""
    session = requests.Session()
    retries = Retry(
        total=config.RETRY_TOTAL,
        backoff_factor=config.RETRY_BACKOFF_FACTOR,
        status_forcelist=config.RETRY_STATUS_FORCELIST,
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_session()

# Extract frequently used config values for backward compatibility
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
BASE = config.BASE_URL
PAGES_ID = config.PAGES_ID
year = config.SEASON_YEAR
wait_time = config.WAIT_TIME
ENABLE_VALIDATION = config.ENABLE_VALIDATION
DIVISIONS = config.DIVISIONS
REPO_ROOT = config.REPO_ROOT

# Import the ratings algorithm processor (needs REPO_ROOT to be defined first)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "process_ratings_algorithm", REPO_ROOT / "scripts" / "SquashLevels style ratings" / "process_ratings_algorithm.py"
)
process_ratings_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_ratings_module)
process_ratings_algorithm = process_ratings_module.process_ratings_algorithm


def url(path, league_id):
    """Build URL for API endpoints (legacy function, consider using config.build_url())."""
    return f"{BASE}/{path}/id/{league_id}/league/Squash/year/{year}/pages_id/{PAGES_ID}.html"


# Convenience derived views from config
all_divisions = config.get_all_divisions()
current_divisions = config.get_enabled_divisions()
weekday_groups = config.get_weekday_groups()

# Save divisions JSON
out_path = save_divisions_json(DIVISIONS, year, REPO_ROOT)
print(f"\n📁 Divisions JSON saved to: {out_path}")

# Get base directories from config
base_directories = config.get_output_directories()

# Ensure the logs directory exists
os.makedirs(base_directories["logs"], exist_ok=True)

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        RotatingFileHandler(
            str(config.get_log_file_path()),
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

# Configure StreamHandler to use utf-8 encoding
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
        handler.setStream(open(handler.stream.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False))


# Scraper functions are now imported from scrapers package
# (scrape_teams_page, scrape_summary_page, scrape_schedules_and_results_page,
#  scrape_ranking_page, scrape_players_page)


def aggregate_wins_home(team, results_df):
    """
    Calculate the number of wins for a given team in their home fixtures.

    Args:
    team (str): The name of the team.
    results_df (DataFrame): The DataFrame containing match results.

    Returns:
    DataFrame: A DataFrame with the number of wins in each rubber for the given team at home.
    """
    # Filter for home matches of the given team
    home_matches = results_df[results_df["Home Team"] == team]

    # Initialize a dictionary to store wins in each rubber
    wins = {f"Wins in Rubber {i}": 0 for i in range(1, max_rubbers + 1)}

    # Iterate through each match
    for index, row in home_matches.iterrows():
        # Assuming 'Rubbers' column is a list of scores like ['3-1', '1-3', ...]
        for i, score in enumerate(row["Rubbers"], start=1):
            if score != "NA" and score != "CR" and score != "WO":
                home_score, away_score = map(int, score.split("-"))
                if home_score > away_score:
                    wins[f"Wins in Rubber {i}"] += 1

    return pd.DataFrame(wins, index=[team])


def aggregate_wins_away(team, results_df):
    """
    Calculate the number of wins for a given team in their away fixtures.

    Args:
    team (str): The name of the team.
    results_df (DataFrame): The DataFrame containing match results.

    Returns:
    DataFrame: A DataFrame with the number of wins in each rubber for the given team away from home.
    """
    # Filter for away matches of the given team
    away_matches = results_df[results_df["Away Team"] == team]

    # Initialize a dictionary to store wins in each rubber
    wins = {f"Wins in Rubber {i}": 0 for i in range(1, max_rubbers + 1)}

    # Iterate through each match
    for index, row in away_matches.iterrows():
        # Assuming 'Rubbers' column is a list of scores like ['3-1', '1-3', ...]
        for i, score in enumerate(row["Rubbers"], start=1):
            if score != "NA" and score != "CR" and score != "WO":
                home_score, away_score = map(int, score.split("-"))
                if away_score > home_score:
                    wins[f"Wins in Rubber {i}"] += 1

    return pd.DataFrame(wins, index=[team])


def update_rubbers(row):
    """
    Function to count rubbers for and against for each team
    """
    logging.debug(f"Updating rubbers for match between {row['Home Team']} and {row['Away Team']}")

    # Update for home team
    rubbers_won[row["Home Team"]] = rubbers_won.get(row["Home Team"], 0) + row["Home Score"]
    rubbers_conceded[row["Home Team"]] = rubbers_conceded.get(row["Home Team"], 0) + row["Away Score"]

    # Update for away team
    rubbers_won[row["Away Team"]] = rubbers_won.get(row["Away Team"], 0) + row["Away Score"]
    rubbers_conceded[row["Away Team"]] = rubbers_conceded.get(row["Away Team"], 0) + row["Home Score"]


def update_counts(row):
    """
    Function to count CRs and WOs For and Against
    """
    home_score, away_score = map(int, row["Overall Score"].split("-"))
    home_wins = away_wins = 0

    for rubber in row["Rubbers"]:
        if rubber == "CR":
            # Count CRs
            if home_wins < home_score:
                cr_given_count[row["Away Team"]] = cr_given_count.get(row["Away Team"], 0) + 1
                cr_received_count[row["Home Team"]] = cr_received_count.get(row["Home Team"], 0) + 1
            else:
                cr_given_count[row["Home Team"]] = cr_given_count.get(row["Home Team"], 0) + 1
                cr_received_count[row["Away Team"]] = cr_received_count.get(row["Away Team"], 0) + 1
        elif rubber == "WO":
            # Count WOs
            if home_wins < home_score:
                wo_given_count[row["Away Team"]] = wo_given_count.get(row["Away Team"], 0) + 1
                wo_received_count[row["Home Team"]] = wo_received_count.get(row["Home Team"], 0) + 1
            else:
                wo_given_count[row["Home Team"]] = wo_given_count.get(row["Home Team"], 0) + 1
                wo_received_count[row["Away Team"]] = wo_received_count.get(row["Away Team"], 0) + 1
        else:
            # Count the rubbers won by each team
            rubber_home, rubber_away = map(int, rubber.split("-"))
            if rubber_home > rubber_away:
                home_wins += 1
            elif rubber_away > rubber_home:
                away_wins += 1


def find_max_players(df, team, column):
    """
    Function to find players with max value in a column, handling ties
    """
    max_value = df[df["Team"] == team][column].max()
    players = df[(df["Team"] == team) & (df[column] == max_value)]["Name of Player"]
    return ", ".join(players) + f" ({max_value})"


def find_max_win_percentage(df, team):
    """
    Function to find players with max win percentage, handling ties
    """
    max_value = df[df["Team"] == team]["Win Percentage"].max()
    players = df[(df["Team"] == team) & (df["Win Percentage"] == max_value)]["Name of Player"]
    return ", ".join(players) + f" ({max_value * 100:.1f}%)"


# Old scraper functions removed - now using scrapers package
# (see scrapers/ranking.py and scrapers/players.py)


def ensure_nonempty_df(df: pd.DataFrame, name: str, div: str, hard_fail: bool = True):
    if df is None or df.empty:
        msg = f"{name} is empty for Division {div}."
        if hard_fail:
            logging.error(msg + " Aborting run to avoid writing empty CSV.")
            raise SystemExit(1)
        else:
            logging.warning(msg + " Will not write CSV for this item.")
            return False
    return True


def safe_save_csv(df: pd.DataFrame, path: str, name: str, div: str, allow_empty: bool = False):
    if (df is None or df.empty) and not allow_empty:
        logging.error(f"Refusing to save empty {name} for Division {div}: {path}")
        raise SystemExit(1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved {name} to {path}")


def validate_and_save(
    validator_class,
    df: pd.DataFrame,
    league_id: str,
    year: str,
    division: str,
    validation_report: ValidationReport = None,
):
    """
    Validate a DataFrame and add result to validation report.

    Args:
        validator_class: The validator class to use
        df: DataFrame to validate
        league_id: League ID
        year: Season year
        division: Division name
        validation_report: ValidationReport instance (optional)

    Returns:
        ValidationResult or None if validation disabled
    """
    if validation_report is None or not ENABLE_VALIDATION:
        return None

    validator = validator_class(league_id=league_id, year=year, division=division)
    result = validator.validate(df)
    validation_report.add_result(result)
    validation_report.save_individual_report(result, division)

    # Log critical errors
    if not result.is_valid:
        logging.warning(
            f"⚠️  Validation FAILED for {result.data_type} in {division}: "
            f"{result.error_count} errors, {result.warning_count} warnings"
        )
    else:
        logging.info(f"✅ Validation passed for {result.data_type} in {division}")

    return result


# Use logging to track progress
logging.info("Starting the scraping process...")

# Initialize validation report if validation is enabled
if ENABLE_VALIDATION:
    validation_report = ValidationReport(output_dir=str(REPO_ROOT / year), year=year)
    logging.info("Data validation is ENABLED - validation reports will be generated")
else:
    validation_report = None
    logging.info("Data validation is DISABLED")

# Only process enabled divisions based on TESTING_MODE configuration
for div in current_divisions.keys():
    logging.info(f"Processing Division {div}")
    league_id = f"D00{all_divisions[div]}"

    # Scrape Schedules and Results page
    try:
        logging.info(f"Scraping Schedules and Results page for Division {div}")
        schedules_df = scrape_schedules_and_results_page(league_id, year, SESSION)
        logging.info(f"Successfully scraped Schedules and Results page for Division {div}")

        # Validate schedules data
        validate_and_save(SchedulesValidator, schedules_df, league_id, year, div, validation_report)
    except Exception as e:
        logging.error(f"Error scraping Schedules and Results page for Division {div}: {e}")
        continue

    # Check if the schedules_df is empty
    if schedules_df.empty:
        logging.warning(f"No data found in schedules_df for Division {div}. Skipping further processing.")
        continue

    # Filter schedules_df to only include matches where 'Result' is not empty
    played_matches_df = schedules_df[schedules_df["Result"].notna() & (schedules_df["Result"] != "")]

    # Check if played_matches_df is empty
    if played_matches_df.empty:
        logging.warning(f"No played matches found in schedules_df for Division {div}. Skipping further processing.")
        match_week = 0
    else:
        # Get the latest match week number for which data is available and ensure it is an integer
        match_week = played_matches_df["Match Week"].max()
        match_week = int(match_week)

    # Create week specific directories
    week_dir = f"week_{match_week}" if match_week > 0 else "week_0"

    # Create directories for each base directory
    for base_dir in base_directories.values():
        # Combine the base directory with the week directory
        full_dir = os.path.join(base_dir, week_dir)
        # Create the directory if it doesn't exist
        os.makedirs(full_dir, exist_ok=True)

    overall_scores_file = os.path.join(base_directories["home_away_data"], week_dir, f"{div}_overall_scores.csv")

    # Try to load existing; otherwise create an empty frame with numeric columns 0..4
    if os.path.exists(overall_scores_file):
        try:
            overall_scores_df = pd.read_csv(overall_scores_file, header=None)
            # Ensure at least 5 columns exist
            if overall_scores_df.shape[1] < 5:
                logging.warning(
                    f"overall_scores_file has {overall_scores_df.shape[1]} columns, expected 5. Creating new DataFrame."
                )
                overall_scores_df = pd.DataFrame(columns=[0, 1, 2, 3, 4])
        except Exception as e:
            logging.exception(f"Failed to read overall_scores_file; creating new 5-col DataFrame: {e}")
            overall_scores_df = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    else:
        overall_scores_df = pd.DataFrame(columns=[0, 1, 2, 3, 4])

    # Save the schedules_df to CSV
    schedules_df_path = os.path.join(base_directories["schedules_df"], week_dir, f"{div}_schedules_df.csv")
    try:
        logging.info(f"Saving schedules_df to {schedules_df_path}")
        schedules_df.to_csv(schedules_df_path, index=False)
        logging.info(f"Successfully saved schedules_df to {schedules_df_path}")
    except Exception as e:
        logging.error(f"Error saving schedules_df to {schedules_df_path}: {e}")

    time.sleep(wait_time)

    # Scrape Team Summary page
    try:
        logging.info(f"Scraping Team Summary page for Division {div}")
        summary_df = scrape_summary_page(league_id, year, SESSION)
        logging.info(f"Successfully scraped Team Summary page for Division {div}")

        # Validate summary data
        validate_and_save(SummaryValidator, summary_df, league_id, year, div, validation_report)
    except Exception as e:
        logging.error(f"Error scraping Team Summary page for Division {div}: {e}")
        raise

    # Validate summary_df is not empty; hard fail if it is
    ensure_nonempty_df(summary_df, "summary_df", div, hard_fail=True)

    # If we reached here, it's non-empty → save it
    week_dir = f"week_{match_week}" if "match_week" in locals() and match_week > 0 else "week_0"
    summary_df_path = os.path.join(base_directories["summary_df"], week_dir, f"{div}_summary_df.csv")
    safe_save_csv(summary_df, summary_df_path, "summary_df", div, allow_empty=False)

    has_summary = True

    time.sleep(wait_time)

    # Scrape Teams page
    try:
        logging.info(f"Scraping Teams page for Division {div}")
        teams_df = scrape_teams_page(league_id, year, SESSION)
        logging.info(f"Successfully scraped Teams page for Division {div}")

        # Validate teams data
        validate_and_save(TeamsValidator, teams_df, league_id, year, div, validation_report)
    except Exception as e:
        logging.error(f"Error scraping Teams page for Division {div}: {e}")
        continue

    # Save the teams_df to CSV
    teams_df_path = os.path.join(base_directories["teams_df"], week_dir, f"{div}_teams_df.csv")
    try:
        logging.info(f"Saving teams_df to {teams_df_path}")
        teams_df.to_csv(teams_df_path, index=False)
        logging.info(f"Successfully saved teams_df to {teams_df_path}")
    except Exception as e:
        logging.error(f"Error saving teams_df to {teams_df_path}: {e}")

    time.sleep(wait_time)

    # Scrape Ranking page
    try:
        logging.info(f"Scraping Ranking page for Division {div}")
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scrape_ranking_page(league_id, year, SESSION)
        logging.info(f"Successfully scraped Ranking page for Division {div}")

        # Validate ranking data
        if ranking_df is not None and not ranking_df.empty:
            validate_and_save(RankingValidator, ranking_df, league_id, year, div, validation_report)
    except Exception as e:
        logging.error(f"Error scraping Ranking page for Division {div}: {e}")
        # Stop execution if an error occurs
        raise

    # Save the ranking_df to CSV if it is not None and not empty
    if ranking_df is not None and not ranking_df.empty:
        ranking_df_path = os.path.join(base_directories["ranking_df"], week_dir, f"{div}_ranking_df.csv")
        try:
            logging.info(f"Saving ranking_df to {ranking_df_path}")
            ranking_df.to_csv(ranking_df_path, index=False)
            logging.info(f"Successfully saved ranking_df to {ranking_df_path}")
        except Exception as e:
            logging.error(f"Error saving ranking_df to {ranking_df_path}: {e}")
    else:
        logging.info(f"No ranking data to save for Division {div}; skipping ranking_df CSV creation.")

    time.sleep(wait_time)

    # Scrape Players page
    try:
        logging.info(f"Scraping Players page for Division {div}")
        players_df = scrape_players_page(league_id, year, SESSION)
        logging.info(f"Successfully scraped Players page for Division {div}")

        # Validate players data
        validate_and_save(PlayersValidator, players_df, league_id, year, div, validation_report)
    except Exception as e:
        logging.error(f"Error scraping Players page for Division {div}: {e}")
        continue

    # Save the players_df to CSV
    players_df_path = os.path.join(base_directories["players_df"], week_dir, f"{div}_players_df.csv")
    try:
        logging.info(f"Saving players_df to {players_df_path}")
        players_df.to_csv(players_df_path, index=False)
        logging.info(f"Successfully saved players_df to {players_df_path}")
    except Exception as e:
        logging.error(f"Error saving players_df to {players_df_path}: {e}")

    time.sleep(wait_time)

    # Get list of players who have played every possible game
    played_every_game_list = []

    if has_summary and ranking_df_filtered is not None and not ranking_df_filtered.empty:
        # Merge team-level "Played" into player rows
        merged_ranking_df = ranking_df_filtered.merge(summary_df[["Team", "Played"]], on="Team", how="inner")
        # Ensure numeric
        merged_ranking_df["Played"] = pd.to_numeric(merged_ranking_df["Played"], errors="coerce").fillna(0).astype(int)
        merged_ranking_df["Games Played"] = (
            pd.to_numeric(merged_ranking_df["Games Played"], errors="coerce").fillna(0).astype(int)
        )

        # Players who have played every game their team has played
        played_every_game_list = (
            merged_ranking_df[merged_ranking_df["Games Played"] == merged_ranking_df["Played"]]
            .apply(lambda row: f"{row['Name of Player']} ({row['Team']})", axis=1)
            .tolist()
        )
    else:
        logging.warning(
            f"No usable summary/ranking data for Division {div}. Unable to determine players who have played every game."
        )

    # Save the summarized_df to CSV if it is not None and not empty
    if summarized_df is not None and not summarized_df.empty:
        summarized_df_path = os.path.join(
            base_directories["summarized_player_tables"], week_dir, f"{div}_summarized_players.csv"
        )
        try:
            logging.info(f"Saving summarized_df to {summarized_df_path}")
            summarized_df.to_csv(summarized_df_path, index=False)
            logging.info(f"Successfully saved summarized_df to {summarized_df_path}")
        except Exception as e:
            logging.error(f"Error saving summarized_df to {summarized_df_path}: {e}")
    else:
        logging.info(f"No summarized data to save for Division {div}; skipping summarized_df CSV creation.")

    # Save the unbeaten_list to a text file (create a blank file if no unbeaten players)
    unbeaten_players_path = os.path.join(base_directories["unbeaten_players"], week_dir, f"{div}.txt")
    try:
        logging.info(f"Saving unbeaten_list to {unbeaten_players_path}")
        with open(unbeaten_players_path, "w") as f:
            for player in unbeaten_list:
                f.write(f"{player}\n")
        logging.info(f"Successfully saved unbeaten_list to {unbeaten_players_path}")
    except Exception as e:
        logging.error(f"Error saving unbeaten_list to {unbeaten_players_path}: {e}")

    # Save list of players who have played every game (create a blank file if none)
    played_every_game_path = os.path.join(base_directories["played_every_game"], week_dir, f"{div}.txt")
    try:
        logging.info(f"Saving played_every_game_list to {played_every_game_path}")
        with open(played_every_game_path, "w") as f:
            for player in played_every_game_list:
                f.write(f"{player}\n")
        logging.info(f"Successfully saved played_every_game_list to {played_every_game_path}")
    except Exception as e:
        logging.error(f"Error saving played_every_game_list to {played_every_game_path}: {e}")

    # Create Results Dataframe

    # Drop unnecessary columns
    schedules_df.drop(columns=["vs", "Time"], inplace=True, errors="ignore")

    # Exclude rows where 'Away Team' is '[BYE]' (indicative of a bye week)
    results_df = schedules_df[schedules_df["Away Team"] != "[BYE]"].copy()

    # Replace NaN values in 'Result' with an empty string before applying str.contains
    results_df["Result"] = results_df["Result"].fillna("")

    # Keep rows where 'Result' contains brackets (indicative of a played match)
    results_df = results_df[results_df["Result"].str.contains(r"\(", na=False)]

    # Check if the results_df is empty
    if results_df.empty:
        logging.warning(f"No data found in results_df for Division {div}. Skipping further processing.")
        continue

    # Apply the parse_result function to the 'Result' column
    results_df[["Overall Score", "Rubbers"]] = results_df["Result"].apply(lambda x: pd.Series(parse_result(x)))

    # Apply the normalize_rubber function to the 'Rubbers' column
    results_df["Rubbers"] = results_df["Rubbers"].apply(lambda lst: [normalize_rubber(x) for x in lst])

    # Drop the original 'Result' column
    results_df.drop(columns=["Result"], inplace=True)

    # Count the number of Rubbers For and Against for each team

    # Splitting the 'Overall Score' into two separate columns
    results_df[["Home Score", "Away Score"]] = results_df["Overall Score"].str.split("-", expand=True).astype(int)

    # Initialize dictionaries to keep track of won and conceded rubbers
    rubbers_won = {}
    rubbers_conceded = {}

    # Create Games Won columns
    results_df[["Home Games Won", "Away Games Won"]] = results_df.apply(count_games_won, axis=1, result_type="expand")

    # Apply the function to each row
    results_df.apply(update_rubbers, axis=1)

    # Convert the dictionaries to DataFrames
    df_rubbers_won = pd.DataFrame(list(rubbers_won.items()), columns=["Team", "Rubbers For"])
    df_rubbers_conceded = pd.DataFrame(list(rubbers_conceded.items()), columns=["Team", "Rubbers Against"])

    # Merge the DataFrames on Team
    rubbers_df = pd.merge(df_rubbers_won, df_rubbers_conceded, on="Team")

    # Count the number Conceded Rubbers and Walkovers

    # Initialize dictionaries to keep track of conceded rubbers and walkovers
    cr_given_count = {}
    cr_received_count = {}
    wo_given_count = {}
    wo_received_count = {}

    # Apply the function to each row
    results_df.apply(update_counts, axis=1)

    # Ensure all teams are included in all counts
    all_teams = set(results_df["Home Team"]).union(set(results_df["Away Team"]))
    for team in all_teams:
        cr_given_count.setdefault(team, 0)
        cr_received_count.setdefault(team, 0)
        wo_given_count.setdefault(team, 0)
        wo_received_count.setdefault(team, 0)

    # Convert the dictionaries to DataFrames
    df_cr_given_count = pd.DataFrame(list(cr_given_count.items()), columns=["Team", "CRs Given"])
    df_cr_received_count = pd.DataFrame(list(cr_received_count.items()), columns=["Team", "CRs Received"])
    df_wo_given_count = pd.DataFrame(list(wo_given_count.items()), columns=["Team", "WOs Given"])
    df_wo_received_count = pd.DataFrame(list(wo_received_count.items()), columns=["Team", "WOs Received"])

    # Merge the DataFrames on Team
    detailed_table_df = pd.merge(df_cr_given_count, df_cr_received_count, on="Team")
    detailed_table_df = pd.merge(detailed_table_df, df_wo_given_count, on="Team")
    detailed_table_df = pd.merge(detailed_table_df, df_wo_received_count, on="Team")

    # If team summary is available, merge rubbers data into it, then merge with detailed table
    if has_summary:
        summary_plus = summary_df.merge(rubbers_df, on="Team", how="left")
        detailed_table_df = summary_plus.merge(detailed_table_df, on="Team", how="left")
    else:
        # no team summary available yet; keep counts-only table
        # (optionally skip writing detailed table this week)
        pass

    # Save detailed league table
    detailed_table_df.to_csv(
        os.path.join(base_directories["detailed_league_tables"], week_dir, f"{div}_detailed_league_table.csv"),
        index=False,
    )

    # Create Remaining Fixtures Dataframe
    # Filter out rows where 'Result' is not empty and does not contain placeholder text
    # Keep rows where 'Result' is empty (None or empty string)
    df_remaining_fixtures = schedules_df[(schedules_df["Result"].isna()) | (schedules_df["Result"] == "")]

    # Filter out rows with byes
    df_remaining_fixtures = df_remaining_fixtures[df_remaining_fixtures["Away Team"] != "[BYE]"]

    # Filter out redundant Results column
    df_remaining_fixtures = df_remaining_fixtures[["Home Team", "Away Team", "Venue", "Match Week", "Date"]]

    # Convert the 'Date' column to datetime if it's not already
    df_remaining_fixtures["Date"] = pd.to_datetime(df_remaining_fixtures["Date"], dayfirst=True)

    # Create remaining fixtures folder if it doesn't exist
    os.makedirs(os.path.join(base_directories["remaining_fixtures"], week_dir), exist_ok=True)

    # Save remaining fixtures
    df_remaining_fixtures.to_csv(
        os.path.join(base_directories["remaining_fixtures"], week_dir, f"{div}_remaining_fixtures.csv"), index=False
    )

    # Filter rows where the 'Date' is earlier than today to create awaiting_results dataframe
    today = pd.Timestamp(datetime.now().date())
    awaiting_results_df = df_remaining_fixtures[df_remaining_fixtures["Date"] < today]
    # Save awaiting results
    awaiting_results_df.to_csv(
        os.path.join(base_directories["awaiting_results"], week_dir, f"{div}_awaiting_results.csv"), index=False
    )

    # Create results dataframe that ignores games where away team plays at home venue
    # Create dictionary of team home venues
    if teams_df is None or teams_df.empty or not {"Team Name", "Home"}.issubset(teams_df.columns):
        logging.warning("teams_df empty/missing columns; skipping venue classification")
        valid_matches_df = results_df.copy()
        neutral_fixtures_df = df_remaining_fixtures.iloc[0:0].copy()
    else:
        team_home_venues = teams_df.set_index("Team Name")["Home"].to_dict()

        def venue_type(row):
            hv = team_home_venues.get(row["Home Team"])
            av = team_home_venues.get(row["Away Team"])
            v = row["Venue"]
            is_home = hv is not None and v == hv
            is_away = av is not None and v == av
            if is_home and not is_away:
                return "home"  # true home
            if is_away and not is_home:
                return "away"  # away team playing at (its) home
            if is_home and is_away:
                return "shared_home"  # both teams’ home venue (common in club leagues)
            return "neutral"  # neither team’s home venue

        results_df = results_df.copy()
        results_df["VenueType"] = results_df.apply(venue_type, axis=1)

        # Use only true home and true away for home/away advantage stats
        valid_matches_df = results_df[results_df["VenueType"].isin(["home", "away"])].copy()

        # Save “neutral-like” fixtures separately (shared_home + neutral)
        neutral_fixtures_df = df_remaining_fixtures.copy()
        if not neutral_fixtures_df.empty:

            def remaining_venue_type(row):
                av = team_home_venues.get(row["Away Team"])
                hv = team_home_venues.get(row["Home Team"])
                v = row["Venue"]
                is_home = hv is not None and v == hv
                is_away = av is not None and v == av
                if is_home and is_away:
                    return "shared_home"
                if (not is_home) and (not is_away):
                    return "neutral"
                return "other"

            neutral_fixtures_df = neutral_fixtures_df[
                neutral_fixtures_df.apply(remaining_venue_type, axis=1).isin(["shared_home", "neutral"])
            ].copy()

    # Create folder for neutral fixtures if it doesn't exist
    os.makedirs(os.path.join(base_directories["neutral_fixtures"], week_dir), exist_ok=True)

    # Save neutral fixtures
    neutral_fixtures_df.to_csv(
        os.path.join(base_directories["neutral_fixtures"], week_dir, f"{div}_neutral_fixtures.csv"), index=False
    )

    # Calculate Home vs Away
    if not valid_matches_df.empty:
        # Split overall score into numeric columns (so later groupbys work)
        valid_matches_df[["Home Overall Score", "Away Overall Score"]] = valid_matches_df["Overall Score"].apply(
            lambda x: pd.Series(split_overall_score(x))
        )

        # League-wide averages
        average_home_overall_score = valid_matches_df["Home Overall Score"].mean()
        average_away_overall_score = valid_matches_df["Away Overall Score"].mean()

        # Winner per fixture (ignore exact ties after tiebreak)
        valid_matches_df["Winner"] = valid_matches_df.apply(home_team_won, axis=1)
        home_win_perc = (
            valid_matches_df[valid_matches_df["Winner"] != "Ignore"]["Winner"]
            .value_counts(normalize=True)
            .get("Home", 0.0)
        )

        # ---- per-team averages (must be inside this block so the columns exist) ----
        average_home_scores = (
            valid_matches_df.groupby("Home Team")["Home Overall Score"].mean().rename("Average Home Score")
        )
        average_away_scores = (
            valid_matches_df.groupby("Away Team")["Away Overall Score"].mean().rename("Average Away Score")
        )
        team_average_scores = pd.concat([average_home_scores, average_away_scores], axis=1).fillna(0.0)

    else:
        logging.warning(f"No results data to calculate home vs away statistics for Division {div}.")
        average_home_overall_score = 0.0
        average_away_overall_score = 0.0
        home_win_perc = 0.0
        # empty placeholder so later code can run without additional branching
        team_average_scores = pd.DataFrame(columns=["Average Home Score", "Average Away Score"])

    # Handle missing values by filling NaN with 0 or using appropriate methods
    team_average_scores["Average Home Score"] = team_average_scores["Average Home Score"].fillna(0)
    team_average_scores["Average Away Score"] = team_average_scores["Average Away Score"].fillna(0)

    # Calculate the difference in home and away scores for each team
    team_average_scores["home_away_diff"] = (
        team_average_scores["Average Home Score"] - team_average_scores["Average Away Score"]
    )

    # Merge with teams_df to get home venue info (only if available)
    if teams_df is not None and not teams_df.empty and {"Team Name", "Home"}.issubset(set(teams_df.columns)):
        # Keep the team name index from team_average_scores and LEFT-join venue info
        team_average_scores = team_average_scores.merge(
            teams_df[["Team Name", "Home"]],
            left_index=True,
            right_on="Team Name",
            how="left",  # don't drop teams if venue lookup is missing
        )
        # Ensure we KEEP the diff column
        cols_to_keep = ["Team Name", "Home", "Average Home Score", "Average Away Score", "home_away_diff"]
        team_average_scores = team_average_scores.reindex(columns=cols_to_keep)
    else:
        # If we can't attach venues, keep the team index as a column instead
        team_average_scores = team_average_scores.reset_index().rename(columns={"index": "Team Name"})
        team_average_scores["Home"] = ""
        cols_to_keep = ["Team Name", "Home", "Average Home Score", "Average Away Score", "home_away_diff"]
        team_average_scores = team_average_scores.reindex(columns=cols_to_keep)

    # Since 'home_away_diff' may not be meaningful at this point, you can add a check
    if team_average_scores["home_away_diff"].isnull().all():
        logging.warning(
            "All 'home_away_diff' values are NaN or zero. Teams may not have played both home and away games yet."
        )
    else:
        # Sort the DataFrame based on 'home_away_diff'
        team_average_scores.sort_values("home_away_diff", ascending=False, inplace=True)

    # Save team_average_scores to csv
    team_average_scores.to_csv(
        os.path.join(base_directories["home_away_data"], week_dir, f"{div}_team_average_scores.csv"), index=False
    )

    # Show home/away split by venue
    if (
        not team_average_scores.empty
        and "Home" in team_average_scores.columns
        and "home_away_diff" in team_average_scores.columns
    ):
        venue_split = team_average_scores.pivot_table(index="Home", values="home_away_diff", aggfunc="mean")
        # pivot with a single `values` returns a Series or single-column DataFrame.
        # Handle both cases safely:
        if isinstance(venue_split, pd.Series):
            venue_split = venue_split.sort_values(ascending=False)
        else:
            venue_split = venue_split.sort_values(by="home_away_diff", ascending=False)
    else:
        logging.warning("Skipping venue split: required columns missing or no data.")
        venue_split = pd.DataFrame()  # or leave it undefined

    # Analyze Teams by Rubber

    # Find the maximum number of rubbers in any match
    max_rubbers = results_df["Rubbers"].apply(len).max()

    # Apply the function to each rubber in the list
    for i in range(1, max_rubbers + 1):
        rubber_column = f"Rubber {i}"
        results_df[f"Winner {rubber_column}"] = results_df.apply(
            lambda row: determine_winner(
                row["Rubbers"][i - 1] if i <= len(row["Rubbers"]) else pd.NA, row["Home Team"], row["Away Team"]
            ),
            axis=1,
        )

    # Aggregate the number of wins for each team in each rubber
    aggregate_wins = pd.DataFrame()
    for i in range(1, max_rubbers + 1):
        rubber_column = f"Rubber {i}"
        winner_column = f"Winner {rubber_column}"
        wins = results_df[winner_column].value_counts().rename(f"Wins in {rubber_column}")
        aggregate_wins = pd.concat([aggregate_wins, wins], axis=1)

    # Fill NaN values in aggregate wins with zeros
    aggregate_wins.fillna(0, inplace=True)

    # Convert wins to integer type
    aggregate_wins = aggregate_wins.astype(int)

    # Create Home Win Percentage by Rubber dataframe
    # Initialize an empty list to store the results for each team
    home_results_list = []

    # Check if valid_matches_df is empty
    if valid_matches_df.empty:
        logging.warning("No valid matches to process for home win percentages.")
        # Create an empty DataFrame with the required columns
        home_results = pd.DataFrame(
            columns=["Team"]
            + [f"Wins in Rubber {i}" for i in range(1, max_rubbers + 1)]
            + [f"Rubber {i}" for i in range(1, max_rubbers + 1)]
            + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]
            + ["avg_win_perc", "Total Rubbers"]
        )

    else:
        # Iterate through each team in the "Home Team" column
        for team in valid_matches_df["Home Team"].unique():
            # Filter for matches where the current team is playing at home
            team_home_fixtures = valid_matches_df[valid_matches_df["Home Team"] == team]

            # Get counts per team
            total_home_matches_per_rubber_counts = {
                f"Rubber {i}": count_valid_matches(team_home_fixtures, i - 1) for i in range(1, max_rubbers + 1)
            }

            # Extract counts for the specific team
            total_home_matches_per_rubber = {
                rubber: counts.get(team, 0) for rubber, counts in total_home_matches_per_rubber_counts.items()
            }

            # Convert the dictionary to a DataFrame
            total_home_matches_df = pd.DataFrame([total_home_matches_per_rubber], index=[team])

            # Calculate total games played by summing all the rubber matches for each team
            total_rubbers_played = total_home_matches_df.sum(axis=1)

            # Merge with aggregate wins for the team's home fixtures and calculate win percentages
            team_combined_home = aggregate_wins_home(team, valid_matches_df).merge(
                total_home_matches_df, left_index=True, right_index=True, how="outer"
            )

            team_combined_home.fillna(0, inplace=True)

            for i in range(1, max_rubbers + 1):
                rubber_column = f"Rubber {i}"
                team_combined_home[f"{rubber_column} Win %"] = (
                    team_combined_home[f"Wins in {rubber_column}"] / team_combined_home[rubber_column]
                ) * 100

            team_combined_home.fillna(0, inplace=True)
            team_combined_home["avg_win_perc"] = team_combined_home[
                [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]
            ].mean(axis=1)

            # Add the total rubbers played to the DataFrame
            team_combined_home["Total Rubbers"] = total_rubbers_played

            # Append the team's results to the list
            home_results_list.append(team_combined_home)

        # Concatenate all team results into a single DataFrame
        home_results = pd.concat(home_results_list)

    # Check if home_results is empty
    if home_results.empty:
        logging.warning("No home results to process.")
        win_percentage_home_df = pd.DataFrame(
            columns=["Team"]
            + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]
            + ["avg_win_perc", "Total Rubbers"]
        )
    else:
        # Sort and format the DataFrame
        home_results_sorted = home_results.sort_values("avg_win_perc", ascending=False)
        home_results_sorted = home_results_sorted.reset_index().rename(columns={"index": "Team"})

        # Selecting and displaying the required columns
        keep_columns_home = (
            ["Team"] + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)] + ["avg_win_perc", "Total Rubbers"]
        )
        win_percentage_home_df = home_results_sorted[keep_columns_home]

    # Save win_percentage_home_df to csv
    win_percentage_home_df.to_csv(
        os.path.join(
            base_directories["team_win_percentage_breakdown_home"],
            week_dir,
            f"{div}_team_win_percentage_breakdown_home.csv",
        ),
        index=False,
    )

    # Create Away Win Percentage by Rubber dataframe
    # Initialize an empty list to store the results for each team
    away_results_list = []

    # Check if valid_matches_df is empty
    if valid_matches_df.empty:
        logging.warning("No valid matches to process for away win percentages.")
        # Create an empty DataFrame with the required columns
        away_results = pd.DataFrame(
            columns=["Team"]
            + [f"Wins in Rubber {i}" for i in range(1, max_rubbers + 1)]
            + [f"Rubber {i}" for i in range(1, max_rubbers + 1)]
            + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]
            + ["avg_win_perc", "Total Rubbers"]
        )
    else:
        # Iterate through each team in the "Away Team" column
        for team in valid_matches_df["Away Team"].unique():
            # Filter for matches where the current team is playing away
            team_away_fixtures = valid_matches_df[valid_matches_df["Away Team"] == team]

            # Get counts per team
            total_away_matches_per_rubber_counts = {
                f"Rubber {i}": count_valid_matches(team_away_fixtures, i - 1) for i in range(1, max_rubbers + 1)
            }

            # Extract counts for the specific team
            total_away_matches_per_rubber = {
                rubber: counts.get(team, 0) for rubber, counts in total_away_matches_per_rubber_counts.items()
            }

            # Convert the dictionary to a DataFrame
            total_away_matches_df = pd.DataFrame([total_away_matches_per_rubber], index=[team])

            # Calculate total games played by summing all the rubber matches for each team
            total_rubbers_played = total_away_matches_df.sum(axis=1)

            # Merge with aggregate wins for the team's away fixtures and calculate win percentages
            team_combined_away = aggregate_wins_away(team, valid_matches_df).merge(
                total_away_matches_df, left_index=True, right_index=True, how="outer"
            )

            team_combined_away.fillna(0, inplace=True)

            for i in range(1, max_rubbers + 1):
                rubber_column = f"Rubber {i}"
                team_combined_away[f"{rubber_column} Win %"] = (
                    team_combined_away[f"Wins in {rubber_column}"] / team_combined_away[rubber_column]
                ) * 100

            team_combined_away.fillna(0, inplace=True)
            team_combined_away["avg_win_perc"] = team_combined_away[
                [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]
            ].mean(axis=1)

            # Add the total rubbers played to the DataFrame
            team_combined_away["Total Rubbers"] = total_rubbers_played

            # Append the team's results to the list
            away_results_list.append(team_combined_away)

        # Concatenate all team results into a single DataFrame
        away_results = pd.concat(away_results_list)

    # Check if away_results is empty
    if away_results.empty:
        logging.warning("No away results to process.")
        win_percentage_away_df = pd.DataFrame(
            columns=["Team"]
            + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]
            + ["avg_win_perc", "Total Rubbers"]
        )
    else:
        # Sort and format the DataFrame
        away_results_sorted = away_results.sort_values("avg_win_perc", ascending=False)
        away_results_sorted = away_results_sorted.reset_index().rename(columns={"index": "Team"})

        # Selecting and displaying the required columns
        keep_columns_away = (
            ["Team"] + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)] + ["avg_win_perc", "Total Rubbers"]
        )

        win_percentage_away_df = away_results_sorted[keep_columns_away]

    # Save win_percentage_away_df to csv
    win_percentage_away_df.to_csv(
        os.path.join(
            base_directories["team_win_percentage_breakdown_away"],
            week_dir,
            f"{div}_team_win_percentage_breakdown_away.csv",
        ),
        index=False,
    )

    # Create Win Percentage Delta Dataframe
    # Merge the two DataFrames on the 'Team' column
    merged_df = win_percentage_home_df.merge(win_percentage_away_df, on="Team", suffixes=("_home", "_away"))

    # Initialize a dictionary to store the win percentage differences
    delta_data = {"Team": merged_df["Team"]}

    # Calculate the differences for each 'Rubber Win %' column
    for i in range(1, max_rubbers + 1):
        rubber_column = f"Rubber {i} Win %"
        delta_data[rubber_column] = merged_df[f"{rubber_column}_home"] - merged_df[f"{rubber_column}_away"]

    # Calculate the difference for the 'avg_win_perc' column
    delta_data["avg_win_perc"] = merged_df["avg_win_perc_home"] - merged_df["avg_win_perc_away"]

    # Create the win_percentage_delta_df DataFrame
    win_percentage_delta_df = pd.DataFrame(delta_data)

    # Sort values by avg_win_perc
    win_percentage_delta_df = win_percentage_delta_df.sort_values("avg_win_perc", ascending=False)

    # Save win_percentage_delta_df to csv
    win_percentage_delta_df.to_csv(
        os.path.join(
            base_directories["team_win_percentage_breakdown_delta"],
            week_dir,
            f"{div}_team_win_percentage_breakdown_delta.csv",
        ),
        index=False,
    )

    # Create overall Win Percentage by Rubber dataframe
    # Calculate total matches for each rubber excluding 'NA', 'CR', and 'WO'
    total_matches_per_rubber = {
        f"Rubber {i}": count_valid_matches(results_df, i - 1) for i in range(1, max_rubbers + 1)
    }

    # Convert the dictionary to a DataFrame with teams as index
    total_matches_df = pd.DataFrame(total_matches_per_rubber)

    # Ensure indices are aligned before merging
    aggregate_wins = aggregate_wins.fillna(0).astype(int)
    total_matches_df = total_matches_df.fillna(0).astype(int)

    # Debug logging to check index alignment
    logging.debug(f"aggregate_wins index: {list(aggregate_wins.index)}")
    logging.debug(f"total_matches_df index: {list(total_matches_df.index)}")

    # Properly merge total matches and aggregate wins based on team names
    combined = aggregate_wins.merge(total_matches_df, left_index=True, right_index=True, how="outer")

    # Replace NaN in wins and total matches with 0
    combined.fillna(0, inplace=True)

    # Calculate win percentage
    for i in range(1, max_rubbers + 1):
        rubber_column = f"Rubber {i}"
        combined[f"{rubber_column} Win %"] = (combined[f"Wins in {rubber_column}"] / combined[rubber_column]) * 100

    # Replace NaN in win % columns 0
    combined.fillna(0, inplace=True)

    # Calculate average win percentage
    combined["avg_win_perc"] = combined[[f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)]].mean(axis=1)

    # Calculate total games played by summing all the rubber matches for each team
    combined["Total Rubbers"] = total_matches_df.sum(axis=1)

    # Sort by total wins
    combined_sorted = combined.sort_values("avg_win_perc", ascending=False)

    # Reset the index
    combined_sorted = combined_sorted.reset_index().rename(columns={"index": "Team"})

    # Filter out unnecessary columns
    keep_columns = (
        ["Team"] + [f"Rubber {i} Win %" for i in range(1, max_rubbers + 1)] + ["avg_win_perc", "Total Rubbers"]
    )

    # Select only the win percentage columns and the avg_win_perc column
    win_percentage_df = combined_sorted[keep_columns]

    # Save win_percentage_df to csv
    win_percentage_df.to_csv(
        os.path.join(
            base_directories["team_win_percentage_breakdown_overall"],
            week_dir,
            f"{div}_team_win_percentage_breakdown.csv",
        ),
        index=False,
    )

    # Ensure the DataFrame has the necessary columns
    for col in [0, 1, 2, 3, 4]:
        if col not in overall_scores_df.columns:
            overall_scores_df[col] = None

    # Prepare the data to update or insert
    new_data = [average_home_overall_score, average_away_overall_score, home_win_perc, today, today]

    # Ensure DataFrame has at least one row before assignment
    if overall_scores_df.empty:
        overall_scores_df = pd.DataFrame([new_data], columns=[0, 1, 2, 3, 4])
    else:
        # Assign the data to the first row
        overall_scores_df.loc[0] = new_data

    # Write the updated DataFrame back to the CSV file
    overall_scores_df.to_csv(overall_scores_file, index=False, header=None)

    # Save the results_df to CSV
    results_df.to_csv(os.path.join(base_directories["results_df"], week_dir, f"{div}_results_df.csv"), index=False)

    # Wait so as not to get a connection error
    time.sleep(wait_time)

# After scraping all divisions for the week, run the player results pipeline
run_player_results_pipeline()

# After player results are created, combine all results and player results for the season
season_base_path = REPO_ROOT / year
combined_results_df, combined_player_results_df = load_all_results_and_player_results(season_base_path)
combined_results_df.to_csv(season_base_path / "combined_results_df.csv", index=False)
combined_player_results_df.to_csv(season_base_path / "combined_player_results_df.csv", index=False)
print("Post-scrape player-results + combine done.")

# Process ratings algorithm after combined files are saved
logging.info("Processing player ratings algorithm...")
try:
    process_ratings_algorithm(base_folder=REPO_ROOT, current_season=year, previous_season="2024-2025")
    logging.info("Player ratings processed successfully")
except Exception as e:
    logging.error(f"Error processing player ratings: {e}")
    logging.exception("Full traceback:")

# Generate and save validation report if validation was enabled
if ENABLE_VALIDATION and validation_report is not None:
    logging.info("\n" + "=" * 70)
    logging.info("GENERATING VALIDATION REPORT")
    logging.info("=" * 70)

    # Print summary to console
    validation_report.print_summary()

    # Save summary reports
    validation_report.save_summary_report()

    # Create and save error summary DataFrame
    error_df = validation_report.create_error_summary_dataframe()
    if not error_df.empty:
        error_summary_path = season_base_path / "validation_reports" / "error_summary.csv"
        error_df.to_csv(error_summary_path, index=False)
        logging.info(f"Saved error summary to {error_summary_path}")

    # Log critical issues
    if validation_report.has_errors():
        logging.warning(f"⚠️  VALIDATION ISSUES DETECTED:")
        failed_validations = validation_report.get_failed_validations()
        for result in failed_validations:
            division = result.metadata.get("division", "N/A")
            logging.warning(
                f"  • {result.data_type} ({division}): " f"{result.error_count} errors, {result.warning_count} warnings"
            )
    else:
        logging.info("✅ All validations passed successfully!")

    logging.info("=" * 70 + "\n")

logging.info("🎉 Scraping and validation complete!")
