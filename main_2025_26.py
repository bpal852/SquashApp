
# Imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
import time
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from utils.divisions_export import save_divisions_json
import re
# Import functions from other scripts
from scripts.create_player_results_database_all_divisions import run_player_results_pipeline
from scripts.create_combined_results import load_all_results_and_player_results 


# Global variables
_SUMMARY_NUMS_RE = re.compile(r"(.*?)[^\d]*?(\d+)\s*(\d+)\s*(\d+)\s*(\d+)$")

def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = build_session()
REQUEST_TIMEOUT = (10, 30)

# Constants and Configurations
BASE = "https://www.hksquash.org.hk/public/index.php/leagues"
PAGES_ID = "25"  # parametrize in case the federation changes it later

def url(path, league_id):
    return f"{BASE}/{path}/id/{league_id}/league/Squash/year/{year}/pages_id/{PAGES_ID}.html"


# Inputs
year = "2025-2026"
wait_time = 30

# Repo root: if this file is at the repo root, parents[0] is fine.
# If you move it into /scripts in future, switch to parents[1].
REPO_ROOT = Path(os.getenv("SQUASHAPP_ROOT", Path(__file__).resolve().parents[0]))

DIVISIONS = {
    # Mondays
    "2":                {"id": 473, "day": "Mon", "enabled": True},
    "6":                {"id": 477, "day": "Mon", "enabled": True},
    "10":               {"id": 482, "day": "Mon", "enabled": True},

    # Tuesdays
    "3":                {"id": 474, "day": "Tue", "enabled": False},
    "4":                {"id": 475, "day": "Tue", "enabled": False},
    "11":               {"id": 483, "day": "Tue", "enabled": False},
    "L2":               {"id": 496, "day": "Tue", "enabled": False},

    # Wednesdays
    "7":                {"id": 478, "day": "Wed", "enabled": True},
    "9":                {"id": 481, "day": "Wed", "enabled": True},
    "12":               {"id": 484, "day": "Wed", "enabled": True},
    "M2":               {"id": 492, "day": "Wed", "enabled": True},

    # Thursdays
    "Premier Main":     {"id": 472, "day": "Thu", "enabled": True},
    "Premier Masters":  {"id": 491, "day": "Thu", "enabled": True},
    "Premier Ladies":   {"id": 495, "day": "Thu", "enabled": True},
    "M3":               {"id": 493, "day": "Thu", "enabled": True},
    "M4":               {"id": 494, "day": "Thu", "enabled": True},

    # Fridays
    "5":                {"id": 476, "day": "Fri", "enabled": True},
    "8A":               {"id": 479, "day": "Fri", "enabled": True},
    "8B":               {"id": 480, "day": "Fri", "enabled": True},
    "13A":              {"id": 485, "day": "Fri", "enabled": True},
    "13B":              {"id": 486, "day": "Fri", "enabled": True},
    "13C":              {"id": 487, "day": "Fri", "enabled": True},
    "L3":               {"id": 497, "day": "Fri", "enabled": True},
    "L4":               {"id": 498, "day": "Fri", "enabled": True},

    # Saturdays
    "14":               {"id": 488, "day": "Sat", "enabled": True},
    "15A":              {"id": 489, "day": "Sat", "enabled": True},
    "15B":              {"id": 490, "day": "Sat", "enabled": True},
}

# Convenience derived views
all_divisions = {k: v["id"] for k, v in DIVISIONS.items()}
current_divisions = {k: v["id"] for k, v in DIVISIONS.items() if v["enabled"]}  # or a filtered subset if you want
weekday_groups = {}
for name, meta in DIVISIONS.items():
    if meta["enabled"]:
        weekday_groups.setdefault(meta["day"], {})[name] = meta["id"]

# Save divisions JSON
out_path = save_divisions_json(DIVISIONS, year, REPO_ROOT)
print(f"Divisions JSON saved to: {out_path}")

# Define base directories
base_directories = {
    'summary_df': str(REPO_ROOT / year / 'summary_df'),
    'teams_df': str(REPO_ROOT / year / 'teams_df'),
    'schedules_df': str(REPO_ROOT / year / 'schedules_df'),
    'ranking_df': str(REPO_ROOT / year / 'ranking_df'),
    'players_df': str(REPO_ROOT / year / 'players_df'),
    'summarized_player_tables': str(REPO_ROOT / year / 'summarized_player_tables'),
    'unbeaten_players': str(REPO_ROOT / year / 'unbeaten_players'),
    'played_every_game': str(REPO_ROOT / year / 'played_every_game'),
    'detailed_league_tables': str(REPO_ROOT / year / 'detailed_league_tables'),
    'awaiting_results': str(REPO_ROOT / year / 'awaiting_results'),
    'home_away_data': str(REPO_ROOT / year / 'home_away_data'),
    'team_win_percentage_breakdown_home': str(REPO_ROOT / year / 'team_win_percentage_breakdown' / 'Home'),
    'team_win_percentage_breakdown_away': str(REPO_ROOT / year / 'team_win_percentage_breakdown' / 'Away'),
    'team_win_percentage_breakdown_delta': str(REPO_ROOT / year / 'team_win_percentage_breakdown' / 'Delta'),
    'team_win_percentage_breakdown_overall': str(REPO_ROOT / year / 'team_win_percentage_breakdown' / 'Overall'),
    'simulated_tables': str(REPO_ROOT / year / 'simulated_tables'),
    'simulated_fixtures': str(REPO_ROOT / year / 'simulated_fixtures'),
    'remaining_fixtures': str(REPO_ROOT / year / 'remaining_fixtures'),
    'neutral_fixtures': str(REPO_ROOT / year / 'neutral_fixtures'),
    'results_df': str(REPO_ROOT / year / 'results_df'),
}

# Ensure the logs directory exists
os.makedirs(REPO_ROOT / year / "logs", exist_ok=True)

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            str(REPO_ROOT / year / "logs" / f"{year}_log.txt"), maxBytes=5*1024*1024, backupCount=5
        ),
        logging.StreamHandler()
    ]
)

def parse_result(result):
    """
    Function to parse the 'result' string
    """
    overall, rubbers = result.split('(')
    rubbers = rubbers.strip(')').split(',')
    return overall, rubbers


def split_overall_score(score):
    """
    Function to split the overall score and return home and away scores
    """
    home_score, away_score = map(int, score.split('-'))
    return home_score, away_score


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
        if rubber_index < len(row['Rubbers']):
            r = row['Rubbers'][rubber_index]
            if pd.notna(r) and r not in ['NA', 'CR', 'WO']:
                valid_matches_count[row['Home Team']] = valid_matches_count.get(row['Home Team'], 0) + 1
                valid_matches_count[row['Away Team']] = valid_matches_count.get(row['Away Team'], 0) + 1
    return valid_matches_count

def _parse_summary_row_text(txt: str):
    """
    Fallback parser: extract Team, Played, Won, Lost, Points from raw text.
    Handles cases like: 'Physical Chess 1 1 0 4' (with weird spacing).
    Returns tuple or None if it doesn't look like a data row.
    """
    txt = " ".join(txt.split())
    m = _SUMMARY_NUMS_RE.match(txt)
    if not m:
        return None
    team = m.group(1).strip()
    p, w, l, pts = map(int, m.groups()[1:])
    # sanity check to avoid header lines like 'playedwonlostpoint'
    if not team or team.lower().startswith("played"):
        return None
    return [team, p, w, l, pts]

def scrape_team_summary_page(league_id, year):
    """
    Scrape Team Summary and return a non-empty DataFrame.
    Tries both site spellings: 'team_summery' (current) then 'team_summary' (fallback).
    """
    summary_paths = ["team_summery", "team_summary"]
    last_error = None

    for path in summary_paths:
        summary_url = url(path, league_id)
        logging.info(f"Scraping team summary page ({path}) for league id: {league_id}, year: {year}...")
        try:
            response = SESSION.get(summary_url, timeout=REQUEST_TIMEOUT)
            logging.debug(f"[{path}] status: {response.status_code}")
            if response.status_code != 200:
                last_error = RuntimeError(f"[{path}] HTTP {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # The page markup uses this structure:
            # <div class="clearfix teamSummary-content-list">
            #   <div class="col-xs-4">Team</div>
            #   <div class="col-xs-2">Played</div> ...
            rows = (soup.select("div.clearfix.teamSummary-content-list")
                    or soup.select("div.teamSummary-content-list")
                    or soup.select("div.teamSummary div[class*='content-list']"))

            data = []
            for idx, row in enumerate(rows):
                cells = [d.get_text(strip=True) for d in row.find_all("div", recursive=False)]
                cells = [c for c in cells if c]
                # skip header-like rows
                joined = "".join(cells).lower()
                if "played" in joined and "won" in joined and "lost" in joined:
                    continue

                if len(cells) >= 5:
                    team = " ".join(cells[:-4]) if len(cells) > 5 else cells[0]
                    tail = cells[-4:]
                    try:
                        p, w, l, pts = map(int, tail)
                        if team and not team.lower().startswith("played"):
                            data.append([team, p, w, l, pts])
                    except Exception:
                        # ignore malformed lines; we also have a fallback below if needed
                        pass

            if not data:
                # as a fallback, try parsing the whole row text with regex (optional)
                # if still empty, try the other spelling
                last_error = ValueError(f"[{path}] parsed 0 data rows")
                continue

            df = pd.DataFrame(data, columns=["Team", "Played", "Won", "Lost", "Points"])
            df[["Played", "Won", "Lost", "Points"]] = df[
                ["Played", "Won", "Lost", "Points"]
            ].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

            logging.info(f"[{path}] Successfully created summary DataFrame with {len(df)} rows")
            return df

        except Exception as e:
            logging.exception(f"[{path}] Error scraping team summary: {e}")
            last_error = e
            continue

    # If we get here, both spellings failed
    logging.error(f"Team summary failed with both slugs for {league_id}: {last_error}")
    raise SystemExit(1)


def scrape_teams_page(league_id, year):
    """
    Function to scrape the Teams page on HK squash website and store the data in a dataframe
    """
    teams_url = url("teams", league_id)

    logging.info(f"Starting scrape_teams_page for league id: {league_id}, year: {year}")
    logging.debug(f"Constructed teams URL: {teams_url}")

    try:
        # Send the HTTP request
        response = SESSION.get(teams_url, timeout=REQUEST_TIMEOUT)
        logging.debug(f"Received response with status code: {response.status_code}")

        # Check if the response is successful
        if response.status_code != 200:
            logging.error(f"Failed to retrieve teams page. Status code: {response.status_code}")
            return pd.DataFrame()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        logging.debug("Parsed HTML content with BeautifulSoup")

        # Find the team data
        team_rows = soup.find_all("div", class_="teams-content-list")
        logging.debug(f"Found {len(team_rows)} team rows")

        # Check if any team data was found
        if not team_rows:
            logging.warning("No team data was found on the teams page.")
            return pd.DataFrame()

        # Initialize a list to hold all the data rows
        team_data_rows = []

        # Iterate over the rows and extract data
        for idx, row in enumerate(team_rows):
            columns = row.find_all("div", recursive=False)
            row_data = [col.text.strip() for col in columns if col.text.strip()]
            if row_data:
                team_data_rows.append(row_data)
                logging.debug(f"Extracted data from row {idx}: {row_data}")
            else:
                logging.debug(f"No data found in row {idx}, skipping")

        # Check if any data was extracted
        if not team_data_rows:
            logging.warning("No data rows were extracted from the teams page.")
            return pd.DataFrame()
        
        # Definte the expected column names
        expected_columns = ["Team Name", "Home", "Convenor", "Email"]

        # Create DataFrame from list of lists
        teams_df = pd.DataFrame(team_data_rows, columns=expected_columns)
        logging.info(f"Successfully created teams DataFrame with {len(teams_df)} rows")

        return teams_df
    
    except Exception as e:
        logging.exception(f"An error occured in scrape_teams_page: {e}")
        return pd.DataFrame()


def scrape_schedules_and_results_page(league_id, year):
    """
    Function to scrape Schedules and Results page from HK squash website and store data in a dataframe
    """
    schedule_url = url("results_schedules", league_id)

    # Add logging to track the progress
    logging.info(f"Scraping schedules and results page for league id: {league_id}, year: {year}...")
    logging.debug(f"Constructed schedule URL: {schedule_url}")

    try:
        # Send the HTTP request
        response = SESSION.get(schedule_url, timeout=REQUEST_TIMEOUT)
        logging.debug(f"Received response with status code: {response.status_code}")

        # Check if the response is successful
        if response.status_code != 200:
            logging.error(f"Failed to retrieve schedules and results page. Status code: {response.status_code}")
            return pd.DataFrame()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        logging.debug("Parsed HTML content with BeautifulSoup")

        # Initialize a list to hold all the data rows
        data_rows = []

        # Iterate over each section in the schedule
        sections = soup.find_all('div', class_='results-schedules-content')
        logging.debug(f"Found {len(sections)} schedule sections")

        for section_idx, section in enumerate(sections) :
            # Extract the match week and date from the title
            title_div = section.find_previous_sibling('div', class_='clearfix results-schedules-title')
            if title_div:
                match_week_and_date = title_div.text.strip()
                try:
                    match_week_str, date = match_week_and_date.split(' - ')
                    # Extract just the number from the match week string
                    match_week = ''.join(filter(str.isdigit, match_week_str))
                    # Convert match_week to integer
                    match_week = int(match_week)
                    logging.debug(f"Section {section_idx}: Match Week: {match_week}, Date: {date}")
                except ValueError as e:
                    logging.warning(f"Section {section_idx}: Error parsing match week and date: {match_week_and_date}")
                    match_week, date = None, None  # Assign None if conversion fails
            else:
                logging.warning(f"Section {section_idx}: No title div found for match week and date")
                match_week, date = None, None

            # Find all 'div' elements with the class 'results-schedules-list' in the section
            schedule_rows = section.find_all('div', class_='results-schedules-list')
            logging.debug(f"Section {section_idx}: Found {len(schedule_rows)} schedule rows")

            # Skip the first row as it's the header
            for row_idx, row in enumerate(schedule_rows[1:], start=1):
                columns = row.find_all('div', recursive=False)
                row_data = [col.text.strip() for col in columns]

                # Ensure the correct number of columns (add empty result if missing)
                if len(row_data) == 5:  # Missing result
                    row_data.append('')  # Add empty result
                    logging.debug(f"Row {row_idx}: Missing result, added empty string")

                # Add match week and date to each row
                row_data.extend([match_week, date])
                data_rows.append(row_data)
                logging.debug(f"Row {row_idx}: Extracted data: {row_data}")

        # Create a DataFrame from the scraped schedule data
        column_names = ['Home Team', 'vs', 'Away Team', 'Venue', 'Time', 'Result', 'Match Week', 'Date']
        df = pd.DataFrame(data_rows, columns=column_names)
        logging.info(f"Successfully created schedules and results DataFrame with {len(df)} rows")

        # Convert 'Match Week' to numeric and handle NaN values
        df['Match Week'] = pd.to_numeric(df['Match Week'], errors='coerce')

        # Drop rows with NaN in 'Match Week' if necessary
        initial_row_count = len(df)
        df = df.dropna(subset=['Match Week'])
        logging.info(f"Dropped {initial_row_count - len(df)} rows with NaN in 'Match Week'")

        # Convert 'Match Week' to integer type
        df['Match Week'] = df['Match Week'].astype(int)

        return df
    
    except Exception as e:
        logging.exception(f"An error occured in scrape_schedules_and_results_page: {e}")
        return pd.DataFrame()


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
    home_matches = results_df[results_df['Home Team'] == team]

    # Initialize a dictionary to store wins in each rubber
    wins = {f'Wins in Rubber {i}': 0 for i in range(1, max_rubbers + 1)}

    # Iterate through each match
    for index, row in home_matches.iterrows():
        # Assuming 'Rubbers' column is a list of scores like ['3-1', '1-3', ...]
        for i, score in enumerate(row['Rubbers'], start=1):
            if score != 'NA' and score != 'CR' and score != 'WO':
                home_score, away_score = map(int, score.split('-'))
                if home_score > away_score:
                    wins[f'Wins in Rubber {i}'] += 1

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
    away_matches = results_df[results_df['Away Team'] == team]

    # Initialize a dictionary to store wins in each rubber
    wins = {f'Wins in Rubber {i}': 0 for i in range(1, max_rubbers + 1)}

    # Iterate through each match
    for index, row in away_matches.iterrows():
        # Assuming 'Rubbers' column is a list of scores like ['3-1', '1-3', ...]
        for i, score in enumerate(row['Rubbers'], start=1):
            if score != 'NA' and score != 'CR' and score != 'WO':
                home_score, away_score = map(int, score.split('-'))
                if away_score > home_score:
                    wins[f'Wins in Rubber {i}'] += 1

    return pd.DataFrame(wins, index=[team])


def update_rubbers(row):
    """
    Function to count rubbers for and against for each team
    """
    logging.debug(f"Updating rubbers for match between {row['Home Team']} and {row['Away Team']}")

    # Update for home team
    rubbers_won[row['Home Team']] = rubbers_won.get(row['Home Team'], 0) + row['Home Score']
    rubbers_conceded[row['Home Team']] = rubbers_conceded.get(row['Home Team'], 0) + row['Away Score']

    # Update for away team
    rubbers_won[row['Away Team']] = rubbers_won.get(row['Away Team'], 0) + row['Away Score']
    rubbers_conceded[row['Away Team']] = rubbers_conceded.get(row['Away Team'], 0) + row['Home Score']


def update_counts(row):
    """
    Function to count CRs and WOs For and Against
    """
    home_score, away_score = map(int, row['Overall Score'].split('-'))
    home_wins = away_wins = 0

    for rubber in row['Rubbers']:
        if rubber == 'CR':
            # Count CRs
            if home_wins < home_score:
                cr_given_count[row['Away Team']] = cr_given_count.get(row['Away Team'], 0) + 1
                cr_received_count[row['Home Team']] = cr_received_count.get(row['Home Team'], 0) + 1
            else:
                cr_given_count[row['Home Team']] = cr_given_count.get(row['Home Team'], 0) + 1
                cr_received_count[row['Away Team']] = cr_received_count.get(row['Away Team'], 0) + 1
        elif rubber == 'WO':
            # Count WOs
            if home_wins < home_score:
                wo_given_count[row['Away Team']] = wo_given_count.get(row['Away Team'], 0) + 1
                wo_received_count[row['Home Team']] = wo_received_count.get(row['Home Team'], 0) + 1
            else:
                wo_given_count[row['Home Team']] = wo_given_count.get(row['Home Team'], 0) + 1
                wo_received_count[row['Away Team']] = wo_received_count.get(row['Away Team'], 0) + 1
        else:
            # Count the rubbers won by each team
            rubber_home, rubber_away = map(int, rubber.split('-'))
            if rubber_home > rubber_away:
                home_wins += 1
            elif rubber_away > rubber_home:
                away_wins += 1


def find_max_players(df, team, column):
    """
    Function to find players with max value in a column, handling ties
    """
    max_value = df[df['Team'] == team][column].max()
    players = df[(df['Team'] == team) & (df[column] == max_value)]['Name of Player']
    return ", ".join(players) + f" ({max_value})"


def find_max_win_percentage(df, team):
    """
    Function to find players with max win percentage, handling ties
    """
    max_value = df[df['Team'] == team]['Win Percentage'].max()
    players = df[(df['Team'] == team) & (df['Win Percentage'] == max_value)]['Name of Player']
    return ", ".join(players) + f" ({max_value * 100:.1f}%)"

def scrape_ranking_page(league_id, year):
    """
    Function to scrape the Ranking page and process it into a DataFrame.
    """
    ranking_url = url("ranking", league_id)

    logging.info(f"Scraping ranking page for league id: {league_id}, year: {year}")
    logging.debug(f"Constructed ranking URL: {ranking_url}")

    # Send the HTTP request
    response = SESSION.get(ranking_url, timeout=REQUEST_TIMEOUT)
    logging.debug(f"Received response with status code: {response.status_code}")

    # Check if the response is successful
    if response.status_code != 200:
        logging.error(f"Failed to retrieve ranking page. Status code: {response.status_code}")
        raise Exception(f"Failed to retrieve ranking page. Status code: {response.status_code}")

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    logging.debug("Parsed HTML content with BeautifulSoup")

    # Find the ranking data
    ranking_rows = soup.find_all("div", class_="clearfix ranking-content-list")
    logging.debug(f"Found {len(ranking_rows)} ranking rows")

    # Initialize a list to hold all the data rows
    ranking_data_rows = []

    # Extract the ranking data from the soup
    for idx, row in enumerate(ranking_rows):
        columns = row.find_all("div", recursive=False)
        row_data = [col.text.strip() for col in columns]
        # Exclude rows that contain "NO DATA" or are empty
        if "NO DATA" in row_data or not row_data or len(row_data) < 8:
            logging.debug(f"Skipping row {idx} due to 'NO DATA' or insufficient data: {row_data}")
            continue
        ranking_data_rows.append(row_data)
        logging.debug(f"Extracted data from row {idx}: {row_data}")

    # Check if any data was extracted
    if not ranking_data_rows:
        logging.warning("No data rows were extracted from the ranking page.")
        return None, None, None, None

    # Create DataFrame
    df = pd.DataFrame(ranking_data_rows, columns=['Position', 'Name of Player', 'Team', 'Average Points',
                                                  'Total Game Points', 'Games Played', 'Won', 'Lost'])
    logging.info(f"Successfully created ranking DataFrame with {len(df)} rows")

    # Get Division Name and add as a column
    try:
        full_division_name = soup.find('a', href=lambda href: href and "leagues/detail/id" in href).text.strip()
        division_number = full_division_name.split("Division ")[-1]
        df['Division'] = division_number
        logging.debug(f"Extracted division number: {division_number}")
    except Exception as e:
        logging.warning(f"Error extracting division number: {e}")
        df['Division'] = ''

    # Convert columns to numeric types, handling errors
    numeric_columns = ['Average Points', 'Total Game Points', 'Games Played', 'Won', 'Lost']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle NaN values
    df['Average Points'] = df['Average Points'].fillna(0.0)
    df['Total Game Points'] = df['Total Game Points'].fillna(0)
    df['Games Played'] = df['Games Played'].fillna(0)
    df['Won'] = df['Won'].fillna(0)
    df['Lost'] = df['Lost'].fillna(0)

    # Convert to appropriate types
    df['Total Game Points'] = df['Total Game Points'].astype(int)
    df['Games Played'] = df['Games Played'].astype(int)
    df['Won'] = df['Won'].astype(int)
    df['Lost'] = df['Lost'].astype(int)
    df['Average Points'] = df['Average Points'].astype(float)

    logging.debug("Converted numeric columns to appropriate data types")

    # Create Win Percentage column, handling division by zero
    df["Win Percentage"] = df.apply(
        lambda row: row["Won"] / row["Games Played"] if row["Games Played"] > 0 else 0, axis=1
    )

    # Create filtered DataFrame
    ranking_df_filtered = df[df["Games Played"] >= 5]
    logging.info(f"Filtered ranking DataFrame to {len(ranking_df_filtered)} rows with 5 or more games played")


    # Check if ranking_df_filtered is empty
    if ranking_df_filtered.empty:
        logging.warning("No players have played enough games to qualify for the table.")
        summarized_df = None
        unbeaten_list = []
    else:
        # Create the summarized DataFrame
        teams = df['Team'].unique()
        summary_data = {
            'Team': [],
            'Most Games': [],
            'Most Wins': [],
            'Highest Win Percentage': []
        }
        for team in teams:
            summary_data['Team'].append(team)
            summary_data['Most Games'].append(find_max_players(ranking_df_filtered, team, 'Games Played'))
            summary_data['Most Wins'].append(find_max_players(ranking_df_filtered, team, 'Won'))
            summary_data['Highest Win Percentage'].append(find_max_win_percentage(ranking_df_filtered, team))

        summarized_df = pd.DataFrame(summary_data).sort_values("Team")
        logging.info(f"Created summarized DataFrame with {len(summarized_df)} teams")

        # Get list of unbeaten players
        unbeaten_list = ranking_df_filtered[
            ranking_df_filtered["Lost"] == 0
            ].apply(lambda row: f"{row['Name of Player']} ({row['Team']})", axis=1).tolist()
        logging.info(f"Found {len(unbeaten_list)} unbeaten players")

    return df, summarized_df, unbeaten_list, ranking_df_filtered


def scrape_players_page(league_id, year):
    """
    Function to scrape the Players page and store data in a DataFrame.
    """

    logging.info(f"Starting scrape_players_page for league_id: {league_id}, year: {year}")

    players_url = url("players", league_id)
    logging.debug(f"Constructed players URL: {players_url}")

    try:
        # Send the HTTP request
        response = SESSION.get(players_url, timeout=REQUEST_TIMEOUT)
        logging.debug(f"Received response with status code: {response.status_code}")

        # Check if the response is successful
        if response.status_code != 200:
            raise RuntimeError(f"Failed to retrieve players page. Status code: {response.status_code}")
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        logging.debug("Parsed HTML content with BeautifulSoup")

        # Dictionary to store the dataframes
        team_dataframes = []

        # Loop through each team's container
        team_containers = soup.find_all("div", class_="players-container")
        logging.debug(f"Found {len(team_containers)} team containers")

        for idx, team_container in enumerate(team_containers):
            # Extract team name
            team_name = None
            try:
                team_name_div = team_container.find("div", string="team name:")
                team_name = team_name_div.find_next_sibling().get_text(strip=True)
            except Exception as e:
                logging.warning(f"Team {idx}: Error extracting team name: {e}")
                continue

            # If this team block explicitly says NO DATA, skip the whole team
            if team_container.get_text(strip=True).upper().find("NO DATA") != -1:
                logging.info(f"Team {idx} ('{team_name}') shows NO DATA — skipping team.")
                continue

            # Extract player data
            player_rows = team_container.find_all("div", class_="players-content-list")
            logging.debug(f"Team {idx}: Found {len(player_rows)} player rows")

            # Initialize a list to store each player's data for this team
            players_data = []

            for player_idx, player in enumerate(player_rows):
                # collect fields
                order_rank_points = [div.get_text(strip=True) for div in player.find_all("div", class_="col-xs-2")]
                player_name = [div.get_text(strip=True) for div in player.find_all("div", class_="col-xs-4")]

                # Build row: [Order] + [Name of Players] + [HKS No., Ranking, Points]
                row = order_rank_points[:1] + player_name + order_rank_points[1:]

                # Keep only well-formed rows of length 5 and that are not header junk
                if len(row) == 5 and row[0].isdigit():
                    players_data.append(row)
                else:
                    # benign skip: headers/format noise produce zero-length or short rows
                    logging.debug(f"Team {idx}, Player {player_idx}: skipping malformed row: {row}")

            if not players_data:
                logging.warning(f"Team '{team_name}' produced no valid player rows; skipping team.")
                continue

            # Create DataFrame
            df = pd.DataFrame(players_data, columns=["Order", "Name of Players", "HKS No.", "Ranking", "Points"])

            # Convert columns to the correct data types
            df['Order'] = pd.to_numeric(df['Order'], errors='coerce').fillna(0).astype(int)
            df['HKS No.'] = pd.to_numeric(df['HKS No.'], errors='coerce').fillna(0).astype(int)
            df['Ranking'] = pd.to_numeric(df['Ranking'], errors='coerce').fillna(0).astype(int)
            df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0.0).astype(float)
            df['Team'] = team_name

            # Rename column
            df = df.rename(columns={"Name of Players": "Player"})

            # Add dataframe to list
            team_dataframes.append(df)
            logging.info(f"Team {idx + 1}: Created DataFrame with {len(df)} rows for team: {team_name}")

            time.sleep(5)

        if not team_dataframes:
            raise ValueError("No valid player data found in any team block on the page.")

        combined_df = pd.concat(team_dataframes, ignore_index=True)
        logging.info(f"Concatenated all team dataframes into a single DataFrame with {len(combined_df)} rows")
        return combined_df  

    except Exception as e:
        logging.exception(f"An error occured in scrape_players_page: {e}")
        return pd.DataFrame()
    

def count_games_won(row):
    """
    Function to count the number of games won by each team in a match,
    handling walkovers (WO) and conceded rubbers (CR) by referring to the 'Overall Score'.
    """
    home_games_won = 0
    away_games_won = 0

    # Calculate the games won from the rubbers, excluding 'CR' and 'WO'
    for rubber in row['Rubbers']:
        if rubber == 'CR' or rubber == 'WO':
            continue
        home, away = map(int, rubber.split('-'))
        home_games_won += home
        away_games_won += away

    # Now handle the 'WO' and 'CR' rubbers by referring to the 'Overall Score'
    if 'WO' in row['Rubbers'] or 'CR' in row['Rubbers']:
        home_overall_score, away_overall_score = map(int, row['Overall Score'].split('-'))
        
        # If the home team has a higher overall score, award the missing games to them
        # Otherwise, award the missing games to the away team
        for rubber in row['Rubbers']:
            if rubber == 'WO' or rubber == 'CR':
                if home_overall_score > away_overall_score:
                    home_games_won += 3
                else:
                    away_games_won += 3

    return home_games_won, away_games_won
    

def home_team_won(row):
    """Function to determine whether the home team or away team
    won the match, using games won as a tiebreaker. If overall score
    and games won are equal, the match is ignored.
    """
    if row['Home Score'] > row['Away Score']:
        return 'Home'
    elif row['Home Score'] < row['Away Score']:
        return 'Away'
    else:
        # If overall scores are equal, use games won as tiebreaker
        if row['Home Games Won'] > row['Away Games Won']:
            return 'Home'
        elif row['Home Games Won'] < row['Away Games Won']:
            return 'Away'
        else:
            return 'Ignore'
        

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


# Use logging to track progress
logging.info("Starting the scraping process...")

# Change dictionary if you want specific week
for div in all_divisions.keys():
    logging.info(f"Processing Division {div}")
    league_id = f"D00{all_divisions[div]}"

    # Scrape Schedules and Results page
    try:
        logging.info(f"Scraping Schedules and Results page for Division {div}")
        schedules_df = scrape_schedules_and_results_page(league_id, year)
        logging.info(f"Successfully scraped Schedules and Results page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Schedules and Results page for Division {div}: {e}")
        continue

    # Check if the schedules_df is empty
    if schedules_df.empty:
        logging.warning(f"No data found in schedules_df for Division {div}. Skipping further processing.")
        continue

    # Filter schedules_df to only include matches where 'Result' is not empty
    played_matches_df = schedules_df[schedules_df['Result'].notna() & (schedules_df['Result'] != '')]

    # Check if played_matches_df is empty
    if played_matches_df.empty:
        logging.warning(f"No played matches found in schedules_df for Division {div}. Skipping further processing.")
        match_week = 0
    else:
        # Get the latest match week number for which data is available and ensure it is an integer
        match_week = played_matches_df['Match Week'].max()
        match_week = int(match_week)

    # Create week specific directories
    week_dir = f"week_{match_week}" if match_week > 0 else "week_0"

    # Create directories for each base directory
    for base_dir in base_directories.values():
        # Combine the base directory with the week directory
        full_dir = os.path.join(base_dir, week_dir)
        # Create the directory if it doesn't exist
        os.makedirs(full_dir, exist_ok=True)
    
    overall_scores_file = os.path.join(
        base_directories['home_away_data'], week_dir, f"{div}_overall_scores.csv"
    )

    # Try to load existing; otherwise create an empty frame with numeric columns 0..4
    if os.path.exists(overall_scores_file):
        try:
            overall_scores_df = pd.read_csv(overall_scores_file, header=None)
        except Exception:
            overall_scores_df = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    else:
        overall_scores_df = pd.DataFrame(columns=[0, 1, 2, 3, 4])

    # Save the schedules_df to CSV
    schedules_df_path = os.path.join(base_directories['schedules_df'], week_dir, f"{div}_schedules_df.csv")
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
        summary_df = scrape_team_summary_page(league_id, year)
        logging.info(f"Successfully scraped Team Summary page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Team Summary page for Division {div}: {e}")
        raise
    
    # Validate summary_df is not empty; hard fail if it is
    ensure_nonempty_df(summary_df, "summary_df", div, hard_fail=True)

    # If we reached here, it's non-empty → save it
    week_dir = f"week_{match_week}" if 'match_week' in locals() and match_week > 0 else "week_0"
    summary_df_path = os.path.join(base_directories['summary_df'], week_dir, f"{div}_summary_df.csv")
    safe_save_csv(summary_df, summary_df_path, "summary_df", div, allow_empty=False)

    has_summary = True

    time.sleep(wait_time)

    # Scrape Teams page
    try:
        logging.info(f"Scraping Teams page for Division {div}")
        teams_df = scrape_teams_page(league_id, year)
        logging.info(f"Successfully scraped Teams page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Teams page for Division {div}: {e}")
        continue

    # Save the teams_df to CSV
    teams_df_path = os.path.join(base_directories['teams_df'], week_dir, f"{div}_teams_df.csv")
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
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scrape_ranking_page(league_id, year)
        logging.info(f"Successfully scraped Ranking page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Ranking page for Division {div}: {e}")
        # Stop execution if an error occurs
        raise

    # Save the ranking_df to CSV if it is not None and not empty
    if ranking_df is not None and not ranking_df.empty:
        ranking_df_path = os.path.join(base_directories['ranking_df'], week_dir, f"{div}_ranking_df.csv")
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
        players_df = scrape_players_page(league_id, year)
        logging.info(f"Successfully scraped Players page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Players page for Division {div}: {e}")
        continue
    
    # Save the players_df to CSV
    players_df_path = os.path.join(base_directories['players_df'], week_dir, f"{div}_players_df.csv")
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
        merged_ranking_df = ranking_df_filtered.merge(
            summary_df[["Team", "Played"]], on="Team", how="inner"
        )
        # Ensure numeric
        merged_ranking_df["Played"] = pd.to_numeric(merged_ranking_df["Played"], errors="coerce").fillna(0).astype(int)
        merged_ranking_df["Games Played"] = pd.to_numeric(merged_ranking_df["Games Played"], errors="coerce").fillna(0).astype(int)

        # Players who have played every game their team has played
        played_every_game_list = merged_ranking_df[
            merged_ranking_df["Games Played"] == merged_ranking_df["Played"]
        ].apply(lambda row: f"{row['Name of Player']} ({row['Team']})", axis=1).tolist()
    else:
        logging.warning(f"No usable summary/ranking data for Division {div}. Unable to determine players who have played every game.")

    # Save the summarized_df to CSV if it is not None and not empty
    if summarized_df is not None and not summarized_df.empty:
        summarized_df_path = os.path.join(base_directories['summarized_player_tables'], week_dir, f"{div}_summarized_players.csv")
        try:
            logging.info(f"Saving summarized_df to {summarized_df_path}")
            summarized_df.to_csv(summarized_df_path, index=False)
            logging.info(f"Successfully saved summarized_df to {summarized_df_path}")
        except Exception as e:
            logging.error(f"Error saving summarized_df to {summarized_df_path}: {e}")
    else:
        logging.info(f"No summarized data to save for Division {div}; skipping summarized_df CSV creation.")

    # Save the unbeaten_list to a text file (create a blank file if no unbeaten players)
    unbeaten_players_path = os.path.join(base_directories['unbeaten_players'], week_dir, f"{div}.txt")
    try:
        logging.info(f"Saving unbeaten_list to {unbeaten_players_path}")
        with open(unbeaten_players_path, 'w') as f:
            for player in unbeaten_list:
                f.write(f"{player}\n")
        logging.info(f"Successfully saved unbeaten_list to {unbeaten_players_path}")
    except Exception as e:
        logging.error(f"Error saving unbeaten_list to {unbeaten_players_path}: {e}")

    # Save list of players who have played every game (create a blank file if none)
    played_every_game_path = os.path.join(base_directories['played_every_game'], week_dir, f"{div}.txt")
    try:
        logging.info(f"Saving played_every_game_list to {played_every_game_path}")
        with open(played_every_game_path, 'w') as f:
            for player in played_every_game_list:
                f.write(f"{player}\n")
        logging.info(f"Successfully saved played_every_game_list to {played_every_game_path}")
    except Exception as e:
        logging.error(f"Error saving played_every_game_list to {played_every_game_path}: {e}")

    # Create Results Dataframe

    # Drop unnecessary columns
    schedules_df.drop(columns=["vs", "Time"], inplace=True, errors="ignore")

    # Exclude rows where 'Away Team' is '[BYE]' (indicative of a bye week)
    results_df = schedules_df[schedules_df['Away Team'] != '[BYE]'].copy()

    # Replace NaN values in 'Result' with an empty string before applying str.contains
    results_df['Result'] = results_df['Result'].fillna('')

    # Keep rows where 'Result' contains brackets (indicative of a played match)
    results_df = results_df[results_df['Result'].str.contains(r'\(')]

    # Check if the results_df is empty
    if results_df.empty:
        logging.warning(f"No data found in results_df for Division {div}. Skipping further processing.")
        continue

    def normalize_rubber(s: str) -> str:
        s = (s or "").strip().upper()
        if s == "W/O":  # unify variant
            return "WO"
        return s

    # Apply the parse_result function to the 'Result' column
    results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(lambda x: pd.Series(parse_result(x)))

    # Apply the normalize_rubber function to the 'Rubbers' column
    results_df['Rubbers'] = results_df['Rubbers'].apply(lambda lst: [normalize_rubber(x) for x in lst])

    # Drop the original 'Result' column
    results_df.drop(columns=['Result'], inplace=True)

    # Count the number of Rubbers For and Against for each team

    # Splitting the 'Overall Score' into two separate columns
    results_df[['Home Score', 'Away Score']] = results_df['Overall Score'].str.split('-', expand=True).astype(int)

    # Initialize dictionaries to keep track of won and conceded rubbers
    rubbers_won = {}
    rubbers_conceded = {}

    # Create Games Won columns
    results_df[['Home Games Won', 'Away Games Won']] = results_df.apply(count_games_won, axis=1, result_type='expand')

    # Apply the function to each row
    results_df.apply(update_rubbers, axis=1)

    # Convert the dictionaries to DataFrames
    df_rubbers_won = pd.DataFrame(list(rubbers_won.items()), columns=['Team', 'Rubbers For'])
    df_rubbers_conceded = pd.DataFrame(list(rubbers_conceded.items()), columns=['Team', 'Rubbers Against'])

    # Merge the DataFrames on Team
    rubbers_df = pd.merge(df_rubbers_won, df_rubbers_conceded, on='Team')

    # Count the number Conceded Rubbers and Walkovers

    # Initialize dictionaries to keep track of conceded rubbers and walkovers
    cr_given_count = {}
    cr_received_count = {}
    wo_given_count = {}
    wo_received_count = {}

    # Apply the function to each row
    results_df.apply(update_counts, axis=1)

    # Ensure all teams are included in all counts
    all_teams = set(results_df['Home Team']).union(set(results_df['Away Team']))
    for team in all_teams:
        cr_given_count.setdefault(team, 0)
        cr_received_count.setdefault(team, 0)
        wo_given_count.setdefault(team, 0)
        wo_received_count.setdefault(team, 0)

    # Convert the dictionaries to DataFrames
    df_cr_given_count = pd.DataFrame(list(cr_given_count.items()), columns=['Team', 'CRs Given'])
    df_cr_received_count = pd.DataFrame(list(cr_received_count.items()), columns=['Team', 'CRs Received'])
    df_wo_given_count = pd.DataFrame(list(wo_given_count.items()), columns=['Team', 'WOs Given'])
    df_wo_received_count = pd.DataFrame(list(wo_received_count.items()), columns=['Team', 'WOs Received'])

    # Merge the DataFrames on Team
    detailed_table_df = pd.merge(df_cr_given_count, df_cr_received_count, on='Team')
    detailed_table_df = pd.merge(detailed_table_df, df_wo_given_count, on='Team')
    detailed_table_df = pd.merge(detailed_table_df, df_wo_received_count, on='Team')

    # If team summary is available, merge rubbers data into it, then merge with detailed table
    if has_summary:
        summary_plus = summary_df.merge(rubbers_df, on="Team", how="left")
        detailed_table_df = summary_plus.merge(detailed_table_df, on="Team", how="left")
    else:
        # no team summary available yet; keep counts-only table
        # (optionally skip writing detailed table this week)
        pass

    # Save detailed league table
    detailed_table_df.to_csv(os.path.join(base_directories['detailed_league_tables'], week_dir, f"{div}_detailed_league_table.csv"), index=False)

    # Create Remaining Fixtures Dataframe
    # Filter out rows where 'Result' is not empty and does not contain placeholder text
    # Keep rows where 'Result' is empty (None or empty string)
    df_remaining_fixtures = schedules_df[
        (schedules_df['Result'].isna()) |
        (schedules_df['Result'] == '')
        ]

    # Filter out rows with byes
    df_remaining_fixtures = df_remaining_fixtures[df_remaining_fixtures["Away Team"] != "[BYE]"]

    # Filter out redundant Results column
    df_remaining_fixtures = df_remaining_fixtures[["Home Team", "Away Team", "Venue", "Match Week", "Date"]]

    # Convert the 'Date' column to datetime if it's not already
    df_remaining_fixtures['Date'] = pd.to_datetime(df_remaining_fixtures['Date'], dayfirst=True)

    # Create remaining fixtures folder if it doesn't exist
    os.makedirs(os.path.join(base_directories['remaining_fixtures'], week_dir), exist_ok=True)

    # Save remaining fixtures
    df_remaining_fixtures.to_csv(
        os.path.join(base_directories['remaining_fixtures'], week_dir, f"{div}_remaining_fixtures.csv"), index=False)

    # Filter rows where the 'Date' is earlier than today to create awaiting_results dataframe
    today = pd.Timestamp(datetime.now().date())
    awaiting_results_df = df_remaining_fixtures[df_remaining_fixtures['Date'] < today]
    # Save awaiting results
    awaiting_results_df.to_csv(
        os.path.join(base_directories['awaiting_results'], week_dir, f"{div}_awaiting_results.csv"), index=False)

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
            v  = row["Venue"]
            is_home  = (hv is not None and v == hv)
            is_away  = (av is not None and v == av)
            if is_home and not is_away:
                return "home"            # true home
            if is_away and not is_home:
                return "away"            # away team playing at (its) home
            if is_home and is_away:
                return "shared_home"     # both teams’ home venue (common in club leagues)
            return "neutral"              # neither team’s home venue

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
                v  = row["Venue"]
                is_home  = (hv is not None and v == hv)
                is_away  = (av is not None and v == av)
                if is_home and is_away:
                    return "shared_home"
                if (not is_home) and (not is_away):
                    return "neutral"
                return "other"
            neutral_fixtures_df = neutral_fixtures_df[
                neutral_fixtures_df.apply(remaining_venue_type, axis=1).isin(["shared_home", "neutral"])
            ].copy()
    
    # Create folder for neutral fixtures if it doesn't exist
    os.makedirs(os.path.join(base_directories['neutral_fixtures'], week_dir), exist_ok=True)

    # Save neutral fixtures
    neutral_fixtures_df.to_csv(
        os.path.join(base_directories['neutral_fixtures'], week_dir, f"{div}_neutral_fixtures.csv"), index=False)

    # Calculate Home vs Away
    if not valid_matches_df.empty:
        # Split overall score into numeric columns (so later groupbys work)
        valid_matches_df[['Home Overall Score', 'Away Overall Score']] = (
            valid_matches_df['Overall Score'].apply(lambda x: pd.Series(split_overall_score(x)))
        )

        # League-wide averages
        average_home_overall_score = valid_matches_df['Home Overall Score'].mean()
        average_away_overall_score = valid_matches_df['Away Overall Score'].mean()

        # Winner per fixture (ignore exact ties after tiebreak)
        valid_matches_df['Winner'] = valid_matches_df.apply(home_team_won, axis=1)
        home_win_perc = (
            valid_matches_df[valid_matches_df["Winner"] != "Ignore"]["Winner"]
            .value_counts(normalize=True)
            .get("Home", 0.0)
        )

        # ---- per-team averages (must be inside this block so the columns exist) ----
        average_home_scores = (
            valid_matches_df.groupby('Home Team')['Home Overall Score']
            .mean().rename('Average Home Score')
        )
        average_away_scores = (
            valid_matches_df.groupby('Away Team')['Away Overall Score']
            .mean().rename('Average Away Score')
        )
        team_average_scores = pd.concat([average_home_scores, average_away_scores], axis=1).fillna(0.0)

    else:
        logging.warning(f"No results data to calculate home vs away statistics for Division {div}.")
        average_home_overall_score = 0.0
        average_away_overall_score = 0.0
        home_win_perc = 0.0
        # empty placeholder so later code can run without additional branching
        team_average_scores = pd.DataFrame(columns=['Average Home Score', 'Average Away Score'])

    # Handle missing values by filling NaN with 0 or using appropriate methods
    team_average_scores['Average Home Score'] = team_average_scores['Average Home Score'].fillna(0)
    team_average_scores['Average Away Score'] = team_average_scores['Average Away Score'].fillna(0)

    # Calculate the difference in home and away scores for each team
    team_average_scores["home_away_diff"] = team_average_scores["Average Home Score"] - team_average_scores[
        "Average Away Score"]

    # Merge with teams_df to get home venue info (only if available)
    if teams_df is not None and not teams_df.empty and {"Team Name", "Home"}.issubset(set(teams_df.columns)):
        # Keep the team name index from team_average_scores and LEFT-join venue info
        team_average_scores = team_average_scores.merge(
            teams_df[["Team Name", "Home"]],
            left_index=True,
            right_on="Team Name",
            how="left"   # don't drop teams if venue lookup is missing
        )
        # Ensure we KEEP the diff column
        cols_to_keep = ["Team Name", "Home", "Average Home Score", "Average Away Score", "home_away_diff"]
        team_average_scores = team_average_scores.reindex(columns=cols_to_keep)
    else:
        # If we can't attach venues, keep the team index as a column instead
        team_average_scores = (
            team_average_scores
            .reset_index()
            .rename(columns={"index": "Team Name"})
        )
        team_average_scores["Home"] = ""
        cols_to_keep = ["Team Name", "Home", "Average Home Score", "Average Away Score", "home_away_diff"]
        team_average_scores = team_average_scores.reindex(columns=cols_to_keep)


    # Since 'home_away_diff' may not be meaningful at this point, you can add a check
    if team_average_scores['home_away_diff'].isnull().all():
        logging.warning("All 'home_away_diff' values are NaN or zero. Teams may not have played both home and away games yet.")
    else:
        # Sort the DataFrame based on 'home_away_diff'
        team_average_scores.sort_values("home_away_diff", ascending=False, inplace=True)

    # Save team_average_scores to csv
    team_average_scores.to_csv(os.path.join(base_directories['home_away_data'], week_dir, f"{div}_team_average_scores.csv"), index=False)

    # Show home/away split by venue
    if (
        not team_average_scores.empty
        and "Home" in team_average_scores.columns
        and "home_away_diff" in team_average_scores.columns
    ):
        venue_split = team_average_scores.pivot_table(
            index="Home",
            values="home_away_diff",
            aggfunc="mean"
        )
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
    max_rubbers = results_df['Rubbers'].apply(len).max()

    # Apply the function to each rubber in the list
    for i in range(1, max_rubbers + 1):
        rubber_column = f'Rubber {i}'
        results_df[f'Winner {rubber_column}'] = results_df.apply(
            lambda row: determine_winner(row['Rubbers'][i - 1] if i <= len(row['Rubbers']) else pd.NA,
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

    # Create Home Win Percentage by Rubber dataframe
    # Initialize an empty list to store the results for each team
    home_results_list = []

    # Check if valid_matches_df is empty
    if valid_matches_df.empty:
        logging.warning("No valid matches to process for home win percentages.")
        # Create an empty DataFrame with the required columns
        home_results = pd.DataFrame(columns=['Team'] +
                                [f'Wins in Rubber {i}' for i in range(1, max_rubbers + 1)] +
                                [f'Rubber {i}' for i in range(1, max_rubbers + 1)] +
                                [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] +
                                ['avg_win_perc', 'Total Rubbers'])
        
    else:
        # Iterate through each team in the "Home Team" column
        for team in valid_matches_df['Home Team'].unique():
            # Filter for matches where the current team is playing at home
            team_home_fixtures = valid_matches_df[valid_matches_df['Home Team'] == team]

            # Get counts per team
            total_home_matches_per_rubber_counts = {f'Rubber {i}': count_valid_matches(team_home_fixtures, i - 1)
                                                    for i in range(1, max_rubbers + 1)}
            
            # Extract counts for the specific team
            total_home_matches_per_rubber = {rubber: counts.get(team, 0) for rubber, counts in total_home_matches_per_rubber_counts.items()}

            # Convert the dictionary to a DataFrame
            total_home_matches_df = pd.DataFrame([total_home_matches_per_rubber], index=[team])

            # Calculate total games played by summing all the rubber matches for each team
            total_rubbers_played = total_home_matches_df.sum(axis=1)

            # Merge with aggregate wins for the team's home fixtures and calculate win percentages
            team_combined_home = aggregate_wins_home(
                team, valid_matches_df
            ).merge(total_home_matches_df, left_index=True, right_index=True, how='outer')

            team_combined_home.fillna(0, inplace=True)

            for i in range(1, max_rubbers + 1):
                rubber_column = f'Rubber {i}'
                team_combined_home[f'{rubber_column} Win %'] = (team_combined_home[f'Wins in {rubber_column}'] /
                                                                team_combined_home[rubber_column]) * 100

            team_combined_home.fillna(0, inplace=True)
            team_combined_home["avg_win_perc"] = team_combined_home[
                [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)]].mean(axis=1)

            # Add the total rubbers played to the DataFrame
            team_combined_home["Total Rubbers"] = total_rubbers_played

            # Append the team's results to the list
            home_results_list.append(team_combined_home)

        # Concatenate all team results into a single DataFrame
        home_results = pd.concat(home_results_list)

    # Check if home_results is empty
    if home_results.empty:
        logging.warning("No home results to process.")
        win_percentage_home_df = pd.DataFrame(columns=['Team'] +
                                          [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] +
                                          ['avg_win_perc', 'Total Rubbers'])
    else:
        # Sort and format the DataFrame
        home_results_sorted = home_results.sort_values("avg_win_perc", ascending=False)
        home_results_sorted = home_results_sorted.reset_index().rename(columns={'index': 'Team'})

        # Selecting and displaying the required columns
        keep_columns_home = (
                ["Team"] + [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] + ['avg_win_perc', "Total Rubbers"]
        )
        win_percentage_home_df = home_results_sorted[keep_columns_home]

    # Save win_percentage_home_df to csv
    win_percentage_home_df.to_csv(os.path.join(base_directories['team_win_percentage_breakdown_home'], week_dir, \
                                               f"{div}_team_win_percentage_breakdown_home.csv"), index=False)

    # Create Away Win Percentage by Rubber dataframe
    # Initialize an empty list to store the results for each team
    away_results_list = []

    # Check if valid_matches_df is empty
    if valid_matches_df.empty:
        logging.warning("No valid matches to process for away win percentages.")
        # Create an empty DataFrame with the required columns
        away_results = pd.DataFrame(columns=['Team'] +
                                [f'Wins in Rubber {i}' for i in range(1, max_rubbers + 1)] +
                                [f'Rubber {i}' for i in range(1, max_rubbers + 1)] +
                                [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] +
                                ['avg_win_perc', 'Total Rubbers'])
    else:
        # Iterate through each team in the "Away Team" column
        for team in valid_matches_df['Away Team'].unique():
            # Filter for matches where the current team is playing away
            team_away_fixtures = valid_matches_df[valid_matches_df['Away Team'] == team]

            # Get counts per team
            total_away_matches_per_rubber_counts = {f'Rubber {i}': count_valid_matches(team_away_fixtures, i - 1)
                                                    for i in range(1, max_rubbers + 1)}
            
            # Extract counts for the specific team
            total_away_matches_per_rubber = {rubber: counts.get(team, 0) for rubber, counts in total_away_matches_per_rubber_counts.items()}

            # Convert the dictionary to a DataFrame
            total_away_matches_df = pd.DataFrame([total_away_matches_per_rubber], index=[team])

            # Calculate total games played by summing all the rubber matches for each team
            total_rubbers_played = total_away_matches_df.sum(axis=1)

            # Merge with aggregate wins for the team's away fixtures and calculate win percentages
            team_combined_away = aggregate_wins_away(
                team, valid_matches_df
            ).merge(total_away_matches_df, left_index=True, right_index=True, how='outer')

            team_combined_away.fillna(0, inplace=True)

            for i in range(1, max_rubbers + 1):
                rubber_column = f'Rubber {i}'
                team_combined_away[f'{rubber_column} Win %'] = (team_combined_away[f'Wins in {rubber_column}'] /
                                                                team_combined_away[rubber_column]) * 100

            team_combined_away.fillna(0, inplace=True)
            team_combined_away["avg_win_perc"] = team_combined_away[
                [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)]].mean(axis=1)

            # Add the total rubbers played to the DataFrame
            team_combined_away["Total Rubbers"] = total_rubbers_played

            # Append the team's results to the list
            away_results_list.append(team_combined_away)

        # Concatenate all team results into a single DataFrame
        away_results = pd.concat(away_results_list)

    # Check if away_results is empty
    if away_results.empty:
        logging.warning("No away results to process.")
        win_percentage_away_df = pd.DataFrame(columns=['Team'] +
                                          [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] +
                                          ['avg_win_perc', 'Total Rubbers'])
    else:
        # Sort and format the DataFrame
        away_results_sorted = away_results.sort_values("avg_win_perc", ascending=False)
        away_results_sorted = away_results_sorted.reset_index().rename(columns={'index': 'Team'})

        # Selecting and displaying the required columns
        keep_columns_away = (
                ["Team"] +
                [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] +
                ['avg_win_perc', "Total Rubbers"]
        )

        win_percentage_away_df = away_results_sorted[keep_columns_away]

    # Save win_percentage_away_df to csv
    win_percentage_away_df.to_csv(os.path.join(base_directories['team_win_percentage_breakdown_away'],
                                               week_dir, f"{div}_team_win_percentage_breakdown_away.csv"), index=False)

    # Create Win Percentage Delta Dataframe
    # Merge the two DataFrames on the 'Team' column
    merged_df = win_percentage_home_df.merge(win_percentage_away_df, on='Team', suffixes=('_home', '_away'))

    # Initialize a dictionary to store the win percentage differences
    delta_data = {'Team': merged_df['Team']}

    # Calculate the differences for each 'Rubber Win %' column
    for i in range(1, max_rubbers + 1):
        rubber_column = f'Rubber {i} Win %'
        delta_data[rubber_column] = merged_df[f'{rubber_column}_home'] - merged_df[f'{rubber_column}_away']

    # Calculate the difference for the 'avg_win_perc' column
    delta_data['avg_win_perc'] = merged_df['avg_win_perc_home'] - merged_df['avg_win_perc_away']

    # Create the win_percentage_delta_df DataFrame
    win_percentage_delta_df = pd.DataFrame(delta_data)

    # Sort values by avg_win_perc
    win_percentage_delta_df = win_percentage_delta_df.sort_values("avg_win_perc", ascending=False)

    # Save win_percentage_delta_df to csv
    win_percentage_delta_df.to_csv(os.path.join(base_directories['team_win_percentage_breakdown_delta'], 
                                                week_dir, f"{div}_team_win_percentage_breakdown_delta.csv"), index=False)

    # Create overall Win Percentage by Rubber dataframe
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

    # Save win_percentage_df to csv
    win_percentage_df.to_csv(os.path.join(base_directories['team_win_percentage_breakdown_overall'], 
                                          week_dir, f"{div}_team_win_percentage_breakdown.csv"), index=False)

    # Ensure the DataFrame has the necessary columns
    for col in [0, 1, 2, 3, 4]:
        if col not in overall_scores_df.columns:
            overall_scores_df[col] = None

    # Prepare the data to update or insert
    new_data = [
        average_home_overall_score,
        average_away_overall_score,
        home_win_perc,
        today,
        today
    ]

    # Assign the data to the first row
    overall_scores_df.loc[0] = new_data

    # Write the updated DataFrame back to the CSV file
    overall_scores_df.to_csv(overall_scores_file, index=False, header=None)

    # Save the results_df to CSV
    results_df.to_csv(os.path.join(base_directories['results_df'], week_dir, f"{div}_results_df.csv"), index=False)

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
