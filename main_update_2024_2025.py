
# Imports
import requests
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

year = "2024-2025"

# League Simulation
home_advantage_factor = 0.05
unpredictability_factor = 0.01  # Adjust this value as needed
num_simulations = 9999
run_projections = 0  # toggle 1/0 to run projections

# Inputs
all_divisions = {
    "Premier Main": 424,
    "2": 425,
    "3": 426,
    "4": 427,
    "5": 428,
    "6": 429,
    "7A": 430,
    "7B": 431,
    "8A": 432,
    "8B": 433,
    "9": 434,
    "10": 435,
    "11": 436,
    "12": 437,
    "13A": 438,
    "13B": 439,
    "14": 440,
    "15A": 441,
    "15B": 442,
    "Premier Masters": 443,
    "M2": 444,
    "M3": 445,
    "M4": 446,
    "Premier Ladies": 447,
    "L2": 448,
    "L3": 449,
    "L4": 450,
    }

monday = {
    "2": 425,
    "6": 429,
    "10": 435
}

tuesday = {
    "3": 426,
    "4": 427,
    "11": 436,
    "L2": 448,
}

wednesday = {
    "7A": 430,
    "7B": 431,
    "9": 434,
    "12": 437,
    "M2": 444
}

thursday = {
    "Premier Main": 424,
    "Premier Ladies": 447,
    "Premier Masters": 443,
    "M3": 445,
    "M4": 446,
}

friday = {
    "5": 428,
    "8A": 432,
    "8B": 433,
    "13A": 438,
    "13B": 439,
    "L3": 449,
    "L4": 450
}

saturday = {
    "14": 440,
    "15A": 441,
    "15B": 442
}

weekend = {
    "5": 428,
    "8A": 432,
    "8B": 433,
    "13A": 438,
    "13B": 439,
    "L3": 449,
    "L4": 450,
    "14": 440,
    "15A": 441,
    "15B": 442
}

# Define base directories
base_directories = {
    'summary_df': f'{year}/summary_df',
    'teams_df': f'{year}/teams_df',
    'schedules_df': f'{year}/schedules_df',
    'ranking_df': f'{year}/ranking_df',
    'players_df': f'{year}/players_df',
    'summarized_player_tables': f'{year}/summarized_player_tables',
    'unbeaten_players': f'{year}/unbeaten_players',
    'played_every_game': f'{year}/played_every_game',
    'detailed_league_tables': f'{year}/detailed_league_tables',
    'awaiting_results': f'{year}/awaiting_results',
    'home_away_data': f'{year}/home_away_data',
    'team_win_percentage_breakdown_home': f'{year}/team_win_percentage_breakdown/Home',
    'team_win_percentage_breakdown_away': f'{year}/team_win_percentage_breakdown/Away',
    'team_win_percentage_breakdown_delta': f'{year}/team_win_percentage_breakdown/Delta',
    'team_win_percentage_breakdown_overall': f'{year}/team_win_percentage_breakdown/Overall',
    'simulated_tables': f'{year}/simulated_tables',
    'simulated_fixtures': f'{year}/simulated_fixtures'
}

# Ensure the logs directory exists
os.makedirs(f"{year}/logs", exist_ok=True)

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            f"{year}/logs/{year}_log.txt", maxBytes=5*1024*1024, backupCount=5
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
        if (rubber_index < len(row['Rubbers']) and
            pd.notna(row['Rubbers'][rubber_index]) and
                row['Rubbers'][rubber_index] not in ['CR', 'WO']):
            valid_matches_count[row['Home Team']] = valid_matches_count.get(row['Home Team'], 0) + 1
            valid_matches_count[row['Away Team']] = valid_matches_count.get(row['Away Team'], 0) + 1
    return valid_matches_count


def simulate_rubber(home_team, away_team, rubber_index, combined, home_advantage_factor, unpredictability_factor):
    """
    A function to simulate the results of a rubber based on each team's previous win percentage,
    factoring in home advantage and unpredictability.
    """
    logging.debug(f"Simulating rubber {rubber_index} between {home_team} and {away_team}")

    # Original win probabilities
    original_home_prob = combined.loc[home_team, f'Rubber {rubber_index} Win %'] / 100
    original_away_prob = combined.loc[away_team, f'Rubber {rubber_index} Win %'] / 100

    # Handle the case when both probabilities are zero
    if original_home_prob == 0 and original_away_prob == 0:
        original_home_prob = 0.5
        original_away_prob = 0.5

    # Adjust for home advantage
    adjusted_home_prob = original_home_prob * (1 + home_advantage_factor)
    adjusted_away_prob = original_away_prob * (1 - home_advantage_factor)

    # Apply unpredictability factor
    final_home_prob = unpredictability_factor * 0.5 + (1 - unpredictability_factor) * adjusted_home_prob
    final_away_prob = unpredictability_factor * 0.5 + (1 - unpredictability_factor) * adjusted_away_prob

    # Normalize the final probabilities to sum up to 1
    final_total_prob = final_home_prob + final_away_prob
    final_home_prob /= final_total_prob
    final_away_prob /= final_total_prob

    # Choose the winner based on the adjusted probabilities
    return np.random.choice([home_team, away_team], p=[final_home_prob, final_away_prob])


def simulate_match(home_team, away_team, max_rubbers, combined, home_advantage_factor, unpredictability_factor):
    """
    A function that applies the simulate_rubber function to all rubbers in a given match
    """
    logging.debug(f"Simulating match between {home_team} and {away_team}")
    results = []
    for i in range(1, max_rubbers + 1):
        winner = simulate_rubber(home_team, away_team, i, combined, home_advantage_factor, unpredictability_factor)
        results.append(winner)
        logging.debug(f"Rubber {i}: Winner is {winner}")
    return results


def calculate_match_points(match_result, home_team, away_team):
    """
    A function that calculates the results of a match
    """
    counter = Counter(match_result)

    # If all rubbers are won by one team
    if len(counter) == 1:
        winner = counter.most_common(1)[0][0]
        winner_points = counter[winner] + 1  # Bonus point for winning
        # Identify the loser as the team that is not the winner
        loser = home_team if winner != home_team else away_team
        loser_points = 0
        return winner, winner_points, loser, loser_points

    # Normal case with at least one rubber won by each team
    elif len(counter) >= 2:
        winner = counter.most_common(1)[0][0]
        loser = counter.most_common(2)[1][0]
        winner_points = counter[winner] + 1  # Bonus point for winning
        loser_points = counter[loser]
        return winner, winner_points, loser, loser_points

    # If unable to determine winner and loser
    return None


def simulate_league(df_fixtures, summary_df, num_simulations, max_rubbers, combined, home_advantage_factor,
                    unpredictability_factor, neutral_fixtures_df):
    """
    A function to simulate the remaining fixtures in a league,
    using the simulate_rubber, simulate_match, and calculate_match_points
    functions
    """

    logging.info(f"Starting league simulation with {num_simulations} simulations")

    # Convert 'Team' column to string in summary_df
    summary_df['Team'] = summary_df['Team'].astype(str)

    # Initialize columns if not present
    for column in ['Played', 'Won', 'Lost', 'Points']:
        if column not in summary_df.columns:
            summary_df[column] = 0
        else:
            summary_df[column] = summary_df[column].astype(int)

    final_tables = []

    # Dictionary to track position counts for each team
    position_counts = defaultdict(lambda: defaultdict(int))

    # Dictionary to store total points gained in each fixture
    fixture_points = defaultdict(lambda: defaultdict(int))

    # Start total runtime timer
    total_start_time = time.time()

    for simulation in range(num_simulations):
        if simulation % 1000 == 0:
            logging.debug(f"Simulation {simulation}/{num_simulations}")

        league_table = summary_df.copy()
        league_table['Simulated Points'] = 0
        league_table['Simulated Played'] = 0
        league_table['Simulated Won'] = 0
        league_table['Simulated Lost'] = 0

        for index, match in df_fixtures.iterrows():
            home_team = match['Home Team']
            away_team = match['Away Team']

            # Filter the neutral_fixtures_df for rows where both home and away teams match
            filtered_df = neutral_fixtures_df[(neutral_fixtures_df['Home Team'] == home_team) &
                                              (neutral_fixtures_df['Away Team'] == away_team)]

            # Check if the filtered dataframe is empty
            is_neutral_fixture = not filtered_df.empty

            # Check if both teams are in the combined data
            if home_team in combined.index and away_team in combined.index:
                # Adjust home_advantage_factor for neutral fixtures
                adjusted_home_advantage = 0 if is_neutral_fixture else home_advantage_factor
                match_result = simulate_match(home_team, away_team, max_rubbers, combined, adjusted_home_advantage,
                                              unpredictability_factor)
                points_result = calculate_match_points(match_result, home_team, away_team)

                # Update simulated results
                if points_result:
                    winner, winner_points, loser, loser_points = points_result
                    league_table.loc[
                        league_table['Team'] == winner, ['Simulated Points', 'Simulated Won', 'Simulated Played']] += [
                        winner_points, 1, 1]
                    league_table.loc[
                        league_table['Team'] == loser, ['Simulated Points', 'Simulated Lost', 'Simulated Played']] += [
                        loser_points, 1, 1]
                    fixture_points[index]['Home'] += winner_points if match['Home Team'] == winner else loser_points
                    fixture_points[index]['Away'] += loser_points if match['Home Team'] == winner else winner_points
                else:
                    logging.warning(f"Could not calculate points for match between {home_team} and {away_team}")
            else:
                logging.warning(f"Teams {home_team} or {away_team} not found in the combined data")

        # Sum the simulated results with the actual results
        league_table['Played'] += league_table['Simulated Played']
        league_table['Won'] += league_table['Simulated Won']
        league_table['Lost'] += league_table['Simulated Lost']
        league_table['Points'] += league_table['Simulated Points']

        # Sort the league table based on Points, with Won and Rubbers For as tiebreakers
        league_table.sort_values(by=['Points', 'Won', 'Rubbers For'], ascending=[False, False, False], inplace=True)
        league_table.reset_index(drop=True, inplace=True)

        # Track the final position of each team in this simulation
        for position, row in league_table.iterrows():
            position_counts[row['Team']][position + 1] += 1

        final_tables.append(league_table)

    # Calculate average points per fixtures
    for index in fixture_points:
        total_simulations = num_simulations
        fixture_points[index]['Home'] = fixture_points[index]['Home'] / total_simulations
        fixture_points[index]['Away'] = fixture_points[index]['Away'] / total_simulations

    # Add average points to df_fixtures (outside the simulation loop)
    df_fixtures['Avg Simulated Home Points'] = df_fixtures.index.map(lambda x: fixture_points[x]['Home'])
    df_fixtures['Avg Simulated Away Points'] = df_fixtures.index.map(lambda x: fixture_points[x]['Away'])

    # End total runtime timer
    total_end_time = time.time()
    run_time = total_end_time - total_start_time

    # Format runtime output
    if run_time > 60:
        minutes = int(run_time // 60)
        seconds = int(run_time % 60)
        logging.info(f"Total simulation runtime: {minutes} minutes and {seconds} seconds.")
    else:
        logging.info(f"Total simulation runtime: {run_time:.2f} seconds.")

    aggregated_table = pd.concat(final_tables).groupby('Team', as_index=False).mean()

    # Convert 'Team' column to string in aggregated_table
    aggregated_table['Team'] = aggregated_table['Team'].astype(str)

    # Merge aggregated results with summary_df
    summary_df = summary_df.merge(aggregated_table, on='Team', how='left', suffixes=('', '_sim'))

    # Sum the simulated results with the actual results
    summary_df['Played'] += summary_df['Simulated Played']
    summary_df['Won'] += summary_df['Simulated Won']
    summary_df['Lost'] += summary_df['Simulated Lost']
    summary_df['Points'] += summary_df['Simulated Points']

    # Drop the redundant '_sim' columns
    summary_df.drop(columns=['Played_sim', 'Won_sim', 'Lost_sim', 'Points_sim',
                             'Simulated Points', 'Simulated Played',
                             'Simulated Won', 'Simulated Lost'], inplace=True)

    summary_df["Played"] = summary_df["Played"].astype(int)

    # Calculate the percentage chance of each team finishing in each position
    for team in position_counts:
        total = sum(position_counts[team].values())
        for position in position_counts[team]:
            position_counts[team][position] = (position_counts[team][position] / total) * 100

    # Convert position_counts to DataFrame
    position_percentage_df = pd.DataFrame.from_dict(position_counts, orient='index').fillna(0)

    # Sort position columns numerically
    position_cols = sorted(position_percentage_df.columns, key=int)
    position_percentage_df = position_percentage_df[position_cols]

    # Calculate the 'Playoffs' probability (sum of positions 1 to 4)
    position_percentage_df['Playoffs'] = position_percentage_df[[1, 2, 3, 4]].sum(axis=1)

    # Add the 'Team' column for merging
    position_percentage_df.reset_index(inplace=True)
    position_percentage_df.rename(columns={'index': 'Team'}, inplace=True)

    # Merge with summary_df
    summary_df = summary_df.merge(position_percentage_df, on='Team', how='left')

    # Reorder final DataFrame columns
    standard_columns = ['Team', 'Played', 'Won', 'Lost', 'Points', 'Playoffs']
    final_column_order = standard_columns + position_cols
    summary_df = summary_df[final_column_order]

    # Sort the DataFrame based on 'Points' and 'Won' as a tiebreaker in descending order
    summary_df.sort_values(by=['Points', "Won", 1], ascending=[False, False, False], inplace=True)

    # Return the updated league table
    return summary_df, df_fixtures


def scrape_team_summary_page(league_id, year):
    """
    Function to scrape the Team Summary page on Hk squash website and store the data in a dataframe
    """
    summary_url = (
        f"https://www.hksquash.org.hk/public/index.php/leagues/team_summery/"
        f"id/{league_id}/league/Squash/year/{year}/pages_id/25.html"
    )

    # Add logging to track the progress
    logging.info(f"Scraping team summary page for league id: {league_id}, year: {year}...")
    logging.debug(f"Constructed summary URL: {summary_url}")

    try:
        # Send the HTTP request
        response = requests.get(summary_url)
        logging.debug(f"Received response with status code: {response.status_code}")

        # Check if the response is successful
        if response.status_code != 200:
            logging.error(f"Failed to retrieve team summary page. Status code: {response.status_code}")
            return pd.DataFrame()
         
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        logging.debug("Parsed HTML content with BeautifulSoup")

        # Find the league table data
        summary_rows = soup.find_all("div", class_="clearfix teamSummary-content-list")
        logging.debug(f"Found {len(summary_rows)} team summary rows")

        # Extract the league table table data from summary_rows soup element
        summary_data_rows = []

        for idx, row in enumerate(summary_rows):
            columns = row.find_all("div", recursive=False)
            row_data = [col.text.strip() for col in columns if col.text.strip()]
            if row_data and len(row_data) > 1:  # To skip header rows and empty rows
                summary_data_rows.append(row_data)
            else:
                logging.debug(f"Skipping row {idx} due to insufficient data: {row_data}")

        # Check if any data was extracted
        if not summary_data_rows:
            logging.warning("No data rows were extracted from the team summary page.")
            return pd.DataFrame()
        
        # Create DataFrame from list of lists
        # Skip the first row as it's the header
        headers = ["Team", "Played", "Won", "Lost", "Points"]
        summary_df = pd.DataFrame(summary_data_rows[1:], columns=headers)
        logging.info(f"Successfully created summary DataFrame with {len(summary_df)} rows")

        # Convert numeric columns to appropriate data types
        summary_df[['Played', 'Won', 'Lost', 'Points']] = summary_df[['Played', 'Won', 'Lost', 'Points']].apply(pd.to_numeric, errors='coerce')
        logging.debug("Converted numeric columns to appropriate data types")

        return summary_df
    
    except Exception as e:
        logging.exception(f"An error occured in scrape_team_summary_page: {e}")
        return pd.DataFrame()


def scrape_teams_page(league_id, year):
    """
    Function to scrape the Teams page on HK squash website and store the data in a dataframe
    """
    teams_url = (
        f"https://www.hksquash.org.hk/public/index.php/leagues/teams/"
        f"id/{league_id}/league/Squash/year/{year}/pages_id/25.html"
    )

    logging.info(f"Starting scrape_teams_page for league id: {league_id}, year: {year}")
    logging.debug(f"Constructed teams URL: {teams_url}")

    try:
        # Send the HTTP request
        response = requests.get(teams_url)
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
    schedule_url = (
        f"https://www.hksquash.org.hk/public/index.php/leagues/results_schedules/"
        f"id/{league_id}/league/Squash/year/{year}/pages_id/25.html"
    )

    # Add logging to track the progress
    logging.info(f"Scraping schedules and results page for league id: {league_id}, year: {year}...")
    logging.debug(f"Constructed schedule URL: {schedule_url}")

    try:
        # Send the HTTP request
        response = requests.get(schedule_url)
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
    ranking_url = (
        f"https://www.hksquash.org.hk/public/index.php/leagues/ranking/id/{league_id}/"
        f"league/Squash/year/{year}/pages_id/25.html"
    )

    logging.info(f"Scraping ranking page for league id: {league_id}, year: {year}...")
    logging.debug(f"Constructed ranking URL: {ranking_url}")

    # Send the HTTP request
    response = requests.get(ranking_url)
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
        raise Exception("No data rows were extracted from the ranking page.")

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

    # Creating the summarized DataFrame
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
        ranking_df_filtered["Lost"] == 0].apply(lambda row: f"{row['Name of Player']} ({row['Team']})", axis=1).tolist()
    logging.info(f"Found {len(unbeaten_list)} unbeaten players")

    return df, summarized_df, unbeaten_list, ranking_df_filtered


def scrape_players_page(league_id, year):
    """
    Function to scrape the Players page and store data in a DataFrame.
    """

    logging.info(f"Starting scrape_players_page for league_id: {league_id}, year: {year}")

    players_url = (f"https://www.hksquash.org.hk/public/index.php/"
                   f"leagues/players/id/{league_id}/league/Squash/year/{year}/pages_id/25.html")
    
    logging.debug(f"Constructed players URL: {players_url}")

    try:
        # Send the HTTP request
        response = requests.get(players_url)
        logging.debug(f"Received response with status code: {response.status_code}")

        # Check if the response is successful
        if response.status_code != 200:
            logging.error(f"Failed to retrieve players page. Status code: {response.status_code}")
            return pd.DataFrame()
        
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
            try:
                team_name_div = team_container.find("div", string="team name:")
                team_name = team_name_div.find_next_sibling().get_text(strip=True)
                logging.debug(f"Team {idx}: Extracted team name: {team_name}")
            except Exception as e:
                logging.warning(f"Team {idx}: Error extracting team name: {e}")
                continue

            # Initialize a list to store each player's data for this team
            players_data = []

            # Extract player data
            player_rows = team_container.find_all("div", class_="players-content-list")
            logging.debug(f"Team {idx}: Found {len(player_rows)} player rows")

            for player_idx, player in enumerate(player_rows):
                # Extract data from 'col-xs-2' and 'col-xs-4' classes
                order_rank_points = [div.get_text(strip=True) for div in player.find_all("div", class_="col-xs-2")]
                player_name = [div.get_text(strip=True) for div in player.find_all("div", class_="col-xs-4")]
                player_data = order_rank_points[:1] + player_name + order_rank_points[1:]
                players_data.append(player_data)
                logging.debug(f"Team {idx}, Player {player_idx}: Extracted data: {player_data}")

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
            logging.info(f"Team {idx}: Created DataFrame with {len(df)} rows for team: {team_name}")

            time.sleep(5)

        if team_dataframes:
            # Concatenate all team dataframes
            combined_df = pd.concat(team_dataframes, ignore_index=True)
            logging.info(f"Concatenated all team dataframes into a single DataFrame with {len(combined_df)} rows")
            return combined_df
        else:
            logging.warning("No team dataframes were created. Returning an empty DataFrame.")
            return pd.DataFrame()    

    except Exception as e:
        logging.exception(f"An error occured in scrape_players_page: {e}")
        return pd.DataFrame()
    

# Use logging to track progress
logging.info("Starting the scraping process...")

# Change dictionary if you want specific week
for div in thursday.keys():
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
    
    # Save the schedules_df to CSV
    schedules_df_path = os.path.join(base_directories['schedules_df'], week_dir, f"{div}_schedules_df.csv")
    try:
        logging.info(f"Saving schedules_df to {schedules_df_path}")
        schedules_df.to_csv(schedules_df_path, index=False)
        logging.info(f"Successfully saved schedules_df to {schedules_df_path}")
    except Exception as e:
        logging.error(f"Error saving schedules_df to {schedules_df_path}: {e}")

    time.sleep(10)

    # Scrape Team Summary page
    try:
        logging.info(f"Scraping Team Summary page for Division {div}")
        summary_df = scrape_team_summary_page(league_id, year)
        logging.info(f"Successfully scraped Team Summary page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Team Summary page for Division {div}: {e}")
        continue

    # Check if the summary_df is empty
    if summary_df.empty:
        logging.warning(f"No data found in summary_df for Division {div}. Skipping further processing.")
        continue

    # Save the summary_df to CSV
    summary_df_path = os.path.join(base_directories['summary_df'], week_dir, f"{div}_summary_df.csv")
    try:
        logging.info(f"Saving summary_df to {summary_df_path}")
        summary_df.to_csv(summary_df_path, index=False)
        logging.info(f"Successfully saved summary_df to {summary_df_path}")
    except Exception as e:
        logging.error(f"Error saving summary_df to {summary_df_path}: {e}")
    
    time.sleep(10)

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

    time.sleep(10)

    # Scrape Ranking page
    try:
        logging.info(f"Scraping Ranking page for Division {div}")
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scrape_ranking_page(league_id, year)
        logging.info(f"Successfully scraped Ranking page for Division {div}")
    except Exception as e:
        logging.error(f"Error scraping Ranking page for Division {div}: {e}")
        # Stop execution if an error occurs
        raise

    # Save the ranking_df to CSV
    ranking_df_path = os.path.join(base_directories['ranking_df'], week_dir, f"{div}_ranking_df.csv")
    try:
        logging.info(f"Saving ranking_df to {ranking_df_path}")
        ranking_df.to_csv(ranking_df_path, index=False)
        logging.info(f"Successfully saved ranking_df to {ranking_df_path}")
    except Exception as e:
        logging.error(f"Error saving ranking_df to {ranking_df_path}: {e}")

    time.sleep(10)

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

    time.sleep(10)

    # Get list of players who have played every possible game
    if not ranking_df_filtered.empty and not summary_df.empty:
        merged_ranking_df = ranking_df_filtered.merge(summary_df[["Team", "Played"]], on="Team", how="inner")
        merged_ranking_df = merged_ranking_df.rename(columns={"Played": "Team Games Played"})
        merged_ranking_df["Team Games Played"] = merged_ranking_df["Team Games Played"].astype(int)
        played_every_game_list = merged_ranking_df[
            (merged_ranking_df["Games Played"] == merged_ranking_df["Team Games Played"])].apply(
            lambda row: f"{row['Name of Player']} ({row['Team']})", axis=1
        ).tolist()
    else:
        logging.warning(f"No ranking data available for Division {div}. Unable to determine players who have played every game.")
        played_every_game_list = []
        unbeaten_list = []

    # Save summarized player tables to CSV
    if not summarized_df.empty:
        summarized_df.to_csv(os.path.join(base_directories['summarized_player_tables'], week_dir, f"{div}_summarized_players.csv"), index=False)
    else:
        logging.warning(f"No summarized player data available for Division {div}")

    # Save list of unbeaten players
    with open(os.path.join(base_directories['unbeaten_players'], week_dir, f"{div}.txt"), "w") as file:
        for item in unbeaten_list:
            file.write(f"{item}\n")

    # Save list of players who have played every game
    with open(os.path.join(base_directories['played_every_game'], week_dir, f"{div}.txt"), "w") as file:
        for item in played_every_game_list:
            file.write(f"{item}\n")

    # Create Results Dataframe

    # Drop unnecessary columns
    schedules_df.drop(columns=['vs', 'Time'], inplace=True)

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

    # Apply the function to the 'Result' column
    results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(lambda x: pd.Series(parse_result(x)))

    # Drop the original 'Result' column
    results_df.drop(columns=['Result'], inplace=True)

    # Replace 'CR' with NaN
    results_df.replace('CR', np.nan, inplace=True)
    results_df.replace('WO', np.nan, inplace=True)

    # Count the number of Rubbers For and Against for each team

    # Splitting the 'Overall Score' into two separate columns
    results_df[['Home Score', 'Away Score']] = results_df['Overall Score'].str.split('-', expand=True).astype(int)

    # Initialize dictionaries to keep track of won and conceded rubbers
    rubbers_won = {}
    rubbers_conceded = {}

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
    summary_df = pd.merge(summary_df, rubbers_df, on="Team")
    detailed_table_df = pd.merge(summary_df, detailed_table_df, on="Team")

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

    # Filter rows where the 'Date' is earlier than today to create awaiting_results dataframe
    today = pd.Timestamp(datetime.now().date())
    awaiting_results_df = df_remaining_fixtures[df_remaining_fixtures['Date'] < today]
    # Save awaiting results
    awaiting_results_df.to_csv(os.path.join(base_directories['awaiting_results'], week_dir, f"{div}_awaiting_results.csv"), index=False)

    # Create results dataframe that ignores games where away team plays at home venue
    # Create dictionary of team home venues
    team_home_venues = teams_df.set_index("Team Name")["Home"].to_dict()
    valid_matches_df = results_df[
        ~results_df.apply(lambda row: team_home_venues.get(row['Away Team']) == row['Venue'], axis=1)].copy()

    # Create dataframe of neutral fixtures
    neutral_fixtures_df = df_remaining_fixtures[
        df_remaining_fixtures.apply(lambda row: team_home_venues.get(row["Away Team"]) == row["Venue"], axis=1)].copy()

    # Calculate Home vs Away
    if not results_df.empty:
        # Apply the function to 'Overall Score' column
        results_df[['Home Overall Score', 'Away Overall Score']] = results_df['Overall Score'].apply(
            lambda x: pd.Series(split_overall_score(x)))

        # Calculate the average score for home and away teams
        average_home_overall_score = results_df['Home Overall Score'].mean()
        average_away_overall_score = results_df['Away Overall Score'].mean()

        # Calculate home win percentage
        home_win_perc = len(
            results_df[results_df["Home Overall Score"] > results_df["Away Overall Score"]]) / len(results_df)
    else:
        logging.warning("No results data to calculate home vs away statistics for Division {div}.")
        average_home_overall_score = 0
        average_away_overall_score = 0
        home_win_perc = 0

    # Path to the overall scores CSV file (excluding week_dir)
    overall_scores_file = os.path.join(base_directories['home_away_data'], f"{div}_overall_scores.csv")

    # Read the existing data from the CSV file into a DataFrame
    if os.path.exists(overall_scores_file):
        overall_scores_df = pd.read_csv(overall_scores_file, header=None)
    else:
        # Create an empty DataFrame with the expected columns
        overall_scores_df = pd.DataFrame(columns=[0, 1, 2, 3, 4])

    # Calculate average home score for each home team
    average_home_scores = results_df.groupby('Home Team')['Home Overall Score'].mean().rename('Average Home Score')

    # Calculate average away score for each away team
    average_away_scores = results_df.groupby('Away Team')['Away Overall Score'].mean().rename('Average Away Score')

    # Combine the two Series into one DataFrame
    team_average_scores = pd.concat([average_home_scores, average_away_scores], axis=1)

    # Check the index name
    print("Index name of team_average_scores:", team_average_scores.index.name)

    # Handle missing values by filling NaN with 0 or using appropriate methods
    team_average_scores['Average Home Score'] = team_average_scores['Average Home Score'].fillna(0)
    team_average_scores['Average Away Score'] = team_average_scores['Average Away Score'].fillna(0)

    # Calculate the difference in home and away scores for each team
    team_average_scores["home_away_diff"] = team_average_scores["Average Home Score"] - team_average_scores[
        "Average Away Score"]

    # Merge with teams_df to get home venue info
    team_average_scores = team_average_scores.merge(teams_df[["Team Name", "Home"]],
                                                    left_on=team_average_scores.index, right_on="Team Name",
                                                    how="inner")
    
    # Reorganise columns and show teams in order of home/away split
    team_average_scores = team_average_scores[
        ["Team Name", "Home", "Average Home Score", "Average Away Score", "home_away_diff"]
    ]

    # Since 'home_away_diff' may not be meaningful at this point, you can add a check
    if team_average_scores['home_away_diff'].isnull().all():
        logging.warning("All 'home_away_diff' values are NaN or zero. Teams may not have played both home and away games yet.")
    else:
        # Sort the DataFrame based on 'home_away_diff'
        team_average_scores.sort_values("home_away_diff", ascending=False, inplace=True)

    # Save team_average_scores to csv
    team_average_scores.to_csv(os.path.join(base_directories['home_away_data'], week_dir, f"{div}_team_average_scores.csv"), index=False)

    # Show home/away split by venue
    venue_split = pd.pivot_table(data=team_average_scores,
                                 index="Home",
                                 values="home_away_diff").sort_values("home_away_diff", ascending=False)

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
        today if run_projections == 1 else None
    ]

    # Assign the data to the first row
    overall_scores_df.loc[0] = new_data

    # Write the updated DataFrame back to the CSV file
    overall_scores_df.to_csv(overall_scores_file, index=False, header=None)

    # Wait so as not to get a connection error
    time.sleep(15)

    # Use run_projections to determine whether to run projections or not
    if run_projections == 1:
        projected_final_table, projected_fixtures = simulate_league(df_remaining_fixtures,
                                                                    summary_df,
                                                                    num_simulations,
                                                                    max_rubbers,
                                                                    combined,
                                                                    home_advantage_factor,
                                                                    unpredictability_factor,
                                                                    neutral_fixtures_df)

        # Save the results
        projected_final_table.to_csv(os.path.join(base_directories['simulated_tables'], week_dir, f"{div}_proj_final_table.csv"), index=False)
        projected_fixtures.to_csv(os.path.join(base_directories['simulated_fixtures'], week_dir, f"{div}_proj_fixtures.csv"), index=False)

    print(div)
