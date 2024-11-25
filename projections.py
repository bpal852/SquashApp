import pandas as pd
import numpy as np
import os
import logging
import time
from datetime import datetime
from collections import Counter, defaultdict

# Set parameters
year = "2024-2025"
num_simulations = 1999
home_advantage_factor = 0.05
unpredictability_factor = 0.4

# Define functions

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the divisions you want to run projections for
divisions = [
    "Premier Main", "2", "3", "4", "5", "6", "7A", "7B", "8A", "8B", "9", "10", "11", "12",
    "13A", "13B", "14", "15A", "15B", "Premier Masters", "M2", "M3", "M4",
    "Premier Ladies", "L2", "L3", "L4"
]

wednesday = ["7A", "7B", "9", "12", "M2"]

# Base directories (ensure they match the ones used in the data gathering script)
base_directories = {
    'summary_df': f'{year}/summary_df',
    'schedules_df': f'{year}/schedules_df',
    'team_win_percentage_breakdown_overall': f'{year}/team_win_percentage_breakdown/Overall',
    'simulated_tables': f'{year}/simulated_tables',
    'simulated_fixtures': f'{year}/simulated_fixtures',
    'remaining_fixtures': f'{year}/remaining_fixtures',
    'neutral_fixtures': f'{year}/neutral_fixtures',
}

# Ensure output directories exist
os.makedirs(base_directories['simulated_tables'], exist_ok=True)
os.makedirs(base_directories['simulated_fixtures'], exist_ok=True)

# Process each division
for div in divisions:
    logging.info(f"Processing projections for Division {div}")

    # Find the latest week directory for the division
    try:
        # Get all week directories for this division's summary_df
        division_summary_path = os.path.join(base_directories['summary_df'])
        week_dirs = [d for d in os.listdir(division_summary_path) if os.path.isdir(os.path.join(division_summary_path, d))]

        # Filter week directories that contain the division's summary file
        valid_week_dirs = []
        for week_dir in week_dirs:
            summary_file = os.path.join(division_summary_path, week_dir, f"{div}_summary_df.csv")
            if os.path.exists(summary_file):
                valid_week_dirs.append(week_dir)

        if not valid_week_dirs:
            logging.warning(f"No valid week directories found for Division {div}. Skipping.")
            continue

        # Sort the week directories and get the latest one
        latest_week_dir = sorted(valid_week_dirs, key=lambda x: int(x.split('_')[1]), reverse=True)[0]
        logging.info(f"Using data from {latest_week_dir} for Division {div}")

    except Exception as e:
        logging.error(f"Error determining latest week directory for Division {div}: {e}")
        continue

    try:
        # Load data
        df_remaining_fixtures_path = os.path.join(
            base_directories['remaining_fixtures'], latest_week_dir, f"{div}_remaining_fixtures.csv")
        summary_df_path = os.path.join(
            base_directories['summary_df'], latest_week_dir, f"{div}_summary_df.csv")
        win_percentage_df_path = os.path.join(
            base_directories['team_win_percentage_breakdown_overall'], latest_week_dir, f"{div}_team_win_percentage_breakdown.csv")
        neutral_fixtures_df_path = os.path.join(
            base_directories['neutral_fixtures'], latest_week_dir, f"{div}_neutral_fixtures.csv")

        df_remaining_fixtures = pd.read_csv(df_remaining_fixtures_path)
        summary_df = pd.read_csv(summary_df_path)
        win_percentage_df = pd.read_csv(win_percentage_df_path)

        # Check if neutral fixtures file exists
        if os.path.exists(neutral_fixtures_df_path):
            neutral_fixtures_df = pd.read_csv(neutral_fixtures_df_path)
        else:
            # If not, create an empty DataFrame with the necessary columns
            neutral_fixtures_df = pd.DataFrame(columns=['Home Team', 'Away Team', 'Venue', 'Match Week', 'Date'])

    except Exception as e:
        logging.error(f"Error loading data for Division {div}: {e}")
        continue

    # Prepare data
    # Ensure 'Team' is string
    win_percentage_df['Team'] = win_percentage_df['Team'].astype(str)

    # Set 'Team' as index
    combined = win_percentage_df.set_index('Team')

    # Determine max_rubbers
    max_rubbers = len([col for col in combined.columns if 'Rubber' in col and 'Win %' in col])

    # Get the list of teams from the fixtures
    teams_in_fixtures = set(df_remaining_fixtures['Home Team']).union(set(df_remaining_fixtures['Away Team']))

    # Get the list of teams in the combined data
    teams_in_combined = set(combined.index)

    # Find missing teams
    missing_teams = teams_in_fixtures - teams_in_combined

    if missing_teams:
        logging.warning(f"The following teams are missing in the combined data for Division {div}: {', '.join(missing_teams)}")
        logging.warning(f"Skipping Division {div} due to missing team data.")
        continue  # Skip to the next division

    # Run simulations
    try:
        projected_final_table, projected_fixtures = simulate_league(
            df_remaining_fixtures,
            summary_df,
            num_simulations,
            max_rubbers,
            combined,
            home_advantage_factor,
            unpredictability_factor,
            neutral_fixtures_df
        )
    except Exception as e:
        logging.error(f"Error running simulations for Division {div}: {e}")
        continue

    # Ensure output directories for this week exist
    simulated_tables_dir = os.path.join(base_directories['simulated_tables'], latest_week_dir)
    simulated_fixtures_dir = os.path.join(base_directories['simulated_fixtures'], latest_week_dir)
    os.makedirs(simulated_tables_dir, exist_ok=True)
    os.makedirs(simulated_fixtures_dir, exist_ok=True)

    # Define the path for the simulation date file
    simulation_date_file = os.path.join(simulated_tables_dir, f"{div}_simulation_date.txt")

    # Save the results
    projected_final_table_path = os.path.join(simulated_tables_dir, f"{div}_proj_final_table.csv")
    projected_fixtures_path = os.path.join(simulated_fixtures_dir, f"{div}_proj_fixtures.csv")
    logging.info(f"Saving projections for Division {div} to {projected_final_table_path} and {projected_fixtures_path}")

    # Save the simulation date to the file
    simulation_date = datetime.now().strftime('%Y-%m-%d')
    with open(simulation_date_file, 'w') as f:
        f.write(simulation_date)
    logging.info(f"Saved simulation date to {simulation_date_file}")

    projected_final_table.to_csv(projected_final_table_path, index=False)
    projected_fixtures.to_csv(projected_fixtures_path, index=False)

    logging.info(f"Finished projections for Division {div}")

logging.info("All projections completed.")