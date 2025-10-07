import pandas as pd
import os
import re
import glob
import logging
import numpy as np

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

# Create a folder for detailed league tables if it doesn't exist
output_folder = os.path.join(base_path, 'detailed_league_tables')
os.makedirs(output_folder, exist_ok=True)

# Iterate through each season directory
for season in season_folders:
    season_path = os.path.join(base_path, season)
    if not os.path.isdir(season_path) or season == excluded_season:
        continue  # Skip if it's not a directory or if it's the excluded season

    # Iterate through each division for the given season
    schedules_path = os.path.join(season_path, "schedules_df")
    summary_path = os.path.join(season_path, "summary_df")

    # Find all schedule files in the season
    for schedule_file in glob.glob(os.path.join(schedules_path, "*.csv")):
        # Extract the division number from the filename
        match = re.search(r"([A-Za-z0-9]+)_schedules_df.csv", os.path.basename(schedule_file))
        if match:
            division_number = match.group(1)
        else:
            logging.warning(f"File name does not match expected pattern: {schedule_file}")
            continue  # Skip this file if it doesn't match the pattern
        
        # Load the corresponding summary file
        summary_file = os.path.join(summary_path, f"{division_number}_summary_df.csv")
        if not os.path.exists(summary_file):
            logging.warning(f"Summary file for division {division_number} in season {season} not found.")
            continue

        # Load DataFrames
        schedules_df = pd.read_csv(schedule_file)
        summary_df = pd.read_csv(summary_file)

        ### Functions
        def parse_result(result):
            """
            Function to parse the 'result' string
            """
            overall, rubbers = result.split('(')
            rubbers = rubbers.strip(')').split(',')
            return overall, rubbers

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

        ### Process Data
        # Drop unnecessary columns
        schedules_df.drop(columns=['vs', 'Time'], inplace=True)

        # Exclude rows where 'Away Team' is '[BYE]'
        results_df = schedules_df[schedules_df['Away Team'] != '[BYE]'].copy()

        # Replace NaN values in 'Result' with an empty string
        results_df['Result'] = results_df['Result'].fillna('')

        # Keep rows where 'Result' contains brackets (indicative of a played match)
        results_df = results_df[results_df['Result'].str.contains(r'\(')]

        # Apply the function to the 'Result' column
        results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(lambda x: pd.Series(parse_result(x)))

        # Drop the original 'Result' column
        results_df.drop(columns=['Result'], inplace=True)

        # Replace 'CR' and 'WO' with NaN
        results_df.replace('CR', np.nan, inplace=True)
        results_df.replace('WO', np.nan, inplace=True)

        # Splitting the 'Overall Score' into two separate columns
        results_df[['Home Score', 'Away Score']] = results_df['Overall Score'].str.split('-', expand=True).astype(int)

        # Initialize dictionaries for rubbers won and conceded
        rubbers_won = {}
        rubbers_conceded = {}

        # Apply the function to each row
        results_df.apply(update_rubbers, axis=1)

        # Convert the dictionaries to DataFrames
        df_rubbers_won = pd.DataFrame(list(rubbers_won.items()), columns=['Team', 'Rubbers For'])
        df_rubbers_conceded = pd.DataFrame(list(rubbers_conceded.items()), columns=['Team', 'Rubbers Against'])

        # Merge the DataFrames on Team
        rubbers_df = pd.merge(df_rubbers_won, df_rubbers_conceded, on='Team')

        # Initialize dictionaries for CRs and WOs
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

        # Merge all the DataFrames on Team
        detailed_table_df = pd.merge(df_cr_given_count, df_cr_received_count, on='Team')
        detailed_table_df = pd.merge(detailed_table_df, df_wo_given_count, on='Team')
        detailed_table_df = pd.merge(detailed_table_df, df_wo_received_count, on='Team')
        detailed_table_df = pd.merge(detailed_table_df, rubbers_df, on='Team')
        detailed_table_df = pd.merge(summary_df, detailed_table_df, on='Team')

        # Save the detailed table for this division
        output_file = os.path.join(season_path, f"detailed_league_tables/{division_number}_detailed_league_table.csv")
        detailed_table_df.to_csv(output_file, index=False)
        logging.info(f"Saved detailed league table for division {division_number} in season {season} to {output_file}")
