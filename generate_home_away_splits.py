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

# Iterate through each season directory
for season in season_folders:
    season_path = os.path.join(base_path, season)
    if not os.path.isdir(season_path) or season == excluded_season:
        continue

    # Create results_df and home_away_data folders for the current season if they don't exist
    season_results_folder = os.path.join(season_path, 'results_df')
    season_home_away_folder = os.path.join(season_path, 'home_away_data')
    
    os.makedirs(season_results_folder, exist_ok=True)
    os.makedirs(season_home_away_folder, exist_ok=True)

    # Iterate through each division for the given season
    schedules_path = os.path.join(season_path, "schedules_df")
    teams_path = os.path.join(season_path, "teams_df")

    # Iterate through each division for the given season
    schedules_path = os.path.join(season_path, "schedules_df")
    teams_path = os.path.join(season_path, "teams_df")

    # Find all schedule files in the season
    for schedule_file in glob.glob(os.path.join(schedules_path, "*.csv")):
        # Extract the division number from the filename
        match = re.search(r"([A-Za-z0-9]+[A-Za-z0-9]*)_schedules_df.csv", os.path.basename(schedule_file))
        if match:
            division_number = match.group(1)
        else:
            logging.warning(f"File name does not match expected pattern: {schedule_file}")
            continue  # Skip this file if it doesn't match the pattern

        # Load the corresponding teams file
        teams_file = os.path.join(teams_path, f"{division_number}_teams_df.csv")
        if not os.path.exists(teams_file):
            logging.warning(f"Teams file for division {division_number} in season {season} not found.")
            continue

        # Load DataFrames
        schedules_df = pd.read_csv(schedule_file)
        teams_df = pd.read_csv(teams_file)

        # Define functions

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


        # Create results_df from schedules_df

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

        # Create Games Won columns
        results_df[['Home Games Won', 'Away Games Won']] = results_df.apply(count_games_won, axis=1, result_type='expand')

        # Create results dataframe that ignores games where away team plays at home venue
        # Create dictionary of team home venues
        team_home_venues = teams_df.set_index("Team Name")["Home"].to_dict()
        valid_matches_df = results_df[
            ~results_df.apply(lambda row: team_home_venues.get(row['Away Team']) == row['Venue'], axis=1)].copy()
        
        # Skip further processing if valid_matches_df is empty
        if valid_matches_df.empty:
            logging.warning(f"No valid matches found for division {division_number} in season {season}. Skipping...")
            continue  # Skip to the next loop iteration if no valid matches
        
        # Calculate the average score for home and away teams, excluding neutral venues
        average_home_overall_score_exc_neutral = valid_matches_df['Home Score'].mean()
        average_away_overall_score_exc_neutral = valid_matches_df['Away Score'].mean()

        # Apply home_team_won function to each row
        valid_matches_df['Winner'] = valid_matches_df.apply(home_team_won, axis=1)

        # Calculate home team win percentage, filtering out matches where the winner is 'Ignore'
        home_win_percentage = valid_matches_df[valid_matches_df["Winner"] != "Ignore"]["Winner"].value_counts(normalize=True)["Home"]

        # Calculate average home score for each home team
        average_home_score = valid_matches_df.groupby("Home Team")["Home Score"].mean().rename("Average Home Score")

        # Calculate average away score for each away team
        average_away_score = valid_matches_df.groupby("Away Team")["Away Score"].mean().rename("Average Away Score")

        # Combine the two Series into one DataFrame
        team_average_scores = pd.concat([average_home_score, average_away_score], axis=1)

        # Calculate the difference in home and away scores for each team
        team_average_scores["Difference"] = team_average_scores["Average Home Score"] - team_average_scores["Average Away Score"]

        # Merge with teams_df to get home venue info
        team_average_scores = team_average_scores.merge(teams_df, left_index=True, right_on="Team Name")

        # Reorganise columns and show teams in order of difference in scores
        team_average_scores = team_average_scores[
            ["Team Name", "Home", "Average Home Score", "Average Away Score", "Difference"]
            ].sort_values("Difference", ascending=False)
        
        # Save the results_df to a CSV file
        results_file = os.path.join(season_path, f"results_df/{division_number}_results_df.csv")
        results_df.to_csv(results_file, index=False)

        # Save the team average scores to a CSV file
        average_scores_file = os.path.join(season_path, f"home_away_data/{division_number}_team_average_scores.csv")
        team_average_scores.to_csv(average_scores_file, index=False)

        # Save average_home_overall_score_exc_neutral, average_away_overall_score_exc_neutral, home_win_percentage to a CSV file
        average_scores_file = os.path.join(season_path, f"home_away_data/{division_number}_overall_scores.csv")
        with open(average_scores_file, 'w') as f:
            f.write(f"{average_home_overall_score_exc_neutral},{average_away_overall_score_exc_neutral},{home_win_percentage}")

        # Add logging
        logging.info(f"Saved team average scores for division {division_number} in season {season} to {average_scores_file}")