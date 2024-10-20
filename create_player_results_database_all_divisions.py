import pandas as pd
import numpy as np
import os
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("create_player_results_database_all_divisions.log"),
        logging.StreamHandler()
    ]
)

# Base directory
base_directory = "C:/Users/bpali/PycharmProjects/SquashApp/2024-2025"

# Divisions to process
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

divisions_to_process = list(all_divisions.keys())

# Define functions

def parse_result(result):
    """
    Parse the 'Result' string into overall score and list of rubbers.
    Example result string: '3-2(3-0,2-3,3-1,1-3,3-2)'
    """
    try:
        if '(' in result and ')' in result:
            overall, rubbers_str = result.split('(')
            overall = overall.strip()
            rubbers_str = rubbers_str.strip(')')
            rubbers = rubbers_str.split(',')
            return overall, rubbers
        else:
            # Handle cases where result is not in expected format
            logging.warning(f"Result string '{result}' is not in expected format.")
            return result.strip(), []
    except Exception as e:
        logging.exception(f"Error parsing result string '{result}': {e}")
        return None, []
    

def determine_winner(rubber_score, home_player, away_player):
    """
    Determine the winner of a rubber based on the score.
    """
    if rubber_score in ['CR', 'WO', 'NA', np.nan]:
        return None  # No winner can be determined
    try:
        home_score, away_score = map(int, rubber_score.split('-'))
        if home_score > away_score:
            return home_player
        elif away_score > home_score:
            return away_player
        else:
            return None  # Draw, unlikely in squash
    except ValueError as e:
        logging.warning(f"Invalid score format '{rubber_score}' for players '{home_player}' vs '{away_player}': {e}")
        return None  # Invalid score format
    

def process_division(division, current_week, previous_week):
    """
    Process a division for a given week. 
    """
    # Construct file paths
    players_df_path = os.path.join(base_directory, "players_df", f"week_{current_week}", f"{division}_players_df.csv")
    schedules_df_path = os.path.join(base_directory, "schedules_df", f"week_{current_week}", f"{division}_schedules_df.csv")
    ranking_df_current_path = os.path.join(base_directory, "ranking_df", f"week_{current_week}", f"{division}_ranking_df.csv")
    ranking_df_previous_path = os.path.join(base_directory, "ranking_df", f"week_{previous_week}", f"{division}_ranking_df.csv")
    
    # Check if current week files exist
    current_files_exist = all([
        os.path.exists(players_df_path),
        os.path.exists(schedules_df_path),
        os.path.exists(ranking_df_current_path)
    ])

    if not current_files_exist:
        logging.warning(f"Data files for Division {division} not found for week {current_week}. Skipping.")
        return
    
    # Load the current week's DataFrames
    try:
        players_df = pd.read_csv(players_df_path)
        schedules_df = pd.read_csv(schedules_df_path)
        ranking_df_current = pd.read_csv(ranking_df_current_path)
    except Exception as e:
        logging.exception(f"Error loading data for Division {division}, Week {current_week}: {e}")
        return

    if current_week == 1:
        # For week 1, there is no previous week data
        # Identify players who have 'Games Played' == 1 in the current ranking_df
        players_played_this_week = ranking_df_current[ranking_df_current['Games Played'] == 1]['Name of Player']
        active_players = set(players_played_this_week)
    else:
        # For weeks after week 1, check if previous week's ranking data exists
        if not os.path.exists(ranking_df_previous_path):
            logging.warning(f"Ranking data for Division {division} not found for previous week {previous_week}. Skipping.")
            return
        ranking_df_previous = pd.read_csv(ranking_df_previous_path)

         # Identify which players played in the current week
        ranking_comparison = ranking_df_current.merge(
            ranking_df_previous[['Name of Player', 'Games Played']],
            on='Name of Player',
            how='left',
            suffixes=('_current', '_previous')
        )
        ranking_comparison['Games Played_previous'] = ranking_comparison['Games Played_previous'].fillna(0)
        ranking_comparison['Games Played_diff'] = ranking_comparison['Games Played_current'] - ranking_comparison['Games Played_previous']
        players_played_this_week = ranking_comparison[ranking_comparison['Games Played_diff'] == 1]['Name of Player']
        active_players = set(players_played_this_week)

    logging.info(f"Number of players who played in Division {division} during week {current_week}: {len(active_players)}")

    if not active_players:
        logging.warning(f"No players played in Division {division} during week {current_week}. Skipping.")
        return  

    # Create Results DataFrame
    # Drop unnecessary columns
    try:
        schedules_df.drop(columns=['vs', 'Time'], inplace=True, errors='ignore')  # Use errors='ignore' to prevent errors
    except Exception as e:
        logging.exception(f"Error dropping columns from schedules_df in Division {division}, Week {current_week}: {e}")
        return

    # Exclude rows where 'Away Team' is '[BYE]' (indicative of a bye week)
    results_df = schedules_df[schedules_df['Away Team'] != '[BYE]'].copy()

    # Replace NaN values in 'Result' with an empty string before applying str.contains
    results_df['Result'] = results_df['Result'].fillna('')

    # Keep rows where 'Result' contains brackets (indicative of a played match)
    results_df = results_df[results_df['Result'].str.contains(r'\(')]

    # Check if results_df is empty
    if results_df.empty:
        logging.info(f"No match results found for Division {division} during week {current_week}.")
        return

    # Apply the function to the 'Result' column
    try:
        results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(lambda x: pd.Series(parse_result(x)))
    except Exception as e:
        logging.exception(f"Error parsing results for Division {division}, Week {current_week}: {e}")
        return

    # Ensure 'Order' is an integer
    try:
        players_df['Order'] = players_df['Order'].astype(int)
    except ValueError as e:
        logging.exception(f"Error converting 'Order' column to int in Division {division}, Week {current_week}: {e}")
        return

    # Filter players_df to include only active players
    players_df_active = players_df[players_df['Player'].isin(active_players)].copy()

    # Check if players_df_active is empty
    if players_df_active.empty:
        logging.info(f"No active players found for Division {division} during week {current_week}.")
        return

    # Create a dictionary for each team mapping Order to Player
    team_players = {}

    for team in players_df_active['Team'].unique():
        team_data = players_df_active[players_df_active['Team'] == team]
        order_to_player = dict(zip(team_data['Order'], team_data['Player']))
        team_players[team] = order_to_player
    
    # Create mapping from 'Rubber Number' to 'Order' for each team
    team_rubber_to_order = {}

    for team in team_players:
        orders = sorted(team_players[team].keys())
        team_rubber_to_order[team] = {}
        for idx, order in enumerate(orders):
            rubber_number = idx + 1  # Because idx starts from 0
            team_rubber_to_order[team][rubber_number] = order

    # Function to get player name based on team and order
    def get_player_name(team, rubber_number):
        """
        Get the player name based on team order and rubber order.
        """
        # Get the Order corresponding to the rubber number for the team
        try:
            order = team_rubber_to_order.get(team, {}).get(rubber_number)
            if order is None:
                return 'Unknown'
            return team_players.get(team, {}).get(order, 'Unknown')
        except Exception as e:
            logging.exception(f"Error getting player name for team '{team}', rubber {rubber_number}: {e}")
            return 'Unknown'
    
    
    # Find the maximum number of rubbers in any match
    max_rubbers = results_df['Result'].apply(lambda x: len(parse_result(x)[1])).max()

    # Assign players to each rubber
    for i in range(1, max_rubbers + 1):
        results_df[f'Home Player {i}'] = results_df.apply(
            lambda row: get_player_name(row['Home Team'], i), axis=1
        )
        results_df[f'Away Player {i}'] = results_df.apply(
            lambda row: get_player_name(row['Away Team'], i), axis=1
        )
        
    # Process results_df to include rubber details
    try:
        results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(
            lambda x: pd.Series(parse_result(x))
        )
    except Exception as e:
        logging.exception(f"Error parsing 'Result' column in Division {division}, Week {current_week}: {e}")
        return
    
    # Ensure 'Rubbers' is a list
    results_df['Rubbers'] = results_df['Rubbers'].apply(lambda x: x if isinstance(x, list) else [x])

    # Split the 'Rubbers' list into separate columns
    try:
        for i in range(1, max_rubbers + 1):
            results_df[f'Rubber {i} Score'] = results_df['Rubbers'].apply(
                lambda rubbers: rubbers[i - 1] if i - 1 < len(rubbers) else None
            )
    except Exception as e:
        logging.exception(f"Error extracting 'Rubber {i} Score' in Division {division}, Week {current_week}: {e}")
        return

    # Generate player match results
    player_match_results = []
    for idx, row in results_df.iterrows():
        try:
            match_date = row['Date']
            venue = row['Venue']
            home_team = row['Home Team']
            away_team = row['Away Team']
            
            for i in range(1, max_rubbers + 1):
                home_player = row[f'Home Player {i}']
                away_player = row[f'Away Player {i}']
                rubber_score = row[f'Rubber {i} Score']
                
                if pd.isna(rubber_score) or home_player == 'Unknown' or away_player == 'Unknown':
                    continue
                
                winner = determine_winner(rubber_score, home_player, away_player)
                if winner is None:
                    result_home = 'Unknown'
                    result_away = 'Unknown'
                else:
                    if winner == home_player:
                        result_home = 'Win'
                        result_away = 'Loss'
                    else:
                        result_home = 'Loss'
                        result_away = 'Win'

                # Adjust the score so that the player's own score is first
                try:
                    home_score, away_score = map(int, rubber_score.split('-'))
                    # For home player
                    own_score_home = home_score
                    opponent_score_home = away_score
                    score_home = f"{own_score_home}-{opponent_score_home}"
                    # For away player
                    own_score_away = away_score
                    opponent_score_away = home_score
                    score_away = f"{own_score_away}-{opponent_score_away}"
                except ValueError:
                    # If unable to parse the score, use the rubber_score as is
                    score_home = rubber_score
                    score_away = rubber_score
                
                # Append home player's result
                player_match_results.append({
                    'Player Name': home_player,
                    'Team': home_team,
                    'Opponent Name': away_player,
                    'Opponent Team': away_team,
                    'Match Date': match_date,
                    'Venue': venue,
                    'Rubber Number': i,
                    'Score': score_home,
                    'Result': result_home,
                    'Home/Away': 'Home'
                })
                
                # Append away player's result
                player_match_results.append({
                    'Player Name': away_player,
                    'Team': away_team,
                    'Opponent Name': home_player,
                    'Opponent Team': home_team,
                    'Match Date': match_date,
                    'Venue': venue,
                    'Rubber Number': i,
                    'Score': score_away,
                    'Result': result_away,
                    'Home/Away': 'Away'
                })
        except Exception as e:
            logging.exception(f"Error processing match at index {idx} in Division {division}, Week {current_week}: {e}")
            continue  # Skip to the next match

    player_results_df = pd.DataFrame(player_match_results)

    # Clean and format the DataFrame
    try:
        player_results_df['Match Date'] = pd.to_datetime(player_results_df['Match Date'], 
                                                         dayfirst=True, 
                                                         errors='coerce')
    except Exception as e:
        logging.exception(f"Error converting 'Match Date' to datetime in Division {division}, Week {current_week}: {e}")
        return

    player_results_df = player_results_df[[
        'Player Name', 'Team', 'Opponent Name', 'Opponent Team', 'Match Date',
        'Venue', 'Rubber Number', 'Score', 'Result', 'Home/Away'
    ]]
    player_results_df['Match Date'] = player_results_df['Match Date'].fillna(pd.NaT)

    # Save the player_results_df
    output_path = os.path.join(base_directory, "player_results", f"week_{current_week}", f"{division}_player_results.csv")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        player_results_df.to_csv(output_path, index=False)
        logging.info(f"Player match results saved to {output_path}")
    except Exception as e:
        logging.exception(f"Error saving player results for Division {division}, Week {current_week}: {e}")
        return

# Run the script
week_numbers = [1, 2, 3]  # Adjust as needed

for current_week in week_numbers:
    previous_week = current_week - 1
    for division in divisions_to_process:
        logging.info(f"Processing Division {division} for Week {current_week}")
        try:
            process_division(division, current_week, previous_week)
        except Exception as e:
            logging.exception(f"Unexpected error processing Division {division}, Week {current_week}: {e}")
            continue  # Proceed to the next division
