import pandas as pd
import numpy as np
import os
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

def build_player_mapping(all_divisions, base_directory):
    """
    Build a mapping of player names to their divisions and teams across all divisions.

    Args:
        all_divisions (dict): Dictionary of division names and their IDs.
        base_directory (str): Base directory path where player_df files are stored.

    Returns:
        dict: A dictionary mapping player names to their division and team.
    """
    player_mapping = {}
    for division in all_divisions.keys():
        players_df_path = os.path.join(base_directory, "players_df", "week_1", f"{division}_players_df.csv")  # Assuming week 1 has all players
        if not os.path.exists(players_df_path):
            logging.warning(f"Players file for Division {division} not found at {players_df_path}. Skipping.")
            continue
        try:
            players_df = pd.read_csv(players_df_path)
            for _, row in players_df.iterrows():
                player_name = row['Player']
                team = row['Team']
                player_mapping[player_name] = {
                    'Division': division,
                    'Team': team,
                    'Order': row['Order']
                }
        except Exception as e:
            logging.exception(f"Error loading players for Division {division}: {e}")
            continue
    return player_mapping


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
    

def determine_winner(rubber_score, home_team, away_team):
    """
    Determine the winner of a rubber based on the score.

    Args:
        rubber_score (str): Score string, e.g., '3-1', 'CR', 'WO'.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.

    Returns:
        tuple: (winner_team, conceded_team)
            winner_team: 'Home', 'Away', or 'Draw'
            conceded_team: 'Home', 'Away', or None
    """
    if rubber_score.upper() in ['CR', 'WO']:
        # Without knowing which team conceded, we return 'Undecided' for now
        return 'Undecided', None
    try:
        home_score, away_score = map(int, rubber_score.split('-'))
        if home_score > away_score:
            return 'Home', None
        elif away_score > home_score:
            return 'Away', None
        else:
            return 'Draw', None  # Draws are rare in squash but handled here
    except ValueError as e:
        logging.warning(f"Invalid score format '{rubber_score}' between '{home_team}' and '{away_team}': {e}")
        return 'Unknown', None

    

def process_division(division, current_week, previous_week, player_mapping, all_divisions, base_directory):
    """
    Process a division for a given week, handling 'CR', 'WO', 'Playing Up' players, and missing data.

    Args:
        division (str): Division name.
        current_week (int): Current week number.
        previous_week (int): Previous week number.
        player_mapping (dict): Global player mapping.
        all_divisions (dict): Dictionary of division names and their IDs.
        base_directory (str): Base directory path where data files are stored.
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
        logging.warning(f"Data files for Division '{division}' not found for week {current_week}. Skipping.")
        return
    
    # Load the current week's DataFrames
    try:
        players_df = pd.read_csv(players_df_path)
        schedules_df = pd.read_csv(schedules_df_path)
        ranking_df_current = pd.read_csv(ranking_df_current_path)
    except Exception as e:
        logging.exception(f"Error loading data for Division '{division}', Week {current_week}: {e}")
        return
    
    # Filter 'schedules_df' to only include matches from 'current_week'
    if 'Match Week' in schedules_df.columns:
        # Ensure 'Match Week' is of integer type
        schedules_df['Match Week'] = schedules_df['Match Week'].astype(int)
        # Filter matches for the current week
        schedules_df = schedules_df[schedules_df['Match Week'] == current_week]
    else:
        logging.warning(f"'Match Week' column not found in schedules_df for Division '{division}'. Cannot filter by week.")
        return

    # Determine active players based on 'Games Played'
    if current_week == 1:
        # For week 1, players with 'Games Played' == 1 are active
        players_played_this_week = ranking_df_current[ranking_df_current['Games Played'] == 1]['Name of Player']
        active_players = set(players_played_this_week)
    else:
        # For weeks after week 1, compare with previous week's 'Games Played'
        if not os.path.exists(ranking_df_previous_path):
            logging.warning(f"Ranking data for Division '{division}' not found for previous week {previous_week}. Skipping.")
            return
        try:
            ranking_df_previous = pd.read_csv(ranking_df_previous_path)
        except Exception as e:
            logging.exception(f"Error loading previous ranking data for Division '{division}', Week {previous_week}: {e}")
            return

        # Merge current and previous rankings to find players who played this week
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

    logging.info(f"Number of players who played in Division '{division}' during week {current_week}: {len(active_players)}")

    if not active_players:
        logging.warning(f"No players played in Division '{division}' during week {current_week}. Skipping.")
        return  

    # Create Results DataFrame
    try:
        # Drop unnecessary columns
        schedules_df.drop(columns=['vs', 'Time'], inplace=True, errors='ignore')
    except Exception as e:
        logging.exception(f"Error dropping columns from schedules_df in Division '{division}', Week {current_week}: {e}")
        return

    # Exclude rows where 'Away Team' is '[BYE]' (indicative of a bye week)
    results_df = schedules_df[schedules_df['Away Team'] != '[BYE]'].copy()

    # Replace NaN values in 'Result' with an empty string before applying str.contains
    results_df['Result'] = results_df['Result'].fillna('')

    # Keep rows where 'Result' contains brackets (indicative of a played match)
    results_df = results_df[results_df['Result'].str.contains(r'\(')]

    # Check if results_df is empty
    if results_df.empty:
        logging.info(f"No match results found for Division '{division}' during week {current_week}.")
        return

    # Replace string 'nan' with actual NaN
    results_df['Result'] = results_df['Result'].replace(to_replace=['nan', 'NaN', 'NAN'], value=np.nan)

    # Now, fill NaN with 'Unknown'
    results_df['Result'] = results_df['Result'].fillna('Unknown')

    # Ensure 'Result' is string type
    results_df['Result'] = results_df['Result'].astype(str)

    # Apply the parse_result function to split 'Result' into 'Overall Score' and 'Rubbers'
    try:
        results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(lambda x: pd.Series(parse_result(x)))
    except Exception as e:
        logging.exception(f"Error parsing results for Division '{division}', Week {current_week}: {e}")
        return

    # Determine maximum number of rubbers in any match
    max_rubbers = results_df['Rubbers'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

    # Assign players to each rubber based on 'Order'
    # For each team, get the list of active players and sort them by 'Order'
    team_players = {}
    for team in players_df['Team'].unique():
        # Get active players for the team
        team_data = players_df[(players_df['Team'] == team) & (players_df['Player'].isin(active_players))]
        # Sort active players by 'Order'
        sorted_players = team_data.sort_values('Order')['Player'].tolist()
        team_players[team] = sorted_players

    # Identify 'playing up' players (active_players not in current division's players_df)
    playing_up_players = active_players - set(players_df['Player'])
    logging.info(f"Number of 'Playing Up' players in Division '{division}' during week {current_week}: {len(playing_up_players)}")

    # Ensure 'playing up' players are included in players_df with an 'Order' higher than existing players
    for player in playing_up_players:
        # Get the team the 'playing up' player is playing for
        team = ranking_df_current.loc[ranking_df_current['Name of Player'] == player, 'Team'].values
        if len(team) > 0:
            team = team[0]
        else:
            continue  # Skip if team is not found

        # Find the maximum 'Order' in the team
        team_orders = players_df[players_df['Team'] == team]['Order']
        if not team_orders.empty:
            max_order = team_orders.max()
        else:
            max_order = 0  # If team has no players, start from 0

        # Assign 'Order' as max_order + 1
        order = max_order + 1

        # Check if the player is already in players_df for this team
        if not ((players_df['Player'] == player) & (players_df['Team'] == team)).any():
            # Add the player to players_df
            new_row = {'Player': player, 'Team': team, 'Order': order}
            players_df = pd.concat([players_df, pd.DataFrame([new_row])], ignore_index=True)

    # Prepare active players list for each team, including 'playing up' players
    team_active_players = {}
    for team in players_df['Team'].unique():
        # Get all players for the team
        team_players = players_df[players_df['Team'] == team]['Player'].tolist()
        # Filter active players
        active_team_players = [player for player in team_players if player in active_players]
        team_active_players[team] = active_team_players

    # Process results_df to include rubber details
    try:
        results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(
            lambda x: pd.Series(parse_result(x))
        )
    except Exception as e:
        logging.exception(f"Error parsing 'Result' column in Division '{division}', Week {current_week}: {e}")
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
        logging.exception(f"Error extracting 'Rubber {i} Score' in Division '{division}', Week {current_week}: {e}")
        return

    # Process each match and assign players to rubbers
    player_match_results = []
    for idx, row in results_df.iterrows():
        try:
            match_date = row['Date']
            venue = row['Venue']
            home_team = row['Home Team']
            away_team = row['Away Team']
            overall_score = row['Overall Score']
            rubbers = row['Rubbers']
            max_rubbers = len(rubbers)
            
            # Initialize rubber results
            rubber_results = []
            home_rubbers_won = 0
            away_rubbers_won = 0
            undecided_rubbers = []
            
            # First pass: Determine winners of rubbers without 'WO' or 'CR'
            for i in range(1, max_rubbers + 1):
                rubber_score = row.get(f'Rubber {i} Score', None)
                winner_team, conceded_team = determine_winner(rubber_score, home_team, away_team)
                rubber_results.append({
                    'Rubber Number': i,
                    'Rubber Score': rubber_score,
                    'Winner Team': winner_team,
                    'Conceded Team': conceded_team
                })
                if winner_team == 'Home':
                    home_rubbers_won += 1
                elif winner_team == 'Away':
                    away_rubbers_won += 1
                elif winner_team == 'Undecided':
                    undecided_rubbers.append(i)
            
            # Second pass: Resolve 'Undecided' rubbers using overall match score
            try:
                home_match_points, away_match_points = map(int, overall_score.split('-'))
            except Exception as e:
                logging.exception(f"Error parsing 'Overall Score' in match index {idx}: {e}")
                continue  # Skip to the next match
            
            rubbers_needed_by_home = home_match_points - home_rubbers_won
            rubbers_needed_by_away = away_match_points - away_rubbers_won
            
            for rubber in rubber_results:
                if rubber['Winner Team'] == 'Undecided':
                    if rubbers_needed_by_home > 0:
                        rubber['Winner Team'] = 'Home'
                        rubber['Conceded Team'] = 'Away'
                        home_rubbers_won += 1
                        rubbers_needed_by_home -= 1
                    elif rubbers_needed_by_away > 0:
                        rubber['Winner Team'] = 'Away'
                        rubber['Conceded Team'] = 'Home'
                        away_rubbers_won += 1
                        rubbers_needed_by_away -= 1
                    else:
                        rubber['Winner Team'] = 'Unknown'
                        rubber['Conceded Team'] = None
            
            # Get active players for each team, including 'playing up' players
            home_team_players = team_active_players.get(home_team, [])
            away_team_players = team_active_players.get(away_team, [])

            # Sort active players by 'Order'
            home_team_data = players_df[(players_df['Team'] == home_team) & (players_df['Player'].isin(home_team_players))]
            home_team_players_sorted = home_team_data.sort_values('Order')['Player'].tolist()

            away_team_data = players_df[(players_df['Team'] == away_team) & (players_df['Player'].isin(away_team_players))]
            away_team_players_sorted = away_team_data.sort_values('Order')['Player'].tolist()
                        
            # Assign players to rubbers, inserting 'Unknown' where teams have conceded
            # Initialize player indexes
            home_player_idx = 0
            away_player_idx = 0

            # Lists to hold final players assigned to rubbers
            home_players_assigned = []
            away_players_assigned = []

            for rubber in rubber_results:
                winner_team = rubber['Winner Team']
                rubber_score = rubber['Rubber Score']
                
                # Determine if either team conceded
                home_conceded = False
                away_conceded = False
                if rubber_score and rubber_score.upper() in ['CR', 'WO']:
                    if winner_team == 'Home':
                        away_conceded = True  # Away team conceded
                    elif winner_team == 'Away':
                        home_conceded = True  # Home team conceded
                    else:
                        # If winner is 'Unknown' or 'Draw', we cannot determine who conceded
                        pass

                # Assign home player
                if home_conceded:
                    home_players_assigned.append('Unknown')
                else:
                    if home_player_idx < len(home_team_players_sorted):
                        home_players_assigned.append(home_team_players_sorted[home_player_idx])
                        home_player_idx += 1
                    else:
                        home_players_assigned.append('Unknown')

                # Assign away player
                if away_conceded:
                    away_players_assigned.append('Unknown')
                else:
                    if away_player_idx < len(away_team_players_sorted):
                        away_players_assigned.append(away_team_players_sorted[away_player_idx])
                        away_player_idx += 1
                    else:
                        away_players_assigned.append('Unknown')

            # Now, generate player match results
            for idx_rubber, rubber in enumerate(rubber_results):
                i = rubber['Rubber Number']
                rubber_score = rubber['Rubber Score']
                winner_team = rubber['Winner Team']
                home_player = home_players_assigned[idx_rubber]
                away_player = away_players_assigned[idx_rubber]

                if winner_team == 'Home':
                    result_home = 'Win'
                    result_away = 'Loss'
                elif winner_team == 'Away':
                    result_home = 'Loss'
                    result_away = 'Win'
                elif winner_team == 'Draw':
                    result_home = 'Draw'
                    result_away = 'Draw'
                else:
                    result_home = 'Unknown'
                    result_away = 'Unknown'

                # Adjust scores for 'WO' or 'CR' rubbers
                if rubber_score and rubber_score.upper() in ['CR', 'WO']:
                    score_home = rubber_score
                    score_away = rubber_score
                    if winner_team == 'Home':
                        away_player = 'Unknown'  # Away team conceded
                    elif winner_team == 'Away':
                        home_player = 'Unknown'  # Home team conceded
                else:
                    try:
                        home_score, away_score = map(int, rubber_score.split('-'))
                        score_home = f"{home_score}-{away_score}"
                        score_away = f"{away_score}-{home_score}"
                    except (ValueError, TypeError):
                        score_home = rubber_score if rubber_score else 'Unknown'
                        score_away = rubber_score if rubber_score else 'Unknown'

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
            logging.exception(f"Error processing match at index {idx} in Division '{division}', Week {current_week}: {e}")
            continue  # Skip to the next match


    # Convert the results to a DataFrame
    player_results_df = pd.DataFrame(player_match_results)

    # Reorder columns and handle missing dates
    player_results_df = player_results_df[[
        'Player Name', 'Team', 'Opponent Name', 'Opponent Team', 'Match Date',
        'Venue', 'Rubber Number', 'Score', 'Result', 'Home/Away'
    ]]
    player_results_df['Match Date'] = player_results_df['Match Date'].fillna(pd.NaT)

    # Save the player_results_df
    output_path = os.path.join(
        base_directory,
        "player_results",
        f"week_{current_week}",
        f"{division}_player_results.csv"
    )
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) 
        player_results_df.to_csv(output_path, index=False)
        logging.info(f"Player match results saved to {output_path}")
    except Exception as e:
        logging.exception(f"Error saving player results for Division '{division}', Week {current_week}: {e}")
        return

    
# Build the global player mapping
player_mapping = build_player_mapping(all_divisions, base_directory)
logging.info(f"Total players mapped across all divisions: {len(player_mapping)}")

# Get current week
current_week = 16

# Run the script for each division and week
for week in range(1, current_week + 1):
    previous_week = week - 1
    for division in all_divisions.keys():
        logging.info(f"Processing Division '{division}' for Week {week}")
        try:
            process_division(
                division=division,
                current_week=week,
                previous_week=previous_week,
                player_mapping=player_mapping,
                all_divisions=all_divisions,
                base_directory=base_directory
            )
        except Exception as e:
            logging.exception(f"Unexpected error processing Division '{division}', Week {week}: {e}")
            continue  # Proceed to the next division