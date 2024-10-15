import pandas as pd
import os
import re
import glob
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("squash_app_debug.log"),  # Log to a file named 'squash_app_debug.log'
        logging.StreamHandler()  # Also output logs to the console
    ]
)

def parse_result(result):
    """
    Function to parse the 'result' string
    """
    overall, rubbers = result.split('(')
    rubbers = rubbers.strip(')').split(',')
    return overall, rubbers


def update_rubbers(row, rubbers_won, rubbers_conceded):
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


def update_counts(row, cr_given_count, cr_received_count, wo_given_count, wo_received_count):
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



def get_divisions_for_season(season_path):
    """
    Function to get the divisions for a given season.
    """
    teams_df_path = os.path.join(season_path, 'teams_df')
    division_files = [f for f in os.listdir(teams_df_path) if f.endswith('_teams_df.csv')]
    divisions = []

    for filename in division_files:
        match = re.match(r'(.*)_teams_df\.csv', filename)
        if match:
            division = match.group(1)
            divisions.append(division)
        else:
            logging.warning(f"Filename {filename} does not match expected pattern.")
    
    return divisions


def generate_detailed_league_tables(season_path):
    """
    Function to generate detailed league tables based on schedules and match results.
    """
    logging.info(f"Starting to generate detailed league tables for season at {season_path}.")

    # Get divisions for the season
    divisions = get_divisions_for_season(season_path)
    
    for division in divisions:
        # Load the summary_df for each division
        summary_file = os.path.join(season_path, 'summary_df', f'{division}_summary_df.csv')
        if not os.path.exists(summary_file):
            logging.warning(f'Summary file not found for division {division} at {season_path}. Skipping.')
            continue

        summary_df = pd.read_csv(summary_file)
        logging.debug(f"Loaded summary_df for division {division} with shape {summary_df.shape}")

        # Load the schedules_df for the division
        schedules_file = os.path.join(season_path, 'schedules_df', f'{division}_schedules_df.csv')
        if not os.path.exists(schedules_file):
            logging.warning(f'Schedules file not found for division {division} at {season_path}. Skipping.')
            continue

        schedules_df = pd.read_csv(schedules_file)
        logging.debug(f"Loaded schedules_df for division {division} with shape {schedules_df.shape}")

        # Create results_df from schedules_df, excluding BYE weeks
        logging.info("Creating results_df by excluding BYE weeks.")
        results_df = schedules_df[schedules_df['Away Team'] != '[BYE]'].copy()

        # Replace NaN values in 'Result' with an empty string before applying str.contains
        logging.info("Replacing NaN values in 'Result' with an empty string.")
        results_df['Result'] = results_df['Result'].fillna('')

        # Keep rows where 'Result' contains brackets (indicative of a played match)
        logging.info("Filtering rows where 'Result' contains brackets indicating a played match.")
        results_df = results_df[results_df['Result'].str.contains(r'\(')]

        # Apply the parse_result function to the 'Result' column
        logging.info("Applying parse_result function to extract 'Overall Score' and 'Rubbers'.")
        results_df[['Overall Score', 'Rubbers']] = results_df['Result'].apply(lambda x: pd.Series(parse_result(x)))

        # Drop the original 'Result' column
        logging.info("Dropping the original 'Result' column.")
        results_df.drop(columns=['Result'], inplace=True)

        # Replace 'CR' and 'WO' with NaN values
        logging.info("Replacing 'CR' and 'WO' values with NaN.")
        results_df.replace('CR', pd.NA, inplace=True)
        results_df.replace('WO', pd.NA, inplace=True)

        # Split the "Overall Score" column into two separate columns (Home Score, Away Score)
        logging.info("Splitting the 'Overall Score' into 'Home Score' and 'Away Score' and converting to integers.")
        results_df[['Home Score', 'Away Score']] = results_df['Overall Score'].str.split('-', expand=True).astype(int)

        # Initialize rubbers_won and rubbers_conceded dictionaries
        logging.info("Initializing dictionaries for rubbers won and conceded.")
        rubbers_won = {}
        rubbers_conceded = {}

        # Apply the update_rubbers function to the results_df dataframe
        logging.info("Applying the update_rubbers function to the results_df.")
        results_df.apply(update_rubbers, axis=1, args=(rubbers_won, rubbers_conceded))

        # Convert the dictionaries to DataFrames
        logging.info("Converting rubbers_won and rubbers_conceded dictionaries to DataFrames.")
        df_rubbers_won = pd.DataFrame(list(rubbers_won.items()), columns=['Team', 'Rubbers For'])
        df_rubbers_conceded = pd.DataFrame(list(rubbers_conceded.items()), columns=['Team', 'Rubbers Against'])

        # Merge the DataFrames on "Team"
        logging.info("Merging rubbers_won and rubbers_conceded DataFrames on 'Team'.")
        rubbers_df = pd.merge(df_rubbers_won, df_rubbers_conceded, on='Team')

        # Initialize dictionaries to keep track of conceded rubbers and walkovers
        logging.info("Initializing dictionaries for CRs and WOs given and received.")
        cr_given_count = {}
        cr_received_count = {}
        wo_given_count = {}
        wo_received_count = {}

        # Apply the update_counts function to each row
        logging.info("Applying the update_counts function to the results_df.")
        results_df.apply(update_counts, axis=1, args=(cr_given_count, cr_received_count, wo_given_count, wo_received_count))

        # Ensure all teams are included in all counts
        logging.info("Ensuring all teams are included in CR and WO counts.")
        all_teams = set(results_df['Home Team']).union(set(results_df['Away Team']))
        for team in all_teams:
            cr_given_count.setdefault(team, 0)
            cr_received_count.setdefault(team, 0)
            wo_given_count.setdefault(team, 0)
            wo_received_count.setdefault(team, 0)

        # Convert the dictionaries to DataFrames
        logging.info("Converting CR and WO dictionaries to DataFrames.")
        df_cr_given_count = pd.DataFrame(list(cr_given_count.items()), columns=['Team', 'CRs Given'])
        df_cr_received_count = pd.DataFrame(list(cr_received_count.items()), columns=['Team', 'CRs Received'])
        df_wo_given_count = pd.DataFrame(list(wo_given_count.items()), columns=['Team', 'WOs Given'])
        df_wo_received_count = pd.DataFrame(list(wo_received_count.items()), columns=['Team', 'WOs Received'])

        # Merge the DataFrames on "Team" to create detailed_table_df
        logging.info("Merging CR and WO DataFrames with rubbers_df on 'Team'.")
        detailed_table_df = pd.merge(df_cr_given_count, df_cr_received_count, on='Team')
        detailed_table_df = pd.merge(detailed_table_df, df_wo_given_count, on='Team')
        detailed_table_df = pd.merge(detailed_table_df, df_wo_received_count, on='Team')
        detailed_table_df = pd.merge(detailed_table_df, rubbers_df, on="Team")

        # Merge the summary_df with the detailed_table_df
        logging.info("Merging the summary_df with the detailed_table_df.")
        final_detailed_table_df = pd.merge(summary_df, detailed_table_df, on="Team")

        logging.info(f"Finished generating detailed league table for division {division}.")

    logging.info("Finished generating detailed league tables for the entire season.")



def generate_played_every_game(season_path):
    """
    Function to generate played-every-game data for each division in a season.
    """
    logging.info(f"Generating played-every-game data for season at {season_path}")
    divisions = get_divisions_for_season(season_path)
    players_df_path = os.path.join(season_path, 'players_df')
    schedules_df_path = os.path.join(season_path, 'schedules_df')
    played_every_game_path = os.path.join(season_path, 'played_every_game')

    os.makedirs(played_every_game_path, exist_ok=True)

    for division in divisions:
        players_file = os.path.join(players_df_path, f'{division}_players_df.csv')
        schedule_file = os.path.join(schedules_df_path, f'{division}_schedules_df.csv')
        if not os.path.exists(players_file) or not os.path.exists(schedule_file):
            logging.warning(f'Players or schedule file not found for division {division}')
            continue

        players_df = pd.read_csv(players_file)
        schedule_df = pd.read_csv(schedule_file)

        team_games = schedule_df.groupby('Team').size().reset_index(name='Total Games')
        player_games = players_df.groupby(['Team', 'Player']).size().reset_index(name='Games Played')

        player_games = pd.merge(player_games, team_games, on='Team', how='left')
        players_every_game = player_games[player_games['Games Played'] == player_games['Total Games']]

        player_list = players_every_game['Player'].tolist()
        output_file = os.path.join(played_every_game_path, f'{division}.txt')
        with open(output_file, 'w') as f:
            for player in player_list:
                f.write(f"{player}\n")
        logging.info(f"Saved played-every-game data for division {division} to {output_file}")


def generate_summarized_player_tables(season_path):
    """
    Function to generate summarized player tables for each division in a season.
    """
    logging.info(f"Generating summarized player tables for season at {season_path}")
    divisions = get_divisions_for_season(season_path)
    ranking_df_path = os.path.join(season_path, 'ranking_df')
    summarized_player_tables_path = os.path.join(season_path, 'summarized_player_tables')

    os.makedirs(summarized_player_tables_path, exist_ok=True)

    for division in divisions:
        ranking_file = os.path.join(ranking_df_path, f'{division}_ranking_df.csv')
        if not os.path.exists(ranking_file):
            logging.warning(f'Ranking file not found for division {division}')
            continue

        ranking_df = pd.read_csv(ranking_file)
        logging.debug(f"Loaded ranking data for division {division} with shape {ranking_df.shape}")

        # Calculate summary statistics
        player_stats = ranking_df.groupby('Player').agg({
            'Games Played': 'sum',
            'Won': 'sum',
            'Lost': 'sum',
            'Total Game Points': 'sum',
        }).reset_index()

        player_stats['Win Percentage'] = player_stats['Won'] / player_stats['Games Played'] * 100
        player_stats['Average Points'] = player_stats['Total Game Points'] / player_stats['Games Played']

        output_file = os.path.join(summarized_player_tables_path, f'{division}_summarized_players.csv')
        player_stats.to_csv(output_file, index=False)
        logging.info(f"Saved summarized player table for division {division} to {output_file}")



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


def calculate_win_percentage_by_location(results_df, aggregate_wins, max_rubbers, location_col, save_path, div, location_type):
    """
    Function to calculate win percentages for each team based on location (home or away).
    """
    logging.info(f"Calculating {location_type} win percentage for division {div}")
    results_list = []
    valid_matches_df = results_df.dropna(subset=['Rubbers'])

    if valid_matches_df.empty:
        logging.warning(f"No valid matches for {location_type} win percentage calculation in division {div}")
        return pd.DataFrame(columns=['Team'] + [f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)] + ['avg_win_perc', 'Total Rubbers'])

    for team in valid_matches_df[location_col].unique():
        logging.debug(f"Processing team {team} for {location_type} matches")
        team_fixtures = valid_matches_df[valid_matches_df[location_col] == team]

        total_matches_per_rubber = {f'Rubber {i}': count_valid_matches(team_fixtures, i - 1) for i in range(1, max_rubbers + 1)}
        total_matches_df = pd.DataFrame([total_matches_per_rubber], index=[team])

        total_rubbers_played = total_matches_df.sum(axis=1)
        team_combined = aggregate_wins.loc[[team]].merge(total_matches_df, left_index=True, right_index=True, how='outer').fillna(0)

        for i in range(1, max_rubbers + 1):
            rubber_column = f'Rubber {i}'
            team_combined[f'{rubber_column} Win %'] = (team_combined[f'Wins in {rubber_column}'] / team_combined[rubber_column]) * 100

        team_combined.fillna(0, inplace=True)
        team_combined["avg_win_perc"] = team_combined[[f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)]].mean(axis=1)
        team_combined["Total Rubbers"] = total_rubbers_played

        results_list.append(team_combined)

    results_df = pd.concat(results_list).reset_index().rename(columns={'index': 'Team'})
    output_file = os.path.join(save_path, f"{div}_team_win_percentage_breakdown_{location_type}.csv")
    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved {location_type} win percentage data for division {div} to {output_file}")
    return results_df


def calculate_delta_win_percentage(home_df, away_df, max_rubbers):
    """
    Function to calculate the delta in win percentages between home and away matches.
    """
    logging.info("Calculating delta win percentage between home and away matches")
    merged_df = home_df.merge(away_df, on='Team', suffixes=('_home', '_away'))

    delta_data = {'Team': merged_df['Team']}
    for i in range(1, max_rubbers + 1):
        rubber_column = f'Rubber {i} Win %'
        delta_data[rubber_column] = merged_df[f'{rubber_column}_home'] - merged_df[f'{rubber_column}_away']

    delta_data['avg_win_perc'] = merged_df['avg_win_perc_home'] - merged_df['avg_win_perc_away']

    delta_df = pd.DataFrame(delta_data).sort_values('avg_win_perc', ascending=False)
    logging.debug(f"Calculated delta win percentage with shape {delta_df.shape}")
    return delta_df


def calculate_overall_win_percentage(results_df, aggregate_wins, max_rubbers):
    """
    Function to calculate the overall win percentage for each team.
    """
    logging.info("Calculating overall win percentage for each team")
    total_matches_per_rubber = {f'Rubber {i}': count_valid_matches(results_df, i - 1) for i in range(1, max_rubbers + 1)}
    total_matches_df = pd.DataFrame(total_matches_per_rubber)

    combined = aggregate_wins.merge(total_matches_df, left_index=True, right_index=True, how='outer').fillna(0)

    for i in range(1, max_rubbers + 1):
        rubber_column = f'Rubber {i}'
        combined[f'{rubber_column} Win %'] = (combined[f'Wins in {rubber_column}'] / combined[rubber_column]) * 100

    combined.fillna(0, inplace=True)
    combined["avg_win_perc"] = combined[[f'Rubber {i} Win %' for i in range(1, max_rubbers + 1)]].mean(axis=1)
    combined["Total Rubbers"] = total_matches_df.sum(axis=1)

    overall_df = combined.reset_index().rename(columns={'index': 'Team'}).sort_values("avg_win_perc", ascending=False)
    logging.debug(f"Overall win percentage DataFrame shape: {overall_df.shape}")
    return overall_df


def generate_team_win_percentage_breakdown(season_path):
    """
    Function to generate team win percentage breakdowns (overall, home, away, and delta)
    for a given season.
    """
    logging.info(f"Generating team win percentage breakdown for season at {season_path}")
    base_directories = {
        'team_win_percentage_breakdown_home': os.path.join(season_path, 'team_win_percentage_breakdown', 'Home'),
        'team_win_percentage_breakdown_away': os.path.join(season_path, 'team_win_percentage_breakdown', 'Away'),
        'team_win_percentage_breakdown_delta': os.path.join(season_path, 'team_win_percentage_breakdown', 'Delta'),
        'team_win_percentage_breakdown_overall': os.path.join(season_path, 'team_win_percentage_breakdown', 'Overall')
    }

    for dir_path in base_directories.values():
        os.makedirs(dir_path, exist_ok=True)

    divisions = get_divisions_for_season(season_path)

    for div in divisions:
        logging.info(f"Processing division {div} for win percentage breakdown")
        results_file = os.path.join(season_path, 'results_df', f"{div}_results_df.csv")
        if not os.path.exists(results_file):
            logging.warning(f"Results file not found for division {div}, skipping...")
            continue
        
        results_df = pd.read_csv(results_file)
        max_rubbers = results_df['Rubbers'].apply(len).max()

        for i in range(1, max_rubbers + 1):
            rubber_column = f'Rubber {i}'
            results_df[f'Winner {rubber_column}'] = results_df.apply(
                lambda row: determine_winner(row['Rubbers'][i - 1] if i <= len(row['Rubbers']) else pd.NA,
                                             row['Home Team'], row['Away Team']), axis=1)

        aggregate_wins = pd.DataFrame()
        for i in range(1, max_rubbers + 1):
            rubber_column = f'Rubber {i}'
            winner_column = f'Winner {rubber_column}'
            wins = results_df[winner_column].value_counts().rename(f'Wins in {rubber_column}')
            aggregate_wins = pd.concat([aggregate_wins, wins], axis=1)

        aggregate_wins.fillna(0, inplace=True)
        aggregate_wins = aggregate_wins.astype(int)

        win_percentage_home_df = calculate_win_percentage_by_location(
            results_df, aggregate_wins, max_rubbers, 'Home Team', base_directories['team_win_percentage_breakdown_home'], div, 'home'
        )

        win_percentage_away_df = calculate_win_percentage_by_location(
            results_df, aggregate_wins, max_rubbers, 'Away Team', base_directories['team_win_percentage_breakdown_away'], div, 'away'
        )

        win_percentage_delta_df = calculate_delta_win_percentage(win_percentage_home_df, win_percentage_away_df, max_rubbers)
        win_percentage_delta_df.to_csv(
            os.path.join(base_directories['team_win_percentage_breakdown_delta'], f"{div}_team_win_percentage_breakdown_delta.csv"),
            index=False
        )
        logging.info(f"Saved win percentage delta data for division {div}")

        win_percentage_overall_df = calculate_overall_win_percentage(results_df, aggregate_wins, max_rubbers)
        win_percentage_overall_df.to_csv(
            os.path.join(base_directories['team_win_percentage_breakdown_overall'], f"{div}_team_win_percentage_breakdown.csv"),
            index=False
        )
        logging.info(f"Saved overall win percentage data for division {div}")



def generate_unbeaten_players(season_path):
    """
    Function to generate a list of unbeaten players for each division in a season.
    """
    logging.info(f"Generating unbeaten players for season at {season_path}")
    divisions = get_divisions_for_season(season_path)
    ranking_df_path = os.path.join(season_path, 'ranking_df')
    unbeaten_players_path = os.path.join(season_path, 'unbeaten_players')

    os.makedirs(unbeaten_players_path, exist_ok=True)

    for division in divisions:
        ranking_file = os.path.join(ranking_df_path, f'{division}_ranking_df.csv')
        if not os.path.exists(ranking_file):
            logging.warning(f'Ranking file not found for division {division}')
            continue

        ranking_df = pd.read_csv(ranking_file)
        unbeaten_df = ranking_df[ranking_df['Lost'] == 0]
        min_games_played = 5
        unbeaten_df = unbeaten_df[unbeaten_df['Games Played'] >= min_games_played]

        player_list = unbeaten_df['Player'].tolist()
        output_file = os.path.join(unbeaten_players_path, f'{division}.txt')
        with open(output_file, 'w') as f:
            for player in player_list:
                f.write(f"{player}\n")
        logging.info(f"Saved unbeaten players list for division {division} to {output_file}")


def process_previous_seasons(base_directory):
    """
    Function to add required data for all previous seasons in the base directory.
    """
    previous_seasons_path = os.path.join(base_directory, 'previous_seasons')

    # List all seasons excluding "2023-2024"
    seasons = [d for d in os.listdir(previous_seasons_path) 
               if os.path.isdir(os.path.join(previous_seasons_path, d)) and d != "2023-2024"]

    for season in seasons:
        season_path = os.path.join(previous_seasons_path, season)
        print(f"Processing season {season}")

        # Generate required data in sequence
        try:
            # Generate detailed league tables first
            generate_detailed_league_tables(season_path)

            # Generate home and away data next
            #generate_home_away_data(season_path)
            
            # Generate played every game data
            #generate_played_every_game(season_path)
            
            # Generate summarized player tables
            #generate_summarized_player_tables(season_path)
            
            # Generate team win percentage breakdown (depends on previous data)
            #generate_team_win_percentage_breakdown(season_path)
            
            # Generate unbeaten players list
            #generate_unbeaten_players(season_path)

            print(f"Completed processing for season {season}")

        except Exception as e:
            print(f"Error processing season {season}: {e}")


if __name__ == "__main__":
    base_directory = "C:/Users/bpali/PycharmProjects/SquashApp/"
    process_previous_seasons(base_directory)

