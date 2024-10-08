import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import glob
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[
        logging.FileHandler("app_debug.log"),  # Log to a file named 'app_debug.log'
        logging.StreamHandler()  # Also output logs to the console (optional)
    ]
)

# Set page configurations
st.set_page_config(
    page_title="HK Squash App",
    page_icon="🇭🇰",
    layout="wide"
)

today = pd.Timestamp(datetime.now().date())

# Define the season
season = "2024-2025"

# Define the base directory
base_directory = "C:/Users/bpali/PycharmProjects/SquashApp/"

# Define the season base path 
season_base_path = os.path.join(base_directory, season)

# Define the team win percentage base path
team_win_percentage_breakdown_path = os.path.join(season_base_path, "team_win_percentage_breakdown")

# Paths to subdirectories
team_win_percentage_breakdown_overall_path = os.path.join(team_win_percentage_breakdown_path, "Overall")
team_win_percentage_breakdown_home_path = os.path.join(team_win_percentage_breakdown_path, "Home")
team_win_percentage_breakdown_away_path = os.path.join(team_win_percentage_breakdown_path, "Away")
team_win_percentage_breakdown_delta_path = os.path.join(team_win_percentage_breakdown_path, "Delta")

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

def get_latest_week_path(data_folder):
    """
    Function to get the path of the latest week folder in a given data folder.
    """
    logging.debug(f"Looking for week folders in {data_folder}")
    week_folder_pattern = os.path.join(data_folder, "week_*")
    week_folders = glob.glob(week_folder_pattern)
    logging.debug(f"Found week folders: {week_folders}")

    if not week_folders:
        logging.warning(f"No week folders found in {data_folder}")
        return None
    
    # Extract week numbers and their corresponding folders
    week_numbers = []
    for folder in week_folders:
        week_name = os.path.basename(folder)
        match = re.match(r"week_(\d+)", week_name)
        if match:
            week_number = int(match.group(1))
            week_numbers.append((week_number, folder))  # Collect as tuple (week_number, folder)
            logging.debug(f"Found week folder: {week_name} in folder {folder}")
        else:
            logging.warning(f"Folder name {week_name} does not match expected pattern.")
    
    if not week_numbers:
        logging.warning("No valid week folders found after processing.")
        return None # No valid week folders found
    
    # Get the folder with the highest week number
    latest_week_folder = max(week_numbers, key=lambda x: x[0])[1]
    logging.info(f"Latest week folder for {data_folder} is {latest_week_folder}")
    return latest_week_folder
    

def load_overall_home_away_data(division):
    """
    Load the overall home/away data for a given division.
    """
    # Define the base path for home_away_data
    home_away_data_base_path = os.path.join(season_base_path, "home_away_data")

    # Get the latest week folder
    latest_home_away_data_week = get_latest_week_path(home_away_data_base_path)

    if latest_home_away_data_week is None:
        logging.warning(f"No home/away data available for division {division}.")
        return None  # Or return an empty DataFrame/list as appropriate

    # Construct the file path
    overall_home_away_path = os.path.join(latest_home_away_data_week, f"{division}_overall_scores.csv")

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



def load_csvs(division):
    try:
        logging.info(f"Loading CSVs for division {division}")
        # Define the paths to the data folders
        detailed_league_tables_path = os.path.join(season_base_path, "detailed_league_tables")
        home_away_data_path = os.path.join(season_base_path, "home_away_data")
        simulated_tables_path = os.path.join(season_base_path, "simulated_tables")
        simulated_fixtures_path = os.path.join(season_base_path, "simulated_fixtures")
        team_win_percentage_breakdown_path = os.path.join(season_base_path, "team_win_percentage_breakdown")
        awaiting_results_path = os.path.join(season_base_path, "awaiting_results")
        summarized_player_tables_path = os.path.join(season_base_path, "summarized_player_tables")

        # Get the latest week paths
        logging.debug("Getting latest week paths for data folders")
        latest_detailed_league_tables_week = get_latest_week_path(detailed_league_tables_path)
        latest_home_away_data_week = get_latest_week_path(home_away_data_path)
        latest_simulated_tables_week = get_latest_week_path(simulated_tables_path)
        latest_simulated_fixtures_week = get_latest_week_path(simulated_fixtures_path)
        latest_awaiting_results_week = get_latest_week_path(awaiting_results_path)
        latest_summarized_player_tables_week = get_latest_week_path(summarized_player_tables_path)

        # Get latest week paths under each subdirectory for team win percentage breakdown
        latest_team_win_percentage_breakdown_overall_week = get_latest_week_path(team_win_percentage_breakdown_overall_path)
        latest_team_win_percentage_breakdown_home_week = get_latest_week_path(team_win_percentage_breakdown_home_path)
        latest_team_win_percentage_breakdown_away_week = get_latest_week_path(team_win_percentage_breakdown_away_path)
        latest_team_win_percentage_breakdown_delta_week = get_latest_week_path(team_win_percentage_breakdown_delta_path)

        # Construct file paths, handling None values for week paths
        logging.debug("Constructing file paths")
        overall_home_away_path = os.path.join(latest_home_away_data_week, f"{division}_overall_scores.csv") \
            if latest_home_away_data_week else None
        final_table_path = os.path.join(latest_simulated_tables_week, f"{division}_proj_final_table.csv") \
            if latest_simulated_tables_week else None
        fixtures_path = os.path.join(latest_simulated_fixtures_week, f"{division}_proj_fixtures.csv") \
            if latest_simulated_fixtures_week else None
        home_away_df_path = os.path.join(latest_home_away_data_week, f"{division}_team_average_scores.csv") \
            if latest_home_away_data_week else None
        team_win_breakdown_overall_path = os.path.join(
            latest_team_win_percentage_breakdown_overall_week,
            f"{division}_team_win_percentage_breakdown.csv"
        ) if latest_team_win_percentage_breakdown_overall_week else None
        team_win_breakdown_home_path = os.path.join(
            latest_team_win_percentage_breakdown_home_week,
            f"{division}_team_win_percentage_breakdown_home.csv"
        ) if latest_team_win_percentage_breakdown_home_week else None
        team_win_breakdown_away_path = os.path.join(
            latest_team_win_percentage_breakdown_away_week,
            f"{division}_team_win_percentage_breakdown_away.csv"
        ) if latest_team_win_percentage_breakdown_away_week else None
        team_win_breakdown_delta_path = os.path.join(
            latest_team_win_percentage_breakdown_delta_week,
            f"{division}_team_win_percentage_breakdown_delta.csv"
        ) if latest_team_win_percentage_breakdown_delta_week else None
        awaiting_results_path = os.path.join(latest_awaiting_results_week, f"{division}_awaiting_results.csv") \
            if latest_awaiting_results_week else None
        detailed_league_table_path = os.path.join(latest_detailed_league_tables_week,
                                                  f"{division}_detailed_league_table.csv") \
            if latest_detailed_league_tables_week else None
        summarized_players_path = os.path.join(latest_summarized_player_tables_week,
                                               f"{division}_summarized_players.csv") \
            if latest_summarized_player_tables_week else None

        # Initialize variables with empty DataFrames
        overall_home_away = pd.DataFrame()
        final_table = pd.DataFrame()
        fixtures = pd.DataFrame()
        home_away_df = pd.DataFrame()
        team_win_breakdown_overall = pd.DataFrame()
        team_win_breakdown_home = pd.DataFrame()
        team_win_breakdown_away = pd.DataFrame()
        team_win_breakdown_delta = pd.DataFrame()
        awaiting_results = pd.DataFrame()
        detailed_league_table = pd.DataFrame()
        summarized_players = pd.DataFrame()

        # Load data files individually
        data_files = {
            'overall_home_away': (overall_home_away_path, 'csv', {'header': None}),
            'final_table': (final_table_path, 'csv', {}),
            'fixtures': (fixtures_path, 'csv', {}),
            'home_away_df': (home_away_df_path, 'csv', {}),
            'team_win_breakdown_overall': (team_win_breakdown_overall_path, 'csv', {}),
            'team_win_breakdown_home': (team_win_breakdown_home_path, 'csv', {}),
            'team_win_breakdown_away': (team_win_breakdown_away_path, 'csv', {}),
            'team_win_breakdown_delta': (team_win_breakdown_delta_path, 'csv', {}),
            'awaiting_results': (awaiting_results_path, 'csv', {}),
            'detailed_league_table': (detailed_league_table_path, 'csv', {}),
            'summarized_players': (summarized_players_path, 'csv', {}),
        }

        data = {}
        for key, (path, file_type, params) in data_files.items():
            if path:
                try:
                    if file_type == 'csv':
                        data[key] = pd.read_csv(path, **params)
                    # Add other file types if needed
                    logging.debug(f"Loaded {key} data from {path}")
                except Exception as e:
                    logging.warning(f"Could not load {key} data from {path}: {e}")
                    data[key] = pd.DataFrame()  # Assign empty DataFrame
            else:
                logging.warning(f"No path available for {key}; setting as empty DataFrame.")
                data[key] = pd.DataFrame()

        # Return the data in the expected order
        return (
            data.get('final_table', pd.DataFrame()),
            data.get('fixtures', pd.DataFrame()),
            data.get('home_away_df', pd.DataFrame()),
            data.get('team_win_breakdown_overall', pd.DataFrame()),
            data.get('team_win_breakdown_home', pd.DataFrame()),
            data.get('team_win_breakdown_away', pd.DataFrame()),
            data.get('team_win_breakdown_delta', pd.DataFrame()),
            data.get('awaiting_results', pd.DataFrame()),
            data.get('detailed_league_table', pd.DataFrame()),
            data.get('overall_home_away', pd.DataFrame()),
            data.get('summarized_players', pd.DataFrame())
        )

    except Exception as e:
        logging.exception(f"An error occurred while loading CSVs for division {division}: {e}")
        st.error(f"Data not found for division {division}. Error: {e}")
        # Return a tuple of empty DataFrames
        empty_df = pd.DataFrame()
        return (empty_df,) * 11



def load_txts(division):
    """
    Load the lists of unbeaten players and players who have played every game for a given division.

    Args:
        division (str): The division name.

    Returns:
        tuple: A tuple containing two lists:
            - unbeaten_players (list): List of unbeaten players.
            - played_every_game (list): List of players who have played every game.
    """
    logging.info(f"Loading TXTs for division {division}")

    # Define the paths to the data folders for the season
    unbeaten_players_base_path = os.path.join(season_base_path, "unbeaten_players")
    played_every_game_base_path = os.path.join(season_base_path, "played_every_game")

    # Get the latest week paths
    latest_unbeaten_players_week = get_latest_week_path(unbeaten_players_base_path)
    latest_played_every_game_week = get_latest_week_path(played_every_game_base_path)

    # Check if the latest week paths are found
    if latest_unbeaten_players_week is None or latest_played_every_game_week is None:
        logging.warning(f"No data available for unbeaten players or played every game for division {division}.")
        st.error(f"Data not available for division {division}.")
        return [], []
    
    # Construct file paths
    unbeaten_file_path = os.path.join(latest_unbeaten_players_week, f"{division}.txt")
    played_every_game_file_path = os.path.join(latest_played_every_game_week, f"{division}.txt")

    unbeaten_players = []
    played_every_game = []

    # Load unbeaten players
    try:
        with open(unbeaten_file_path, "r") as file:
            unbeaten_players = [line.strip() for line in file]
            logging.debug(f"Loaded unbeaten players: {unbeaten_players}")
    except FileNotFoundError:
        logging.warning(f"No unbeaten players found for division {division} at {unbeaten_file_path}")

    # Load players who have played every game
    try:
        with open(played_every_game_file_path, "r") as file:
            played_every_game = [line.strip() for line in file]
            logging.debug(f"Loaded players who have played every game: {played_every_game}")
    except FileNotFoundError:
        logging.warning(f"No players who have played every game found for division {division} at {played_every_game_file_path}")

    return unbeaten_players, played_every_game


def load_player_rankings():
    """
    Function to load player rankings CSVs and add club information
    """
    logging.info("Loading player rankings")

    # Define the path to the ranking data for the season
    ranking_df_path = os.path.join(season_base_path, "ranking_df")
    latest_ranking_week = get_latest_week_path(ranking_df_path)

    if latest_ranking_week is None:
        logging.error("No ranking data found.")
        st.error("No ranking data found.")
        return pd.DataFrame()
    
    logging.debug(f"Latest ranking week folder: {latest_ranking_week}")

    # Get all CSV files in the latest ranking week directory
    files = glob.glob(os.path.join(latest_ranking_week, "*.csv"))
    logging.debug(f"Ranking data files found: {files}")

    if not files:
        logging.error("No ranking data files found in the latest week.")
        st.error("No ranking data files found in the latest week.")
        return pd.DataFrame()

    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            logging.debug(f"Loaded ranking data from {file}")
        except Exception as e:
            logging.warning(f"Error reading file {file}: {e}")
            continue
    
    if not dataframes:
        logging.error("No valid ranking data files loaded.")
        st.error("No valid ranking data files loaded.")
        return pd.DataFrame()

    all_rankings_df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Combined all rankings data into a single DataFrame with shape {all_rankings_df.shape}")

    # List of clubs
    clubs = [
        "Hong Kong Cricket Club", "Hong Kong Football Club", "Kowloon Cricket Club",
        "Ladies Recreation Club", "Royal Hong Kong Yacht Club", "United Services Recreation Club",
        "Fusion Squash Club", "Sha Tin Squash Rackets Club", "X-Alpha", "TNG", "RELAY", "YLJR",
        "i-Mask Advance Squash Club", "Vitality Squash", "Twister", "Friend Club",
        "North District Sports Association", "Physical Chess", "Electrify Squash", "Global Squash",
        "Squashathon", "Hong Kong Racketlon Association", "The Squash Club", "Happy Squash",
        "Star River", "Kinetic", "Smart Squash", "The Hong Kong Jockey Club", "Young Player",
        "Hong Kong Club", "8 Virtues"
    ]

    def determine_club(team_name):
        """
        Function to create club column from team name
        """
        # Check for the specific case
        if team_name == "FC 3":
            return "Friend Club"

        # Existing logic for other cases
        for club in clubs:
            if club.lower() in team_name.lower():
                return club
        return team_name # Return the team name if no club is matched


    # Apply the function to the 'Team' column
    if "Team" in all_rankings_df.columns:
        all_rankings_df['Club'] = all_rankings_df['Team'].apply(determine_club)
    else:
        st.error("Column 'Team' not found in ranking data.")
        all_rankings_df["Club"] = "Unknown"

    return all_rankings_df

# testing


def main():
    logging.info("Application started")
    # Initialize 'data' in session_state with default empty DataFrames if it doesn't exist
    if 'data' not in st.session_state:
        empty_df = pd.DataFrame()
        st.session_state['data'] = {
            'division_data': {},
            'all_rankings_df': pd.DataFrame(),
            'data_loaded': False,
            'current_division': None
        }
        logging.debug("Initialized session state data")

    # Title
    st.title("HK Squash League App")

    with st.sidebar:
        division_selection = st.radio("**Select View**:", ["Select a Division", "All Divisions"])
        division = None  # Initialize division

        if division_selection == "Select a Division":
            division = st.selectbox("**Select a Division:**", list(all_divisions.keys()))
            sections = [
                "Detailed Division Table",
                "Home/Away Splits",
                "Division Player Stats",
                "Projections",
                "Rubber Win Percentage"
            ]
        else:
            division = "All"  # Handle the "All Divisions" case
            sections = [
                "Player Info"
            ]

        sections_box = st.selectbox("**Select a section:**", sections)
        logging.info(f"User selected division: {division}")
        logging.info(f"User selected section: {sections_box}")

        about = st.expander("**About**")
        about.write("""The aim of this application is to take publicly available data from 
            [hksquash.org.hk](https://www.hksquash.org.hk/public/index.php/leagues/index/league/Squash/pages_id/25.html)
            and provide insights to players and convenors involved in the Hong Kong Squash League.
            \nThis application is not affiliated with the Squash Association of Hong Kong, China.""")

        contact = st.expander("**Contact**")
        contact.write("For any queries, email bpalitherland@gmail.com")

    # Only attempt to load division-specific data if a specific division is selected
    if division != "All":
        logging.debug(f"Processing division-specific data for {division}")
        if (st.session_state['data'].get('current_division') != division or
            not st.session_state['data'].get('data_loaded')):
            # Load data for the selected division
            division_data = load_csvs(division)
            st.session_state['data']['division_data'][division] = division_data
            st.session_state['data']['current_division'] = division
            st.session_state['data']['data_loaded'] = True

            # Load TXTs only for a specific division
            unbeaten_players, played_every_game = load_txts(division)
            st.session_state['data']['unbeaten_players'] = unbeaten_players
            st.session_state['data']['played_every_game'] = played_every_game
        else:
            # Retrieve data from session state
            division_data = st.session_state['data']['division_data'].get(division)
            unbeaten_players = st.session_state['data'].get('unbeaten_players', [])
            played_every_game = st.session_state['data'].get('played_every_game', [])

        # Unpack the division data
        (simulated_table, simulated_fixtures, home_away_df, team_win_breakdown_overall,
         team_win_breakdown_home, team_win_breakdown_away, team_win_breakdown_delta,
         awaiting_results, detailed_league_table, overall_home_away, summarized_players) = division_data
        
        # Handle cases where data might be empty
        if overall_home_away is not None and not overall_home_away.empty and overall_home_away.shape[1] > 4:
            # Extract the raw date values
            simulation_date_value = overall_home_away.iloc[0, 4]
            date_value = overall_home_away.iloc[0, 3]

            # Convert to datetime with error handling
            simulation_date_raw = pd.to_datetime(simulation_date_value, errors='coerce')
            date_raw = pd.to_datetime(date_value, errors='coerce')

            # Check and handle NaT values for simulation_date
            if pd.isnull(simulation_date_raw):
                simulation_date = None  # Do not assign today's date
                logging.warning("Simulation date is not available.")
            else:
                simulation_date = simulation_date_raw.strftime('%Y-%m-%d')

            # Check and handle NaT values for date
            if pd.isnull(date_raw):
                date = None  # Do not assign today's date
                logging.warning("Date is not available.")
            else:
                date = date_raw.strftime('%Y-%m-%d')
        else:
            simulation_date = date = None
            logging.warning("overall_home_away DataFrame is empty or does not have enough columns; dates are not available.")
    
    else:
        logging.debug("Processing data for all divisions")
        # Handle the "All Divisions" case separately
        if not st.session_state["data"].get("all_rankings_loaded", False):
            # Load all_rankings_df
            all_rankings_df = load_player_rankings()
            st.session_state["data"]["all_rankings_df"] = all_rankings_df
            st.session_state["data"]["all_rankings_loaded"] = True
        else:
            all_rankings_df = st.session_state["data"]["all_rankings_df"]

    # Now proceed based on the selected section
    if division != "All":
        if sections_box == "Detailed Division Table":
            # Header
            st.header(f"Detailed Division Table - Division {division}")
            st.write(f"**Last updated:** {date}")

            if not awaiting_results.empty:
                # Line break
                st.write('<br>', unsafe_allow_html=True)
                st.subheader("Still awaiting these results:")
                styled_awaiting_results = awaiting_results.style.hide(axis='index')
                st.write(styled_awaiting_results.to_html(escape=False), unsafe_allow_html=True)
                st.write('<br>', unsafe_allow_html=True)

            # Line break
            st.write('<br>', unsafe_allow_html=True)

            if not detailed_league_table.empty:
                # Apply styles to the DataFrame
                styled_df = detailed_league_table.style.set_properties(**{'text-align': 'right'}).hide(axis='index')
                styled_df = styled_df.set_properties(subset=['Team'], **{'text-align': 'left'})
                styled_df = styled_df.bar(subset=['Points'], color='#87CEEB', vmin=0)

                # Convert styled DataFrame to HTML
                html = styled_df.to_html(escape=False)

                # Display in Streamlit
                st.write(html, unsafe_allow_html=True)
            else:
                st.info(f"No detailed league table available for Division {division}.")

                # Note
                st.write('<br>', unsafe_allow_html=True)
                st.write("**Note:**  \nCR stands for Conceded Rubber.  \nWO stands for Walkover. Teams are penalized \
                         one point for each walkover given.")
                
        elif sections_box == "Home/Away Splits":
            # Header
            st.header(f"Home/Away Splits - Division {division}")

            # Load and display overall scores
            overall_scores = load_overall_home_away_data(division)
            if overall_scores:
                # Line break
                st.write('<br>', unsafe_allow_html=True)
                st.subheader("Overall split:")

                # Sizes for the pie chart
                sizes = [float(overall_scores[0]), float(overall_scores[1])]

                # Update labels and colors
                labels = ['Home', 'Away']
                colors = ['#ff9999', '#66b3ff']

                # Set font properties to Calibri
                prop = fm.FontProperties(family='Calibri')

                # Create columns using st.columns
                # Adjust the fractions to control the width of each column
                col1, col2 = st.columns([1, 1])

                # Plotting the chart in the first column
                with col1:
                    # Create a pie chart with larger font size for labels
                    fig, ax = plt.subplots(figsize=(8, 6))
                    pie, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='',
                                                   textprops={'fontsize': 16, 'fontproperties': prop})
                    
                    # Draw a white circle in the middle to create the donut shape
                    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                    fig = plt.gcf()
                    fig.gca().add_artist(centre_circle)

                    # Place the absolute value texts inside their respective segments
                    for i, (slice, value) in enumerate(zip(pie, sizes)):
                        angle = (slice.theta2 + slice.theta1) / 2
                        x, y = slice.r * np.cos(np.deg2rad(angle)), slice.r * np.sin(np.deg2rad(angle))
                        ax.text(x * 0.85, y * 0.85, f'{value:.2f}', ha='center', va='center', fontproperties=prop, fontsize=14,
                                color='black')
                        
                    # Add text in the center of the circle
                    percentage = float(overall_scores[2]) * 100 if len(overall_scores) > 2 else 0
                    plt.text(0, 0, f"Home Win\n{percentage:.1f}%", ha='center', va='center', fontproperties=prop,
                             fontsize=14)
                    
                    # Add title
                    plt.title(f"Average Home/Away Rubbers Won in Division {division}", fontproperties=prop, size=16)

                    # Ensure the pie chart is a circle
                    ax.axis('equal')
                    plt.tight_layout()

                    # Display the plot in the Streamlit app
                    st.pyplot(fig)

                # Use the second column for other content or leave it empty
                with col2:
                    st.write("")  # You can add other content here if needed

                # Line break
                st.write('<br>', unsafe_allow_html=True)
                st.subheader("Split by team:")

                if not home_away_df.empty:
                    # Rename columns appropriately
                    home_away_df = home_away_df.rename(columns={
                        'home_away_diff': 'Difference',
                        'Home': 'Home Venue',  # Renaming existing 'Home' column to 'Home Venue'
                        'Average Home Score': 'Home',
                        'Average Away Score': 'Away'
                    })

                    # Ensure that the 'Home', 'Away', and 'Difference' columns are numeric
                    home_away_df[['Home', 'Away', "Difference"]] = (home_away_df[['Home', 'Away', "Difference"]]
                                                                    .apply(pd.to_numeric, errors='coerce'))
                    
                    # Determine the range for the colormap
                    vmin = home_away_df[['Home', 'Away']].min().min()
                    vmax = home_away_df[['Home', 'Away']].max().max()

                    # Apply a color gradient using 'Blues' colormap to 'Home' and 'Away' columns
                    colormap_blues = 'Blues'
                    styled_home_away_df = (home_away_df.style.background_gradient(
                        cmap=colormap_blues,
                        vmin=vmin,
                        vmax=vmax,
                        subset=['Home', 'Away']
                    ).set_properties(subset=['Home', 'Away'], **{'text-align': 'right'})
                                           .format("{:.2f}", subset=['Home', 'Away', 'Difference']))
                    
                    # Apply a color gradient using 'OrRd' colormap to 'Difference' column
                    colormap_orrd = 'OrRd'
                    if 'Difference' in home_away_df.columns:
                        styled_home_away_df = (styled_home_away_df.background_gradient(
                            cmap=colormap_orrd,
                            subset=['Difference']
                        ).set_properties(subset=['Difference'], **{'text-align': 'right'}))

                    # Hide the index
                    styled_home_away_df = styled_home_away_df.hide(axis='index')

                    # Display the styled DataFrame in Streamlit
                    st.write(styled_home_away_df.to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.info(f"No home/away data available for Division {division}.")

                # Line break
                st.write('<br>', unsafe_allow_html=True)

                # Note
                st.write("**Note:**  \nMatches where the home team and away team share a \
                        home venue are ignored in the calculation")
                
                st.write(
                    """
                    Since 2016-17 (ignoring the incomplete 2019-20 and 2021-22 seasons), 
                    the overall home advantage factor is 0.5294, meaning that in a best-of-5-rubber division
                    teams win an average of 2.65 rubbers at home compared to 2.35 away.
                    """)
                
            else:
                st.info(f"No overall home/away data available.")

        elif sections_box == "Rubber Win Percentage":
            # Header
            st.header("Rubber Win Percentage - Division {division}")

            # Function to apply common formatting
            def format_dataframe(df):
                if df is not None and not df.empty:

                    # Rename avg_win_perc column
                    df = df.rename(columns={"avg_win_perc": "Average"})
                    # Select only numeric columns for vmin and vmax calculation
                    numeric_cols_raw = [col for col in df.columns if 'Win' in col]

                    # Convert these columns to numeric type, handling non-numeric values
                    for col in numeric_cols_raw:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Determine the range for the colormap
                    vmin = df[numeric_cols_raw].min().min()
                    vmax = df[numeric_cols_raw].max().max()

                    # Format numeric columns for display
                    def format_float(x):
                        try:
                            return f'{float(x):.1f}'
                        except ValueError:
                            return x

                    df[numeric_cols_raw + ["Average"]] = df[numeric_cols_raw + ["Average"]].map(format_float)

                    # Check if "Total Rubbers" is in the DataFrame and format it as integer
                    if 'Total Rubbers' in df.columns:
                        df['Total Rubbers'] = df['Total Rubbers'].astype(int)

                    colormap_blues = 'Blues'
                    cols_for_blues_gradient = numeric_cols_raw
                    styled_df = df.style.background_gradient(
                        cmap=colormap_blues,
                        vmin=vmin,
                        vmax=vmax,
                        subset=cols_for_blues_gradient
                    ).set_properties(subset=cols_for_blues_gradient, **{'text-align': 'right'})

                    # Set right alignment for "Total Rubbers"
                    if 'Total Rubbers' in df.columns:
                        styled_df = styled_df.set_properties(subset=['Total Rubbers'], **{'text-align': 'right'})

                    colormap_oranges = 'OrRd'
                    if 'Average' in df.columns:
                        styled_df = styled_df.background_gradient(
                            cmap=colormap_oranges,
                            subset=['Average']
                        ).set_properties(subset=['Average'], **{'text-align': 'right'})

                    styled_df = styled_df.hide(axis='index')
                    return styled_df
                else:
                    return "DataFrame is empty or not loaded."


            def format_dataframe_delta(df):
                if df is not None and not df.empty:

                    # Rename avg_win_perc column
                    df = df.rename(columns={"avg_win_perc": "Average"})
                    # Select only numeric columns for vmin and vmax calculation
                    numeric_cols_raw = [col for col in df.columns if 'Win' in col]

                    # Convert these columns to numeric type, handling non-numeric values
                    for col in numeric_cols_raw:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Determine the range for the colormap
                    vmin = df[numeric_cols_raw].min().min()
                    vmax = df[numeric_cols_raw].max().max()

                    # Format numeric columns for display
                    def format_float(x):
                        try:
                            return f'{float(x):.1f}'
                        except ValueError:
                            return x

                    df[numeric_cols_raw + ["Average"]] = df[numeric_cols_raw + ["Average"]].map(format_float)

                    # Check if "Total Rubbers" is in the DataFrame and format it as integer
                    if 'Total Rubbers' in df.columns:
                        df['Total Rubbers'] = df['Total Rubbers'].astype(int)

                    colormap_blues = 'RdYlBu'
                    cols_for_blues_gradient = numeric_cols_raw
                    styled_df = df.style.background_gradient(
                        cmap=colormap_blues,
                        vmin=vmin,
                        vmax=vmax,
                        subset=cols_for_blues_gradient
                    ).set_properties(subset=cols_for_blues_gradient, **{'text-align': 'right'})

                    # Set right alignment for "Total Rubbers"
                    if 'Total Rubbers' in df.columns:
                        styled_df = styled_df.set_properties(subset=['Total Rubbers'], **{'text-align': 'right'})

                    colormap_oranges = 'RdYlBu'
                    if 'Average' in df.columns:
                        styled_df = styled_df.background_gradient(
                            cmap=colormap_oranges,
                            subset=['Average']
                        ).set_properties(subset=['Average'], **{'text-align': 'right'})

                    styled_df = styled_df.hide(axis='index')
                    return styled_df
                else:
                    return "DataFrame is empty or not loaded."

            # Radio button for user to choose the DataFrame
            option = st.radio("Select Team Win Breakdown View:", 
                            ['Overall', 'Home', 'Away', 'H/A Delta'], horizontal=True)

            st.write('<br>', unsafe_allow_html=True)
            st.subheader("Team win percentage by rubber:")

            # Apply formatting and display the selected DataFrame
            dataframes = {
                'Overall': team_win_breakdown_overall,
                'Home': team_win_breakdown_home,
                'Away': team_win_breakdown_away,
                'H/A Delta': team_win_breakdown_delta
            }

            selected_df = dataframes.get(option)
            if selected_df is not None and not selected_df.empty:
                if option == "H/A Delta":
                    st.write(format_dataframe_delta(selected_df).to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.write(format_dataframe(selected_df).to_html(escape=False), unsafe_allow_html=True)
            else:
                st.error("Selected data is not available.")

            # Note
            st.write('<br>', unsafe_allow_html=True)
            st.write(
                "**Note:**  \nOnly rubbers that were played are included. Conceded Rubbers \
                and Walkovers are ignored.  \nMatches where the home team and away team share \
                a home venue are ignored in the Home and Away tables.")

    elif sections_box == "Projections":

        # Load and display overall scores
        st.header("Projections")
        st.write(f"**Date of last simulation:** {simulation_date}")

        if not awaiting_results.empty:
            # Line break
            st.write('<br>', unsafe_allow_html=True)
            st.subheader("Still awaiting these results:")
            styled_awaiting_results = awaiting_results.style.hide(axis='index')
            st.write(styled_awaiting_results.to_html(escape=False), unsafe_allow_html=True)
            st.write('<br>', unsafe_allow_html=True)
            st.write('<br>', unsafe_allow_html=True)

        if simulated_fixtures.empty:
            st.write("No more remaining fixtures!")
        else:
            # Convert the "Match Week" column to integers
            simulated_fixtures['Match Week'] = simulated_fixtures['Match Week'].astype(int)

            # Adjust the "Date" column format
            simulated_fixtures['Date'] = pd.to_datetime(simulated_fixtures['Date']).dt.date

            # Rename columns
            simulated_fixtures = simulated_fixtures.rename(columns={
                "Avg Simulated Home Points": "Proj. Home Pts",
                "Avg Simulated Away Points": "Proj. Away Pts"
            })

            # Round values in simulated_fixtures DataFrame except for "Match Week"
            numeric_cols_simulated_fixtures = simulated_fixtures.select_dtypes(include=['float', 'int']).columns.drop(
                'Match Week')
            simulated_fixtures[numeric_cols_simulated_fixtures] = simulated_fixtures[
                numeric_cols_simulated_fixtures].map(lambda x: f'{x:.2f}')

            # Ensure the columns are numeric for vmin and vmax calculation
            simulated_fixtures_numeric = simulated_fixtures.copy()
            simulated_fixtures_numeric[numeric_cols_simulated_fixtures] = simulated_fixtures_numeric[
                numeric_cols_simulated_fixtures].apply(pd.to_numeric, errors='coerce')

            # Get the range of match weeks
            min_week = simulated_fixtures_numeric['Match Week'].min()
            max_week = simulated_fixtures_numeric['Match Week'].max()

            # Create columns for the slider or display the week directly if only one is available
            col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed
            with col1:
                if min_week < max_week:
                    selected_week = st.slider("**Select Match Week:**", int(min_week), int(max_week), 
                                              value=int(min_week), step=1)
                else:
                    selected_week = int(min_week)
                    st.write(f"**Match Week:** {selected_week}")  # Display the week directly without a slider

            # Filter the fixtures based on the selected match week
            filtered_fixtures = simulated_fixtures_numeric[simulated_fixtures_numeric["Match Week"] == selected_week]

            # Determine the range for the colormap
            vmin = filtered_fixtures[["Proj. Home Pts", "Proj. Away Pts"]].min().min()
            vmax = filtered_fixtures[["Proj. Home Pts", "Proj. Away Pts"]].max().max()

            # Apply styling to the filtered DataFrame
            styled_filtered_fixtures = (
                filtered_fixtures.style.background_gradient(
                    cmap='Blues',
                    vmin=vmin,
                    vmax=vmax,
                    subset=numeric_cols_simulated_fixtures)
                .set_properties(subset=['Match Week', 'Proj. Home Pts', 'Proj. Away Pts'], **{'text-align': 'right'})
                .format("{:.2f}", subset=["Proj. Home Pts", "Proj. Away Pts"])
                .hide(axis='index'))

            # Display the styled DataFrame in Streamlit
            st.subheader(f"Projected Fixtures for Match Week {selected_week}:")
            st.write(styled_filtered_fixtures.to_html(escape=False), unsafe_allow_html=True)

        if not simulated_table.empty:
            # Line break and subheader
            st.write('<br>', unsafe_allow_html=True)
            st.subheader("Projected Final Table:")

            # Convert 'Played' column to integers
            if 'Played' in simulated_table.columns:
                simulated_table['Played'] = simulated_table['Played'].astype(int)

            # Round values in simulated_table DataFrame except for 'Played'
            numeric_cols_simulated_table = simulated_table.select_dtypes(include=['float', 'int']).columns
            cols_to_round = numeric_cols_simulated_table.drop('Played')

            # Columns to exclude from gradient formatting
            cols_to_exclude = {'Played', 'Won', 'Lost', 'Points', 'Playoffs'}
            cols_for_blues_gradient = [col for col in cols_to_round if col not in cols_to_exclude]

            # Determine the range for the colormap
            vmin = simulated_table[cols_for_blues_gradient].min().min()
            vmax = simulated_table[cols_for_blues_gradient].max().max()

            # Apply a color gradient using 'Blues' colormap to selected numeric columns
            styled_simulated_table = simulated_table.style.background_gradient(
                cmap='Blues',
                vmin=vmin,
                vmax=vmax,
                subset=cols_for_blues_gradient
            ).set_properties(subset=cols_for_blues_gradient + ["Played", "Won", "Lost"], **{'text-align': 'right'})

            # Apply a color gradient using 'OrRd' colormap to 'Playoffs' column
            if 'Playoffs' in simulated_table.columns:
                styled_simulated_table = styled_simulated_table.background_gradient(
                    cmap='OrRd', subset=['Playoffs']
                ).set_properties(subset=['Playoffs'], **{'text-align': 'right'})

            # Apply bar chart formatting to the 'Points' column
            styled_simulated_table = styled_simulated_table.bar(
                subset=['Points'], color='#87CEEB'
            )

            # Round all numeric columns
            styled_simulated_table = styled_simulated_table.format("{:.1f}", subset=cols_to_round)

            # Apply custom formatting for zero values in cols_for_blues_gradient
            styled_simulated_table = styled_simulated_table.format(
                lambda x: f"<span style='color: #f7fbff;'>{x:.1f}</span>" if x == 0 else f"{x:.1f}",
                subset=cols_for_blues_gradient
            )

            # Hide the index
            styled_simulated_table = styled_simulated_table.hide(axis='index')

            # Display the styled DataFrame in Streamlit
            st.write(styled_simulated_table.to_html(escape=False), unsafe_allow_html=True)

            # Note
            st.write('<br>', unsafe_allow_html=True)
            st.write("**Note:**  \nThe projected final table is the average result of simulating the remaining \
                     fixtures 10,000 times.  \nFixtures are simulated using teams' average rubber win percentage, \
                     factoring in home advantage.")

    elif sections_box == "Division Player Stats":

        def extract_names(cell_value):

            """
            Function to extract names from the cell
            """
            # Using regex to extract names before parentheses or commas
            names = re.findall(r"([\w\s]+)(?=\s\(|,|$)", cell_value)
            return set(names)


        # Custom styling function
        def highlight_row_if_same_player(row):
            # Extract player names from each column
            games_names = extract_names(row["Most Games"])
            wins_names = extract_names(row["Most Wins"])
            win_percentage_names = extract_names(row["Highest Win Percentage"])

            # Highlight color
            highlight_color = 'background-color: #FFF2CC'
            # Default color
            default_color = ''  # or 'background-color: none'

            # Check if each column has exactly one name and it's the same across all three columns
            unique_names = games_names.union(wins_names).union(win_percentage_names)
            if len(unique_names) == 1 and len(games_names) == len(wins_names) == len(win_percentage_names) == 1:
                # The set union of all names will have exactly one element if all columns have the same single name
                return [
                    highlight_color if col in ["Most Games", "Most Wins", "Highest Win Percentage"] else default_color
                    for col in row.index]
            else:
                # Return default color for all columns if the condition is not met
                return [default_color for _ in row.index]

        # Load and display overall scores
        st.header("Player Stats")
        st.write(f"**Last Updated:** {date}")

        if not summarized_players.empty:
            # Apply styles to the DataFrame
            styled_df = summarized_players.style.set_properties(**{'text-align': 'left'}).hide(axis='index')

            # Apply the styling function to the DataFrame
            styled_df = styled_df.apply(highlight_row_if_same_player, axis=1)

            # Convert styled DataFrame to HTML
            html = styled_df.to_html(escape=False)

            st.write(html, unsafe_allow_html=True)
        else:
            st.info(f"No player stats available for Division {division}.")

        # Line break
        st.write('<br>', unsafe_allow_html=True)

        # Show list of unbeaten players
        if len(unbeaten_players) == 0:
            st.write("**There are no unbeaten players.**")
        elif len(unbeaten_players) == 1:
            st.write(f"**The following player is unbeaten:**  \n{', '.join(unbeaten_players)}")
        else:
            unbeaten_players_list = '<br>'.join(unbeaten_players)
            st.markdown(f"**The following players "
                        f"are unbeaten:**<br>{unbeaten_players_list}", unsafe_allow_html=True)

        # Show list of players who have played in every game for their team
        if len(played_every_game) == 0:
            st.write("No player has played in all of their team's games")
        elif len(played_every_game) == 1:
            st.write(f"**The following player has played in all of their "
                     f"team's games:**  \n{', '.join(played_every_game)}")
        else:
            every_game_list = '<br>'.join(played_every_game)
            st.markdown(f"**The following players "
                        f"have played every game:**<br>{every_game_list}", unsafe_allow_html=True)

        # Line break
        st.write("**Note:** Players must have played 5+ games to qualify.")

    elif sections_box == "Player Info":

        def aggregate_club(x):
            """
            Function to aggregate clubs in Club column
            """
            unique_clubs = x.unique()
            if len(unique_clubs) == 1:  # If only one unique club
                return unique_clubs[0]
            else:
                return ', '.join(sorted(unique_clubs))


        def aggregate_club_overall(x):
            """
            Function to aggregate clubs in Club column for 'Overall' selection
            """
            unique_clubs = x.unique()
            if len(unique_clubs) == 1:  # If only one unique club
                return unique_clubs[0]
            else:
                return ', '.join(sorted(unique_clubs))  # Join multiple club names with commas


        # List of clubs
        clubs = ["Overall"] + sorted([
            "Hong Kong Cricket Club", "Hong Kong Football Club", "Kowloon Cricket Club",
            "Ladies Recreation Club", "Royal Hong Kong Yacht Club", "United Services Recreation Club",
            "Fusion Squash Club", "Sha Tin Squash Rackets Club", "X-Alpha", "TNG", "RELAY", "YLJR",
            "i-Mask Advance Squash Club", "Vitality Squash", "Twister", "Friend Club",
            "North District Sports Association", "Physical Chess", "Electrify Squash", "Global Squash",
            "Squashathon", "Hong Kong Racketlon Association", "The Squash Club", "Happy Squash",
            "Star River", "Kinetic", "Smart Squash", "The Hong Kong Jockey Club", "Young Player",
            "Hong Kong Club", "8 Virtues"
        ])

        # Title
        st.title("**Player Rankings**")

        # Line break
        st.write('<br>', unsafe_allow_html=True)

        # Adjust the fractions to control the width of each column
        col1, col2 = st.columns([1, 3])

        # Find the index for "Hong Kong Cricket Club"
        default_club_index = clubs.index("Hong Kong Cricket Club")

        # Plotting the chart in the first column
        with col1:
            club = st.selectbox("**Select club:**", clubs, index=default_club_index)

        if club != "Overall":
            aggregated_df = all_rankings_df.groupby(['Name of Player', 'Club']).agg({
                'Division': lambda x: ', '.join(sorted(set(str(d) for d in x))),  # Aggregate divisions
                'Average Points': 'mean',  # Calculate the mean of average points
                'Total Game Points': 'sum',  # Sum of total game points
                'Games Played': 'sum',  # Sum of games played
                'Won': 'sum',  # Sum of games won
                'Lost': 'sum',  # Sum of games lost
            }).reset_index()
        else:
            aggregated_df = all_rankings_df.groupby('Name of Player').agg({
                'Club': aggregate_club_overall,  # Use the custom aggregation for clubs
                'Division': lambda x: ', '.join(sorted(set(x.astype(str)))),  # Aggregate divisions
                'Average Points': 'mean',  # Calculate the mean of average points
                'Total Game Points': 'sum',  # Sum of total game points
                'Games Played': 'sum',  # Sum of games played
                'Won': 'sum',  # Sum of games won
                'Lost': 'sum',  # Sum of games lost
            }).reset_index()

        # Now calculate 'Win Percentage' outside the aggregation step
        aggregated_df['Win Percentage'] = (aggregated_df['Won'] / aggregated_df['Games Played'] * 100).fillna(0)

        # Create Avg Pts column
        aggregated_df['Avg Pts'] = (aggregated_df['Total Game Points'] / aggregated_df['Games Played']).fillna(0)

        # Continue with your reduced dataframe and further logic
        aggregated_df_reduced = aggregated_df[[
            "Name of Player", "Club", "Division", "Games Played", "Won", "Lost", "Win Percentage", "Avg Pts"
        ]].rename(columns={"Name of Player": "Player", "Games Played": "Games", "Win Percentage": "Win %"})

        # Sort functionality
        with col1:
            sort_column = st.selectbox("**Sort by:**", aggregated_df_reduced.columns,
                                       index=aggregated_df_reduced.columns.get_loc("Games"))
        sort_order = st.radio("**Sort order**", ["Ascending", "Descending"], index=1)

        # If a specific club is selected (not "Overall"), filter the DataFrame by that club
        if club != "Overall":
            # Ensures sorted_df is only defined within this block
            filtered_df = aggregated_df_reduced[aggregated_df_reduced["Club"] == club]
            # Drop Club column if needed
            filtered_df = filtered_df.drop(columns='Club',
                                           errors='ignore')  # Use errors='ignore' to avoid error if 'Club' column does not exist
        else:
            # Ensure that filtered_df is defined even when "Overall" is selected
            filtered_df = aggregated_df_reduced

        # Sort the DataFrame based on user selection
        if sort_order == "Ascending":
            sorted_df = filtered_df.sort_values(by=sort_column, ascending=True)
        else:
            sorted_df = filtered_df.sort_values(by=sort_column, ascending=False)

        # Apply styles and formatting to sorted_df
        sorted_df = sorted_df.style.set_properties(
            subset=['Player', "Division"], **{'text-align': 'left'}).hide(axis="index")
        sorted_df = sorted_df.set_properties(
            subset=['Games', "Won", "Lost", "Win %", "Avg Pts"], **{'text-align': 'right'}
        )
        sorted_df = sorted_df.format("{:.1f}", subset=["Win %", "Avg Pts"])

        # Convert DataFrame to HTML, hide the index, and apply minimal styling for spacing
        html = sorted_df.to_html()

        # Line break
        st.write('<br>', unsafe_allow_html=True)

        # Display the sorted DataFrame
        st.write(html, unsafe_allow_html=True)


def generate_styled_html(df, numeric_cols, blues_cols, orrd_cols):
    styled_df = df.copy()
    styled_df[numeric_cols] = styled_df[numeric_cols].map(lambda x: f'{x:.2f}')

    # Apply 'Blues' gradient
    styled_df = styled_df.style.background_gradient(cmap='Blues', subset=blues_cols) \
        .set_properties(subset=blues_cols, **{'text-align': 'right'})

    # Apply 'OrRd' gradient
    if orrd_cols:
        styled_df = styled_df.background_gradient(cmap='OrRd', subset=orrd_cols) \
            .set_properties(subset=orrd_cols, **{'text-align': 'right'})

    styled_df = styled_df.hide(axis='index')
    return styled_df.to_html(escape=False)

if __name__ == "__main__":
    main()
