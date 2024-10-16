import os
import pandas as pd
import logging
import glob
import re

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

# Define the season
season = "2024-2025"

# Define the base directory
base_directory = "C:/Users/bpali/PycharmProjects/SquashApp/"

# Define the season base path 
season_base_path = os.path.join(base_directory, season)

# Define the awaiting results directory
awaiting_results_directory = os.path.join(season_base_path, "awaiting_results")


# After loop ends, print out all awaiting results
for div in all_divisions.keys():
    # Get highest week from awaiting_results directory
    week_folders = glob.glob(os.path.join(awaiting_results_directory, "week_*"))
    week_numbers = []
    for folder in week_folders:
        week_name = os.path.basename(folder)
        match = re.match(r'week_(\d+)', week_name)
        if match:
            week_number = int(match.group(1))
            week_numbers.append((week_number, folder))
    # Sort week folders by week number in descending order
    week_numbers.sort(reverse=True)
    sorted_week_folders = [folder for _, folder in week_numbers]

    # Initialize variables
    awaiting_results_dataframes = []
    divisions_loaded = set()
    divisions_to_load = set(all_divisions.keys())

    # Iterate over week folders
    for week_folder in sorted_week_folders:
        if not divisions_to_load:
            break

        # Get all CSV files in the week folder
        csv_files = glob.glob(os.path.join(week_folder, "*.csv"))
        for file in csv_files:
            # Extract division from file name
            file_name = os.path.basename(file)
            division_match = re.match(r'(.*)_awaiting_results.csv', file_name)
            if division_match:
                division = division_match.group(1)
                if division in divisions_to_load:
                    try:
                        df = pd.read_csv(file)
                        # add division column to dataframe
                        df['Division'] = division
                        awaiting_results_dataframes.append(df)
                        divisions_loaded.add(division)
                        divisions_to_load.remove(division)
                    except Exception as e:
                        logging.error(f"An error occurred while reading awaiting results for Division {division}: {e}")
                        break
            else:
                logging.warning(f"Unexpected file found in week folder: {file}")
        if not divisions_to_load:
            break

    if not awaiting_results_dataframes:
        logging.error(f"No awaiting results found for Division {div}.")

    # Concatenate all dataframes
    awaiting_results_df = pd.concat(awaiting_results_dataframes, ignore_index=True)

    # Sort by data in ascending order
    awaiting_results_df.sort_values(by=['Date'], ascending=True, inplace=True)

# Show the home team, away team, division, and date in awaiting results
print("Awaiting Results:")
print(awaiting_results_df[['Home Team', 'Away Team', 'Division', 'Date']])
