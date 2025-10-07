import os

def main():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    current_season = "2025-2026"  # or import from a config
    base_folder = os.path.join(base_directory, current_season, "players_df")

    divisions = [
        "Premier Main", "2", "3", "4", "5", "6", "7", "8A", "8B", "9", "10", "11", "12",
        "13A", "13B", "13C", "14", "15A", "15B", "Premier Masters", "M2", "M3", "M4",
        "Premier Ladies", "L2", "L3", "L4"
    ]

    weeks = range(1, 23)  # weeks 1..19

    missing_files = []

    for week in weeks:
        week_folder = os.path.join(base_folder, f"week_{week}")
        for division in divisions:
            # Construct the filename like "Premier Ladies_players_df.csv" or "7A_players_df.csv"
            filename = f"{division}_players_df.csv"
            file_path = os.path.join(week_folder, filename)
            if not os.path.exists(file_path):
                missing_files.append((week, division))

    # Print out any missing files
    if missing_files:
        print("Missing CSV files:")
        for (w, div) in missing_files:
            print(f"  Week {w}, Division '{div}'")
    else:
        print("All CSVs appear to be present.")

if __name__ == "__main__":
    main()
