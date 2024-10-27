import pandas as pd

# Load the combined results dataframe
results_df = pd.read_csv(r"C:\Users\bpali\PycharmProjects\SquashApp\2024-2025\combined_results_df.csv")

hkcc = "Hong Kong Cricket Club"

# Filter the results dataframe to only include matches where home team or away team contains "Hong Kong Cricket Club"
hkcc_results_df = results_df[(results_df["Home Team"].str.contains(hkcc)) | (results_df["Away Team"].str.contains(hkcc))]

# Make sure the date column is in datetime format
hkcc_results_df["Date"] = pd.to_datetime(hkcc_results_df["Date"], dayfirst=True, errors="coerce")

# Sort in ascending order by date
hkcc_results_df = hkcc_results_df.sort_values(by="Date")

# Remove unnecessary columns
columns_to_keep = ["Home Team", "Away Team", "Venue", "Match Week", "Date", "Overall Score", "Rubbers", "Division"]
hkcc_results_df = hkcc_results_df[columns_to_keep]

# Save the filtered results dataframe to a new CSV file
hkcc_results_df.to_csv(r"C:\Users\bpali\PycharmProjects\SquashApp\2024-2025\hkcc_results_df.csv", index=False)