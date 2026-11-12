import re
import pandas as pd
from pathlib import Path

LADIES_DIVS = {"Premier Ladies", "L2", "L3", "L4"}
MASTERS_DIVS = {"Premier Masters", "M2", "M3", "M4"}


def get_league(division):
    if division in LADIES_DIVS:
        return "Ladies"
    elif division in MASTERS_DIVS:
        return "Masters"
    return "Main"


def get_division_num(division, league):
    if league == "Ladies":
        return 1 if division == "Premier Ladies" else int(division[1:])
    elif league == "Masters":
        return 1 if division == "Premier Masters" else int(division[1:])
    else:
        return 1 if division == "Premier Main" else int(re.sub(r"[^0-9]", "", division))


teams_dir = Path(__file__).resolve().parents[1] / "2026-2027" / "teams_df" / "week_1"

dfs = []
for csv_file in sorted(teams_dir.glob("*_teams_df.csv")):
    division = csv_file.stem.replace("_teams_df", "")
    df = pd.read_csv(csv_file)
    df["Division"] = division
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined["League"] = combined["Division"].apply(get_league)
combined["Division Number"] = combined.apply(
    lambda row: get_division_num(row["Division"], row["League"]), axis=1
)
combined = combined[["Team Name", "Home", "Convenor", "Email", "Division", "League", "Division Number"]]

league_order = pd.CategoricalDtype(["Main", "Masters", "Ladies"], ordered=True)
combined["League"] = combined["League"].astype(league_order)
combined = combined.sort_values(["League", "Division Number"]).reset_index(drop=True)

output_dir = Path(r"C:\Users\bpali\PycharmProjects\SquashApp\outputs\emails")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "teams_contact_list.csv"
combined.to_csv(output_path, index=False)
print(f"Saved {len(combined)} rows to {output_path}")
