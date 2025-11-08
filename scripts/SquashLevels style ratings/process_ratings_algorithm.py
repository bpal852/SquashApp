import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Base ratings by division (same as initial ratings script)
# ------------------------------------------------------------------
division_base_ratings = {
    # Numeric divisions
    '2': 8000,
    '3': 6000,
    '4': 4000,
    '5': 3000,
    '6': 2000,
    '7': 1800,
    '8': 1600,
    '9': 1400,
    '10': 1200,
    '11': 1000,
    '12': 800,
    '13': 600,
    '14': 400,
    '15': 200,
    # Special names
    'premier main': 18000,
    'premier masters': 4000,  # div 4
    'm2': 2000,  # div 6
    'm3': 1800,  # div 7
    'm4': 1200,  # div 10
    'premier ladies': 8000,  # div 2
    'l2': 2000,  # div 6
    'l3': 1600,  # div 8
    'l4': 600,  # div 13
}


def get_base_rating(division_str):
    d = str(division_str).lower().strip()
    if d in division_base_ratings:
        return division_base_ratings[d]
    match = re.match(r'^(\d+)', d)
    if match and match.group(1) in division_base_ratings:
        return division_base_ratings[match.group(1)]
    return 1000.0


# ------------------------------------------------------------------
# Glicko-2 configuration
# ------------------------------------------------------------------
GLICKO_BASE_RATING = 1500.0
DISPLAY_OFFSET = 4000.0
# Scale chosen so a 22K SquashLevels spread maps to roughly 400 Glicko points.
DISPLAY_SCALE = 55.0
GLICKO_SCALE = 173.7178
GLICKO_Q = math.log(10) / 400
MAX_RD = 350.0
MIN_RD = 20.0
ESTABLISHED_RD = 80.0
NEW_PLAYER_RD = 180.0
DEFAULT_VOLATILITY = 0.06
MIN_VOLATILITY = 0.01
MAX_VOLATILITY = 0.5
TAU = 0.5
PERIOD_LENGTH_DAYS = 14
OFFSEASON_PERIODS = 8
VOLATILITY_EPSILON = 1e-6
PERFORMANCE_SCALE = 0.6


def display_to_glicko(display_rating):
    return GLICKO_BASE_RATING + (float(display_rating) - DISPLAY_OFFSET) / DISPLAY_SCALE


def glicko_to_display(glicko_rating):
    return DISPLAY_OFFSET + (glicko_rating - GLICKO_BASE_RATING) * DISPLAY_SCALE


def rating_to_mu(rating):
    return (rating - GLICKO_BASE_RATING) / GLICKO_SCALE


def mu_to_rating(mu):
    return GLICKO_BASE_RATING + mu * GLICKO_SCALE


def rd_to_phi(rd):
    return rd / GLICKO_SCALE


def phi_to_rd(phi):
    return phi * GLICKO_SCALE


def g(phi):
    return 1.0 / math.sqrt(1.0 + (3.0 * (GLICKO_Q ** 2) * (phi ** 2)) / (math.pi ** 2))


def _stable_logistic(x):
    if x >= 0:
        exp_neg = math.exp(-x)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(x)
    return exp_pos / (1.0 + exp_pos)


def volatility_function(x, delta, phi, v, a):
    exp_x = math.exp(x)
    numerator = exp_x * (delta ** 2 - phi ** 2 - v - exp_x)
    denominator = 2.0 * (phi ** 2 + v + exp_x) ** 2
    return (numerator / denominator) - ((x - a) / (TAU ** 2))


def update_volatility(phi, sigma, delta, v):
    a = math.log(sigma ** 2)
    if delta ** 2 > phi ** 2 + v:
        b = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        b = a - k * TAU
        while volatility_function(b, delta, phi, v, a) < 0:
            k += 1
            b = a - k * TAU

    f_a = volatility_function(a, delta, phi, v, a)
    f_b = volatility_function(b, delta, phi, v, a)

    while abs(b - a) > VOLATILITY_EPSILON:
        c = a + (a - b) * f_a / (f_b - f_a)
        f_c = volatility_function(c, delta, phi, v, a)
        if f_c * f_b < 0:
            a = b
            f_a = f_b
        else:
            f_a /= 2.0
        b = c
        f_b = f_c

    return math.exp(a / 2.0)


def inflate_rd_for_inactivity(player_record, periods):
    if periods <= 0:
        return
    phi = rd_to_phi(player_record["rd"])
    sigma = player_record["volatility"]
    for _ in range(periods):
        phi = math.sqrt(phi ** 2 + sigma ** 2)
    player_record["rd"] = min(phi_to_rd(phi), MAX_RD)


def score_to_fraction(score_str):
    score = str(score_str).strip()
    if not score:
        return 0.5, 0.5
    score = score.replace(" ", "")
    match = re.match(r"(\d+)[:\-](\d+)", score)
    if not match:
        return 0.5, 0.5
    winner_sets = int(match.group(1))
    loser_sets = int(match.group(2))
    diff = winner_sets - loser_sets
    if diff >= 3:
        return 0.60, 0.40
    if diff == 2:
        return 0.57, 0.43
    if diff == 1:
        return 0.54, 0.46
    if diff == 0:
        return 0.50, 0.50
    total_sets = winner_sets + loser_sets
    if total_sets == 0:
        return 0.5, 0.5
    win_fraction = winner_sets / total_sets
    return win_fraction, 1.0 - win_fraction


def process_ratings_algorithm(base_folder=None, current_season="2025-2026", previous_season="2024-2025"):
    """Process ratings using a fortnightly Glicko-2 update."""
    if base_folder is None:
        base_folder = Path(__file__).parent.parent.parent
    else:
        base_folder = Path(base_folder)

    previous_season_folder = base_folder / "previous_seasons" / previous_season
    current_season_folder = base_folder / current_season

    logging.info("=" * 70)
    logging.info("PROCESSING PLAYER RATINGS (GLICKO-2)")
    logging.info("=" * 70)
    logging.info(f"Loading final ratings from {previous_season} season...")

    previous_ratings_csv = previous_season_folder / "ratio_results.csv"
    player_data = {}

    if os.path.exists(previous_ratings_csv):
        df_prev = pd.read_csv(previous_ratings_csv)
        logging.info(f"Loaded {len(df_prev)} players from previous season")
        for _, row in df_prev.iterrows():
            hks = row.get("HKS_No")
            if pd.isna(hks):
                continue
            key = int(hks)
            final_rating_display = row.get("Final Rating", 1000.0)
            glicko_rating = display_to_glicko(final_rating_display)
            rd_value = row.get("Glicko RD", ESTABLISHED_RD)
            if pd.isna(rd_value):
                rd_value = ESTABLISHED_RD
            rd_value = max(MIN_RD, min(float(rd_value), MAX_RD))
            volatility_value = row.get("Glicko Volatility", DEFAULT_VOLATILITY)
            if pd.isna(volatility_value):
                volatility_value = DEFAULT_VOLATILITY
            volatility_value = max(MIN_VOLATILITY, min(float(volatility_value), MAX_VOLATILITY))
            prior_matches = row.get("Matches Played", 0)
            if pd.isna(prior_matches):
                prior_matches = 0
            prior_teams_raw = str(row.get("Teams", "")).strip()
            prior_teams = set()
            if prior_teams_raw:
                prior_teams = {team.strip() for team in prior_teams_raw.split(",") if team.strip()}
            player_data[key] = {
                "player": str(row.get("Player", "")).strip(),
                "rating": float(glicko_rating),
                "rd": float(rd_value),
                "volatility": float(volatility_value),
                "matches_played": int(prior_matches),
                "teams": prior_teams,
                "last_period": -OFFSEASON_PERIODS - 1,
            }
        logging.info(f"Initialized {len(player_data)} returning players")
    else:
        logging.warning(f"Previous season ratings not found at {previous_ratings_csv}")
        logging.warning("Will use division-based initial ratings for all players")

    current_init_csv = current_season_folder / "all_initial_ratings.csv"
    if os.path.exists(current_init_csv):
        df_current_init = pd.read_csv(current_init_csv)
        df_current_init["HKS_No"] = pd.to_numeric(df_current_init.get("HKS No."), errors="coerce")
        df_current_init.rename(columns={"Initial Rating": "Initial_Rating"}, inplace=True)
        new_players_count = 0
        for row in df_current_init.itertuples(index=False):
            hks = getattr(row, "HKS_No", None)
            if pd.isna(hks):
                continue
            key = int(hks)
            if key in player_data:
                player_data[key]["teams"].add(str(getattr(row, "Team", "")).strip())
                continue
            initial_rating_display = getattr(row, "Initial_Rating", 1000.0)
            glicko_rating = display_to_glicko(initial_rating_display)
            player_data[key] = {
                "player": str(getattr(row, "Player", "")).strip(),
                "rating": float(glicko_rating),
                "rd": float(NEW_PLAYER_RD),
                "volatility": DEFAULT_VOLATILITY,
                "matches_played": 0,
                "teams": {str(getattr(row, "Team", "")).strip()},
                "last_period": None,
            }
            new_players_count += 1
        logging.info(f"Added {new_players_count} new players from current season initial ratings")
    else:
        logging.info(f"No initial ratings file found at {current_init_csv}")

    results_csv = current_season_folder / "combined_player_results_df.csv"
    logging.info(f"Loading match results from {results_csv}")
    df = pd.read_csv(results_csv)
    df.rename(columns={"HKS No.": "HKS_No", "Opponent HKS No.": "Opponent_HKS_No"}, inplace=True)
    df["HKS_No"] = pd.to_numeric(df["HKS_No"], errors="coerce")
    df["Opponent_HKS_No"] = pd.to_numeric(df["Opponent_HKS_No"], errors="coerce")
    df["Match Date"] = pd.to_datetime(df["Match Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Match Date"])
    df = df.sort_values(["Match Date", "Division", "Team", "Rubber Number"]).reset_index(drop=True)

    skip_mask = (
        df["Player Name"].eq("Unknown")
        | df["Opponent Name"].eq("Unknown")
        | df["Score"].isin(["CR", "WO"])
    )
    df = df[~skip_mask].copy()

    df["Pre Winner Rating"] = pd.NA
    df["Post Winner Rating"] = pd.NA
    df["Pre Loser Rating"] = pd.NA
    df["Post Loser Rating"] = pd.NA
    df["Winner Matches Played"] = pd.NA
    df["Loser Matches Played"] = pd.NA

    df_wins = df[df["Result"] == "Win"].copy()
    if df_wins.empty:
        logging.warning("No matches marked as wins were found; exiting early")
        return

    min_date = df_wins["Match Date"].min()
    df_wins["Period"] = ((df_wins["Match Date"] - min_date).dt.days // PERIOD_LENGTH_DAYS).astype(int)

    matches_processed = 0
    match_entries = []

    for period_id in sorted(df_wins["Period"].unique()):
        period_rows = df_wins[df_wins["Period"] == period_id].sort_values("Match Date")
        period_players = set()
        period_entries = []

        for idx, row in period_rows.iterrows():
            winner_hks = row.get("HKS_No")
            loser_hks = row.get("Opponent_HKS_No")
            if pd.isna(winner_hks) or pd.isna(loser_hks):
                continue
            winner_key = int(winner_hks)
            loser_key = int(loser_hks)
            winner_name = str(row.get("Player Name", "")).strip()
            loser_name = str(row.get("Opponent Name", "")).strip()
            winner_team = str(row.get("Team", "")).strip()
            loser_team = str(row.get("Opponent Team", "")).strip()
            division = row.get("Division", "")

            def ensure_player(key, name, team):
                if key not in player_data:
                    initial_display = get_base_rating(division)
                    initial_rating = display_to_glicko(initial_display)
                    player_data[key] = {
                        "player": name,
                        "rating": float(initial_rating),
                        "rd": float(NEW_PLAYER_RD),
                        "volatility": DEFAULT_VOLATILITY,
                        "matches_played": 0,
                        "teams": {team} if team else set(),
                        "last_period": period_id - 1,
                    }
                    logging.info(
                        f"New player: '{name}' (HKS {key}) initialized to {initial_display:.1f} for division {division}"
                    )
                else:
                    player_record = player_data[key]
                    if team:
                        player_record["teams"].add(team)
                    if not player_record["player"] and name:
                        player_record["player"] = name
                    if player_record["last_period"] is None:
                        player_record["last_period"] = period_id - 1

            ensure_player(winner_key, winner_name, winner_team)
            ensure_player(loser_key, loser_name, loser_team)

            period_players.add(winner_key)
            period_players.add(loser_key)

            winner_score, loser_score = score_to_fraction(row.get("Score", ""))
            period_entries.append(
                {
                    "index": idx,
                    "winner_key": winner_key,
                    "loser_key": loser_key,
                    "winner_score": winner_score,
                    "loser_score": loser_score,
                    "date": row.get("Match Date"),
                }
            )

        if not period_entries:
            continue

        period_snapshot = {}
        for key in period_players:
            record = player_data[key]
            if record["last_period"] is None:
                record["last_period"] = period_id - 1
            gap_periods = period_id - record["last_period"] - 1
            if gap_periods > 0:
                inflate_rd_for_inactivity(record, gap_periods)
            period_snapshot[key] = {
                "rating": record["rating"],
                "rd": record["rd"],
                "volatility": record["volatility"],
                "matches_played": record["matches_played"],
            }

        player_matches = defaultdict(list)
        player_match_counts = defaultdict(int)
        period_play_counts = defaultdict(int)
        period_match_entries = []

        for entry in period_entries:
            player_matches[entry["winner_key"]].append(
                (entry["loser_key"], entry["winner_score"])
            )
            player_matches[entry["loser_key"]].append(
                (entry["winner_key"], entry["loser_score"])
            )
            player_match_counts[entry["winner_key"]] += 1
            player_match_counts[entry["loser_key"]] += 1

            winner_pre = glicko_to_display(period_snapshot[entry["winner_key"]]["rating"])
            loser_pre = glicko_to_display(period_snapshot[entry["loser_key"]]["rating"])

            period_play_counts[entry["winner_key"]] += 1
            period_play_counts[entry["loser_key"]] += 1
            winner_post_matches = period_snapshot[entry["winner_key"]]["matches_played"] + period_play_counts[entry["winner_key"]]
            loser_post_matches = period_snapshot[entry["loser_key"]]["matches_played"] + period_play_counts[entry["loser_key"]]

            match_entry = {
                "index": entry["index"],
                "winner_key": entry["winner_key"],
                "loser_key": entry["loser_key"],
                "winner_pre_rating": winner_pre,
                "loser_pre_rating": loser_pre,
                "winner_post_rating": None,
                "loser_post_rating": None,
                "winner_post_matches": winner_post_matches,
                "loser_post_matches": loser_post_matches,
                "winner_score": entry["winner_score"],
                "loser_score": entry["loser_score"],
            }
            period_match_entries.append(match_entry)
            match_entries.append(match_entry)

        updates = {}
        for key in period_players:
            snapshot = period_snapshot[key]
            matches = player_matches.get(key, [])
            mu = rating_to_mu(snapshot["rating"])
            phi = rd_to_phi(snapshot["rd"])
            sigma = snapshot["volatility"]

            if not matches:
                phi = math.sqrt(phi ** 2 + sigma ** 2)
                updates[key] = {
                    "rating": snapshot["rating"],
                    "rd": min(phi_to_rd(phi), MAX_RD),
                    "volatility": sigma,
                    "matches_played": snapshot["matches_played"],
                }
                continue

            v_inv = 0.0
            delta_sum = 0.0
            for opponent_key, actual_score in matches:
                opponent_snapshot = period_snapshot[opponent_key]
                mu_j = rating_to_mu(opponent_snapshot["rating"])
                phi_j = rd_to_phi(opponent_snapshot["rd"])
                g_phi_j = g(phi_j)
                expected = _stable_logistic(g_phi_j * (mu - mu_j))
                v_inv += (g_phi_j ** 2) * expected * (1.0 - expected)
                delta_sum += g_phi_j * (actual_score - expected)

            if v_inv == 0:
                updates[key] = {
                    "rating": snapshot["rating"],
                    "rd": snapshot["rd"],
                    "volatility": sigma,
                    "matches_played": snapshot["matches_played"] + player_match_counts.get(key, 0),
                }
                continue

            v = 1.0 / v_inv
            delta_sum *= PERFORMANCE_SCALE
            delta = v * delta_sum
            sigma_prime = update_volatility(phi, sigma, delta, v)
            sigma_prime = max(MIN_VOLATILITY, min(sigma_prime, MAX_VOLATILITY))
            phi_star = math.sqrt(phi ** 2 + sigma_prime ** 2)
            phi_prime = 1.0 / math.sqrt((1.0 / (phi_star ** 2)) + (1.0 / v))
            mu_prime = mu + (phi_prime ** 2) * delta_sum

            updates[key] = {
                "rating": mu_to_rating(mu_prime),
                "rd": min(max(phi_to_rd(phi_prime), MIN_RD), MAX_RD),
                "volatility": sigma_prime,
                "matches_played": snapshot["matches_played"] + player_match_counts.get(key, 0),
            }

        for key, update in updates.items():
            player_record = player_data[key]
            player_record["rating"] = update["rating"]
            player_record["rd"] = update["rd"]
            player_record["volatility"] = update["volatility"]
            player_record["matches_played"] = update["matches_played"]
            player_record["last_period"] = period_id

        updated_display_cache = {
            key: glicko_to_display(update["rating"])
            for key, update in updates.items()
        }

        for entry in period_match_entries:
            winner_key = entry["winner_key"]
            loser_key = entry["loser_key"]
            entry["winner_post_rating"] = updated_display_cache.get(
                winner_key,
                glicko_to_display(player_data[winner_key]["rating"]),
            )
            entry["loser_post_rating"] = updated_display_cache.get(
                loser_key,
                glicko_to_display(player_data[loser_key]["rating"]),
            )

        matches_processed += len(period_entries)

    if matches_processed == 0:
        logging.warning("No matches processed after filtering")
        return

    for entry in match_entries:
        idx = entry["index"]
        df.at[idx, "Pre Winner Rating"] = entry["winner_pre_rating"]
        df.at[idx, "Pre Loser Rating"] = entry["loser_pre_rating"]
        df.at[idx, "Post Winner Rating"] = entry["winner_post_rating"]
        df.at[idx, "Post Loser Rating"] = entry["loser_post_rating"]
        df.at[idx, "Winner Matches Played"] = entry["winner_post_matches"]
        df.at[idx, "Loser Matches Played"] = entry["loser_post_matches"]

    df_wins = df[df["Result"] == "Win"].copy()
    df_wins = df_wins.dropna(subset=["Pre Winner Rating", "Post Winner Rating"])
    df_wins["rel_diff"] = (
        (df_wins["Post Winner Rating"].astype(float) - df_wins["Pre Winner Rating"].astype(float))
        / df_wins["Pre Winner Rating"].astype(float)
        * 100.0
    )

    updated_csv = current_season_folder / "combined_player_results_df_updated.csv"
    df_wins.to_csv(updated_csv, index=False)
    logging.info(f"Saved updated matches to {updated_csv}")

    final_rows = []
    for key, info in player_data.items():
        final_rows.append(
            {
                "Player": info.get("player", ""),
                "HKS_No": key,
                "Teams": ", ".join(sorted(info.get("teams", []))),
                "Final Rating": glicko_to_display(info["rating"]),
                "Matches Played": info.get("matches_played", 0),
                "Glicko Rating": info["rating"],
                "Glicko RD": info["rd"],
                "Glicko Volatility": info["volatility"],
            }
        )

    final_df = pd.DataFrame(final_rows)
    final_df.sort_values("Final Rating", ascending=False, inplace=True, ignore_index=True)

    ratio_csv = current_season_folder / "ratio_results.csv"
    final_df.to_csv(ratio_csv, index=False)
    logging.info("=" * 70)
    logging.info(f"Processed {matches_processed} matches")
    logging.info(f"Updated matches CSV => {updated_csv}")
    logging.info(f"Final rating results => {ratio_csv}")
    logging.info("=" * 70)
    logging.info("PLAYER RATINGS ALGORITHM COMPLETED")


if __name__ == "__main__":
    process_ratings_algorithm()
