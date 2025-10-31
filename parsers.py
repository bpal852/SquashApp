"""
Parser functions for squash league data processing.

These functions handle parsing and processing of scraped data from the HK Squash website.
"""

import logging
import re

import pandas as pd

# Global regex for summary row parsing
_SUMMARY_NUMS_RE = re.compile(r"^(.*?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$")


def parse_result(result):
    """
    Function to parse the 'result' string from scraped data.

    Args:
        result (str): Result string in format "3-2(3-1,1-3,3-2,3-1,1-3)"

    Returns:
        tuple: (overall_score, rubbers_list)

    Examples:
        >>> parse_result("3-2(3-1,1-3,3-2,3-1,1-3)")
        ('3-2', ['3-1', '1-3', '3-2', '3-1', '1-3'])
    """
    if not isinstance(result, str) or "(" not in result:
        return pd.NA, []

    try:
        overall, rubbers_part = result.split("(", 1)
        overall = overall.strip()

        # Find the closing bracket and extract content
        if ")" in rubbers_part:
            rubbers_part = rubbers_part[: rubbers_part.rfind(")")]

        rubbers = [r.strip() for r in rubbers_part.split(",") if r.strip()]
        return overall, rubbers
    except Exception as e:
        logging.warning(f"Error parsing result '{result}': {e}")
        return pd.NA, []


def split_overall_score(score):
    """
    Function to split the overall score and return home and away scores.

    Args:
        score (str): Score string like "3-2"

    Returns:
        tuple: (home_score, away_score) as integers

    Examples:
        >>> split_overall_score("3-2")
        (3, 2)
    """
    if not isinstance(score, str) or "-" not in score:
        return 0, 0

    try:
        home_score, away_score = map(int, score.split("-"))
        return home_score, away_score
    except (ValueError, TypeError) as e:
        logging.warning(f"Error splitting score '{score}': {e}")
        return 0, 0


def normalize_rubber(s):
    """
    Normalize rubber score strings to handle various formats.

    Args:
        s (str): Raw rubber score string

    Returns:
        str: Normalized rubber score

    Examples:
        >>> normalize_rubber("w/o")
        'WO'
        >>> normalize_rubber(" CR ")
        'CR'
    """
    if pd.isna(s) or s is None:
        return "NA"

    s = str(s).strip().upper()
    if s in ["W/O", "WO"]:
        return "WO"
    if s in ["CR", "CONCEDED"]:
        return "CR"
    if s in ["", "NA", "N/A"]:
        return "NA"

    return s


def determine_winner(rubber_score, home_team, away_team):
    """
    Function to determine the winner of a rubber.

    Args:
        rubber_score (str): Score like "3-1" or "CR" or "WO"
        home_team (str): Home team name
        away_team (str): Away team name

    Returns:
        str or pd.NA: Team name that won, or pd.NA if indeterminate
    """
    if pd.isna(rubber_score) or rubber_score in ["CR", "WO", "NA"]:
        return pd.NA

    try:
        home_score, away_score = map(int, str(rubber_score).split("-"))
        return home_team if home_score > away_score else away_team
    except (ValueError, TypeError, AttributeError):
        return pd.NA


def count_games_won(row):
    """
    Function to count the number of games won by each team in a match,
    handling walkovers (WO) and conceded rubbers (CR) by referring to the 'Overall Score'.

    Args:
        row (pd.Series): DataFrame row with 'Rubbers' and 'Overall Score' columns

    Returns:
        tuple: (home_games_won, away_games_won) as integers
    """
    home_games_won = 0
    away_games_won = 0

    try:
        # Get rubbers list, handle if it's not a list
        rubbers = row.get("Rubbers", [])
        if not isinstance(rubbers, list):
            rubbers = []

        # Calculate the games won from the rubbers, excluding 'CR' and 'WO'
        cr_wo_count = 0
        for rubber in rubbers:
            if rubber in ("CR", "WO", "NA"):
                cr_wo_count += 1
                continue
            try:
                home, away = map(int, str(rubber).split("-"))
                home_games_won += home
                away_games_won += away
            except (ValueError, TypeError, AttributeError):
                continue

        # Handle the 'WO' and 'CR' rubbers by referring to the 'Overall Score'
        if cr_wo_count > 0:
            overall_score = row.get("Overall Score", "0-0")
            try:
                home_overall_score, away_overall_score = map(int, str(overall_score).split("-"))

                # For each CR/WO rubber, award 3-0 to the overall winner
                for _ in range(cr_wo_count):
                    if home_overall_score > away_overall_score:
                        home_games_won += 3
                    else:
                        away_games_won += 3

            except (ValueError, TypeError, AttributeError):
                pass

    except Exception as e:
        logging.warning(f"Error in count_games_won for row: {e}")

    return int(home_games_won), int(away_games_won)


def count_valid_matches(df, rubber_index):
    """
    Function to count matches excluding 'NA', 'CR', and 'WO'.

    Args:
        df (pd.DataFrame): Results DataFrame
        rubber_index (int): Index of rubber to check (0-based)

    Returns:
        dict: Count of valid matches per team
    """
    valid_matches_count = {}

    if df.empty:
        return valid_matches_count

    for _, row in df.iterrows():
        rubbers = row.get("Rubbers", [])
        if not isinstance(rubbers, list) or rubber_index >= len(rubbers):
            continue

        rubber = rubbers[rubber_index]
        if pd.notna(rubber) and rubber not in ["NA", "CR", "WO"]:
            home_team = row.get("Home Team", "")
            away_team = row.get("Away Team", "")

            valid_matches_count[home_team] = valid_matches_count.get(home_team, 0) + 1
            valid_matches_count[away_team] = valid_matches_count.get(away_team, 0) + 1

    return valid_matches_count


def _parse_summary_row_text(txt):
    """
    Fallback parser: extract Team, Played, Won, Lost, Points from raw text.
    Handles cases like: 'Physical Chess 1 1 0 4' (with weird spacing).

    Args:
        txt (str): Raw text from summary table

    Returns:
        list or None: [team, played, won, lost, points] or None if no match
    """
    if not isinstance(txt, str):
        return None

    txt = " ".join(txt.split())
    m = _SUMMARY_NUMS_RE.match(txt)
    if not m:
        return None
    team = m.group(1).strip()
    p, w, l, pts = map(int, m.groups()[1:])
    # sanity check to avoid header lines like 'playedwonlostpoint'
    if not team or team.lower().startswith("played"):
        return None
    return [team, p, w, l, pts]


def home_team_won(row):
    """
    Function to determine whether the home team or away team won the match,
    using games won as a tiebreaker. If overall score and games won are equal,
    the match is ignored.

    Args:
        row (pd.Series): DataFrame row with score information

    Returns:
        str: 'Home', 'Away', or 'Ignore'
    """
    try:
        home_score = row.get("Home Score", 0)
        away_score = row.get("Away Score", 0)
        home_games = row.get("Home Games Won", 0)
        away_games = row.get("Away Games Won", 0)

        if home_score > away_score:
            return "Home"
        elif home_score < away_score:
            return "Away"
        else:
            # If overall scores are equal, use games won as tiebreaker
            if home_games > away_games:
                return "Home"
            elif home_games < away_games:
                return "Away"
            else:
                return "Ignore"
    except Exception:
        return "Ignore"
