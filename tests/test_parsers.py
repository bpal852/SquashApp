"""
Unit tests for parser functions.

Tests all the core parsing functions with edge cases including:
- CR/WO/NA handling
- Malformed strings
- Empty lists
- Type errors
"""

import pytest
import pandas as pd
from parsers import (
    parse_result,
    split_overall_score,
    normalize_rubber,
    determine_winner,
    count_games_won,
    count_valid_matches,
    _parse_summary_row_text,
    home_team_won
)


class TestParseResult:
    """Test the parse_result function."""
    
    def test_normal_result(self):
        result = "3-2(3-1,1-3,3-2,3-1,1-3)"
        overall, rubbers = parse_result(result)
        assert overall == "3-2"
        assert rubbers == ["3-1", "1-3", "3-2", "3-1", "1-3"]
    
    def test_result_with_spaces(self):
        result = " 3-1 ( 3-0 , 0-3 , 3-1 ) "
        overall, rubbers = parse_result(result)
        assert overall == "3-1"
        assert rubbers == ["3-0", "0-3", "3-1"]
    
    def test_result_with_wo_cr(self):
        result = "3-0(3-1,WO,CR)"
        overall, rubbers = parse_result(result)
        assert overall == "3-0"
        assert rubbers == ["3-1", "WO", "CR"]
    
    def test_empty_string(self):
        overall, rubbers = parse_result("")
        assert pd.isna(overall)
        assert rubbers == []
    
    def test_none_input(self):
        overall, rubbers = parse_result(None)
        assert pd.isna(overall)
        assert rubbers == []
    
    def test_no_brackets(self):
        result = "3-2"
        overall, rubbers = parse_result(result)
        assert pd.isna(overall)
        assert rubbers == []
    
    def test_malformed_brackets(self):
        result = "3-2(3-1,1-3"  # missing closing bracket
        overall, rubbers = parse_result(result)
        assert overall == "3-2"
        assert rubbers == ["3-1", "1-3"]
    
    def test_empty_rubbers(self):
        result = "3-0()"
        overall, rubbers = parse_result(result)
        assert overall == "3-0"
        assert rubbers == []


class TestSplitOverallScore:
    """Test the split_overall_score function."""
    
    def test_normal_score(self):
        home, away = split_overall_score("3-2")
        assert home == 3
        assert away == 2
    
    def test_zero_score(self):
        home, away = split_overall_score("0-0")
        assert home == 0
        assert away == 0
    
    def test_high_score(self):
        home, away = split_overall_score("5-0")
        assert home == 5
        assert away == 0
    
    def test_empty_string(self):
        home, away = split_overall_score("")
        assert home == 0
        assert away == 0
    
    def test_none_input(self):
        home, away = split_overall_score(None)
        assert home == 0
        assert away == 0
    
    def test_no_dash(self):
        home, away = split_overall_score("32")
        assert home == 0
        assert away == 0
    
    def test_non_numeric(self):
        home, away = split_overall_score("a-b")
        assert home == 0
        assert away == 0
    
    def test_multiple_dashes(self):
        home, away = split_overall_score("3-2-1")
        assert home == 0
        assert away == 0


class TestNormalizeRubber:
    """Test the normalize_rubber function."""
    
    def test_normal_score(self):
        assert normalize_rubber("3-1") == "3-1"
    
    def test_wo_variations(self):
        assert normalize_rubber("W/O") == "WO"
        assert normalize_rubber("w/o") == "WO"
        assert normalize_rubber("WO") == "WO"
        assert normalize_rubber(" wo ") == "WO"
    
    def test_cr_variations(self):
        assert normalize_rubber("CR") == "CR"
        assert normalize_rubber("cr") == "CR"
        assert normalize_rubber(" CR ") == "CR"
        assert normalize_rubber("CONCEDED") == "CR"
    
    def test_na_variations(self):
        assert normalize_rubber("NA") == "NA"
        assert normalize_rubber("N/A") == "NA"
        assert normalize_rubber("") == "NA"
        assert normalize_rubber(" ") == "NA"
    
    def test_none_input(self):
        assert normalize_rubber(None) == "NA"
    
    def test_pd_na_input(self):
        assert normalize_rubber(pd.NA) == "NA"
    
    def test_numeric_input(self):
        assert normalize_rubber(123) == "123"


class TestDetermineWinner:
    """Test the determine_winner function."""
    
    def test_home_wins(self):
        winner = determine_winner("3-1", "Home Team", "Away Team")
        assert winner == "Home Team"
    
    def test_away_wins(self):
        winner = determine_winner("1-3", "Home Team", "Away Team")
        assert winner == "Away Team"
    
    def test_tie_score(self):
        winner = determine_winner("2-2", "Home Team", "Away Team")
        assert winner == "Away Team"  # away wins ties
    
    def test_cr_rubber(self):
        winner = determine_winner("CR", "Home Team", "Away Team")
        assert pd.isna(winner)
    
    def test_wo_rubber(self):
        winner = determine_winner("WO", "Home Team", "Away Team")
        assert pd.isna(winner)
    
    def test_na_rubber(self):
        winner = determine_winner("NA", "Home Team", "Away Team")
        assert pd.isna(winner)
    
    def test_none_input(self):
        winner = determine_winner(None, "Home Team", "Away Team")
        assert pd.isna(winner)
    
    def test_malformed_score(self):
        winner = determine_winner("abc", "Home Team", "Away Team")
        assert pd.isna(winner)


class TestCountGamesWon:
    """Test the count_games_won function."""
    
    def test_normal_match(self):
        row = pd.Series({
            'Rubbers': ['3-1', '1-3', '3-2'],
            'Overall Score': '2-1'
        })
        home, away = count_games_won(row)
        assert home == 7  # 3+1+3
        assert away == 6  # 1+3+2
    
    def test_match_with_wo(self):
        row = pd.Series({
            'Rubbers': ['3-1', 'WO', '3-2'],
            'Overall Score': '3-0'
        })
        home, away = count_games_won(row)
        assert home == 9  # 3+3+3 (WO gets allocated 3-0 to winner)
        assert away == 3  # 1+0+2
    
    def test_match_with_cr(self):
        row = pd.Series({
            'Rubbers': ['3-1', 'CR', '1-3'],
            'Overall Score': '2-1'
        })
        home, away = count_games_won(row)
        assert home == 7  # 3+3+1 (CR gets allocated 3-0 to winner)
        assert away == 4  # 1+0+3
    
    def test_empty_rubbers(self):
        row = pd.Series({
            'Rubbers': [],
            'Overall Score': '0-0'
        })
        home, away = count_games_won(row)
        assert home == 0
        assert away == 0
    
    def test_missing_rubbers(self):
        row = pd.Series({
            'Overall Score': '2-1'
        })
        home, away = count_games_won(row)
        assert home == 0
        assert away == 0
    
    def test_non_list_rubbers(self):
        row = pd.Series({
            'Rubbers': "3-1,1-3",  # string instead of list
            'Overall Score': '1-1'
        })
        home, away = count_games_won(row)
        assert home == 0
        assert away == 0
    
    def test_malformed_rubber_scores(self):
        row = pd.Series({
            'Rubbers': ['3-1', 'abc', '1-3'],
            'Overall Score': '2-1'
        })
        home, away = count_games_won(row)
        assert home == 4  # 3+1
        assert away == 4  # 1+3


class TestCountValidMatches:
    """Test the count_valid_matches function."""
    
    def test_normal_matches(self):
        df = pd.DataFrame({
            'Home Team': ['Team A', 'Team B'],
            'Away Team': ['Team B', 'Team C'],
            'Rubbers': [['3-1', '1-3'], ['3-0', '0-3']]
        })
        counts = count_valid_matches(df, 0)  # First rubber
        expected = {'Team A': 1, 'Team B': 2, 'Team C': 1}
        assert counts == expected
    
    def test_with_invalid_rubbers(self):
        df = pd.DataFrame({
            'Home Team': ['Team A', 'Team B', 'Team C'],
            'Away Team': ['Team B', 'Team C', 'Team A'],
            'Rubbers': [['3-1', '1-3'], ['WO', '0-3'], ['CR', 'NA']]
        })
        counts = count_valid_matches(df, 0)  # First rubber
        expected = {'Team A': 1, 'Team B': 1}  # Only first match valid
        assert counts == expected
    
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        counts = count_valid_matches(df, 0)
        assert counts == {}
    
    def test_rubber_index_out_of_range(self):
        df = pd.DataFrame({
            'Home Team': ['Team A'],
            'Away Team': ['Team B'],
            'Rubbers': [['3-1']]  # Only one rubber
        })
        counts = count_valid_matches(df, 5)  # Ask for 6th rubber
        assert counts == {}


class TestParseSummaryRowText:
    """Test the _parse_summary_row_text function."""
    
    def test_normal_row(self):
        result = _parse_summary_row_text("Physical Chess 5 4 1 12")
        assert result == ["Physical Chess", 5, 4, 1, 12]
    
    def test_row_with_extra_spaces(self):
        result = _parse_summary_row_text("Team   Name   10   8   2   24")
        assert result == ["Team Name", 10, 8, 2, 24]  # Spaces are normalized
    
    def test_team_name_with_numbers(self):
        result = _parse_summary_row_text("Team 123 5 4 1 12")
        assert result == ["Team 123", 5, 4, 1, 12]
    
    def test_header_row(self):
        result = _parse_summary_row_text("Team Played Won Lost Points")
        assert result is None
    
    def test_played_header_variation(self):
        result = _parse_summary_row_text("playedwonlostpoints")
        assert result is None
    
    def test_empty_string(self):
        result = _parse_summary_row_text("")
        assert result is None
    
    def test_none_input(self):
        result = _parse_summary_row_text(None)
        assert result is None
    
    def test_insufficient_numbers(self):
        result = _parse_summary_row_text("Team Name 5 4")
        assert result is None
    
    def test_no_numbers(self):
        result = _parse_summary_row_text("Team Name")
        assert result is None


class TestHomeTeamWon:
    """Test the home_team_won function."""
    
    def test_home_wins_by_score(self):
        row = pd.Series({
            'Home Score': 3,
            'Away Score': 1,
            'Home Games Won': 9,
            'Away Games Won': 5
        })
        assert home_team_won(row) == 'Home'
    
    def test_away_wins_by_score(self):
        row = pd.Series({
            'Home Score': 1,
            'Away Score': 3,
            'Home Games Won': 5,
            'Away Games Won': 9
        })
        assert home_team_won(row) == 'Away'
    
    def test_tie_score_home_wins_games(self):
        row = pd.Series({
            'Home Score': 2,
            'Away Score': 2,
            'Home Games Won': 8,
            'Away Games Won': 6
        })
        assert home_team_won(row) == 'Home'
    
    def test_tie_score_away_wins_games(self):
        row = pd.Series({
            'Home Score': 2,
            'Away Score': 2,
            'Home Games Won': 6,
            'Away Games Won': 8
        })
        assert home_team_won(row) == 'Away'
    
    def test_complete_tie(self):
        row = pd.Series({
            'Home Score': 2,
            'Away Score': 2,
            'Home Games Won': 7,
            'Away Games Won': 7
        })
        assert home_team_won(row) == 'Ignore'
    
    def test_missing_data(self):
        row = pd.Series({})
        assert home_team_won(row) == 'Ignore'
    
    def test_malformed_data(self):
        row = pd.Series({
            'Home Score': 'abc',
            'Away Score': 2,
            'Home Games Won': 7,
            'Away Games Won': 7
        })
        assert home_team_won(row) == 'Ignore'


if __name__ == "__main__":
    pytest.main([__file__])