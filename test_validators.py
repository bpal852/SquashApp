"""
Test script to demonstrate validators with sample data.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from validators import (
    TeamsValidator,
    SummaryValidator,
    SchedulesValidator,
    RankingValidator,
    PlayersValidator,
    ValidationReport,
)


def test_teams_validator():
    """Test TeamsValidator with sample data."""
    print("\n" + "="*70)
    print("Testing TeamsValidator")
    print("="*70)
    
    # Good data
    good_data = pd.DataFrame({
        'Team Name': ['Team A', 'Team B', 'Team C'],
        'Home': ['Venue 1', 'Venue 2', 'Venue 3'],
        'Convenor': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
    })
    
    validator = TeamsValidator(league_id="D00473", year="2025-2026", division="Test")
    result = validator.validate(good_data)
    print(result.summary())
    
    # Bad data - missing email @
    bad_data = pd.DataFrame({
        'Team Name': ['Team A', 'Team B'],
        'Home': ['Venue 1', 'Venue 2'],
        'Convenor': ['John Doe', 'Jane Smith'],
        'Email': ['john.email.com', 'jane@email.com'],  # First one missing @
    })
    
    result = validator.validate(bad_data)
    print(result.summary())


def test_summary_validator():
    """Test SummaryValidator with sample data."""
    print("\n" + "="*70)
    print("Testing SummaryValidator")
    print("="*70)
    
    # Good data
    good_data = pd.DataFrame({
        'Team': ['Team A', 'Team B', 'Team C'],
        'Played': [4, 4, 4],
        'Won': [3, 2, 1],
        'Lost': [1, 2, 3],
        'Points': [15, 10, 5],
    })
    
    validator = SummaryValidator(league_id="D00473", year="2025-2026", division="Test")
    result = validator.validate(good_data)
    print(result.summary())
    
    # Bad data - inconsistent
    bad_data = pd.DataFrame({
        'Team': ['Team A', 'Team B'],
        'Played': [4, 4],
        'Won': [3, 2],
        'Lost': [2, 2],  # Won + Lost != Played for Team A
        'Points': [15, 10],
    })
    
    result = validator.validate(bad_data)
    print(result.summary())


def test_schedules_validator():
    """Test SchedulesValidator with sample data."""
    print("\n" + "="*70)
    print("Testing SchedulesValidator")
    print("="*70)
    
    # Good data
    good_data = pd.DataFrame({
        'Home Team': ['Team A', 'Team B', 'Team C'],
        'vs': ['vs', 'vs', 'vs'],
        'Away Team': ['Team B', 'Team C', 'Team A'],
        'Venue': ['Venue 1', 'Venue 2', 'Venue 3'],
        'Time': ['19:00', '19:30', '20:00'],
        'Result': ['3-2', '', '4-1'],
        'Match Week': [1, 1, 2],
        'Date': pd.to_datetime(['2025-10-01', '2025-10-01', '2025-10-08']),
    })
    
    validator = SchedulesValidator(league_id="D00473", year="2025-2026", division="Test")
    result = validator.validate(good_data)
    print(result.summary())


def test_ranking_validator():
    """Test RankingValidator with sample data."""
    print("\n" + "="*70)
    print("Testing RankingValidator")
    print("="*70)
    
    # Good data
    good_data = pd.DataFrame({
        'Player': ['Alice', 'Bob', 'Charlie'],
        'Games': [10, 8, 6],
        'Won': [8, 4, 3],
        'Lost': [2, 4, 3],
        'W %': [80.0, 50.0, 50.0],
        '5-0': [2, 1, 0],
        '4-1': [3, 2, 1],
        '3-2': [3, 1, 2],
        '2-3': [1, 2, 2],
        '1-4': [1, 1, 1],
        '0-5': [0, 1, 0],
        'Team': ['Team A', 'Team B', 'Team C'],
    })
    
    validator = RankingValidator(league_id="D00473", year="2025-2026", division="Test")
    result = validator.validate(good_data)
    print(result.summary())


def test_players_validator():
    """Test PlayersValidator with sample data."""
    print("\n" + "="*70)
    print("Testing PlayersValidator")
    print("="*70)
    
    # Good data
    good_data = pd.DataFrame({
        'Order': [1, 2, 3, 1, 2],
        'Player': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
        'HKS No.': [12345, 23456, 34567, 45678, 56789],
        'Ranking': [150, 200, 250, 180, 220],
        'Points': [3.5, 2.8, 2.3, 3.2, 2.6],
        'Team': ['Team A', 'Team A', 'Team A', 'Team B', 'Team B'],
    })
    
    validator = PlayersValidator(league_id="D00473", year="2025-2026", division="Test")
    result = validator.validate(good_data)
    print(result.summary())


def test_validation_report():
    """Test ValidationReport with multiple validations."""
    print("\n" + "="*70)
    print("Testing ValidationReport")
    print("="*70)
    
    report = ValidationReport(output_dir=".", year="2025-2026")
    
    # Add some test validations
    teams_data = pd.DataFrame({
        'Team Name': ['Team A', 'Team B'],
        'Home': ['Venue 1', 'Venue 2'],
        'Convenor': ['John', 'Jane'],
        'Email': ['john@email.com', 'jane@email.com'],
    })
    
    validator = TeamsValidator(league_id="D00473", year="2025-2026", division="Test")
    result = validator.validate(teams_data)
    report.add_result(result)
    
    # Print summary
    report.print_summary()
    
    # Create error summary DataFrame
    df = report.create_error_summary_dataframe()
    print("\nError Summary DataFrame:")
    print(df)


if __name__ == '__main__':
    print("\n" + "🧪 VALIDATORS DEMONSTRATION" + "\n")
    print("This script demonstrates the validators with sample data.")
    print("Both good and bad data examples are shown.")
    
    test_teams_validator()
    test_summary_validator()
    test_schedules_validator()
    test_ranking_validator()
    test_players_validator()
    test_validation_report()
    
    print("\n✅ All validator demonstrations complete!\n")
