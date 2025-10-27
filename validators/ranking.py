"""
Validator for player ranking data.
"""

import pandas as pd
from .base import BaseValidator, ValidationResult


class RankingValidator(BaseValidator):
    """Validates player ranking data."""
    
    REQUIRED_COLUMNS = {'Player', 'Games', 'Won', 'Lost', 'W %', 'Team'}
    SCORE_COLUMNS = {'5-0', '4-1', '3-2', '2-3', '1-4', '0-5'}
    EXPECTED_MIN_PLAYERS = 20  # Most divisions have at least 20 players
    EXPECTED_MAX_PLAYERS = 200  # Most divisions have at most 200 players
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate ranking DataFrame."""
        result = ValidationResult(
            validator_name='RankingValidator',
            data_type='ranking',
            is_valid=True
        )
        
        # Store metadata
        result.metadata['league_id'] = self.league_id
        result.metadata['year'] = self.year
        if self.division:
            result.metadata['division'] = self.division
        
        # Check not empty
        if not self._check_not_empty(df, result):
            return result
        
        result.metadata['row_count'] = len(df)
        
        # Check required columns
        if not self._check_required_columns(df, self.REQUIRED_COLUMNS, result):
            return result
        
        # Check no null values in critical columns
        critical_columns = {'Player', 'Games', 'Won', 'Lost', 'Team'}
        self._check_no_null_values(df, critical_columns, result)
        
        # Check data types
        expected_types = {
            'Player': 'str',
            'Games': 'int',
            'Won': 'int',
            'Lost': 'int',
            'W %': 'float',
            'Team': 'str',
        }
        self._check_data_types(df, expected_types, result)
        
        # Check row count is reasonable
        self._check_row_count_range(df, self.EXPECTED_MIN_PLAYERS,
                                    self.EXPECTED_MAX_PLAYERS, result)
        
        # Check value ranges
        if 'Games' in df.columns:
            self._check_value_range(df, 'Games', 0, 100, result)
            
            # Check for players with 0 games
            zero_games = (df['Games'] == 0).sum()
            if zero_games > 0:
                result.add_warning('Games',
                    f'{zero_games} players have 0 games (should they be excluded?)')
        
        if 'Won' in df.columns:
            self._check_value_range(df, 'Won', 0, 100, result)
        
        if 'Lost' in df.columns:
            self._check_value_range(df, 'Lost', 0, 100, result)
        
        if 'W %' in df.columns:
            self._check_value_range(df, 'W %', 0.0, 100.0, result)
            
            # Check for win% = 100 (unbeaten players)
            unbeaten = (df['W %'] == 100.0).sum()
            if unbeaten > 0:
                result.add_info('W %', f'{unbeaten} unbeaten players')
        
        # Check mathematical consistency: Won + Lost should equal Games
        if all(col in df.columns for col in ['Games', 'Won', 'Lost']):
            inconsistent = df[df['Games'] != df['Won'] + df['Lost']]
            if len(inconsistent) > 0:
                result.add_error('consistency',
                    f'{len(inconsistent)} players have Won+Lost != Games')
        
        # Check win percentage calculation
        if all(col in df.columns for col in ['Won', 'Games', 'W %']):
            # Filter out players with 0 games
            with_games = df[df['Games'] > 0].copy()
            if len(with_games) > 0:
                expected_pct = (with_games['Won'] / with_games['Games'] * 100).round(1)
                actual_pct = with_games['W %'].round(1)
                mismatch = (expected_pct != actual_pct).sum()
                
                if mismatch > 0:
                    result.add_warning('W %',
                        f'{mismatch} players have incorrect win percentage')
        
        # Check score distribution columns if present
        score_cols_present = [col for col in self.SCORE_COLUMNS if col in df.columns]
        if score_cols_present:
            result.add_info('score_columns',
                f'Score distribution columns present: {score_cols_present}')
            
            # Check that score columns sum to Games
            if 'Games' in df.columns:
                score_sum = df[score_cols_present].sum(axis=1)
                games = df['Games']
                mismatch = (score_sum != games).sum()
                
                if mismatch > 0:
                    result.add_warning('score_distribution',
                        f'{mismatch} players have score distribution sum != Games')
        
        # Check for duplicate players
        if 'Player' in df.columns:
            duplicates = df['Player'].duplicated().sum()
            if duplicates > 0:
                result.add_warning('Player',
                    f'{duplicates} duplicate player names (same name, different teams?)')
        
        # Team distribution
        if 'Team' in df.columns:
            team_counts = df['Team'].value_counts()
            result.add_info('Team',
                f'{team_counts.nunique()} unique teams, avg {team_counts.mean():.1f} players/team')
            
            # Check for teams with very few players
            small_teams = team_counts[team_counts < 3]
            if len(small_teams) > 0:
                result.add_warning('Team',
                    f'{len(small_teams)} teams have < 3 players: {small_teams.index.tolist()}')
        
        self._log_result(result)
        return result
