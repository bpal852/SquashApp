"""
Validator for team summary data.
"""

import pandas as pd
from .base import BaseValidator, ValidationResult


class SummaryValidator(BaseValidator):
    """Validates team summary/standings data."""
    
    REQUIRED_COLUMNS = {'Team', 'Played', 'Won', 'Lost', 'Points'}
    EXPECTED_MIN_TEAMS = 4
    EXPECTED_MAX_TEAMS = 20
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate summary DataFrame."""
        result = ValidationResult(
            validator_name='SummaryValidator',
            data_type='summary',
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
        self._check_no_null_values(df, self.REQUIRED_COLUMNS, result)
        
        # Check data types
        expected_types = {
            'Team': 'str',
            'Played': 'int',
            'Won': 'int',
            'Lost': 'int',
            'Points': 'int',
        }
        self._check_data_types(df, expected_types, result)
        
        # Check row count is reasonable
        self._check_row_count_range(df, self.EXPECTED_MIN_TEAMS,
                                    self.EXPECTED_MAX_TEAMS, result)
        
        # Check for duplicate teams
        if 'Team' in df.columns:
            duplicates = df['Team'].duplicated().sum()
            if duplicates > 0:
                result.add_error('Team', f'Found {duplicates} duplicate teams')
        
        # Check value ranges
        if 'Played' in df.columns:
            self._check_value_range(df, 'Played', 0, 100, result)
            
            # Check if all teams have 0 played
            if df['Played'].max() == 0:
                result.add_warning('Played', 'All teams have 0 games played (season not started?)')
        
        if 'Won' in df.columns:
            self._check_value_range(df, 'Won', 0, 100, result)
        
        if 'Lost' in df.columns:
            self._check_value_range(df, 'Lost', 0, 100, result)
        
        if 'Points' in df.columns:
            self._check_value_range(df, 'Points', 0, 500, result)
        
        # Check mathematical consistency: Won + Lost should equal Played
        if all(col in df.columns for col in ['Played', 'Won', 'Lost']):
            inconsistent = df[df['Played'] != df['Won'] + df['Lost']]
            if len(inconsistent) > 0:
                result.add_error('consistency',
                    f'{len(inconsistent)} teams have Won+Lost != Played')
                for idx, row in inconsistent.iterrows():
                    result.add_error('consistency',
                        f"Team '{row['Team']}': {row['Won']}+{row['Lost']} != {row['Played']}")
        
        # Check points calculation (typically 5 points per win)
        if all(col in df.columns for col in ['Won', 'Points']):
            expected_points = df['Won'] * 5
            points_mismatch = df[df['Points'] != expected_points]
            if len(points_mismatch) > 0:
                result.add_warning('Points',
                    f'{len(points_mismatch)} teams have points != (wins * 5)')
        
        self._log_result(result)
        return result
