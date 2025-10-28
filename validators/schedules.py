"""
Validator for schedules and results data.
"""

import pandas as pd
from .base import BaseValidator, ValidationResult


class SchedulesValidator(BaseValidator):
    """Validates schedules and results data."""
    
    REQUIRED_COLUMNS = {'Home Team', 'vs', 'Away Team', 'Venue', 'Time', 
                       'Result', 'Match Week', 'Date'}
    EXPECTED_MIN_MATCHES = 10  # Even small divisions have at least 10 matches
    EXPECTED_MAX_MATCHES = 500  # Large divisions might have up to 500 matches
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate schedules DataFrame."""
        result = ValidationResult(
            validator_name='SchedulesValidator',
            data_type='schedules',
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
        critical_columns = {'Home Team', 'Away Team', 'vs', 'Match Week'}
        self._check_no_null_values(df, critical_columns, result)
        
        # Check data types
        expected_types = {
            'Home Team': 'str',
            'vs': 'str',
            'Away Team': 'str',
            'Venue': 'str',
            'Time': 'str',
            'Result': 'str',
            'Match Week': 'int',
            'Date': 'datetime',
        }
        self._check_data_types(df, expected_types, result)
        
        # Check row count is reasonable
        self._check_row_count_range(df, self.EXPECTED_MIN_MATCHES,
                                    self.EXPECTED_MAX_MATCHES, result)
        
        # Check "vs" column always equals "vs"
        if 'vs' in df.columns:
            non_vs = df[df['vs'] != 'vs']
            if len(non_vs) > 0:
                result.add_error('vs', f'{len(non_vs)} rows have vs != "vs"')
        
        # Check for same home and away team
        if 'Home Team' in df.columns and 'Away Team' in df.columns:
            same_team = df[df['Home Team'] == df['Away Team']]
            if len(same_team) > 0:
                result.add_error('teams',
                    f'{len(same_team)} matches have same home and away team')
        
        # Check Match Week values
        if 'Match Week' in df.columns:
            self._check_value_range(df, 'Match Week', 0, 30, result)
            
            week_counts = df['Match Week'].value_counts()
            result.add_info('Match Week',
                f'Data spans {week_counts.nunique()} weeks (weeks {df["Match Week"].min()}-{df["Match Week"].max()})')
        
        # Check Result format (should be like "3-2" or empty)
        if 'Result' in df.columns:
            played_matches = df[df['Result'].notna() & (df['Result'] != '')]
            total_played = len(played_matches)
            total_unplayed = len(df) - total_played
            
            result.add_info('Result',
                f'{total_played} matches played, {total_unplayed} upcoming/unplayed')
            
            if total_played > 0:
                # Check result format
                invalid_results = played_matches[
                    ~played_matches['Result'].str.match(r'^\d+-\d+$|^WO$|^CR$', na=False)
                ]
                if len(invalid_results) > 0:
                    result.add_warning('Result',
                        f'{len(invalid_results)} results have unexpected format')
        
        # Check Date values
        if 'Date' in df.columns:
            try:
                date_col = pd.to_datetime(df['Date'], errors='coerce')
                invalid_dates = date_col.isna().sum()
                if invalid_dates > 0:
                    result.add_warning('Date',
                        f'{invalid_dates} rows have invalid dates')
                
                # Check for dates in reasonable range (within 2 years)
                current_year = int(self.year.split('-')[0])
                valid_dates = date_col.dropna()
                if len(valid_dates) > 0:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    
                    if min_date.year < current_year - 1:
                        result.add_warning('Date',
                            f'Dates start from {min_date.date()} (before season)')
                    
                    if max_date.year > current_year + 1:
                        result.add_warning('Date',
                            f'Dates extend to {max_date.date()} (after season)')
            except Exception as e:
                result.add_warning('Date', f'Error validating dates: {e}')
        
        self._log_result(result)
        return result
